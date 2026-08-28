from __future__ import annotations

import torch
from torch import einsum
from torch.nn import Module, ModuleList, Parameter, ReLU

from einops.layers.torch import Rearrange

import einx

from bdh_cq.bdh_cq import (
    LinearNoBias,
    LayerNormNoParams,
    default,
    exists
)
from bdh_cq.rotary import apply_rotary_emb

# order-2 poly attention (https://arxiv.org/abs/2602.02422), adapted for linear attention

class HigherOrderBDHLayer(Module):
    def __init__(
        self,
        dim,
        *,
        heads,
        dim_queries_keys,
        qk_activation = ReLU(),
        ff_activation = ReLU(),
        include_pass1_mass = True,
        omit_self = True,
        eps = 1e-6
    ):
        super().__init__()
        dim_inner_qk = dim_queries_keys * heads

        self.to_qks = ModuleList([LinearNoBias(dim, dim_inner_qk) for _ in range(3)])

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.qk_activation = qk_activation

        self.include_pass1_mass = include_pass1_mass
        self.omit_self = omit_self
        self.eps = eps

        self.post_attn_norm = LayerNormNoParams(dim)
        self.post_ff_norm = LayerNormNoParams(dim)

        # the feedforward part, gates from the root (pass-2 query) stream

        self.proj_up = Parameter(torch.randn(heads, dim, dim_queries_keys) * 0.02)

        self.ff_act = ff_activation

        self.merge_heads = Rearrange('b h n d -> b n (h d)')
        self.proj_out = LinearNoBias(dim_queries_keys * heads, dim)

    def forward(
        self,
        tokens,
        memories = None,
        rotary_emb = None,
        return_memories = False
    ):
        # the three sparse qk streams, relu activated - the exact same features as BDH

        eps = self.eps

        q1, q2, q3 = map(
            lambda to_qk: self.split_heads(self.qk_activation(to_qk(tokens))),
            self.to_qks
        )

        # the root stream also gates the ff

        gates = q1

        # relative positions - applied to the queries and keys only, never the values

        if exists(rotary_emb):
            q1, q2, q3 = (apply_rotary_emb(rotary_emb, t) for t in (q1, q2, q3))

        # the values are the tokens, no projection

        v3 = tokens

        # pass 1 - kernel attention from q2 to q3, aggregating the tokens

        S3_local = einsum('b h n d, b n e -> b h d e', q3, v3)
        z3_local = q3.sum(dim = -2)

        S3 = S3_local if not exists(memories) else S3_local + memories[0]
        z3 = z3_local if not exists(memories) else z3_local + memories[1]

        msg_un = einsum('b h n d, b h d e -> b h n e', q2, S3)
        s23 = einsum('b h n d, b h d -> b h n', q2, z3)

        # omit attention to self

        if self.omit_self:
            self_terms = einsum('b h n d, b h n d -> b h n', q2, q3)
            msg_un = msg_un - einx.multiply('b h n, b n e -> b h n e', self_terms, v3)
            s23 = s23 - self_terms

        # pass 2 - kernel attention from q1 to q2, weighted by the pass-1 mass

        if self.include_pass1_mass:
            msg_norm = msg_un
            s23_norm = s23
        else:
            msg_norm = einx.divide('b h n e, b h n -> b h n e', msg_un, s23.clamp_min(eps))
            s23_norm = s23.clamp_min(eps)

        S12 = einsum('b h n d, b h n e -> b h d e', q2, msg_norm)
        z12 = einsum('b h n d, b h n -> b h d', q2, s23_norm)

        num = einsum('b h n d, b h d e -> b h n e', q1, S12)
        den = einsum('b h n d, b h d -> b h n', q1, z12)

        # omit attention to self - subtracted from the queries directly, the
        # self contribution to the pass-2 aggregate is a rank-one per position

        if self.omit_self:
            q1_dot_q2 = einsum('b h n d, b h n d -> b h n', q1, q2)
            num = num - einx.multiply('b h n, b h n e -> b h n e', q1_dot_q2, msg_norm)
            den = den - q1_dot_q2 * s23_norm

        out = einx.divide('b h n e, b h n -> b h n e', num, den + eps)

        # fully masked query rows (all keys masked) should output zero

        out = einx.where('b h n, b h n e, b h n e -> b h n e', den != 0., out, torch.zeros_like(out))

        # post attn norm

        attn_out = self.post_attn_norm(out)

        # the interesting ff glu variant - the root sparse input gates the projection

        projected = einsum('b h n d, h d e -> b h n e', attn_out, self.proj_up)

        projected = self.ff_act(projected * gates)

        out = self.merge_heads(projected)

        out = self.proj_out(out)

        out = self.post_ff_norm(out)

        # maybe return the local pass-1 key value stats as the fast weight memory

        if not return_memories:
            return out

        memories = (S3_local, z3_local)

        return out, memories

    # the order-2 memory is a pair of tensors per layer

    @staticmethod
    def combine_memories(new_memory, prev_memory):
        if exists(prev_memory):
            return tuple(l + p for l, p in zip(new_memory, prev_memory))

        return new_memory
