"""Protocol tests for BDHVideoReasoningWrapper. FakeVAE only, no MiniMax-H3."""

from __future__ import annotations

import sys

import pytest
import torch

from bdh_cq.bdh_cq import Memory
from bdh_cq.video import (
    BILLION_MODEL_KWARGS,
    CANVAS_START,
    EOS,
    IN,
    N_CELLS,
    NUM_TOKENS,
    OUT,
    PAD,
    BDHVideoReasoningWrapper,
    apply_residual,
    composite_shift,
    encode_task,
    integrate_u,
    lock_t0,
    make_video_model,
    sample_task,
    warp_still,
)
from bdh_cq.video_probes import LAST_FRAME_L1, pixel_last_frame_l1
from bdh_cq.video_vae import (
    LATENT_CH,
    LATENT_HW,
    LATENT_T,
    FakeVideoVAE,
    decode_pixels,
    encode_mean_video,
)

H3_MARKERS = (
    "klvae",
    "minimax_h3_video_vae",
    "vae_cnn",
    "vae_vit",
    "vae_processor",
    "vae_module",
)


@pytest.fixture
def vae():
    return FakeVideoVAE(seed=0)


@pytest.fixture
def wrapper():
    return BDHVideoReasoningWrapper(make_video_model(scale="tiny"))


@pytest.fixture
def task_latents(vae):
    return encode_task(vae, sample_task("identity", seed=0))


def test_protocol_does_not_import_h3():
    assert [name for name in sys.modules if any(m in name for m in H3_MARKERS)] == []


def test_billion_scale_is_about_one_billion_tied_block_params():
    dim = BILLION_MODEL_KWARGS["dim"]
    qk = BILLION_MODEL_KWARGS["dim_qk_heads"]
    heads = BILLION_MODEL_KWARGS["heads"]
    assert qk % heads == 0
    # to_qk + proj_up + proj_out; depth is weight-tied.
    assert 0.95e9 < 3 * dim * qk < 1.10e9


def test_ingest_task_canvas_is_h0(wrapper, task_latents):
    memories = wrapper.ingest_task(task_latents)
    assert memories.embeds.shape[-2] == N_CELLS
    assert memories.embeds.shape == (1, N_CELLS, wrapper.bdh.dim)


def test_canvas_is_full_still_volume(wrapper, task_latents):
    z = task_latents["query_in"]
    n = int(z.shape[2] * z.shape[3] * z.shape[4])
    widths = []
    original = wrapper.patch_in.forward

    def spy(cells):
        widths.append(cells.shape[-2])
        return original(cells)

    wrapper.patch_in.forward = spy
    memories = wrapper.ingest_task(task_latents)
    assert n in widths
    assert n == N_CELLS
    assert memories.embeds.shape[-2] == n
    spatial = z.shape[3] * z.shape[4]
    assert spatial not in widths or widths.count(n) >= 1


def test_ingest_writes_s(wrapper, task_latents):
    memories = wrapper.ingest_task(task_latents)
    assert memories.fast_weight_memories is not None
    assert any(
        layer is not None and layer.abs().sum() > 0
        for layer in memories.fast_weight_memories
    )


def test_canvas_can_skip_writing_s(wrapper, task_latents):
    wrapper.canvas_update_memory = False
    skipped = wrapper.ingest_task(task_latents)
    wrapper.canvas_update_memory = True
    written = wrapper.ingest_task(task_latents)
    assert skipped.embeds.shape[-2] == N_CELLS
    assert written.embeds.shape[-2] == N_CELLS
    differ = False
    for left, right in zip(skipped.fast_weight_memories, written.fast_weight_memories):
        if left is None or right is None:
            continue
        if not torch.equal(left, right):
            differ = True
            break
    assert differ, "canvas update_memory=True should write extra mass into S"


def _clone_fast(memories: Memory):
    return [layer.detach().clone() for layer in memories.fast_weight_memories]


def test_s_frozen_during_reason(wrapper, task_latents):
    memories = wrapper.ingest_task(task_latents)
    before = _clone_fast(memories)
    wrapper.reason(memories, steps=3)
    after = memories.fast_weight_memories
    for left, right in zip(before, after):
        assert torch.equal(left, right)


def test_residual_list_grows_by_depth(wrapper, task_latents):
    memories = wrapper.ingest_task(task_latents)
    steps = 3
    _z, residual = wrapper.reason(memories, steps, return_residual=True)
    assert len(residual) == 1 + steps * wrapper.bdh.depth


def test_reason_output_shape(wrapper, task_latents):
    memories = wrapper.ingest_task(task_latents)
    z_hat = wrapper.reason(memories, steps=2)
    assert z_hat.shape == (1, LATENT_CH, LATENT_T, LATENT_HW, LATENT_HW)


def test_reason_rejects_raw_canvas(wrapper, task_latents):
    memories = wrapper.ingest_task(task_latents)
    with pytest.raises(TypeError, match="raw canvas"):
        wrapper.reason(memories.embeds, steps=1)
    bad = Memory(memories.tokens_seen, memories.embeds[:, :32], memories.fast_weight_memories)
    with pytest.raises(ValueError, match=str(N_CELLS)):
        wrapper.reason(bad, steps=1)


def test_vae_cells_never_index_token_embed(wrapper, task_latents):
    seen = []
    original = wrapper.bdh.token_embed.forward

    def spy(ids):
        seen.append(ids.detach().cpu())
        return original(ids)

    wrapper.bdh.token_embed.forward = spy
    loss = wrapper.train_loss(task_latents, steps=1)
    loss.backward()
    assert seen
    allowed = {PAD, IN, OUT, EOS, CANVAS_START}
    for ids in seen:
        assert ids.dtype in (torch.int32, torch.int64, torch.long)
        extra = set(ids.unique().tolist()) - allowed
        assert extra == set(), extra
        assert int(ids.max()) < NUM_TOKENS


def test_sequence_is_post_embed_normed(wrapper, task_latents):
    shapes = []
    original = wrapper.bdh.post_embed_norm.forward

    def spy(tokens):
        shapes.append(tuple(tokens.shape))
        return original(tokens)

    wrapper.bdh.post_embed_norm.forward = spy
    wrapper.ingest_task(task_latents)
    assert shapes, "ingest must call post_embed_norm on marker+patch sequences"
    assert all(rank == 3 for rank in (len(shape) for shape in shapes))
    # demos are 899 tokens (1+448+1+448+1), canvas is 449 (1+448)
    assert any(shape[-2] == 899 for shape in shapes)
    assert any(shape[-2] == N_CELLS + 1 for shape in shapes)


def test_backward_through_to_latent_into_bdh(wrapper, task_latents):
    loss = wrapper.train_loss(task_latents, steps=1)
    assert torch.isfinite(loss)
    loss.backward()
    assert wrapper.to_latent.weight.grad is not None
    assert wrapper.to_occupancy.weight.grad is not None
    assert wrapper.to_occupancy.weight.grad.abs().sum() > 0
    assert wrapper.to_centroid.weight.grad is not None
    assert wrapper.to_centroid.bias.grad is not None
    assert wrapper.to_centroid.bias.grad.abs().sum() > 0
    assert wrapper.patch_in.weight.grad is not None
    assert wrapper.bdh.block.to_qk.weight.grad is not None
    assert not hasattr(wrapper, "empty_cell")


def test_occupancy_gate_is_spatial_softmax_not_sigmoid(wrapper, task_latents):
    from bdh_cq.video_probes import spatial_softmax_gate

    memories = wrapper.ingest_task(task_latents)
    hidden = memories.embeds
    z = wrapper.to_latent_volume(hidden)
    logits = wrapper.occupancy_logits(hidden)
    gate = spatial_softmax_gate(logits)
    assert gate.shape == (1, wrapper.latent_t, wrapper.latent_h, wrapper.latent_w)
    assert float(gate.amax().detach()) <= 1.0 + 1e-5
    # peaked dest must beat a corner; independent sigmoid of zeros is flat 0.5
    assert not torch.allclose(gate, torch.full_like(gate, 0.5))
    # Gaussian occupancy: compact, not the uniform ones-gate wash
    g = gate[:, -1].detach()
    assert float(g.amax() / g.mean()) > 4.0
    assert z.shape[1] == 24


def test_centroid_head_gets_location_grad(vae):
    task = encode_task(vae, sample_task("translate", seed=0))
    ranked = BDHVideoReasoningWrapper(make_video_model(scale="tiny"))
    loss, parts = ranked.train_loss(task, steps=1, return_parts=True)
    assert "shift_mse" in parts
    loss.backward()
    assert ranked.to_centroid.weight.grad is not None
    assert float(ranked.to_centroid.weight.grad.abs().sum()) > 0


def test_reason_r0_is_linear_of_h0(wrapper, task_latents):
    memories = wrapper.ingest_task(task_latents)
    z0 = wrapper.reason(memories, steps=0)
    assert torch.allclose(z0, wrapper.to_latent_volume(memories.embeds))


def test_low_rank_motion_volume_shape_and_t0_lock(wrapper, task_latents):
    ranked = BDHVideoReasoningWrapper(make_video_model(scale="tiny"), motion_rank=4)
    memories = ranked.ingest_task(task_latents)
    raw = ranked.reason(memories, steps=0)
    flowed = ranked.reason(memories, steps=0, still=task_latents["query_in"])
    z = task_latents["query_out"]
    assert raw.shape == z.shape
    assert flowed.shape == z.shape
    assert torch.equal(flowed[:, :, 0], task_latents["query_in"][:, :, 0])
    token = ranked.motion_token(memories.embeds)
    assert token.shape == (1, 4)
    loss = ranked.train_loss(task_latents, steps=1)
    assert torch.isfinite(loss)
    loss.backward()
    assert ranked.to_motion.weight.grad is not None
    assert ranked.to_bases.weight.grad is not None


def test_train_loss_parts_logs_rec_dt(wrapper, task_latents):
    loss, parts = wrapper.train_loss(task_latents, steps=1, return_parts=True)
    assert torch.isfinite(loss)
    assert "rec" in parts and "dt" in parts
    assert parts["rec"] >= 0 and parts["dt"] >= 0
    assert parts["last_mse"] >= 0
    assert parts["last_app"] >= 0
    assert parts["last_mot"] >= 0
    assert parts["occupancy_ce"] >= 0
    assert parts["energy_kl"] >= 0
    assert parts["occupancy_dice"] >= 0
    assert parts["fp_energy"] >= 0
    assert parts["mass_rel"] >= 0


def test_residual_motion_code_strips_appearance():
    ranked = BDHVideoReasoningWrapper(make_video_model(scale="tiny"), motion_rank=4)
    residual = torch.zeros(1, LATENT_CH, LATENT_T, LATENT_HW, LATENT_HW)
    residual[:, 0, 1:] = torch.linspace(0.1, 1.0, LATENT_T - 1).view(1, 1, -1, 1, 1)
    still_a = torch.zeros_like(residual)
    still_a[:, 3] = 1.2
    still_b = torch.zeros_like(residual)
    still_b[:, 7] = -0.8
    z_a = still_a + residual
    z_b = still_b + residual
    code_a = ranked.residual_motion_code(z_a)
    code_b = ranked.residual_motion_code(z_b)
    assert code_a.shape == (1, 4)
    assert torch.allclose(code_a, code_b, atol=1e-6)
    assert not torch.allclose(code_a, torch.zeros_like(code_a))
    assert "motion_basis" in dict(ranked.named_buffers())
    assert all(name != "motion_basis" for name, _ in ranked.named_parameters())


def test_ingest_and_reason_never_see_query_out():
    import inspect

    assert "query_out" not in inspect.getsource(BDHVideoReasoningWrapper.ingest_task)
    assert "query_out" not in inspect.getsource(BDHVideoReasoningWrapper.reason)
    src = inspect.getsource(BDHVideoReasoningWrapper.train_loss)
    assert "residual_motion_code(target)" in src
    assert ".detach()" in src


def test_train_loss_supervises_motion_token(task_latents):
    ranked = BDHVideoReasoningWrapper(make_video_model(scale="tiny"), motion_rank=4)
    loss, parts = ranked.train_loss(
        task_latents, steps=1, lambda_m=1.0, lambda_ctx=1.0, return_parts=True
    )
    assert torch.isfinite(loss)
    assert parts["m_mse"] >= 0
    assert parts["ctx_mse"] >= 0
    assert "demo_m_var" in parts
    loss.backward()
    assert ranked.to_motion.weight.grad is not None
    assert ranked.motion_basis.grad is None


def test_motion_token_overfit_tracks_residual_teacher(vae):
    task = encode_task(vae, sample_task("translate", seed=1))
    ranked = BDHVideoReasoningWrapper(make_video_model(scale="tiny"), motion_rank=4)
    opt = torch.optim.AdamW(ranked.parameters(), lr=3e-3)
    start = ranked.train_loss(task, 1, lambda_m=1.0, lambda_ctx=1.0, return_parts=True)[1]
    for _ in range(30):
        loss = ranked.train_loss(task, 1, lambda_m=1.0, lambda_ctx=1.0)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    end = ranked.train_loss(task, 1, lambda_m=1.0, lambda_ctx=1.0, return_parts=True)[1]
    assert end["m_mse"] < start["m_mse"]
    assert end["rec"] < start["rec"] * 1.05


def test_warp_still_moves_mass_and_locks_t0():
    still = torch.zeros(1, 24, 7, 8, 8)
    still[:, 0, :, 1, 1] = 1.0
    z = warp_still(still, torch.tensor([[2.0, 0.0]]))
    assert torch.allclose(z[:, :, 0], still[:, :, 0])
    assert float(z[0, 0, -1, 3, 1]) > 0.5
    assert float(z[0, 0, -1, 1, 1]) < 0.5


def test_composite_shift_keeps_bg_and_erases_source():
    still = torch.zeros(1, 24, 7, 8, 8)
    still[:, 0, :, :, :] = 0.05
    still[:, 0, :, 1, 1] = 1.0
    z = composite_shift(still, torch.tensor([[2.0, 0.0]]))
    assert torch.allclose(z[:, :, 0], still[:, :, 0])
    assert float(z[0, 0, -1, 3, 1]) > 0.5
    assert float(z[0, 0, -1, 1, 1]) < 0.2
    # checker/bg away from the sprite stays put; full-frame warp would move it
    assert torch.allclose(z[:, 0, -1, 6, 6], still[:, 0, 0, 6, 6])


def test_composite_shift_oracle_last_l1_under_eight():
    vae = FakeVideoVAE(seed=0, spatial=8)
    task = encode_task(vae, sample_task("translate", seed=0))
    z = composite_shift(task["query_in"], task["query_shift"], spatial=8)
    pred = decode_pixels(vae, z)
    gt = decode_pixels(vae, task["query_out"])
    last_l1 = pixel_last_frame_l1(pred, gt)
    warped = decode_pixels(vae, warp_still(task["query_in"], task["query_shift"], spatial=8))
    ghost = decode_pixels(
        vae, apply_residual(warp_still(task["query_in"], task["query_shift"], spatial=8), task["query_in"])
    )
    assert last_l1 <= LAST_FRAME_L1
    assert last_l1 < pixel_last_frame_l1(warped, gt)
    assert last_l1 < pixel_last_frame_l1(ghost, gt)


def test_shift_head_overfit_moves_last_frame(vae):
    task = encode_task(vae, sample_task("translate", seed=0))
    wrapper = BDHVideoReasoningWrapper(
        make_video_model(scale="tiny"), motion_rank=-1, canvas_update_memory=False
    )
    opt = torch.optim.AdamW(wrapper.parameters(), lr=1e-2)
    start = wrapper.train_loss(task, 2, lambda_last=4.0, lambda_energy=1.0, return_parts=True)[1]
    for _ in range(80):
        loss = wrapper.train_loss(task, 2, lambda_last=4.0, lambda_energy=1.0)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    end = wrapper.train_loss(task, 2, lambda_last=4.0, lambda_energy=1.0, return_parts=True)[1]
    assert end["shift_mse"] < start["shift_mse"] * 0.5
    assert end["last_mse"] < start["last_mse"] * 0.8


def test_demo_shift_feeds_the_shift_head(vae):
    """The demo-average energy-centroid delta is stashed at ingest and drives
    the shift head (probe: hidden.mean carries no readable displacement)."""
    from bdh_cq.video_probes import energy_centroid

    task = encode_task(vae, sample_task("translate", seed=0))
    wrapper = BDHVideoReasoningWrapper(
        make_video_model(scale="tiny"), canvas_update_memory=False, motion_rank=0
    )
    mem = wrapper.ingest_task(task)
    want = torch.stack(
        [
            energy_centroid(z_out)[:, -1] - energy_centroid(z_in)[:, 0]
            for z_in, z_out in zip(task["demo_in"], task["demo_out"])
        ]
    ).mean(dim=0)
    assert wrapper._demo_shift is not None
    assert torch.allclose(wrapper._demo_shift, want, atol=1e-5)
    # demo_to_shift is init 2*I and to_centroid is zero-init, so the untrained
    # head starts at 2 * demo_shift (right direction, query is a higher level).
    shift0 = wrapper._shift_from_hidden(mem.embeds)
    assert torch.allclose(shift0, 2.0 * wrapper._demo_shift, atol=1e-4)


def test_apply_residual_locks_t0_without_cumsum():
    still = torch.zeros(1, 24, 7, 2, 2)
    still[:, :, :, 0, 0] = 1.0
    u = torch.zeros(1, 24, 7, 2, 2)
    u[:, 0, 1] = 0.5
    u[:, 0, 2] = 0.5
    z = apply_residual(u, still)
    assert torch.equal(z[:, :, 0], still[:, :, 0])
    assert torch.allclose(z[:, 0, 1], still[:, 0, 0] + 0.5)
    assert torch.allclose(z[:, 0, 2], still[:, 0, 0] + 0.5)


def test_lock_t0_is_absolute_answer_not_residual():
    still = torch.zeros(1, 24, 7, 2, 2)
    still[:, :, :, 0, 0] = 1.0
    volume = torch.zeros(1, 24, 7, 2, 2)
    volume[:, 0, 1, 1, 1] = 0.9
    z = lock_t0(volume, still)
    assert torch.equal(z[:, :, 0], still[:, :, 0])
    assert torch.allclose(z[:, 0, 1, 1, 1], torch.tensor(0.9))
    assert torch.equal(z[:, :, 1, 0, 0], torch.zeros(1, 24))


def test_integrate_u_locks_t0_and_cums_motion():
    still = torch.zeros(1, 24, 7, 2, 2)
    still[:, :, :, 0, 0] = 1.0
    u = torch.zeros(1, 24, 7, 2, 2)
    u[:, 0, 1] = 0.5
    u[:, 0, 2] = 0.5
    z = integrate_u(u, still)
    assert z.shape == still.shape
    assert torch.equal(z[:, :, 0], still[:, :, 0])
    assert torch.allclose(z[:, 0, 1], still[:, 0, 0] + 0.5)
    assert torch.allclose(z[:, 0, 2], still[:, 0, 0] + 1.0)
    assert torch.allclose(z[:, 0, 3], still[:, 0, 0] + 1.0)


def test_train_loss_uses_integrated_still(wrapper, task_latents):
    loss = wrapper.train_loss(task_latents, steps=1)
    assert torch.isfinite(loss)
    memories = wrapper.ingest_task(task_latents)
    raw = wrapper.reason(memories, steps=0)
    flowed = wrapper.reason(memories, steps=0, still=task_latents["query_in"])
    assert flowed.shape == raw.shape
    assert torch.equal(flowed[:, :, 0], task_latents["query_in"][:, :, 0])


def test_admit_clip_is_library_policy_not_decode_only():
    from bdh_cq.video_probes import admit_clip

    assert admit_clip(latent_vs_still=0.9, decode_t_l1=0.01, mode="decode") is True
    assert admit_clip(latent_vs_still=0.9, decode_t_l1=0.001, mode="decode") is False
    assert admit_clip(latent_vs_still=0.9, decode_t_l1=0.001, mode="latent_or_decode") is True
    assert admit_clip(latent_vs_still=0.001, decode_t_l1=0.001, mode="latent_or_decode") is False


def test_reason_does_not_contain_pixel_gate():
    import inspect

    src = inspect.getsource(BDHVideoReasoningWrapper.reason)
    assert "is_still_clip" not in src
    assert "STILL_L1" not in src
    assert "admit_clip" not in src
    loss_src = inspect.getsource(BDHVideoReasoningWrapper.train_loss)
    assert "is_still_clip" not in loss_src
    assert "pixel_temporal_l1" not in loss_src


def test_reason_locks_t0_not_residual_add():
    import inspect

    src = inspect.getsource(BDHVideoReasoningWrapper.reason)
    assert "lock_t0(volume" in src
    assert "apply_residual(volume" not in src
    vol_src = inspect.getsource(BDHVideoReasoningWrapper.to_latent_volume)
    assert "composite_shift" in vol_src


def test_still_clip_is_a_fail():
    from bdh_cq.video_probes import is_still_clip, pixel_temporal_l1

    still = torch.rand(1, 3, 1, 16, 16).expand(1, 3, 22, 16, 16).contiguous()
    moving = still.clone()
    moving[:, :, 10:] = 1.0
    assert is_still_clip(still)
    assert pixel_temporal_l1(still) == 0.0
    assert not is_still_clip(moving)


def test_wrapper_non_square_canvas():
    vae = FakeVideoVAE(seed=0)
    video = torch.rand(1, 3, 22, 64, 48)
    z = encode_mean_video(vae, video)
    assert z.shape == (1, LATENT_CH, LATENT_T, 4, 3)
    wrapper = BDHVideoReasoningWrapper(make_video_model(scale="tiny"))
    latents = dict(
        name="portrait-debug",
        demo_in=[z, z, z],
        demo_out=[z, z, z],
        query_in=z,
        query_out=z,
    )
    memories = wrapper.ingest_task(latents)
    assert memories.embeds.shape[-2] == 84
    z_hat = wrapper.reason(memories, steps=1)
    assert z_hat.shape == z.shape
    loss = wrapper.train_loss(latents, steps=1)
    assert torch.isfinite(loss)
