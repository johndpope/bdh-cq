"""BDH-CQ video ingest → relax → render. FakeVAE is enough for the protocol.

Demos write S in 128-token chunks. The query canvas is one unchunked pass
so Memory.embeds is H_0 of length N_CELLS. Reasoning holds S at S_K.
VAE cells always enter as floats; they never index token_embed.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from einops import rearrange
from torch import Tensor, nn
from torch.nn import Module

from bdh_cq.bdh_cq import BDH, Memory, default, exists
from bdh_cq.video_probes import (
    OCC_FRAC,
    admit_clip,
    energy_centroid,
    gaussian_occupancy_logits,
    last_frame_appearance_mse,
    last_frame_energy_kl,
    last_frame_fp_energy,
    last_frame_latent_mse,
    last_frame_motion_mse,
    last_frame_mass_rel,
    last_frame_occupancy_ce,
    last_frame_occupancy_dice,
    latent_dt_mse,
    latent_mse,
    latent_mse_peak,
    latent_vs_still_mse,
    pixel_temporal_l1,
    spatial_softmax_gate,
)
from bdh_cq.video_tasks import VIDEO_TASKS
from bdh_cq.video_vae import (
    CLIP_FRAMES,
    LATENT_CH,
    LATENT_T,
    N_CELLS,
    decode_pixels,
    encode_mean_video,
)

# markers — VAE cells never use this table

PAD, IN, OUT, EOS, CANVAS_START = 0, 1, 2, 3, 4
NUM_TOKENS = 8
CHUNK_SIZE = 128
MAX_REASONING_STEPS = 8

PROTOCOL_MODEL_KWARGS = dict(
    dim=256,
    depth=4,
    heads=4,
    dim_qk_heads=1024,
    attn_residual=True,
    attn_residual_depth_bias_distance=1,
)

VIDEO_MODEL_KWARGS = dict(
    dim=512,
    depth=8,
    heads=8,
    dim_qk_heads=4096,
    rotary_dim=64,
    attn_residual=True,
    attn_residual_depth_bias_distance=1,
)

TINY_MODEL_KWARGS = dict(
    dim=64,
    depth=2,
    heads=4,
    dim_qk_heads=512,
    attn_residual=True,
    attn_residual_depth_bias_distance=1,
)

# ~1.007B in the tied block: 3 * dim * dim_qk_heads. Depth is recurrent.
# dim=4096 + checkpoint fits 24GB; dim=8192 OOMed the RTX PRO 4000 on fwd.
BILLION_MODEL_KWARGS = dict(
    dim=4096,
    depth=2,
    heads=8,
    dim_qk_heads=81920,
    rotary_dim=64,
    attn_residual=True,
    attn_residual_depth_bias_distance=1,
    gradient_checkpoint=True,
)


def make_video_model(scale: str = "protocol", **overrides) -> BDH:
    tables = {
        "tiny": TINY_MODEL_KWARGS,
        "protocol": PROTOCOL_MODEL_KWARGS,
        "gpu": VIDEO_MODEL_KWARGS,
        "billion": BILLION_MODEL_KWARGS,
    }
    if scale not in tables:
        raise ValueError(f"scale must be one of {sorted(tables)}, got {scale!r}")
    return BDH(num_tokens=NUM_TOKENS, **{**tables[scale], **overrides})


def pixel_to_ncthw(frames: np.ndarray, clip_frames: int | None = None) -> Tensor:
    """uint8 (H,W,3) or (T,H,W,3) -> float (1,3,T,H,W) in [0, 1]."""
    array = np.asarray(frames)
    if array.ndim == 3:
        n = CLIP_FRAMES if clip_frames is None else int(clip_frames)
        array = np.repeat(array[None, ...], n, axis=0)
    if array.ndim != 4:
        raise ValueError(f"expected (H,W,3) or (T,H,W,3), got {array.shape}")
    tensor = torch.from_numpy(np.ascontiguousarray(array)).float().div_(255.0)
    return rearrange(tensor, "t h w c -> 1 c t h w")


def _bright_centroid_hw(frame: np.ndarray) -> tuple[float, float]:
    """Pixel centroid of the bright sprite. Used as the shift-head teacher."""
    gray = np.asarray(frame, dtype=np.float32).mean(axis=-1)
    ys, xs = np.nonzero(gray > 80.0)
    if ys.size == 0:
        return 0.0, 0.0
    return float(ys.mean()), float(xs.mean())


def integrate_u(u: Tensor, still: Tensor) -> Tensor:
    """Lock t=0 to the query still; later latent times are still + cumsum(u[:,:,1:])."""
    if u.ndim != 5 or still.ndim != 5:
        raise ValueError(f"expected 5D u and still, got {tuple(u.shape)} {tuple(still.shape)}")
    z0 = still[:, :, :1]
    if u.shape[2] < 2:
        return z0.expand_as(u).contiguous()
    motion = torch.cumsum(u[:, :, 1:], dim=2)
    return torch.cat([z0, z0 + motion], dim=2)


def warp_still(still: Tensor, shift_yx: Tensor, spatial: int = 8) -> Tensor:
    """Translate the query still. shift_yx is (B, 2) latent cells (dy, dx).

    Warp at pixel resolution then avg-pool so a 32px sprite lands on the
    same FakeVAE cells as a pixel-space translate. Latent bilinear of a
    4x4 block cannot hit lastL1 <= 8/255 even with the oracle shift.
    t=0 stays put; later times lerp to the full shift.
    """
    if still.ndim != 5 or shift_yx.ndim != 2 or shift_yx.shape[-1] != 2:
        raise ValueError(
            f"expected still (B,C,T,H,W) and shift (B,2), got "
            f"{tuple(still.shape)} {tuple(shift_yx.shape)}"
        )
    if spatial < 1:
        raise ValueError(f"spatial must be positive, got {spatial}")
    batch, _, t_len, height, width = still.shape
    z0 = still[:, :, 0]
    pix = z0.repeat_interleave(spatial, dim=-2).repeat_interleave(spatial, dim=-1)
    _, _, pix_h, pix_w = pix.shape
    ys = torch.linspace(-1, 1, pix_h, device=still.device, dtype=still.dtype)
    xs = torch.linspace(-1, 1, pix_w, device=still.device, dtype=still.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    base = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)
    frames = []
    denom_y = max(pix_h - 1, 1)
    denom_x = max(pix_w - 1, 1)
    for ti in range(t_len):
        alpha = 0.0 if t_len < 2 else ti / (t_len - 1)
        dy = shift_yx[:, 0] * float(spatial) * alpha
        dx = shift_yx[:, 1] * float(spatial) * alpha
        delta = torch.stack(
            (2.0 * dx / denom_x, 2.0 * dy / denom_y), dim=-1
        ).view(batch, 1, 1, 2)
        warped_pix = torch.nn.functional.grid_sample(
            pix, base - delta, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        warped = torch.nn.functional.avg_pool2d(
            warped_pix, kernel_size=spatial, stride=spatial
        )
        frames.append(warped)
    return torch.stack(frames, dim=2)


def composite_shift(
    still: Tensor,
    shift_yx: Tensor,
    spatial: int = 8,
    occ_frac: float = OCC_FRAC,
) -> Tensor:
    """Translate the bright sprite; keep the query still's background.

    Full-frame warp_still also shifts the checker, so oracle lastL1 floors
    at ~18/255 even with the true (dy, dx). Punch source, fill with bg
    mean, paste the warped sprite. t=0 is identity. Grad flows through
    grid_sample to shift_yx.
    """
    if still.ndim != 5 or shift_yx.ndim != 2 or shift_yx.shape[-1] != 2:
        raise ValueError(
            f"expected still (B,C,T,H,W) and shift (B,2), got "
            f"{tuple(still.shape)} {tuple(shift_yx.shape)}"
        )
    energy = still[:, :, :1].pow(2).mean(dim=1, keepdim=True)
    peak = energy.flatten(2).amax(dim=-1).clamp_min(1e-8).view(-1, 1, 1, 1, 1)
    src_gate = (energy >= occ_frac * peak).to(dtype=still.dtype)
    t_len = still.shape[2]
    gate_vol = src_gate.expand(-1, 1, t_len, -1, -1)
    moved_gate = warp_still(gate_vol, shift_yx, spatial=spatial).clamp(0, 1)
    moved = warp_still(still * src_gate, shift_yx, spatial=spatial)
    bg_only = still * (1.0 - src_gate)
    denom = (1.0 - src_gate).flatten(-2).sum(dim=-1).clamp_min(1.0)
    fill = (bg_only.flatten(-2).sum(dim=-1) / denom).view(*still.shape[:3], 1, 1)
    punched = torch.where(src_gate.bool(), fill.expand_as(still), still)
    return punched * (1.0 - moved_gate) + moved


def apply_residual(u: Tensor, still: Tensor) -> Tensor:
    """Lock t=0 to the query still; later times are still + u (no cumsum).

    Rank-0 Linear predicting absolute z replaced the sprite and flooded.
    Residual keeps appearance; the head only paints the motion delta.
    """
    if u.ndim != 5 or still.ndim != 5:
        raise ValueError(f"expected 5D u and still, got {tuple(u.shape)} {tuple(still.shape)}")
    z0 = still[:, :, :1]
    if u.shape[2] < 2:
        return z0.expand_as(u).contiguous()
    return torch.cat([z0, z0 + u[:, :, 1:]], dim=2)


def lock_t0(volume: Tensor, still: Tensor) -> Tensor:
    """t=0 is the query still; later times are G(H_r) as an absolute grid.

    Residual `still + gated_u` cannot erase the start sprite: occupancy is
    low there on the answer, so the gate zeros the residual and the still
    blob stays. Absolute last time is the paper output grid.
    """
    if volume.ndim != 5 or still.ndim != 5:
        raise ValueError(
            f"expected 5D volume and still, got {tuple(volume.shape)} {tuple(still.shape)}"
        )
    z0 = still[:, :, :1]
    if volume.shape[2] < 2:
        return z0.expand_as(volume).contiguous()
    return torch.cat([z0, volume[:, :, 1:]], dim=2)


@torch.no_grad()
def encode_task(
    vae: nn.Module,
    task: dict[str, Any],
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Oracle task -> VAE posterior means. Inputs are still-clips of frame 0."""

    def encode_still(first_frame: np.ndarray) -> Tensor:
        video = pixel_to_ncthw(first_frame)
        if exists(device):
            video = video.to(device)
        return encode_mean_video(vae, video)

    def encode_clip(clip: np.ndarray) -> Tensor:
        video = pixel_to_ncthw(clip)
        if exists(device):
            video = video.to(device)
        return encode_mean_video(vae, video)

    demo_in, demo_out = [], []
    for _level, first, clip in task["train"]:
        demo_in.append(encode_still(first))
        demo_out.append(encode_clip(clip))

    _level, query_first, query_clip = task["test"][0]
    query_in = encode_still(query_first)
    query_out = encode_clip(query_clip)
    # Latent-grid dest, not pixel centroid / spatial. Pixel 7.875 overshoots
    # FakeVAE last-time energy (~7.42) and trips motL1 while lastL1 is fine.
    src = energy_centroid(query_in)[:, 0]
    dst = energy_centroid(query_out)[:, -1]
    query_shift = (dst - src).to(dtype=query_out.dtype)
    return dict(
        name=task["name"],
        demo_in=demo_in,
        demo_out=demo_out,
        query_in=query_in,
        query_out=query_out,
        query_shift=query_shift,
    )


@torch.no_grad()
def encode_aval_library(
    vae: nn.Module,
    root: str | Path = "data/aval_clips",
    height: int = 1280,
    width: int = 720,
    device: torch.device | str | None = None,
    tag: str = "h3",
    limit: int | None = None,
    allow_raw: bool = True,
    clip_frames: int = CLIP_FRAMES,
    gate_decode: bool = True,
    admit: str = "decode",
) -> list[dict[str, Any]]:
    """Encode max-Δ windows. Admission is a library filter, not a layer in F.

    Always records src_t_l1, latent_vs_still, decode_t_l1. Drops a clip only
    via admit_clip (decode-visible, or latent_or_decode). Identity/sprite
    families are not filtered here.
    """
    from bdh_cq.aval_clips import (
        best_motion_window,
        decode_mp4,
        list_processed,
        load_processed,
        proc_path,
        raw_path,
    )

    root = Path(root)
    clip_frames = int(clip_frames)
    cache = root / f"latents_{tag}_{height}x{width}_f{clip_frames}_motion.pt"
    scores_path = root / f"latents_{tag}_{height}x{width}_f{clip_frames}_scores.jsonl"
    paths = list_processed(root, height=height, width=width)
    names = [path.stem for path in paths]
    raw_dir = root / "raw"
    if raw_dir.is_dir():
        raw_names = sorted(path.stem for path in raw_dir.glob("*.mp4"))
        if allow_raw and (not names or len(raw_names) >= len(names)):
            names = raw_names
    if limit is not None:
        names = names[: int(limit)]
    if len(names) < 4:
        raise RuntimeError(f"need AVAL proc or raw under {root}, have {len(names)}")

    library: list[dict[str, Any]] = []
    dropped: list[str] = []
    if cache.is_file():
        loaded = torch.load(cache, map_location="cpu", weights_only=False)
        wanted = set(names)
        library = [item for item in loaded if item.get("name") in wanted]
        for item in library:
            if "latent_vs_still" not in item:
                item["latent_vs_still"] = latent_vs_still_mse(item["z_out"])
        print(f"resume {cache} n={len(library)}/{len(names)}", flush=True)

    done = {item["name"] for item in library}

    def persist() -> None:
        torch.save(library, cache)

    for index, name in enumerate(names, start=1):
        if name in done:
            print(f"encode {index}/{len(names)} {name} cached", flush=True)
            continue
        raw = raw_path(root, name)
        proc = proc_path(root, name, height=height, width=width)
        if allow_raw and raw.is_file():
            full = decode_mp4(raw, height=height, width=width)
        elif proc.is_file():
            full = load_processed(proc)
        else:
            raise FileNotFoundError(f"no raw or processed clip for {name}")
        frames, start, src_l1 = best_motion_window(full, length=clip_frames)
        z_in = encode_mean_video(
            vae, pixel_to_ncthw(frames[0], clip_frames=clip_frames).to(device)
        ).cpu()
        z_out = encode_mean_video(vae, pixel_to_ncthw(frames).to(device)).cpu()
        latent_still = latent_vs_still_mse(z_out)
        gt_l1 = None
        if gate_decode:
            recon = decode_pixels(vae, z_out.to(device))
            gt_l1 = pixel_temporal_l1(recon)
        keep = admit_clip(
            latent_vs_still=latent_still,
            decode_t_l1=gt_l1 if gate_decode else 1.0,
            mode=admit,
        )
        row = dict(
            name=name,
            start=start,
            src_t_l1=src_l1,
            latent_vs_still=latent_still,
            decode_t_l1=gt_l1,
            admit=keep,
            mode=admit,
        )
        with scores_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        if not keep:
            dropped.append(name)
            print(
                f"encode {index}/{len(names)} {name} DROP "
                f"src={src_l1:.5f} latent={latent_still:.4f} "
                f"gt={gt_l1 if gt_l1 is None else f'{gt_l1:.5f}'}",
                flush=True,
            )
            persist()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        library.append(
            dict(
                name=name,
                z_in=z_in,
                z_out=z_out,
                start=start,
                src_t_l1=src_l1,
                latent_vs_still=latent_still,
                gt_t_l1=gt_l1,
            )
        )
        done.add(name)
        persist()
        print(
            f"encode {index}/{len(names)} {name} {tuple(z_out.shape)} "
            f"start={start} src={src_l1:.5f} latent={latent_still:.4f} "
            f"gt={gt_l1 if gt_l1 is None else f'{gt_l1:.5f}'}",
            flush=True,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(
        f"wrote {cache} keep={len(library)} drop={len(dropped)} "
        f"admit={admit} scores={scores_path}",
        flush=True,
    )
    if len(library) < 4:
        raise RuntimeError(
            f"motion library too small: {len(library)} kept, {len(dropped)} dropped "
            f"(need 4). Try clip_frames=39."
        )
    return library


@torch.no_grad()
def encode_icq_transfer_library(
    vae: nn.Module,
    root: str | Path = "data/icq_transfer",
    height: int = 512,
    width: int = 288,
    device: torch.device | str | None = None,
    tag: str = "h3",
    clip_frames: int = CLIP_FRAMES,
) -> list[dict[str, Any]]:
    """Encode idle stills and motion clips. No library still-gate."""
    from bdh_cq.icq_transfer import (
        IDENTITIES,
        MOTION_ACTIONS,
        load_clip,
        load_idle,
    )

    root = Path(root)
    clip_frames = int(clip_frames)
    cache = root / f"latents_{tag}_{height}x{width}_f{clip_frames}_transfer.pt"
    library: list[dict[str, Any]] = []
    if cache.is_file():
        library = torch.load(cache, map_location="cpu", weights_only=False)
        print(f"resume {cache} n={len(library)}", flush=True)
    done = {(item["id"], item["action"]) for item in library}

    def persist() -> None:
        torch.save(library, cache)

    for ident in IDENTITIES:
        idle = load_idle(root, ident, height=height, width=width)
        z_in = encode_mean_video(
            vae, pixel_to_ncthw(idle, clip_frames=clip_frames).to(device)
        ).cpu()
        for action in MOTION_ACTIONS:
            key = (ident, action)
            if key in done:
                print(f"encode transfer {ident}/{action} cached", flush=True)
                continue
            frames = load_clip(root, ident, action, height=height, width=width)
            z_out = encode_mean_video(vae, pixel_to_ncthw(frames).to(device)).cpu()
            library.append(
                dict(
                    name=f"{ident}_{action}",
                    id=ident,
                    action=action,
                    z_in=z_in,
                    z_out=z_out,
                )
            )
            done.add(key)
            persist()
            print(
                f"encode transfer {ident}/{action} {tuple(z_out.shape)}",
                flush=True,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    from bdh_cq.imtalker_motion import attach_motion_tokens, load_token_cache

    token_cache = load_token_cache(root)
    n_tok = attach_motion_tokens(library, token_cache)
    print(f"wrote {cache} n={len(library)} (no still-gate) m_star={n_tok}", flush=True)
    if len(library) < 4:
        raise RuntimeError(f"transfer library too small: {len(library)}")
    return library


def sample_icq_transfer_task_latents(
    library: list[dict[str, Any]],
    seed: int,
    n_demos: int = 3,
    action: str | None = None,
    query_id: str | None = None,
    holdout_query: set[str] | None = None,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    rng = random.Random(seed)
    actions = sorted({item["action"] for item in library})
    if not actions:
        raise RuntimeError("empty transfer library")
    if action is None:
        action = rng.choice(actions)
    pool = [item for item in library if item["action"] == action]
    if holdout_query:
        query_pool = [item for item in pool if item["id"] not in holdout_query]
    else:
        query_pool = pool
    if query_id is not None:
        query_pool = [item for item in query_pool if item["id"] == query_id]
    if not query_pool:
        raise RuntimeError(
            f"no transfer item for action={action!r} query_id={query_id!r} "
            f"holdout={holdout_query}"
        )
    query = rng.choice(query_pool)
    demos = [item for item in pool if item["id"] != query["id"]]
    if len(demos) < n_demos:
        raise RuntimeError(f"need {n_demos} demos besides {query['id']}, have {len(demos)}")
    if len(demos) > n_demos:
        demos = rng.sample(demos, n_demos)

    def to_dev(z: Tensor) -> Tensor:
        return z.to(device) if exists(device) else z

    task = dict(
        name="icq_transfer",
        params=dict(
            query=query["id"],
            demos=[item["id"] for item in demos],
            action=action,
        ),
        demo_in=[to_dev(item["z_in"]) for item in demos],
        demo_out=[to_dev(item["z_out"]) for item in demos],
        query_in=to_dev(query["z_in"]),
        query_out=to_dev(query["z_out"]),
    )
    if query.get("m_star") is not None and all(item.get("m_star") is not None for item in demos):
        task["query_m"] = to_dev(query["m_star"]).unsqueeze(0)
        task["demo_m"] = [to_dev(item["m_star"]).unsqueeze(0) for item in demos]
    return task


def sample_aval_task_latents(
    library: list[dict[str, Any]],
    seed: int,
    n_demos: int = 3,
    holdout: set[str] | None = None,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    pool = [item for item in library if holdout is None or item["name"] not in holdout]
    if len(pool) < n_demos + 1:
        raise RuntimeError(f"latent library too small: {len(pool)}")
    rng = random.Random(seed)
    pick = rng.sample(pool, n_demos + 1)

    def to_dev(z: Tensor) -> Tensor:
        return z.to(device) if exists(device) else z

    demos, query = pick[:-1], pick[-1]
    return dict(
        name="aval",
        params=dict(
            query=[query["name"]],
            demos=[item["name"] for item in demos],
        ),
        demo_in=[to_dev(item["z_in"]) for item in demos],
        demo_out=[to_dev(item["z_out"]) for item in demos],
        query_in=to_dev(query["z_in"]),
        query_out=to_dev(query["z_out"]),
    )


def sample_task(
    family: str = "identity",
    seed: int = 0,
    n_demos: int = 3,
    **clip_kwargs,
) -> dict[str, Any]:
    if family == "aval":
        from bdh_cq.aval_clips import sample_aval_task

        return sample_aval_task(seed=seed, n_demos=n_demos, **clip_kwargs)
    if family == "icq_transfer":
        from bdh_cq.icq_transfer import sample_icq_transfer_task

        return sample_icq_transfer_task(seed=seed, n_demos=n_demos, **clip_kwargs)
    if family not in VIDEO_TASKS:
        raise KeyError(
            f"unknown family {family!r}, have {sorted(VIDEO_TASKS)} or 'aval'|'icq_transfer'"
        )
    if clip_kwargs:
        raise TypeError(
            f"clip kwargs {sorted(clip_kwargs)} only apply to family='aval'|'icq_transfer'"
        )
    return VIDEO_TASKS[family]().generate(seed=seed, n_demos=n_demos)


class BDHVideoReasoningWrapper(Module):
    def __init__(
        self,
        bdh: BDH,
        to_latent: nn.Linear | None = None,
        canvas_update_memory: bool = True,
        motion_rank: int = 0,
    ):
        super().__init__()
        self.bdh = bdh
        self.patch_in = nn.Linear(LATENT_CH, bdh.dim)
        if to_latent is None:
            self.to_latent = nn.Linear(bdh.dim, LATENT_CH)
            nn.init.zeros_(self.to_latent.weight)
            nn.init.zeros_(self.to_latent.bias)
        else:
            self.to_latent = to_latent
        # Occupancy logit residual per cell. The peaked map is a Gaussian at
        # a pooled (dy, dx) — per-cell Linear on prefix-sum H is uniform, so
        # occupancy CE stuck at log(H*W) and G painted a gray wash.
        self.to_occupancy = nn.Linear(bdh.dim, 1)
        nn.init.zeros_(self.to_occupancy.weight)
        nn.init.zeros_(self.to_occupancy.bias)
        self.latent_t = LATENT_T
        self.latent_h = None
        self.latent_w = None
        self._query_still: Tensor | None = None
        self._source_yx: Tensor | None = None
        # False: query still seeds H_0 only; demos keep S (paper split).
        self.canvas_update_memory = canvas_update_memory
        # 0 = residual volume. k>0 = IMF rank-k. k<0 = 2D shift of the still.
        self.motion_rank = int(motion_rank)
        if self.motion_rank <= 0:
            self.warp_spatial = 8
        if self.motion_rank == 0:
            self.to_centroid = nn.Linear(bdh.dim, 2)
            nn.init.zeros_(self.to_centroid.weight)
            nn.init.zeros_(self.to_centroid.bias)
            # The query displacement is not linearly decodable from a
            # mean-pooled reasoned hidden (probe: R^2 ~ -0.4 for hidden.mean
            # vs +0.9 for the demo centroid delta). Feed the demo-average
            # displacement in explicitly; this 2x2 (init 2*I: demos are levels
            # 1-2, query 3-4) learns the per-axis scale, to_centroid a residual.
            self.demo_to_shift = nn.Linear(2, 2)
            with torch.no_grad():
                self.demo_to_shift.weight.copy_(2.0 * torch.eye(2))
                self.demo_to_shift.bias.zero_()
            self.log_occ_sigma = nn.Parameter(torch.zeros(()))
        if self.motion_rank < 0:
            self.to_shift = nn.Linear(bdh.dim, 2)
            nn.init.zeros_(self.to_shift.weight)
            nn.init.zeros_(self.to_shift.bias)
        if self.motion_rank > 0:
            k = self.motion_rank
            self.to_motion = nn.Linear(bdh.dim, k)
            self.to_bases = nn.Linear(bdh.dim, k * LATENT_CH)
            self.time_code = nn.Parameter(torch.zeros(32, k))
            nn.init.normal_(self.time_code, std=0.02)
            # Frozen residual teacher. Not a Parameter — m* cannot collapse.
            pad_t = 32
            basis = torch.empty(k, LATENT_CH * pad_t)
            nn.init.normal_(basis, std=1.0 / (LATENT_CH * pad_t) ** 0.5)
            self.register_buffer("motion_basis", basis, persistent=True)

    def _set_grid(self, z: Tensor) -> int:
        if z.ndim != 5:
            raise ValueError(f"expected (B, C, T, H, W), got {tuple(z.shape)}")
        _, _, latent_t, latent_h, latent_w = z.shape
        cells = latent_t * latent_h * latent_w
        self.latent_t = latent_t
        self.latent_h = latent_h
        self.latent_w = latent_w
        return cells

    @property
    def n_cells(self) -> int:
        if self.latent_h is None or self.latent_w is None:
            return N_CELLS
        return self.latent_t * self.latent_h * self.latent_w

    def _marker(self, index: int, batch: int, ref: Tensor) -> Tensor:
        ids = torch.full((batch, 1), index, device=ref.device, dtype=torch.long)
        return self.bdh.token_embed(ids)

    def _raster(self, z: Tensor) -> Tensor:
        cells = rearrange(z, "b c t h w -> b (t h w) c")
        return self.patch_in(cells)

    def _ingest_floats(
        self,
        tokens: Tensor,
        memories: Memory | None,
        update_memory: bool = True,
    ) -> Memory:
        _, memories = self.bdh(
            tokens,
            memories=memories,
            return_memory=True,
            return_logits=False,
            update_memory=update_memory,
        )
        return memories

    def _demo_tokens(self, z_in: Tensor, z_out: Tensor) -> Tensor:
        batch = z_in.shape[0]
        tokens = torch.cat(
            [
                self._marker(IN, batch, z_in),
                self._raster(z_in),
                self._marker(OUT, batch, z_in),
                self._raster(z_out),
                self._marker(EOS, batch, z_in),
            ],
            dim=1,
        )
        return self.bdh.post_embed_norm(tokens)

    def _canvas_tokens(self, z_query_in: Tensor) -> Tensor:
        """H_0 is the full still-volume z_query_in, not t=0 plus empty_cell."""
        batch = z_query_in.shape[0]
        cells = self._set_grid(z_query_in)
        canvas = self._raster(z_query_in)
        if canvas.shape[1] != cells:
            raise RuntimeError(f"canvas length {canvas.shape[1]} != {cells}")
        tokens = torch.cat(
            [self._marker(CANVAS_START, batch, z_query_in), canvas],
            dim=1,
        )
        return self.bdh.post_embed_norm(tokens)

    def ingest_task(self, task_latents: dict[str, Any], n_demos: int = 3) -> Memory:
        demo_in = task_latents["demo_in"][:n_demos]
        demo_out = task_latents["demo_out"][:n_demos]
        if len(demo_in) != len(demo_out):
            raise ValueError("demo_in / demo_out length mismatch")

        memories: Memory | None = None
        # 1B on 24GB: BPTT through ~189 demo chunks OOMs on S. Demos still
        # write S; the graph starts at the query canvas / reason().
        no_grad_demos = getattr(self.bdh, "gradient_checkpoint", False)
        for z_in, z_out in zip(demo_in, demo_out):
            tokens = self._demo_tokens(z_in, z_out)
            for chunk in tokens.split(CHUNK_SIZE, dim=1):
                if no_grad_demos:
                    with torch.no_grad():
                        memories = self._ingest_floats(
                            chunk, memories, update_memory=True
                        )
                else:
                    memories = self._ingest_floats(chunk, memories, update_memory=True)

        canvas = self._canvas_tokens(task_latents["query_in"])
        memories = self._ingest_floats(
            canvas, memories, update_memory=self.canvas_update_memory
        )
        embeds = memories.embeds[:, 1:, :]
        cells = self.n_cells
        if embeds.shape[-2] != cells:
            raise RuntimeError(
                f"H_0 length {embeds.shape[-2]} != {cells} "
                "(canvas must be one unchunked pass; do not ingest it at CHUNK_SIZE)"
            )
        still = task_latents["query_in"].detach()
        self._query_still = still
        self._source_yx = energy_centroid(still)[:, 0].detach()
        # Demo-average displacement: the shift signal the head cannot recover
        # from the reasoned hidden. Direction/axis is right; the query is a
        # higher level than the demos, so demo_to_shift learns the scale.
        self._demo_shift = None
        if self.motion_rank == 0 and demo_in:
            deltas = [
                energy_centroid(z_out)[:, -1] - energy_centroid(z_in)[:, 0]
                for z_in, z_out in zip(demo_in, demo_out)
            ]
            self._demo_shift = torch.stack(deltas, dim=0).mean(dim=0).detach()
        return Memory(memories.tokens_seen, embeds, memories.fast_weight_memories)

    def _occ_sigma(self) -> Tensor:
        # 32px sprite is 4 cells at FakeVAE spatial=8. σ=4 is a 9-cell wash.
        return (1.5 * self.log_occ_sigma.exp()).clamp(0.5, 2.0)

    def _shift_from_hidden(self, hidden: Tensor) -> Tensor:
        corr = self.to_centroid(hidden.mean(dim=1))
        demo = getattr(self, "_demo_shift", None)
        if demo is None:
            return corr
        demo = demo.to(device=hidden.device, dtype=hidden.dtype)
        if demo.shape[0] != hidden.shape[0]:
            demo = demo.expand(hidden.shape[0], 2)
        return self.demo_to_shift(demo) + corr

    def _occupancy_from_hidden(self, hidden: Tensor) -> tuple[Tensor | None, Tensor]:
        residual = rearrange(
            self.to_occupancy(hidden),
            "b (t h w) 1 -> b 1 t h w",
            t=self.latent_t,
            h=self.latent_h,
            w=self.latent_w,
        )
        if self.motion_rank != 0:
            return None, residual
        shift = self._shift_from_hidden(hidden)
        source = self._source_yx
        if source is None:
            source = torch.tensor(
                [(self.latent_h - 1) * 0.5, (self.latent_w - 1) * 0.5],
                device=hidden.device,
                dtype=hidden.dtype,
            ).view(1, 2).expand(hidden.shape[0], 2)
        else:
            source = source.to(device=hidden.device, dtype=hidden.dtype)
            if source.shape[0] != hidden.shape[0]:
                source = source.expand(hidden.shape[0], 2)
        alpha = torch.linspace(
            0.0, 1.0, self.latent_t, device=hidden.device, dtype=hidden.dtype
        )
        centroid = source[:, None, :] + shift[:, None, :] * alpha[None, :, None]
        gauss = gaussian_occupancy_logits(
            centroid, self.latent_h, self.latent_w, sigma=self._occ_sigma()
        )
        return shift, residual + gauss[:, None]

    def to_latent_volume(self, hidden: Tensor) -> Tensor:
        if self.latent_h is None or self.latent_w is None:
            raise RuntimeError("ingest_task must run before to_latent_volume")
        if hidden.shape[-2] != self.n_cells:
            raise ValueError(f"expected {self.n_cells} cells, got {hidden.shape[-2]}")
        if self.motion_rank < 0:
            raise RuntimeError("shift head warps the still in reason(); not to_latent_volume")
        if self.motion_rank <= 0:
            appearance = self.to_latent(hidden)
            shift, logits = self._occupancy_from_hidden(hidden)
            gate = rearrange(
                spatial_softmax_gate(logits[:, 0]), "b t h w -> b (t h w) 1"
            )
            residual = rearrange(
                appearance * gate,
                "b (t h w) c -> b c t h w",
                t=self.latent_t,
                h=self.latent_h,
                w=self.latent_w,
            )
            still = self._query_still
            if still is None or shift is None:
                return residual
            # Composite keeps checker bg and erases the source sprite.
            # Adding gated Linear residual undid dest after ~60 steps
            # (motL1 0.036 -> 0.076). Keep residual for an L2 stay-near-zero.
            still = still.to(device=hidden.device, dtype=hidden.dtype)
            self._last_residual = residual
            return composite_shift(still, shift, spatial=self.warp_spatial)
        return self._low_rank_u(hidden)

    def occupancy_logits(self, hidden: Tensor) -> Tensor:
        """(B, N, dim) -> (B, 1, T, H, W) occupancy logits. Not energy of z."""
        if self.latent_h is None or self.latent_w is None:
            raise RuntimeError("ingest_task must run before occupancy_logits")
        if hidden.shape[-2] != self.n_cells:
            raise ValueError(f"expected {self.n_cells} cells, got {hidden.shape[-2]}")
        return self._occupancy_from_hidden(hidden)[1]

    def shift_token(self, hidden: Tensor) -> Tensor:
        """(B, N, dim) -> (B, 2) dy, dx in latent cells."""
        if self.motion_rank >= 0:
            raise RuntimeError("shift_token requires motion_rank < 0")
        return self.to_shift(hidden.mean(dim=1))

    def motion_token(self, hidden: Tensor) -> Tensor:
        """(B, N, dim) -> (B, k). Mean over time of per-slice tokens."""
        return self.motion_token_seq(hidden).mean(dim=1)

    def motion_token_seq(self, hidden: Tensor) -> Tensor:
        """(B, N, dim) -> (B, T', k). One animation token per latent time."""
        if self.motion_rank <= 0:
            raise RuntimeError("motion_token_seq requires motion_rank > 0")
        if self.latent_h is None or self.latent_w is None:
            raise RuntimeError("ingest_task must run before motion_token_seq")
        cells = rearrange(
            hidden,
            "b (t h w) d -> b t h w d",
            t=self.latent_t,
            h=self.latent_h,
            w=self.latent_w,
        )
        return self.to_motion(cells.mean(dim=(2, 3)))

    def residual_motion_code(self, z: Tensor) -> Tensor:
        """Frozen teacher: (B,C,T,H,W) -> (B,k) from z - z[:,:,0]. No identity."""
        if self.motion_rank <= 0:
            raise RuntimeError("residual_motion_code requires motion_rank > 0")
        if z.ndim != 5:
            raise ValueError(f"expected (B,C,T,H,W), got {tuple(z.shape)}")
        pad_t = self.motion_basis.shape[1] // LATENT_CH
        delta = z - z[:, :, :1]
        pooled = delta.mean(dim=(-2, -1))
        t_len = pooled.shape[-1]
        if t_len < pad_t:
            pooled = torch.nn.functional.pad(pooled, (0, pad_t - t_len))
        elif t_len > pad_t:
            pooled = pooled[:, :, :pad_t]
        flat = rearrange(pooled, "b c t -> b (c t)")
        return flat @ self.motion_basis.t()

    def _low_rank_u(self, hidden: Tensor) -> Tensor:
        """u[c,t,h,w] = sum_k A[h,w,k,c] * m[t,k]. A from t=0, m from each H_t."""
        batch = hidden.shape[0]
        t_len, height, width = self.latent_t, self.latent_h, self.latent_w
        k = self.motion_rank
        if t_len > self.time_code.shape[0]:
            raise RuntimeError(f"T'={t_len} exceeds time_code {self.time_code.shape[0]}")
        cells = rearrange(
            hidden, "b (t h w) d -> b t h w d", t=t_len, h=height, w=width
        )
        # Per latent time: m_t from H_t. Identity bases A from t=0.
        motion = self.to_motion(cells.mean(dim=(2, 3)))
        bases = self.to_bases(cells[:, 0])
        bases = bases.view(batch, height, width, k, LATENT_CH)
        return torch.einsum("bhwkc,btk->bcthw", bases, motion)

    def reason(
        self,
        memories: Memory,
        steps: int,
        return_all: bool = False,
        return_residual: bool = False,
        still: Tensor | None = None,
    ):
        if not isinstance(memories, Memory):
            raise TypeError(
                "reason() takes Memory from ingest_task, not a raw canvas tensor"
            )
        hidden = memories.embeds
        cells = self.n_cells
        if hidden.ndim != 3 or hidden.shape[-2] != cells:
            raise ValueError(
                f"reason() requires embeds.shape[-2]=={cells}, got {tuple(hidden.shape)}"
            )

        all_block_outputs = [hidden]
        hiddens = [hidden]
        for _ in range(steps):
            _, memories = self.bdh(
                hidden,
                memories=memories,
                return_memory=True,
                return_logits=False,
                update_memory=False,
                all_block_outputs=all_block_outputs,
                total_reasoning_iterations=steps,
            )
            hidden = memories.embeds
            hiddens.append(hidden)
        self._last_hidden = hiddens[-1]
        self._step_hiddens = hiddens

        if self.motion_rank < 0:
            if still is None:
                raise RuntimeError("shift head needs the query still")
            volumes = [
                composite_shift(still, self.shift_token(h), spatial=self.warp_spatial)
                for h in hiddens
            ]
        else:
            volumes = [self.to_latent_volume(h) for h in hiddens]
            if still is not None:
                if self.motion_rank > 0:
                    volumes = [integrate_u(volume, still) for volume in volumes]
                else:
                    # Absolute last time is the paper output grid. apply_residual
                    # added the query still on top of the already-composited
                    # warp and ghosted the source sprite (~sprite-frac lastL1).
                    volumes = [lock_t0(volume, still) for volume in volumes]

        out = volumes if return_all else volumes[-1]
        if return_residual:
            return out, all_block_outputs
        return out

    def train_loss(
        self,
        task_latents: dict[str, Any],
        steps: int,
        lambda_dt: float = 1.0,
        lambda_m: float = 1.0,
        lambda_ctx: float = 1.0,
        peak_power: float = 0.0,
        lambda_last: float = 4.0,
        lambda_energy: float = 1.0,
        lambda_mass: float = 1.0,
        return_parts: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, float]]:
        memories = self.ingest_task(task_latents)
        volumes = self.reason(
            memories, steps, return_all=True, still=task_latents["query_in"]
        )
        target = task_latents["query_out"]
        recs = torch.stack(
            [latent_mse_peak(z_hat, target, power=peak_power) for z_hat in volumes]
        )
        dts = torch.stack([latent_dt_mse(z_hat, target) for z_hat in volumes])
        # G_theta(H_R) is the answer. Occupancy CE on r=0 (query still = source)
        # taught dest-empty / source-occupied and left a smear at H_R.
        z_last = volumes[-1]
        apps = last_frame_appearance_mse(z_last, target)
        lasts = last_frame_latent_mse(z_last, target)
        mots = last_frame_motion_mse(z_last, target)
        kls = last_frame_energy_kl(z_last, target)
        dices = last_frame_occupancy_dice(z_last, target)
        masses = last_frame_mass_rel(z_last, target)
        fps = last_frame_fp_energy(z_last, target)
        step_hiddens = getattr(self, "_step_hiddens", None)
        if self.motion_rank == 0 and step_hiddens is not None:
            # Last-time Gaussian CE. Volume CE matched FakeVAE intermediate
            # times that are not linear lerps and overshot the dest.
            ces = last_frame_occupancy_ce(
                self.occupancy_logits(step_hiddens[-1]), target
            )
        else:
            ces = last_frame_occupancy_ce(z_last, target)
        # rec/dt: deep-supervise every r. Last-frame occupancy/appearance/mass
        # only on H_R. FP energy kills smear outside the GT block.
        # Rank-k / IMTalker: occupancy CE is ~2e3 on H3 posterior mean — skip it.
        sprite_occ = self.motion_rank == 0 and float(target.abs().mean().detach()) < 0.5
        if sprite_occ:
            residual = getattr(self, "_last_residual", None)
            res_l2 = (
                residual.pow(2).mean()
                if residual is not None
                else z_last.new_zeros(())
            )
            loss = (
                recs.mean()
                + lambda_dt * dts.mean()
                + lambda_last * (apps + lasts + mots)
                + lambda_energy * (ces + kls + dices)
                + lambda_mass * (masses + fps)
                + res_l2
            )
            parts_res = float(res_l2.detach())
        else:
            parts_res = 0.0
            loss = (
                recs.mean()
                + lambda_dt * dts.mean()
                + lambda_last * mots
            )
        parts = dict(
            rec=float(recs.mean().detach()),
            dt=float(dts.mean().detach()),
            last_mse=float(lasts.detach()),
            last_app=float(apps.detach()),
            last_mot=float(mots.detach()),
            occupancy_ce=float(ces.detach()),
            energy_kl=float(kls.detach()),
            occupancy_dice=float(dices.detach()),
            mass_rel=float(masses.detach()),
            fp_energy=float(fps.detach()),
            residual_l2=parts_res,
        )
        if self.motion_rank <= 0 and (
            self.motion_rank < 0 or hasattr(self, "to_centroid")
        ):
            if self.motion_rank < 0:
                shift_hat = self.shift_token(self._last_hidden)
            else:
                shift_hat = self._shift_from_hidden(self._last_hidden)
            still = task_latents.get("query_in")
            cents = energy_centroid(target)
            if still is not None:
                src = energy_centroid(still)[:, 0]
            else:
                src = cents[:, 0]
            shift_star = (cents[:, -1] - src).detach()
            shift_mse = (shift_hat - shift_star).pow(2).mean()
            # Pixel-centroid teacher overshot dest and lastL1 was already
            # under 8/255; 4x so energy dest actually moves G.
            loss = loss + 4.0 * shift_mse
            parts.update(
                shift_mse=float(shift_mse.detach()),
                shift_dy=float(shift_hat[0, 0].detach()),
                shift_dx=float(shift_hat[0, 1].detach()),
            )
        if self.motion_rank > 0:
            m_hat = self.motion_token_seq(memories.embeds)
            if task_latents.get("query_m") is not None:
                m_star = task_latents["query_m"].detach()
                if m_star.ndim == 2:
                    m_star = m_star.unsqueeze(0)
                demo_codes = []
                for code in task_latents.get("demo_m") or []:
                    token = code.detach()
                    if token.ndim == 2:
                        token = token.unsqueeze(0)
                    demo_codes.append(token)
            else:
                m_star = self.residual_motion_code(target).detach().unsqueeze(1)
                m_star = m_star.expand_as(m_hat)
                demo_codes = [
                    self.residual_motion_code(z_out).detach().unsqueeze(1).expand_as(m_hat)
                    for z_out in task_latents["demo_out"]
                ]
            m_mse = (m_hat - m_star).pow(2).mean()
            m_ctx = torch.stack(demo_codes, dim=0).mean(dim=0)
            ctx_mse = (m_hat - m_ctx).pow(2).mean()
            demo_var = torch.stack(demo_codes, dim=0).var(dim=0).mean()
            loss = loss + lambda_m * m_mse + lambda_ctx * ctx_mse
            parts.update(
                motion_l2=float(m_hat.pow(2).mean().detach()),
                m_mse=float(m_mse.detach()),
                ctx_mse=float(ctx_mse.detach()),
                demo_m_var=float(demo_var.detach()),
            )
        if not return_parts:
            return loss
        return loss, parts
