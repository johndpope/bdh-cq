"""Latent-grid probes for v0 video oracles. No BDH, no MiniMax-H3."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from bdh_cq.video_vae import LATENT_HW, LATENT_T


def latent_mse(z_hat: Tensor, z_star: Tensor) -> Tensor:
    return (z_hat - z_star).pow(2).mean()


def latent_mse_peak(z_hat: Tensor, z_star: Tensor, power: float = 0.0) -> Tensor:
    """Uniform rec if power<=0; later latent times weigh more if power>0."""
    err = (z_hat - z_star).pow(2)
    if power <= 0:
        return err.mean()
    t = err.shape[2]
    idx = torch.arange(t, device=err.device, dtype=err.dtype)
    weight = (idx / max(t - 1, 1)).pow(power) + 0.1
    weight = weight / weight.mean()
    return (err * weight.view(1, 1, t, 1, 1)).mean()


def latent_mse_motion(z_hat: Tensor, z_star: Tensor, floor: float = 0.05) -> Tensor:
    """Weight rec by GT |z_t - z_0| so a small moving sprite is not drowned by bg."""
    err = (z_hat - z_star).pow(2)
    motion = (z_star - z_star[:, :, :1]).pow(2).mean(dim=1, keepdim=True)
    weight = motion / (motion.mean() + 1e-8)
    weight = weight.clamp(min=floor)
    return (err * weight).mean()


def latent_dt_mse(z_hat: Tensor, z_star: Tensor) -> Tensor:
    d_hat = z_hat[:, :, 1:] - z_hat[:, :, :-1]
    d_star = z_star[:, :, 1:] - z_star[:, :, :-1]
    return (d_hat - d_star).pow(2).mean()


def pixel_psnr(pred: Tensor, target: Tensor) -> float:
    mse = float((pred - target).pow(2).mean())
    if mse <= 0:
        return math.inf
    return 10.0 * math.log10(1.0 / mse)


def still_from_first(z: Tensor) -> Tensor:
    """repeat(z[:, :, :1], T) on the latent time axis."""
    return z[:, :, :1].expand(-1, -1, z.shape[2], -1, -1).contiguous()


def copy_detector(z_hat: Tensor, z_star: Tensor) -> dict[str, float]:
    still = still_from_first(z_star)
    hat_self = still_from_first(z_hat)
    return {
        "mse_hat_vs_still": float(latent_mse(z_hat, still)),
        "mse_star_vs_still": float(latent_mse(z_star, still)),
        "mse_hat_vs_own_still": float(latent_mse(z_hat, hat_self)),
    }


STILL_L1 = 1.0 / 255.0
# Policy knob for library admission, not a layer in F. H3 talking-head
# copy_star is ~0.8 even when G is still; FakeVAE stills are ~1e-3.
LATENT_STILL_MSE = 0.05


def latent_vs_still_mse(z: Tensor) -> float:
    """How far z is from repeat(z[:,:,0]). Cheap; no G."""
    return float(latent_mse(z, still_from_first(z)).detach())


def admit_clip(
    *,
    latent_vs_still: float,
    decode_t_l1: float | None,
    mode: str = "decode",
) -> bool:
    """Library filter only. Never call from reason() / train_loss.

    decode: keep if G-roundtrip is a clip (film-out visible).
    latent_or_decode: keep unless *both* latent and decode say still.
    """
    decode_still = decode_t_l1 is not None and decode_t_l1 < STILL_L1
    latent_still = latent_vs_still < LATENT_STILL_MSE
    if mode == "decode":
        return not decode_still
    if mode == "latent_or_decode":
        return not (latent_still and decode_still)
    raise ValueError(f"admit mode must be decode|latent_or_decode, got {mode!r}")


def pixel_temporal_l1(video: Tensor) -> float:
    """Mean |frame[t+1]-frame[t]| over a (B,3,T,H,W) clip in [0,1]."""
    if video.shape[2] < 2:
        return 0.0
    return float((video[:, :, 1:] - video[:, :, :-1]).abs().mean())


def pixel_t0_t21_l1(video: Tensor) -> float:
    """End-to-end |last - first|. Talking-head morphs fail consecutive tL1."""
    if video.shape[2] < 2:
        return 0.0
    return float((video[:, :, -1] - video[:, :, 0]).abs().mean())


def pixel_last_frame_l1(pred: Tensor, target: Tensor) -> float:
    """Mean |pred[-1] - target[-1]| over a (B,3,T,H,W) pair in [0,1]."""
    return float((pred[:, :, -1] - target[:, :, -1]).abs().mean())


# 8/255: compact dest block, not a wash. Full-frame warp_still of the
# still also translates the checker and floors at ~18/255; composite_shift
# (sprite onto static bg) sits at ~11/255, under this gate. Wash vs square
# is ~30-50/255. Peak check kills a mean-gray that sneaks L1.
LAST_FRAME_L1 = 8.0 / 255.0
# Idle vs laugh on a dark 9:16 canvas is only ~0.037 mean L1. Weight by
# |GT last - GT first| so the mouth counts. Not a sprite fail gate:
# FakeVAE trail already puts oracle motL1 ~0.047.
LAST_FRAME_MOTION_L1 = 0.05
# CLASS_WEIGHTS analogue: bg 0.5, occupied colors 3.0. Occupied = energy
# at least OCC_FRAC of the last-frame peak (sprite ~6-8% of FakeVAE spatial=8).
OCC_WEIGHT = 3.0
BG_WEIGHT = 0.5
OCC_FRAC = 0.35


def pixel_last_frame_motion_l1(pred: Tensor, target: Tensor) -> float:
    """Last-frame L1 weighted by where the GT clip actually changed."""
    err = (pred[:, :, -1] - target[:, :, -1]).abs()
    weight = (target[:, :, -1] - target[:, :, 0]).abs()
    weight = weight / (weight.mean() + 1e-8)
    weight = weight.clamp(min=0.1, max=20.0)
    return float((err * weight).mean())


def last_frame_occupied_frac(video: Tensor, occ_frac: float = OCC_FRAC) -> float:
    """Fraction of last-frame pixels at least occ_frac of the peak. Wash~1, 32px~0.06."""
    last = video[:, :, -1].mean(dim=1)
    peak = last.flatten(1).amax(dim=-1).clamp_min(1e-8).view(-1, 1, 1)
    return float((last >= occ_frac * peak).to(dtype=last.dtype).mean())


def last_frame_mismatch(
    pred: Tensor, target: Tensor, max_l1: float = LAST_FRAME_L1
) -> bool:
    """ICQ fail: the query's last frame does not match GT.

    Pass is a compact bright dest block with lastL1 <= 8/255 (paper output
    grid, pixel-eval on FakeVAE). motL1 is a talking-head mouth probe: oracle
    composite_shift already floors at ~0.047 from the FakeVAE trail, and a
    0.4-cell overshoot of the pixel-centroid teacher trips 0.05 while lastL1
    is still under the gate. Compact dest (frac within 2x of GT) is not a fail.

    A temporally-still GT (identity / stamp_copy / recolor) is not exempt from
    the L1 gate — a converged identity roundtrips well under 8/255, and
    stamp_copy / recolor must actually apply the edit. The still-vs-still
    CLIP_FAIL is suppressed by the caller, not here; the motion-weighted
    fallback below is skipped only when GT has no motion to weight by.
    """
    gt_is_still = pixel_t0_t21_l1(target) < STILL_L1
    if pixel_last_frame_l1(pred, target) > max_l1:
        return True
    pred_peak = float(pred[:, :, -1].amax())
    tgt_peak = float(target[:, :, -1].amax())
    if pred_peak < 0.5 * max(tgt_peak, 1e-6):
        return True
    pred_frac = last_frame_occupied_frac(pred)
    tgt_frac = last_frame_occupied_frac(target)
    if tgt_frac > 0.0 and pred_frac > 2.0 * tgt_frac:
        return True
    if pred_frac >= 0.5 * max(tgt_frac, 1e-8):
        return False
    if gt_is_still:
        return False
    return pixel_last_frame_motion_l1(pred, target) > LAST_FRAME_MOTION_L1


def last_frame_latent_mse(z_hat: Tensor, z_star: Tensor) -> Tensor:
    """MSE on the last latent time only. Uniform rec can hide a wash here."""
    return (z_hat[:, :, -1] - z_star[:, :, -1]).pow(2).mean()


def last_frame_motion_mse(z_hat: Tensor, z_star: Tensor) -> Tensor:
    """Last-frame latent MSE weighted by |z_star[-1] - z_star[0]| (mouth, not bg)."""
    err = (z_hat[:, :, -1] - z_star[:, :, -1]).pow(2)
    weight = (z_star[:, :, -1] - z_star[:, :, 0]).pow(2).mean(dim=1, keepdim=True)
    weight = weight / (weight.mean() + 1e-8)
    weight = weight.clamp(min=0.1, max=20.0)
    return (err * weight).mean()


def last_frame_support(z_star: Tensor) -> Tensor:
    """(B, H, W) last-time cells whose energy is the answer support."""
    energy = energy_map(z_star)[:, -1]
    peak = energy.flatten(1).amax(dim=-1).clamp_min(1e-8).view(-1, 1, 1)
    return (energy >= OCC_FRAC * peak).to(dtype=energy.dtype)


def last_frame_occupancy_ce(logits: Tensor, z_star: Tensor) -> Tensor:
    """Spatial-softmax CE: which last-time cells are the answer.

    `logits` is (B, H, W), (B, 1, H, W), (B, 1, T, H, W), or a (B, C, T, H, W)
    volume (energy of last time used as logits). Softmax is over space, not
    channels — competition the omit-self prefix-sum does not have.
    """
    if logits.ndim == 5:
        logits = logits[:, 0, -1] if logits.shape[1] == 1 else energy_map(logits)[:, -1]
    elif logits.ndim == 4:
        logits = logits[:, 0]
    elif logits.ndim != 3:
        raise ValueError(f"occupancy logits rank {logits.ndim} not in {{3,4,5}}")
    energy = energy_map(z_star)[:, -1].flatten(1).clamp_min(0)
    target = energy / energy.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    log_pred = logits.flatten(1).log_softmax(dim=-1)
    return -(target * log_pred).sum(dim=-1).mean()


def volume_occupancy_ce(logits: Tensor, z_star: Tensor) -> Tensor:
    """Spatial-softmax CE at every latent time. Uniform logits = ln(H*W).

    Last-time-only CE left t=1..T-2 as a ones-gate wash once appearance grew.
    """
    if logits.ndim == 5:
        logits = logits[:, 0] if logits.shape[1] == 1 else energy_map(logits)
    elif logits.ndim == 3:
        return last_frame_occupancy_ce(logits, z_star)
    elif logits.ndim != 4:
        raise ValueError(f"volume occupancy logits rank {logits.ndim} not in {{3,4,5}}")
    energy = energy_map(z_star).flatten(2).clamp_min(0)
    target = energy / energy.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    log_pred = logits.flatten(2).log_softmax(dim=-1)
    return -(target * log_pred).sum(dim=-1).mean()


def gaussian_occupancy_logits(
    centroid_yx: Tensor,
    height: int,
    width: int,
    sigma: float | Tensor = 1.5,
) -> Tensor:
    """Compact spatial logits peaked at centroid_yx. (B, T, 2) or (B, 2) -> (B, T, H, W).

    Per-cell Linear(dim,1) on omit-self prefix-sum hidden is spatially uniform,
    so occupancy CE stuck at log(H*W). One (y, x) cannot cancel with itself.
    """
    if height < 1 or width < 1:
        raise ValueError(f"occupancy grid must be positive, got {height}x{width}")
    if centroid_yx.ndim == 2:
        centroid_yx = centroid_yx.unsqueeze(1)
    if centroid_yx.ndim != 3 or centroid_yx.shape[-1] != 2:
        raise ValueError(f"centroid_yx expected (B, T, 2) or (B, 2), got {tuple(centroid_yx.shape)}")
    ys = torch.arange(height, device=centroid_yx.device, dtype=centroid_yx.dtype)
    xs = torch.arange(width, device=centroid_yx.device, dtype=centroid_yx.dtype)
    cy = centroid_yx[..., 0][:, :, None, None]
    cx = centroid_yx[..., 1][:, :, None, None]
    dist2 = (ys.view(1, 1, height, 1) - cy).pow(2) + (xs.view(1, 1, 1, width) - cx).pow(2)
    sigma_t = torch.as_tensor(sigma, device=centroid_yx.device, dtype=centroid_yx.dtype)
    sigma_t = sigma_t.clamp_min(0.5)
    while sigma_t.ndim < dist2.ndim:
        sigma_t = sigma_t.unsqueeze(-1)
    return -dist2 / (2.0 * sigma_t.pow(2))


def spatial_softmax_gate(logits: Tensor) -> Tensor:
    """Peak-normalized spatial softmax. Independent sigmoid cannot kill smear.

    Accepts (B, H, W), (B, T, H, W), or (B, 1, T, H, W). Softmax over H*W per
    time, then /max so the dest peak stays 1 (compact bright block). Uniform
    logits -> ones (appearance is zeros-init so z stays 0).
    """
    squeeze_t = False
    if logits.ndim == 5:
        logits = logits[:, 0]
    if logits.ndim == 3:
        logits = logits.unsqueeze(1)
        squeeze_t = True
    elif logits.ndim != 4:
        raise ValueError(f"occupancy gate rank {logits.ndim} not in {{3,4,5}}")
    batch, t_len, height, width = logits.shape
    flat = logits.reshape(batch, t_len, height * width)
    pred = flat.log_softmax(dim=-1).exp()
    gate = pred / pred.amax(dim=-1, keepdim=True).clamp_min(1e-8)
    gate = gate.reshape(batch, t_len, height, width)
    if squeeze_t:
        return gate[:, 0]
    return gate


def last_frame_fp_energy(z_hat: Tensor, z_star: Tensor) -> Tensor:
    """Mean last-frame energy on GT background. Penalizes a wide smear/flood."""
    support = last_frame_support(z_star)
    energy = energy_map(z_hat)[:, -1]
    return (energy * (1.0 - support)).mean()


def last_frame_appearance_mse(
    z_hat: Tensor,
    z_star: Tensor,
    occ_weight: float = OCC_WEIGHT,
    bg_weight: float = BG_WEIGHT,
) -> Tensor:
    """Last-time L2 with CLASS_WEIGHTS, inverse-frequency over space.

    Per-cell x3/x0.5 still lets 92% background dominate (0.46 vs 0.24). Divide
    by occupied/bg fraction so dest cells own the loss the way rare colors do
    in icq.CLASS_WEIGHTS.
    """
    err = (z_hat[:, :, -1] - z_star[:, :, -1]).pow(2)
    support = last_frame_support(z_star)
    occ_frac = support.flatten(1).mean(dim=-1).clamp_min(1e-4).view(-1, 1, 1)
    bg_frac = (1.0 - occ_frac).clamp_min(1e-4)
    weight = support * (occ_weight / occ_frac) + (1.0 - support) * (
        bg_weight / bg_frac
    )
    weight = weight / weight.flatten(1).mean(dim=-1).view(-1, 1, 1).clamp_min(1e-8)
    return (err * weight[:, None]).mean()


def last_frame_energy_kl(z_hat: Tensor, z_star: Tensor) -> Tensor:
    """Spatial KL of last-frame energy. A wash is near-uniform; GT is peaked."""
    e_hat = energy_map(z_hat)[:, -1].flatten(1).clamp_min(0)
    e_star = energy_map(z_star)[:, -1].flatten(1).clamp_min(0)
    p = e_hat / e_hat.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    q = e_star / e_star.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return (q * (q.clamp_min(1e-8).log() - p.clamp_min(1e-8).log())).sum(dim=-1).mean()


def last_frame_occupancy_dice(z_hat: Tensor, z_star: Tensor) -> Tensor:
    """1 - Dice of last-frame energy. Penalizes both wash (no overlap) and flood."""
    e_hat = energy_map(z_hat)[:, -1]
    e_star = energy_map(z_star)[:, -1]
    scale = e_star.flatten(1).amax(dim=-1).clamp_min(1e-6).view(-1, 1, 1)
    a = (e_hat / scale).clamp(0, 1)
    b = (e_star / scale).clamp(0, 1)
    inter = (a * b).flatten(1).sum(dim=-1)
    den = a.flatten(1).sum(dim=-1) + b.flatten(1).sum(dim=-1)
    return (1.0 - 2.0 * inter / (den + 1e-8)).mean()


def last_frame_mass_rel(z_hat: Tensor, z_star: Tensor) -> Tensor:
    """|pred last-frame energy - GT| / GT. Stops occupancy from flooding the canvas."""
    m_hat = energy_map(z_hat)[:, -1].sum(dim=(-2, -1))
    m_star = energy_map(z_star)[:, -1].sum(dim=(-2, -1))
    return ((m_hat - m_star).abs() / m_star.clamp_min(1e-8)).mean()


def is_still_clip(video: Tensor, min_l1: float = STILL_L1) -> bool:
    """A clip that does not move is a failed generation."""
    return pixel_temporal_l1(video) < min_l1


def energy_map(z: Tensor) -> Tensor:
    """(B, C, T, H, W) -> (B, T, H, W) mean-square energy."""
    return z.pow(2).mean(dim=1)


def energy_centroid(z: Tensor) -> Tensor:
    """Sprite centroid on the 8x8 latent grid. Returns (B, T, 2) as (y, x)."""
    energy = energy_map(z)
    b, t, height, width = energy.shape
    med = energy.reshape(b, t, -1).median(dim=-1).values[..., None, None]
    mass = (energy - med).clamp_min(0)
    mass = mass / (mass.sum(dim=(-2, -1), keepdim=True) + 1e-8)
    ys = torch.arange(height, device=z.device, dtype=mass.dtype)
    xs = torch.arange(width, device=z.device, dtype=mass.dtype)
    cy = (mass * ys[None, None, :, None]).sum(dim=(-2, -1))
    cx = (mass * xs[None, None, None, :]).sum(dim=(-2, -1))
    return torch.stack([cy, cx], dim=-1)


def _as_bt2(centroids: Tensor, batch: int, frames: int) -> Tensor:
    if centroids.ndim == 2:
        centroids = centroids.unsqueeze(0).expand(batch, -1, -1)
    if centroids.shape[-2] != frames:
        raise ValueError(f"centroid T={centroids.shape[-2]} != {frames}")
    return centroids


def identity_probe(z_hat: Tensor, z_star: Tensor, mse_bound: float = 1e-3) -> bool:
    return float(latent_mse(z_hat, z_star)) <= mse_bound


def translate_probe(
    z_hat: Tensor,
    *,
    expected_centroids: Tensor,
    centroid_tol: float = 1.0,
    bg_mask: Tensor | None = None,
    bg_phase_tol: float = 0.05,
) -> bool:
    """Sprite centroid tracks the oracle; masked background phase ~ 0."""
    cents = energy_centroid(z_hat)
    expected = _as_bt2(expected_centroids, cents.shape[0], cents.shape[1]).to(
        device=cents.device, dtype=cents.dtype
    )
    if float((cents - expected).norm(dim=-1).max()) > centroid_tol:
        return False
    if bg_mask is None:
        return True
    bg = z_hat * bg_mask.to(device=z_hat.device, dtype=z_hat.dtype)
    phase = (bg[:, :, 1:] - bg[:, :, :1]).abs().mean()
    return float(phase) <= bg_phase_tol


def pan_probe(
    z_hat: Tensor,
    *,
    expected_world_centroids: Tensor,
    pan_latent: Tensor,
    centroid_tol: float = 1.0,
    bg_mask: Tensor | None = None,
    bg_phase_ref: Tensor | None = None,
    bg_phase_tol: float = 0.1,
) -> bool:
    """World-static sprite: screen centroid + pan; background follows pan."""
    cents = energy_centroid(z_hat)
    pan = _as_bt2(pan_latent, cents.shape[0], cents.shape[1]).to(
        device=cents.device, dtype=cents.dtype
    )
    world = cents + pan
    expected = _as_bt2(expected_world_centroids, cents.shape[0], cents.shape[1]).to(
        device=cents.device, dtype=cents.dtype
    )
    if float((world - expected).norm(dim=-1).max()) > centroid_tol:
        return False
    if bg_mask is None or bg_phase_ref is None:
        return True
    mask = bg_mask.to(device=z_hat.device, dtype=z_hat.dtype)
    phase = (z_hat * mask).mean(dim=(1, 3, 4))
    ref = bg_phase_ref.to(device=z_hat.device, dtype=z_hat.dtype)
    return float((phase - ref).abs().mean()) <= bg_phase_tol


def translate_pan_probe(
    z_hat: Tensor,
    *,
    expected_centroids: Tensor,
    expected_world_centroids: Tensor,
    pan_latent: Tensor,
    centroid_tol: float = 1.0,
    bg_mask: Tensor | None = None,
    bg_phase_tol: float = 0.05,
    bg_phase_ref: Tensor | None = None,
    pan_bg_phase_tol: float = 0.1,
) -> bool:
    """Composition cliff: both translate and pan probes must pass."""
    return translate_probe(
        z_hat,
        expected_centroids=expected_centroids,
        centroid_tol=centroid_tol,
        bg_mask=None,
        bg_phase_tol=bg_phase_tol,
    ) and pan_probe(
        z_hat,
        expected_world_centroids=expected_world_centroids,
        pan_latent=pan_latent,
        centroid_tol=centroid_tol,
        bg_mask=bg_mask,
        bg_phase_ref=bg_phase_ref,
        bg_phase_tol=pan_bg_phase_tol,
    )


def stamp_copy_probe(
    z_hat: Tensor,
    *,
    anchors: Tensor,
    sprite_patch: Tensor,
    match_tol: float = 0.15,
) -> bool:
    """Each latent-grid anchor matches the sprite patch."""
    # anchors: (N, 2) as (y, x) top-left on the 8x8 grid
    # sprite_patch: (C, ph, pw)
    _, channels, _, _, _ = z_hat.shape
    ph, pw = sprite_patch.shape[-2:]
    patch = sprite_patch.to(device=z_hat.device, dtype=z_hat.dtype)
    if patch.ndim == 3:
        patch = patch[None, :, None, :, :]  # (1, C, 1, ph, pw) broadcasts over B, T
    for y, x in anchors.tolist():
        y_i, x_i = int(y), int(x)
        cell = z_hat[:, :, :, y_i : y_i + ph, x_i : x_i + pw]
        if cell.shape[-2:] != (ph, pw):
            return False
        if float((cell - patch).pow(2).mean()) > match_tol:
            return False
    return channels == sprite_patch.shape[0]


def seed_extend_probe(
    z_hat: Tensor,
    *,
    row: int,
    expected_extent: float,
    tol: float = 1.0,
    last_frame: bool = True,
) -> bool:
    """Filled extent along the bar row is within 1 latent cell of target."""
    energy = energy_map(z_hat)
    line = energy[:, -1 if last_frame else 0, row, :]
    peak = line.amax(dim=-1, keepdim=True).clamp_min(1e-8)
    filled = (line > 0.5 * peak).to(dtype=line.dtype).sum(dim=-1)
    return bool(((filled - expected_extent).abs() <= tol).all())


def recolor_probe(
    z_hat: Tensor,
    *,
    sprite_mask: Tensor,
    source_ref: Tensor,
    target_ref: Tensor,
) -> bool:
    """Sprite-region mean is closer to the target color than the source."""
    mask = sprite_mask.to(device=z_hat.device, dtype=torch.bool)
    if mask.shape[-2:] != (LATENT_HW, LATENT_HW) and mask.shape != (LATENT_T, LATENT_HW, LATENT_HW):
        raise ValueError(f"sprite_mask spatial shape {tuple(mask.shape)} is not a latent grid")
    if mask.ndim == 2:
        cells = z_hat[:, :, :, mask]
    else:
        cells = z_hat[:, :, mask]
    mean = cells.mean(dim=tuple(range(2, cells.ndim)))
    target = target_ref.to(device=z_hat.device, dtype=z_hat.dtype).reshape(1, -1)
    source = source_ref.to(device=z_hat.device, dtype=z_hat.dtype).reshape(1, -1)
    d_tgt = (mean - target).pow(2).mean()
    d_src = (mean - source).pow(2).mean()
    return float(d_tgt) < float(d_src)
