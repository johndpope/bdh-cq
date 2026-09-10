"""Last-frame ICQ gate: wash vs compact square is a fail, not a PSNR pass."""

from __future__ import annotations

import math

import torch

from bdh_cq.video_probes import (
    LAST_FRAME_L1,
    LAST_FRAME_MOTION_L1,
    gaussian_occupancy_logits,
    last_frame_appearance_mse,
    last_frame_energy_kl,
    last_frame_fp_energy,
    last_frame_latent_mse,
    last_frame_mass_rel,
    last_frame_mismatch,
    last_frame_occupied_frac,
    last_frame_occupancy_ce,
    last_frame_occupancy_dice,
    last_frame_support,
    pixel_last_frame_l1,
    pixel_last_frame_motion_l1,
    spatial_softmax_gate,
    volume_occupancy_ce,
)


def _square_clip(value: float = 1.0) -> torch.Tensor:
    clip = torch.zeros(1, 3, 22, 32, 32)
    clip[:, :, -1, 8:16, 8:16] = value
    return clip


def test_last_frame_l1_gate_is_eight_over_255():
    assert LAST_FRAME_L1 == 8.0 / 255.0


def test_last_frame_mismatch_wash_vs_square():
    gt = _square_clip(1.0)
    wash = torch.full((1, 3, 22, 32, 32), 0.2)
    assert last_frame_mismatch(wash, gt)
    assert pixel_last_frame_l1(wash, gt) > LAST_FRAME_L1


def test_last_frame_mismatch_identical_passes():
    gt = _square_clip(1.0)
    assert not last_frame_mismatch(gt, gt)
    assert pixel_last_frame_l1(gt, gt) == 0.0


def test_last_frame_mismatch_blurry_square_can_pass():
    gt = _square_clip(1.0)
    pred = gt.clone()
    pred[:, :, -1, 8:16, 8:16] = 0.97
    assert not last_frame_mismatch(pred, gt)


def test_last_frame_mismatch_dim_peak_fails_even_if_l1_is_modest():
    gt = _square_clip(1.0)
    dim = torch.zeros_like(gt)
    dim[:, :, -1, 8:16, 8:16] = 0.2
    assert last_frame_mismatch(dim, gt)


def test_last_frame_mismatch_idle_vs_open_mouth():
    """Dark-bg mean L1 can be modest while the mouth is wrong."""
    gt = torch.zeros(1, 3, 22, 64, 64)
    gt[:, :, :, 16:48, 16:48] = 0.25
    gt[:, :, -1, 36:48, 22:42] = 1.0
    pred = gt.clone()
    pred[:, :, -1] = pred[:, :, 0]
    # Mean L1 can sit between 8/255 and 20/255; motion-weighted still fails.
    assert pixel_last_frame_l1(pred, gt) < 20.0 / 255.0
    assert last_frame_mismatch(pred, gt)
    assert pixel_last_frame_motion_l1(pred, gt) > pixel_last_frame_l1(pred, gt)


def test_last_frame_mismatch_compact_dest_passes_despite_motl1():
    """FakeVAE trail: compact 32x32 vs 40px GT, lastL1 < 8/255, motL1 > 0.05."""
    gt = torch.zeros(1, 3, 22, 128, 128)
    gt[:, :, -1, 88:128, 0:32] = 0.96
    pred = torch.zeros_like(gt)
    pred[:, :, -1, 96:128, 0:32] = 0.93
    assert pixel_last_frame_l1(pred, gt) <= LAST_FRAME_L1
    assert pixel_last_frame_motion_l1(pred, gt) > LAST_FRAME_MOTION_L1
    assert last_frame_occupied_frac(pred) < last_frame_occupied_frac(gt)
    assert not last_frame_mismatch(pred, gt)


def test_energy_kl_wash_exceeds_matched_square():
    z_gt = torch.zeros(1, 24, 7, 8, 8)
    z_gt[:, :3, -1, 2:6, 2:6] = 0.9
    z_wash = torch.full((1, 24, 7, 8, 8), 0.2)
    z_ok = z_gt.clone()
    kl_wash = float(last_frame_energy_kl(z_wash, z_gt))
    kl_ok = float(last_frame_energy_kl(z_ok, z_gt))
    dice_wash = float(last_frame_occupancy_dice(z_wash, z_gt))
    dice_ok = float(last_frame_occupancy_dice(z_ok, z_gt))
    assert kl_wash > kl_ok + 0.5
    assert dice_wash > dice_ok + 0.3
    assert float(last_frame_latent_mse(z_ok, z_gt)) == 0.0
    assert float(last_frame_mass_rel(z_ok, z_gt)) == 0.0
    assert float(last_frame_mass_rel(z_wash, z_gt)) > 0.3
    z_flood = torch.ones_like(z_gt)
    assert float(last_frame_mass_rel(z_flood, z_gt)) > 1.0


def test_occupancy_ce_uniform_exceeds_peaked_support():
    z_gt = torch.zeros(1, 24, 7, 8, 8)
    z_gt[:, :3, -1, 2:6, 2:6] = 0.9
    logits_wash = torch.zeros(1, 8, 8)
    logits_ok = torch.full((1, 8, 8), -4.0)
    logits_ok[:, 2:6, 2:6] = 4.0
    ce_wash = float(last_frame_occupancy_ce(logits_wash, z_gt))
    ce_ok = float(last_frame_occupancy_ce(logits_ok, z_gt))
    assert ce_wash > ce_ok + 0.5


def test_occupancy_ce_logits_have_grad_at_zero_energy():
    z_gt = torch.zeros(1, 24, 7, 8, 8)
    z_gt[:, :3, -1, 2:6, 2:6] = 0.9
    logits = torch.zeros(1, 8, 8, requires_grad=True)
    last_frame_occupancy_ce(logits, z_gt).backward()
    assert logits.grad is not None
    # dest cell must increase; a corner bg cell must decrease
    assert float(logits.grad[0, 3, 3]) < 0
    assert float(logits.grad[0, 0, 0]) > 0


def test_appearance_mse_upweights_occupied():
    z_gt = torch.zeros(1, 24, 7, 8, 8)
    z_gt[:, :3, -1, 2:6, 2:6] = 0.9
    miss_sprite = z_gt.clone()
    miss_sprite[:, :, -1, 2:6, 2:6] = 0
    miss_bg = z_gt.clone()
    support = last_frame_support(z_gt)
    miss_bg[:, :, -1] = torch.where(
        support[:, None].bool(), z_gt[:, :, -1], torch.full_like(z_gt[:, :, -1], 0.2)
    )
    assert float(last_frame_appearance_mse(z_gt, z_gt)) == 0.0
    assert float(last_frame_appearance_mse(miss_sprite, z_gt)) > float(
        last_frame_appearance_mse(miss_bg, z_gt)
    )
    assert tuple(support.shape) == (1, 8, 8)
    assert int(support[:, 2:6, 2:6].sum()) == 16
    assert int(support.sum()) == 16


def test_gaussian_occupancy_is_compact_and_tracks_centroid():
    dest = torch.tensor([[3.5, 4.5]])
    logits = gaussian_occupancy_logits(dest, 8, 8, sigma=1.5)
    assert tuple(logits.shape) == (1, 1, 8, 8)
    peak = int(logits.reshape(1, -1).argmax(dim=-1)[0])
    assert peak // 8 in (3, 4) and peak % 8 in (4, 5)
    gate = spatial_softmax_gate(logits)
    assert float(gate.amax() / gate.mean()) > 4.0
    z_gt = torch.zeros(1, 24, 7, 8, 8)
    z_gt[:, :3, -1, 2:6, 2:6] = 0.9
    ce_ok = float(last_frame_occupancy_ce(logits[:, 0], z_gt))
    ce_wash = float(last_frame_occupancy_ce(torch.zeros(1, 8, 8), z_gt))
    assert ce_ok < ce_wash - 0.5


def test_spatial_softmax_gate_peaks_dest_and_kills_bg():
    logits = torch.full((1, 8, 8), -4.0)
    logits[:, 2:6, 2:6] = 4.0
    gate = spatial_softmax_gate(logits)
    assert tuple(gate.shape) == (1, 8, 8)
    assert float(gate[:, 2:6, 2:6].min()) > 0.9
    assert float(gate[:, 0, 0]) < 0.01
    uniform = spatial_softmax_gate(torch.zeros(1, 8, 8))
    assert torch.allclose(uniform, torch.ones_like(uniform))


def test_volume_occupancy_ce_uniform_is_log_hw():
    z_gt = torch.zeros(1, 24, 7, 8, 8)
    z_gt[:, :3, :, 2:6, 2:6] = 0.9
    ce = float(volume_occupancy_ce(torch.zeros(1, 7, 8, 8), z_gt))
    assert abs(ce - math.log(64)) < 1e-4


def test_gaussian_occupancy_logits_peak_at_centroid():
    logits = gaussian_occupancy_logits(torch.tensor([[[3.0, 4.0]]]), 8, 8, sigma=1.0)
    assert tuple(logits.shape) == (1, 1, 8, 8)
    idx = int(logits.reshape(-1).argmax().item())
    assert (idx // 8, idx % 8) == (3, 4)
    gate = spatial_softmax_gate(logits)
    assert float(gate[0, 0, 3, 4]) > 0.5
    assert float(gate[0, 0, 0, 0]) < 0.05


def test_fp_energy_flood_exceeds_compact():
    z_gt = torch.zeros(1, 24, 7, 8, 8)
    z_gt[:, :3, -1, 2:6, 2:6] = 0.9
    z_flood = torch.ones_like(z_gt)
    z_smear = z_gt.clone()
    z_smear[:, :3, -1, 1:7, 1:7] = 0.6
    assert float(last_frame_fp_energy(z_gt, z_gt)) == 0.0
    assert float(last_frame_fp_energy(z_flood, z_gt)) > float(
        last_frame_fp_energy(z_smear, z_gt)
    )
    assert float(last_frame_fp_energy(z_smear, z_gt)) > 0.0
