"""FakeVAE GT path. Must not import MiniMax-H3. Real-VAE tests skip without weights."""

from __future__ import annotations

import sys

import pytest
import torch

from bdh_cq.video_probes import (
    copy_detector,
    energy_centroid,
    identity_probe,
    latent_dt_mse,
    latent_mse,
    recolor_probe,
    seed_extend_probe,
    stamp_copy_probe,
    still_from_first,
    translate_probe,
)
from bdh_cq.video_vae import (
    CLIP_FRAMES,
    DEFAULT_H3_ROOT,
    DEFAULT_VAE_PATH,
    FakeVideoVAE,
    LATENT_CH,
    LATENT_HW,
    LATENT_T,
    LONG_FRAMES,
    PORTRAIT_H,
    PORTRAIT_W,
    decode_pixels,
    encode_mean_video,
    frames_for_latent_t,
    latent_t_for_frames,
    load_visual_vae,
    n_cells,
    visual_vae_weights_available,
)

H3_MARKERS = (
    "klvae",
    "minimax_h3_video_vae",
    "vae_cnn",
    "vae_vit",
    "vae_processor",
    "vae_module",
)


def _h3_modules():
    return [name for name in sys.modules if any(marker in name for marker in H3_MARKERS)]


def test_fake_vae_does_not_import_h3():
    assert _h3_modules() == []


def test_legal_frame_lengths():
    assert latent_t_for_frames(22) == 7
    assert latent_t_for_frames(39) == 12
    assert frames_for_latent_t(7) == 22
    assert frames_for_latent_t(12) == 39
    assert LONG_FRAMES == 39


def test_fake_vae_long_clip():
    vae = FakeVideoVAE()
    video = torch.rand(1, 3, LONG_FRAMES, 128, 128)
    z = encode_mean_video(vae, video)
    assert z.shape == (1, LATENT_CH, 12, LATENT_HW, LATENT_HW)
    pixels = decode_pixels(vae, z)
    assert pixels.shape == (1, 3, LONG_FRAMES, 128, 128)


def test_fake_vae_shape():
    vae = FakeVideoVAE()
    video = torch.rand(2, 3, CLIP_FRAMES, 128, 128)
    z = encode_mean_video(vae, video)
    assert z.shape == (2, LATENT_CH, LATENT_T, LATENT_HW, LATENT_HW)
    assert LATENT_T == 7
    assert LATENT_T != CLIP_FRAMES // 4


def test_fake_vae_identity_rgb_roundtrip_sprite_block():
    vae = FakeVideoVAE(seed=0, spatial=8)
    video = torch.zeros(1, 3, CLIP_FRAMES, 128, 128)
    for t in range(CLIP_FRAMES):
        x0 = 16 + 2 * t
        video[0, 1, t, 32:64, x0 : x0 + 32] = 1.0
    z = encode_mean_video(vae, video)
    assert z.shape == (1, LATENT_CH, LATENT_T, 16, 16)
    pix = decode_pixels(vae, z)
    assert pix.shape == video.shape
    assert float(pix[0, 1].max()) > 0.5
    # last frame's block sits further right than the first
    assert float(pix[0, 1, -1, 48, 80:].mean()) > float(pix[0, 1, 0, 48, 80:].mean())


def test_fake_vae_portrait_canvas():
    vae = FakeVideoVAE()
    video = torch.rand(1, 3, CLIP_FRAMES, PORTRAIT_H, PORTRAIT_W)
    z = encode_mean_video(vae, video)
    assert z.shape == (1, LATENT_CH, LATENT_T, 32, 18)
    assert n_cells(PORTRAIT_H, PORTRAIT_W) == 7 * 32 * 18
    pixels = decode_pixels(vae, z)
    assert pixels.shape == (1, 3, CLIP_FRAMES, PORTRAIT_H, PORTRAIT_W)


def test_fake_vae_temporal_is_resample_not_stride_four():
    vae = FakeVideoVAE()
    still = torch.zeros(1, 3, CLIP_FRAMES, 128, 128)
    still[:, 0] = 0.2
    late = still.clone()
    late[:, :, -4:] = 1.0
    z_still = encode_mean_video(vae, still)
    z_late = encode_mean_video(vae, late)
    assert not torch.allclose(z_still[:, :, -1], z_late[:, :, -1], atol=1e-5)
    assert z_late.shape[2] == 7


def test_fake_vae_still_clip_is_temporally_constant():
    vae = FakeVideoVAE()
    frame = torch.rand(1, 3, 1, 128, 128)
    video = frame.expand(1, 3, CLIP_FRAMES, 128, 128).contiguous()
    z = encode_mean_video(vae, video)
    assert torch.allclose(z[:, :, 1:], z[:, :, :1], atol=1e-5)


def test_fake_vae_is_deterministic():
    a = FakeVideoVAE(seed=0)
    b = FakeVideoVAE(seed=0)
    video = torch.rand(1, 3, CLIP_FRAMES, 128, 128)
    z1 = encode_mean_video(a, video)
    z2 = encode_mean_video(a, video)
    z3 = encode_mean_video(b, video)
    assert torch.equal(z1, z2)
    assert torch.equal(z1, z3)


def test_fake_vae_decode_shape():
    vae = FakeVideoVAE()
    video = torch.rand(1, 3, CLIP_FRAMES, 128, 128)
    z = encode_mean_video(vae, video)
    recon = decode_pixels(vae, z)
    assert recon.shape == (1, 3, CLIP_FRAMES, 128, 128)
    assert recon.min() >= 0 and recon.max() <= 1


def test_identity_probe_on_matching_latents():
    z = torch.randn(1, LATENT_CH, LATENT_T, LATENT_HW, LATENT_HW)
    assert identity_probe(z, z, mse_bound=1e-12)
    assert not identity_probe(z, z + 1.0, mse_bound=1e-3)


def test_copy_detector_still_vs_motion():
    still = torch.randn(1, LATENT_CH, 1, LATENT_HW, LATENT_HW).expand(
        1, LATENT_CH, LATENT_T, LATENT_HW, LATENT_HW
    ).contiguous()
    motion = still.clone()
    motion[:, :, 3:] += 1.0
    still_stats = copy_detector(still, still)
    motion_stats = copy_detector(motion, motion)
    assert still_stats["mse_star_vs_still"] == pytest.approx(0.0, abs=1e-6)
    assert motion_stats["mse_star_vs_still"] > still_stats["mse_star_vs_still"]
    assert torch.equal(still_from_first(motion)[:, :, :1], motion[:, :, :1])


def test_translate_probe_tracks_sprite_on_fake_vae():
    vae = FakeVideoVAE()
    video = torch.zeros(1, 3, CLIP_FRAMES, 128, 128)
    for t in range(CLIP_FRAMES):
        x = 16 + 4 * t
        y = 32
        video[0, :, t, y : y + 24, x : x + 24] = 1.0
    z = encode_mean_video(vae, video)
    cents = energy_centroid(z)[0]
    assert cents[0, 1] < cents[-1, 1]
    assert translate_probe(z, expected_centroids=cents, centroid_tol=1e-5)


def test_stamp_copy_seed_extend_recolor_probes():
    z = torch.zeros(1, LATENT_CH, LATENT_T, LATENT_HW, LATENT_HW)
    patch = torch.ones(LATENT_CH, 2, 2)
    z[:, :, :, 1:3, 2:4] = 1.0
    z[:, :, :, 4:6, 5:7] = 1.0
    anchors = torch.tensor([[1, 2], [4, 5]])
    assert stamp_copy_probe(z, anchors=anchors, sprite_patch=patch, match_tol=1e-6)

    bar = torch.zeros(1, LATENT_CH, LATENT_T, LATENT_HW, LATENT_HW)
    bar[:, :, -1, 3, :5] = 1.0
    assert seed_extend_probe(bar, row=3, expected_extent=5, tol=1.0)

    mask = torch.zeros(LATENT_HW, LATENT_HW, dtype=torch.bool)
    mask[2:4, 2:4] = True
    colored = torch.zeros(1, LATENT_CH, LATENT_T, LATENT_HW, LATENT_HW)
    target = torch.linspace(0, 1, LATENT_CH)
    source = torch.zeros(LATENT_CH)
    colored[:, :, :, mask] = target.view(1, LATENT_CH, 1, 1)
    assert recolor_probe(colored, sprite_mask=mask, source_ref=source, target_ref=target)


def test_latent_losses_shapes():
    z_hat = torch.randn(2, LATENT_CH, LATENT_T, LATENT_HW, LATENT_HW)
    z_star = torch.randn(2, LATENT_CH, LATENT_T, LATENT_HW, LATENT_HW)
    assert latent_mse(z_hat, z_star).ndim == 0
    assert latent_dt_mse(z_hat, z_star).ndim == 0


@pytest.mark.skipif(
    not visual_vae_weights_available(),
    reason="H3 VisualVAE weights not present",
)
def test_gt_is_mean_not_sample():
    vae = load_visual_vae(DEFAULT_H3_ROOT, DEFAULT_VAE_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = vae.to(device)
    clip = torch.rand(1, 3, CLIP_FRAMES, 128, 128, device=device)
    z1 = encode_mean_video(vae, clip)
    z2 = encode_mean_video(vae, clip)
    assert z1.shape == (1, LATENT_CH, LATENT_T, LATENT_HW, LATENT_HW)
    assert torch.equal(z1, z2)

    from bdh_cq.video_vae import _prepare_video

    prepared = _prepare_video(vae, clip)
    sampled_a = vae.encode_base(prepared, False)
    sampled_b = vae.encode_base(prepared, False)
    assert sampled_a.shape == z1.shape
    assert not torch.equal(sampled_a, sampled_b)
    assert not torch.equal(sampled_a, z1)
