"""Frozen VisualVAE GT factory for BDH-CQ video v0.

FakeVideoVAE is CPU-testable and does not import MiniMax-H3. The real
path loads AutoencoderKLLegacy from FL2VA/video_vae plus a Comfy fp16
state dict — never MiniMaxH3VideoVAE.from_pretrained, never the 33B
transformer. GT is encode_temporal + DiagonalGaussianDistribution.mean.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

LATENT_CH = 24
VAE_SPATIAL = 16
VAE_TEMPORAL = 4  # documented compression; T' is NOT T / 4
CLIP_FRAMES = 22  # production-legal shortest: T = 17k+5, k>=1
LONG_FRAMES = 39  # next legal length, T'=12
LATENT_T = 7  # VAEProcessor.get_latent_length(22)
CANVAS = 128  # tests / sprite oracles
LATENT_HW = CANVAS // VAE_SPATIAL  # 8
N_CELLS = LATENT_T * LATENT_HW * LATENT_HW  # 448
VAE_CLIP_LENGTH = 17
VAE_TOKEN_DROP = 3  # get_latent_length: T' = ((T-5)//17)*5 + 2
# 16-aligned 9:16. 512x288 is the cheap canvas. Native AVAL film-out is 1280x720
# (source clips are 1280x720). Do not upsample a 288-wide decode — re-encode at native.
PORTRAIT_H = 512
PORTRAIT_W = 288
PORTRAIT_N_CELLS = LATENT_T * (PORTRAIT_H // VAE_SPATIAL) * (PORTRAIT_W // VAE_SPATIAL)  # 4032
NATIVE_H = 1280
NATIVE_W = 720
NATIVE_N_CELLS = LATENT_T * (NATIVE_H // VAE_SPATIAL) * (NATIVE_W // VAE_SPATIAL)  # 7*80*45=25200


def align_hw(height: int, width: int) -> tuple[int, int]:
    if height <= 0 or width <= 0:
        raise ValueError(f"canvas must be positive, got {height}x{width}")
    if height % VAE_SPATIAL or width % VAE_SPATIAL:
        raise ValueError(
            f"canvas {height}x{width} is not aligned to {VAE_SPATIAL} "
            "(VAE spatial ratio)"
        )
    return height, width


def latent_hw(height: int, width: int) -> tuple[int, int]:
    height, width = align_hw(height, width)
    return height // VAE_SPATIAL, width // VAE_SPATIAL


def n_cells(height: int, width: int, latent_t: int = LATENT_T) -> int:
    lh, lw = latent_hw(height, width)
    return latent_t * lh * lw


def is_legal_frames(frames: int) -> bool:
    """Production lengths that survive get_suitable_video_length: 22, 39, 56, …"""
    return frames >= CLIP_FRAMES and (frames - 5) % VAE_CLIP_LENGTH == 0


def latent_t_for_frames(frames: int) -> int:
    if not is_legal_frames(frames):
        raise ValueError(f"illegal clip length {frames}; need 17k+5, k>=1 (22, 39, 56, …)")
    k = (frames - 5) // VAE_CLIP_LENGTH
    return k * 5 + 2


def frames_for_latent_t(latent_t: int) -> int:
    if latent_t < LATENT_T or (latent_t - 2) % 5 != 0:
        raise ValueError(f"illegal latent T {latent_t}; need 5k+2, k>=1 (7, 12, 17, …)")
    k = (latent_t - 2) // 5
    return VAE_CLIP_LENGTH * k + 5

DEFAULT_H3_ROOT = os.environ.get(
    "MINIMAX_H3_ROOT",
    "/home/johndpope/Documents/GitHub/MiniMax-H3",
)
DEFAULT_VAE_PATH = os.environ.get(
    "MINIMAX_H3_VAE",
    "/run/media/johndpope/2TB/minimax-h3-nvfp4/vae/"
    "minimax_h3_video_vae_fp16.safetensors",
)

_DIT_STATS_KEYS = ("latents_mean", "latents_std")
_STATE_PREFIXES = ("model.", "vae.")
_H3_VAE_PKG = "_minimax_h3_video_vae"


def visual_vae_weights_available(
    h3_root: str | Path = DEFAULT_H3_ROOT,
    vae_path: str | Path = DEFAULT_VAE_PATH,
) -> bool:
    root = Path(h3_root)
    return (root / "FL2VA" / "video_vae" / "klvae.py").is_file() and Path(vae_path).is_file()


def _assert_clip(video_ncthw: Tensor) -> tuple[int, int, int, int, int]:
    if video_ncthw.ndim != 5:
        raise ValueError(f"expected (B, 3, T, H, W), got {tuple(video_ncthw.shape)}")
    b, c, t, h, w = video_ncthw.shape
    if c != 3:
        raise ValueError(f"expected 3 RGB channels, got {c}")
    if not is_legal_frames(t):
        raise ValueError(f"clips must be 17k+5 frames (22, 39, 56, …), got {t}")
    align_hw(h, w)
    return b, c, t, h, w


def _resample_time(video: Tensor, frames: int) -> Tensor:
    b, c, t, h, w = video.shape
    if t == frames:
        return video
    flat = video.permute(0, 1, 3, 4, 2).reshape(b, c * h * w, t)
    flat = F.interpolate(flat, size=frames, mode="linear", align_corners=False)
    return flat.reshape(b, c, h, w, frames).permute(0, 1, 4, 2, 3)


def _spatial_pool(video: Tensor, factor: int = VAE_SPATIAL) -> Tensor:
    b, c, t, h, w = video.shape
    frames = video.permute(0, 2, 1, 3, 4).contiguous().reshape(b * t, c, h, w)
    pooled = F.avg_pool2d(frames, kernel_size=factor, stride=factor)
    _, _, nh, nw = pooled.shape
    return pooled.reshape(b, t, c, nh, nw).permute(0, 2, 1, 3, 4).contiguous()


def _spatial_unpool(latent: Tensor, factor: int = VAE_SPATIAL) -> Tensor:
    return latent.repeat_interleave(factor, dim=-2).repeat_interleave(factor, dim=-1)


class FakeVideoVAE(nn.Module):
    """No H3 import. Spatial pool, temporal 22->7, frozen RGB copy into 24ch.

    Default spatial=16 matches H3. Sprite canvases use spatial=8 so a 32px
    sprite is 4x4 cells instead of 2x2 smear. Channels 0-2 are RGB; 3-23 zero.
    """

    def __init__(self, seed: int = 0, spatial: int = VAE_SPATIAL):
        super().__init__()
        if spatial < 1 or (spatial & (spatial - 1)):
            raise ValueError(f"spatial must be a positive power of two, got {spatial}")
        self.spatial = int(spatial)
        self.rgb_to_z = nn.Conv3d(3, LATENT_CH, kernel_size=1, bias=False)
        self.z_to_rgb = nn.Conv3d(LATENT_CH, 3, kernel_size=1, bias=False)
        with torch.no_grad():
            w_enc = torch.zeros(LATENT_CH, 3, 1, 1, 1)
            w_dec = torch.zeros(3, LATENT_CH, 1, 1, 1)
            for c in range(3):
                w_enc[c, c] = 1.0
                w_dec[c, c] = 1.0
            self.rgb_to_z.weight.copy_(w_enc)
            self.z_to_rgb.weight.copy_(w_dec)
        self.rgb_to_z.requires_grad_(False)
        self.z_to_rgb.requires_grad_(False)
        self.eval()

    def encode_mean(self, video_ncthw: Tensor) -> Tensor:
        _assert_clip(video_ncthw)
        x = _spatial_pool(video_ncthw, factor=self.spatial)
        x = _resample_time(x, latent_t_for_frames(video_ncthw.shape[2]))
        weight = self.rgb_to_z.weight.to(device=x.device, dtype=x.dtype)
        return F.conv3d(x, weight)

    def decode(self, z: Tensor) -> Tensor:
        if z.ndim != 5 or z.shape[1] != LATENT_CH:
            raise ValueError(
                f"expected (B, {LATENT_CH}, T', H', W'), got {tuple(z.shape)}"
            )
        frames = frames_for_latent_t(z.shape[2])
        weight = self.z_to_rgb.weight.to(device=z.device, dtype=z.dtype)
        rgb = F.conv3d(z, weight)
        rgb = _resample_time(rgb, frames)
        rgb = _spatial_unpool(rgb, factor=self.spatial)
        return rgb.clamp(0, 1)


def _video_vae_dir(h3_root: str | Path) -> Path:
    vae_dir = Path(h3_root) / "FL2VA" / "video_vae"
    if not (vae_dir / "klvae.py").is_file():
        raise FileNotFoundError(f"AutoencoderKLLegacy sources not found under {vae_dir}")
    return vae_dir


def _h3_video_vae_package(vae_dir: Path) -> types.ModuleType:
    """Load FL2VA/video_vae as a real package so relative imports resolve.

    Do not put this dir on sys.path as a top-level `klvae` module — those
    files use relative imports. Do not add FL2VA or the 33B transformer.
    """
    if _H3_VAE_PKG not in sys.modules:
        pkg = types.ModuleType(_H3_VAE_PKG)
        pkg.__path__ = [str(vae_dir)]
        pkg.__file__ = str(vae_dir)
        pkg.__package__ = _H3_VAE_PKG
        sys.modules[_H3_VAE_PKG] = pkg
    return sys.modules[_H3_VAE_PKG]


def _import_h3(name: str):
    return importlib.import_module(f"{_H3_VAE_PKG}.{name}")


def _ensure_vae_parallel_state() -> None:
    state = _import_h3("parallel").get_parallel_state()
    if not isinstance(state, dict):
        raise TypeError("get_parallel_state() must return a dict")
    if state:
        return
    state.update(
        {
            "group_size": 1,
            "group_rank": 0,
            "local_process_group": None,
            "sp_size": 1,
            "sp_rank": 0,
            "sp_enabled": False,
            "sp_process_group": None,
            "tp_size": 1,
            "tp_rank": 0,
        }
    )


def _strip_state_dict_prefix(state: dict) -> dict:
    out = {}
    for key, value in state.items():
        name = key
        for prefix in _STATE_PREFIXES:
            if name.startswith(prefix):
                name = name[len(prefix) :]
        out[name] = value
    for drop in _DIT_STATS_KEYS:
        out.pop(drop, None)
    return out


def _load_kwargs_from_wrapper_config(wrapper_config: dict) -> dict:
    return {
        "clip_length": int(wrapper_config["vae_clip_length"]),
        "token_drop": int(wrapper_config["vae_token_drop"]),
        "encoder_tiling": int(wrapper_config["vae_encoder_tiling"]),
        "decoder_tiling": int(wrapper_config["vae_decoder_tiling"]),
        "parallel_tiling": int(wrapper_config["vae_parallel_tiling"]),
        "tile_size": int(wrapper_config["vae_tile_size"]),
        "tile_overlap_min": int(wrapper_config["vae_tile_overlap_min"]),
        "encoder_parallel": int(wrapper_config["vae_encoder_parallel"]),
        "decoder_parallel": int(wrapper_config["vae_decoder_parallel"]),
        "chunk_dim": int(wrapper_config["vae_chunk_dim"]),
    }


def load_visual_vae(h3_root: str | Path, vae_path: str | Path) -> nn.Module:
    """Load frozen AutoencoderKLLegacy from split code + Comfy fp16 weights.

    Do NOT call MiniMaxH3VideoVAE.from_pretrained(h3_root): that expects
    {h3_root}/FL2VA/video_vae/source/model.safetensors, which is absent.
    """
    vae_dir = _video_vae_dir(h3_root)
    weights = Path(vae_path)
    if not weights.is_file():
        raise FileNotFoundError(f"VAE weights not found: {weights}")

    wrapper_config_path = vae_dir / "config.json"
    with wrapper_config_path.open("r", encoding="utf-8") as handle:
        wrapper_config = json.load(handle)
    load_kwargs = _load_kwargs_from_wrapper_config(wrapper_config)

    _h3_video_vae_package(vae_dir)
    if bool(load_kwargs["parallel_tiling"]):
        _ensure_vae_parallel_state()

    AutoencoderKLLegacy = _import_h3("klvae").AutoencoderKLLegacy

    source_path = vae_dir / wrapper_config.get("source_path", "source")
    source_config = AutoencoderKLLegacy.load_config(str(source_path))
    model, _unused = AutoencoderKLLegacy.from_config(
        source_config, return_unused_kwargs=True, **load_kwargs
    )

    import safetensors.torch

    state = _strip_state_dict_prefix(safetensors.torch.load_file(str(weights)))
    model.load_state_dict(state, strict=True)
    dtype = next(iter(state.values())).dtype
    model.to(dtype=dtype)
    model.eval()
    model.requires_grad_(False)
    return model


def _unwrap_legacy(vae: nn.Module) -> nn.Module:
    if isinstance(vae, FakeVideoVAE):
        return vae
    inner = getattr(vae, "model", vae)
    return inner


def _module_device_dtype(module: nn.Module) -> tuple[torch.device, torch.dtype]:
    param = next(module.parameters())
    return param.device, param.dtype


def _diagonal_gaussian():
    vae_module = sys.modules.get(f"{_H3_VAE_PKG}.vae_module")
    if vae_module is None:
        raise RuntimeError("load_visual_vae() must run before encode_mean_video on a real VAE")
    return vae_module.DiagonalGaussianDistribution


def _prepare_video(legacy: nn.Module, video_ncthw: Tensor) -> Tensor:
    """Reuse encode_videos preprocessing; do not call encode_videos."""
    _assert_clip(video_ncthw)
    processor = legacy.processor
    device, dtype = _module_device_dtype(legacy)
    video = video_ncthw.to(device=device)
    used = processor.get_suitable_video_length(video.shape[2])
    video = video[:, :, :used]
    _, _, _, height, width = video.shape
    new_h, new_w = processor._align_to_total_patch_size(height, width)
    video = processor._crop_to_align(video, new_h, new_w, is_video=True)
    video = processor.transform_tensor(video)
    return video.to(dtype=dtype)


@torch.no_grad()
def encode_mean_video(vae: nn.Module, video_ncthw: Tensor) -> Tensor:
    """(B, 3, 22, 128, 128) in [0, 1] -> (B, 24, 7, 8, 8) posterior mean.

    Real VAE: encode_temporal + DiagonalGaussianDistribution.mean.
    Never encode_base / encode_videos / encode_images / .sample().
    """
    if isinstance(vae, FakeVideoVAE):
        z = vae.encode_mean(video_ncthw)
    else:
        legacy = _unwrap_legacy(vae)
        video = _prepare_video(legacy, video_ncthw)
        moments = legacy.encode_temporal(video)
        z = _diagonal_gaussian()(moments).mean
    _, _, frames, height, width = video_ncthw.shape
    if isinstance(vae, FakeVideoVAE):
        spatial = vae.spatial
        lh, lw = height // spatial, width // spatial
    else:
        lh, lw = latent_hw(height, width)
    expected = (video_ncthw.shape[0], LATENT_CH, latent_t_for_frames(frames), lh, lw)
    if tuple(z.shape) != expected:
        raise RuntimeError(f"encode_mean_video produced {tuple(z.shape)}, expected {expected}")
    return z


@torch.no_grad()
def decode_pixels(vae: nn.Module, z: Tensor) -> Tensor:
    """(B, 24, 7, 8, 8) -> (B, 3, 22, 128, 128) in [0, 1]."""
    frames = frames_for_latent_t(z.shape[2])
    if isinstance(vae, FakeVideoVAE):
        pixels = vae.decode(z)
    else:
        legacy = _unwrap_legacy(vae)
        device, dtype = _module_device_dtype(legacy)
        recon = legacy.decode_base(z.to(device=device, dtype=dtype), frame_num=frames)
        pixels = legacy.processor.revert_tensor(recon)
    _, _, _, lh, lw = z.shape
    spatial = vae.spatial if isinstance(vae, FakeVideoVAE) else VAE_SPATIAL
    expected = (z.shape[0], 3, frames, lh * spatial, lw * spatial)
    if tuple(pixels.shape) != expected:
        raise RuntimeError(f"decode_pixels produced {tuple(pixels.shape)}, expected {expected}")
    return pixels.clamp(0, 1)
