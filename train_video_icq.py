"""POC trainer for BDH-CQ video: FakeVAE or frozen H3 VAE, demos-only, no text.

    WANDB_MODE=offline python train_video_icq.py --family aval --device cuda --wandb
"""

from __future__ import annotations

import os
import random
import subprocess
from pathlib import Path

import fire
import numpy as np
import torch

from bdh_cq.aval_clips import DEFAULT_CACHE, default_holdout, list_processed
from bdh_cq.icq_transfer import DEFAULT_ROOT as TRANSFER_ROOT, MOTION_ACTIONS
from bdh_cq.video import (
    MAX_REASONING_STEPS,
    BDHVideoReasoningWrapper,
    encode_aval_library,
    encode_icq_transfer_library,
    encode_task,
    make_video_model,
    sample_aval_task_latents,
    sample_icq_transfer_task_latents,
    sample_task,
)
from bdh_cq.video_probes import (
    copy_detector,
    identity_probe,
    is_still_clip,
    last_frame_mismatch,
    latent_dt_mse,
    latent_mse,
    pixel_last_frame_l1,
    pixel_last_frame_motion_l1,
    pixel_psnr,
    pixel_t0_t21_l1,
    pixel_temporal_l1,
)
from bdh_cq.video_tasks import CANVAS as SPRITE_CANVAS, VIDEO_TASKS
from bdh_cq.video_vae import (
    DEFAULT_H3_ROOT,
    DEFAULT_VAE_PATH,
    LONG_FRAMES,
    NATIVE_H,
    NATIVE_W,
    PORTRAIT_H,
    PORTRAIT_W,
    FakeVideoVAE,
    decode_pixels,
    latent_t_for_frames,
    load_visual_vae,
    n_cells,
    visual_vae_weights_available,
)

GRAD_CLIP = 1.0
GRAD_EXPLODE = 20000.0  # H3+IMTalker step-0 pre-clip ~5e3; 500/5000 aborted live runs
LOSS_EXPLODE = 5000.0  # H3 last-frame CE can be hundreds; 100 aborted a live run


def _park_vae(vae, device: torch.device) -> None:
    """H3 decoder is eval-only. Keep it off the train GPU."""
    vae.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _eval(wrapper, task, steps: int) -> dict:
    wrapper.eval()
    with torch.no_grad():
        memories = wrapper.ingest_task(task)
        z_hat = wrapper.reason(memories, steps=steps, still=task["query_in"])
        rec = float(latent_mse(z_hat, task["query_out"]).detach())
        dt = float(latent_dt_mse(z_hat, task["query_out"]).detach())
        probe = identity_probe(z_hat, task["query_out"])
        copy = copy_detector(z_hat, task["query_out"])
    wrapper.train()
    return dict(mse=rec, dt=dt, identity_probe=probe, z_hat=z_hat, **copy)


def _to_uint8_hwc(ncthw: torch.Tensor, t: int) -> np.ndarray:
    frame = ncthw[0, :, t].detach().float().clamp(0, 1).cpu()
    return (frame.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)


def _write_png(path: Path, hwc: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width, _ = hwc.shape
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-i",
            "pipe:0",
            "-frames:v",
            "1",
            "-update",
            "1",
            str(path),
        ],
        input=np.ascontiguousarray(hwc).tobytes(),
        capture_output=True,
        check=False,
    )


def save_recon_strip(path: Path, pred: torch.Tensor, target: torch.Tensor) -> Path:
    """Debug contact sheet only. The deliverable is save_pair_mp4."""
    t = pred.shape[2]
    idx = (0, t // 2, t - 1)
    still = _to_uint8_hwc(target, 0)
    pred_row = np.concatenate([still] + [_to_uint8_hwc(pred, i) for i in idx], axis=1)
    gt_row = np.concatenate([still] + [_to_uint8_hwc(target, i) for i in idx], axis=1)
    sheet = np.concatenate([pred_row, gt_row], axis=0)
    _write_png(path, sheet)
    return path


def ncthw_to_thwc(video: torch.Tensor) -> np.ndarray:
    frames = video[0].detach().float().clamp(0, 1).cpu().permute(1, 2, 3, 0)
    return (frames.numpy() * 255.0).round().astype(np.uint8)


def save_clip_mp4(path: Path, video: torch.Tensor, fps: int = 24) -> Path:
    """(1,3,T,H,W) in [0,1] -> 24 fps H.264 clip. A still is a failed clip."""
    path.parent.mkdir(parents=True, exist_ok=True)
    thwc = ncthw_to_thwc(video)
    t, height, width, _ = thwc.shape
    proc = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            str(path),
        ],
        input=np.ascontiguousarray(thwc).tobytes(),
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0 or not path.is_file():
        err = proc.stderr.decode("utf-8", errors="replace")[-400:]
        raise RuntimeError(f"ffmpeg clip failed {path}: {err}")
    return path


def save_pair_mp4(path: Path, pred: torch.Tensor, target: torch.Tensor, fps: int = 24) -> Path:
    """Left = pred, right = GT. 22 frames @ 24 fps."""
    stacked = torch.cat([pred, target], dim=-1)
    return save_clip_mp4(path, stacked, fps=fps)


def _maybe_wandb(enabled: bool, cfg: dict):
    if not enabled:
        return None
    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.setdefault("WANDB_DIR", str(Path("logs")))
    import wandb

    Path("logs").mkdir(parents=True, exist_ok=True)
    return wandb.init(project="bdh-cq-video", config=cfg, mode="offline", dir="logs")


def _load_vae(kind: str, device: torch.device, seed: int, spatial: int = 16):
    if kind == "fake":
        return FakeVideoVAE(seed=seed, spatial=spatial).to(device)
    if kind != "h3":
        raise ValueError(f"--vae must be fake|h3, got {kind!r}")
    if not visual_vae_weights_available():
        raise FileNotFoundError(
            f"H3 VAE missing. h3-root={DEFAULT_H3_ROOT} vae={DEFAULT_VAE_PATH}"
        )
    # Frozen GT factory only: AutoencoderKLLegacy + Comfy fp16, never the 33B.
    vae = load_visual_vae(DEFAULT_H3_ROOT, DEFAULT_VAE_PATH).to(device)
    vae.eval()
    vae.requires_grad_(False)
    return vae


def run(
    device: str = "cpu",
    family: str = "identity",
    steps: int = 40,
    seed: int = 0,
    scale: str = "tiny",
    overfit: bool = True,
    lambda_dt: float = 1.0,
    eval_every: int = 10,
    max_reasoning: int = MAX_REASONING_STEPS,
    min_reasoning: int = 0,
    clip_dir: str = str(DEFAULT_CACHE),
    fetch: bool = False,
    height: int | None = None,
    width: int | None = None,
    vae: str = "fake",
    wandb: bool = True,
    recon_dir: str = "logs/recon",
    ckpt: str = "logs/bdh_video.pt",
    explode_grad: float = GRAD_EXPLODE,
    library_limit: int | None = None,
    allow_raw: bool = False,
    resume: bool = False,
    clip_frames: int = 22,
    gate_decode: bool = True,
    admit: str = "decode",
    sprite_mix: float = 0.25,
    canvas_write_s: bool = True,
    motion_rank: int = 0,
    decode: str = "auto",
    query_cue_frames: int = 0,
    query_id: str = "id_d",
    transfer_action: str = "laugh",
    heldout_query_id: str = "id_c",
    lambda_m: float = 1.0,
    lambda_ctx: float = 1.0,
    peak_power: float = 1.0,
    lambda_last: float = 4.0,
    lambda_energy: float = 1.0,
    lambda_mass: float = 1.0,
    motion_teacher: str = "residual",
    imtalker_root: str = "/media/2TB/IMTalker",
    imtalker_ckpt: str = "/media/2TB/IMTalker/checkpoints/renderer.ckpt",
    amp: bool | None = None,
    explode_loss: float = LOSS_EXPLODE,
):
    torch.manual_seed(seed)
    random.seed(seed)
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    if device == "cpu":
        torch.set_num_threads(8)

    if height is None or width is None:
        if family in VIDEO_TASKS:
            height, width = SPRITE_CANVAS, SPRITE_CANVAS
        elif family == "aval" and vae == "h3":
            height, width = NATIVE_H, NATIVE_W
        else:
            height, width = PORTRAIT_H, PORTRAIT_W
    if family == "icq_transfer":
        if (height, width) == (NATIVE_H, NATIVE_W):
            height, width = PORTRAIT_H, PORTRAIT_W
        if clip_dir == str(DEFAULT_CACHE):
            clip_dir = str(TRANSFER_ROOT)
        if transfer_action not in MOTION_ACTIONS:
            raise KeyError(f"--transfer_action must be {MOTION_ACTIONS}, got {transfer_action!r}")
        if motion_teacher not in ("residual", "imtalker"):
            raise KeyError(f"--motion_teacher must be residual|imtalker, got {motion_teacher!r}")
        if motion_teacher == "imtalker" and motion_rank == 0:
            motion_rank = 32

    clip_kwargs = {}
    aval_library = None
    aval_holdout: set[str] | None = None
    transfer_library = None
    if family == "icq_transfer":
        clip_kwargs = dict(
            root=clip_dir,
            action=transfer_action,
            query_id=query_id,
            height=height,
            width=width,
        )
        print(
            f"icq_transfer canvas {height}x{width} "
            f"N={n_cells(height, width, latent_t_for_frames(clip_frames))} "
            f"query={query_id} action={transfer_action} (no admit_clip)",
            flush=True,
        )
    elif family == "aval":
        from bdh_cq.aval_clips import fetch_clips

        have = list_processed(clip_dir, height=height, width=width)
        raw_ok = Path(clip_dir).joinpath("raw").is_dir()
        if fetch and not have and not raw_ok:
            fetch_clips(root=clip_dir, height=height, width=width)
        aval_holdout = default_holdout(clip_dir, height=height, width=width)
        clip_kwargs = dict(
            root=clip_dir,
            holdout=aval_holdout,
            height=height,
            width=width,
        )
        print(
            f"aval canvas {height}x{width} N={n_cells(height, width, latent_t_for_frames(clip_frames))} "
            f"frames={clip_frames} holdout={len(aval_holdout or [])}",
            flush=True,
        )
    elif family not in VIDEO_TASKS:
        raise KeyError(f"unknown family {family!r}")

    if family in VIDEO_TASKS or family in ("aval", "icq_transfer"):
        # Demos write S; query still is H_0 only.
        canvas_write_s = False

    # copy: the answer edits the still in place (no rigid motion) so the
    # demo displacement is meaningless. shift: composite the still by (dy, dx).
    COPY_FAMILIES = {"identity", "stamp_copy", "recolor"}
    PAN_FAMILIES = {"pan", "translate_pan"}
    if decode == "auto":
        if family in COPY_FAMILIES:
            decode = "copy"
        elif family in PAN_FAMILIES:
            decode = "pan"
        else:
            decode = "shift"
    if decode not in ("shift", "copy", "pan"):
        raise ValueError(f"--decode must be auto|shift|copy|pan, got {decode!r}")

    device_t = torch.device(device)
    fake_spatial = 8 if family in VIDEO_TASKS else 16
    vae_mod = _load_vae(vae, device_t, seed, spatial=fake_spatial)
    if family == "aval":
        def _encode(frames: int):
            return encode_aval_library(
                vae_mod,
                root=clip_dir,
                height=height,
                width=width,
                device=device_t,
                tag=vae,
                limit=library_limit,
                allow_raw=allow_raw,
                clip_frames=frames,
                gate_decode=gate_decode,
                admit=admit,
            )

        try:
            aval_library = _encode(clip_frames)
        except RuntimeError as exc:
            if clip_frames == 22 and "motion library too small" in str(exc):
                print(f"22-frame gate empty, retry {LONG_FRAMES}", flush=True)
                clip_frames = LONG_FRAMES
                aval_library = _encode(clip_frames)
            else:
                raise
        names = [item["name"] for item in aval_library]
        print(
            f"aval library {len(names)} latents tag={vae} frames={clip_frames} "
            f"allow_raw={allow_raw} gate_decode={gate_decode}",
            flush=True,
        )
        if len(names) > 10:
            aval_holdout = set(random.Random(1).sample(names, 10))
        elif aval_holdout:
            aval_holdout = {name for name in aval_holdout if name in set(names)}
        clip_kwargs["holdout"] = aval_holdout
        _park_vae(vae_mod, device_t)
        print("vae parked on cpu for train", flush=True)
    elif family == "icq_transfer":
        transfer_library = encode_icq_transfer_library(
            vae_mod,
            root=clip_dir,
            height=height,
            width=width,
            device=device_t,
            tag=vae,
            clip_frames=clip_frames,
        )
        n_star = sum(1 for item in transfer_library if item.get("m_star") is not None)
        print(
            f"icq_transfer library {len(transfer_library)} latents tag={vae} "
            f"m_star={n_star} teacher={motion_teacher} (admit_clip skipped)",
            flush=True,
        )
        if motion_teacher == "imtalker" and n_star < 4:
            raise RuntimeError(
                "imtalker teacher needs token cache. Run: "
                "python -m bdh_cq.imtalker_motion --clip_root data/icq_transfer"
            )
        _park_vae(vae_mod, device_t)
        print("vae parked on cpu for train", flush=True)
    # FakeVAE sprite path is spatial 8; the real H3 latent is 16.
    warp_spatial = 8 if (vae == "fake" and family in VIDEO_TASKS) else 16
    wrapper = BDHVideoReasoningWrapper(
        make_video_model(scale=scale),
        canvas_update_memory=canvas_write_s,
        motion_rank=motion_rank,
        decode=decode,
        warp_spatial=warp_spatial,
    ).to(device_t)
    use_amp = bool(amp) if amp is not None else (
        device_t.type == "cuda" and scale == "billion"
    )
    amp_dtype = torch.bfloat16 if use_amp else None
    if use_amp:
        print(f"amp bfloat16 scale={scale}", flush=True)
    occ_params, other_params = [], []
    for name, param in wrapper.named_parameters():
        if any(
            key in name
            for key in (
                "occupancy_bias",
                "to_occupancy",
                "to_centroid",
                "demo_to_shift",
                "occ_prior_scale",
                "log_occ_sigma",
            )
        ):
            occ_params.append(param)
        else:
            other_params.append(param)
    param_groups = [dict(params=other_params, weight_decay=0.1)]
    if occ_params:
        param_groups.append(dict(params=occ_params, lr=1e-2, weight_decay=0.0))
    # AdamW moments are 8GB at 1B fp32 — OOM next to 18GB activations on 24GB.
    if scale == "billion":
        opt = torch.optim.SGD(wrapper.parameters(), lr=0.05, momentum=0.0)
        print("optim SGD (no AdamW moments / no momentum buffer at 1B on 24GB)", flush=True)
    else:
        opt = torch.optim.AdamW(param_groups, lr=1e-3)
    if resume:
        ckpt_path = Path(ckpt)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"--resume set but missing ckpt {ckpt_path}")
        blob = torch.load(ckpt_path, map_location=device_t, weights_only=False)
        wrapper.load_state_dict(blob["wrapper"])
        if blob.get("opt") is not None:
            opt.load_state_dict(blob["opt"])
        print(
            f"resume {ckpt_path} from_step={blob.get('step')} loss={blob.get('loss')}",
            flush=True,
        )
    rng = random.Random(seed)
    recon_path = Path(recon_dir)
    recon_path.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)

    n_params = sum(p.numel() for p in wrapper.parameters())
    run_cfg = dict(
        family=family,
        scale=scale,
        steps=steps,
        overfit=overfit,
        height=height,
        width=width,
        vae=vae,
        device=str(device_t),
        n_params=n_params,
        lambda_dt=lambda_dt,
        clip_frames=clip_frames,
        gate_decode=gate_decode,
        admit=admit,
        sprite_mix=sprite_mix,
        canvas_write_s=canvas_write_s,
        motion_rank=motion_rank,
        decode=decode,
        query_id=query_id,
        transfer_action=transfer_action,
        heldout_query_id=heldout_query_id,
        lambda_m=lambda_m,
        lambda_ctx=lambda_ctx,
        peak_power=peak_power,
        lambda_last=lambda_last,
        lambda_energy=lambda_energy,
        lambda_mass=lambda_mass,
        motion_teacher=motion_teacher,
    )
    wb = _maybe_wandb(wandb, run_cfg)

    sprite_pool: list[dict] = []
    if (not overfit) and sprite_mix > 0 and family == "aval":
        for mix_i in range(4):
            for mix_family in ("identity", "translate"):
                sprite_pool.append(
                    encode_task(
                        vae_mod,
                        sample_task(mix_family, seed=seed + 1_000 + mix_i * 2),
                        device=device_t,
                    )
                )
        print(
            f"sprite_mix={sprite_mix} pool={len(sprite_pool)} "
            f"{[t['name'] for t in sprite_pool]}",
            flush=True,
        )

    cached = None
    if overfit:
        if transfer_library is not None:
            cached = sample_icq_transfer_task_latents(
                transfer_library,
                seed=seed,
                action=transfer_action,
                query_id=query_id,
                device=device_t,
            )
            if motion_teacher == "imtalker" and cached.get("query_m") is None:
                raise RuntimeError("overfit task missing query_m; extract IMTalker tokens first")
        elif aval_library is not None:
            cached = sample_aval_task_latents(
                aval_library, seed=seed, holdout=aval_holdout, device=device_t
            )
        else:
            cached = encode_task(
                vae_mod, sample_task(family, seed=seed, **clip_kwargs),
                device=device_t, query_cue_frames=query_cue_frames,
            )
        print(f"overfit task {cached.get('name')} {cached.get('params')}", flush=True)

    print(
        f"poc family={family} scale={scale} steps={steps} overfit={overfit} "
        f"vae={vae} canvas={height}x{width} device={device_t} params={n_params:,} "
        f"canvas_write_s={canvas_write_s} motion_rank={motion_rank} decode={decode} "
        f"lambda_m={lambda_m} lambda_ctx={lambda_ctx} peak_power={peak_power} "
        f"lambda_last={lambda_last} lambda_energy={lambda_energy} "
        f"lambda_mass={lambda_mass}",
        flush=True,
    )

    history = []
    query_last_fail = True
    for step in range(steps):
        if cached is None:
            task_seed = rng.randrange(2**31)
            if sprite_pool and rng.random() < sprite_mix:
                task = sprite_pool[task_seed % len(sprite_pool)]
            elif transfer_library is not None:
                task = sample_icq_transfer_task_latents(
                    transfer_library,
                    seed=task_seed,
                    holdout_query={heldout_query_id} if heldout_query_id else None,
                    device=device_t,
                )
            elif aval_library is not None:
                task = sample_aval_task_latents(
                    aval_library,
                    seed=task_seed,
                    holdout=aval_holdout,
                    device=device_t,
                )
            else:
                task = encode_task(
                    vae_mod,
                    sample_task(family, seed=task_seed, **clip_kwargs),
                    device=device_t,
                    query_cue_frames=query_cue_frames,
                )
        else:
            task = cached

        lo = min(int(min_reasoning), int(max_reasoning))
        reasoning_steps = rng.randint(lo, max_reasoning)
        with torch.autocast(
            device_type=device_t.type,
            dtype=amp_dtype or torch.float32,
            enabled=use_amp,
        ):
            loss, parts = wrapper.train_loss(
                task,
                reasoning_steps,
                lambda_dt=lambda_dt,
                lambda_m=lambda_m,
                lambda_ctx=lambda_ctx,
                peak_power=peak_power,
                lambda_last=lambda_last,
                lambda_energy=lambda_energy,
                lambda_mass=lambda_mass,
                return_parts=True,
            )
        loss = loss.float()
        item = float(loss.detach())
        if not torch.isfinite(loss.detach()) or item > explode_loss:
            raise RuntimeError(f"exploding/non-finite loss at step {step}: {item}")

        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(wrapper.parameters(), GRAD_CLIP))
        if not np.isfinite(grad_norm) or grad_norm > explode_grad:
            raise RuntimeError(
                f"exploding grad at step {step}: pre-clip norm {grad_norm:.4f} "
                f"(cap {explode_grad})"
            )
        opt.step()
        opt.zero_grad(set_to_none=True)

        history.append(item)
        row = dict(
            loss=item,
            rec=parts["rec"],
            dt=parts["dt"],
            R=reasoning_steps,
            grad_norm=grad_norm,
            grad_clipped=grad_norm > GRAD_CLIP,
            **{k: v for k, v in parts.items() if k not in ("rec", "dt")},
        )

        do_eval = step % eval_every == 0 or step == steps - 1
        if do_eval:
            line = (
                f"step {step:5d}  loss {item:.4f}  R={reasoning_steps}  "
                f"grad {grad_norm:.4f}"
            )
            eval_task = cached if cached is not None else task
            metrics = _eval(wrapper, eval_task, max_reasoning)
            row.update(
                train_mse=metrics["mse"],
                dt_mse=metrics["dt"],
                copy_hat=metrics["mse_hat_vs_still"],
                copy_star=metrics["mse_star_vs_still"],
                identity_probe=int(metrics["identity_probe"]),
            )
            line += (
                f"  mse {metrics['mse']:.4f}  dt {metrics['dt']:.4f}  "
                f"copy_hat {metrics['mse_hat_vs_still']:.4f}  "
                f"copy_star {metrics['mse_star_vs_still']:.4f}"
            )
            try:
                vae_mod.to(device_t)
                pred_pix = decode_pixels(vae_mod, metrics["z_hat"])
                gt_pix = decode_pixels(vae_mod, eval_task["query_out"])
                psnr = pixel_psnr(pred_pix, gt_pix)
                pred_l1 = pixel_temporal_l1(pred_pix)
                gt_l1 = pixel_temporal_l1(gt_pix)
                pred_end = pixel_t0_t21_l1(pred_pix)
                gt_end = pixel_t0_t21_l1(gt_pix)
                # A still pred is only a failure when the GT actually moves;
                # identity / stamp_copy / recolor GT is a still by design.
                still_fail = is_still_clip(pred_pix) and not is_still_clip(gt_pix)
                last_l1 = pixel_last_frame_l1(pred_pix, gt_pix)
                last_mot = pixel_last_frame_motion_l1(pred_pix, gt_pix)
                last_fail = last_frame_mismatch(pred_pix, gt_pix)
                query_last_fail = last_fail
                row.update(
                    psnr=psnr,
                    pred_t_l1=pred_l1,
                    gt_t_l1=gt_l1,
                    pred_t0_t21=pred_end,
                    gt_t0_t21=gt_end,
                    clip_fail=int(still_fail),
                    last_frame_l1=last_l1,
                    last_frame_motion_l1=last_mot,
                    last_frame_fail=int(last_fail),
                )
                clip_path = recon_path / f"step_{step:05d}.mp4"
                pred_path = recon_path / f"step_{step:05d}_pred.mp4"
                save_clip_mp4(pred_path, pred_pix)
                save_pair_mp4(clip_path, pred_pix, gt_pix)
                tag = "CLIP_FAIL still" if still_fail else "clip"
                if last_fail:
                    tag = "LAST_FRAME_FAIL"
                line += (
                    f"  psnr {psnr:.2f}  tL1 pred {pred_l1:.5f} gt {gt_l1:.5f}  "
                    f"t0_t21 pred {pred_end:.5f} gt {gt_end:.5f}  "
                    f"lastL1 {last_l1:.5f}  motL1 {last_mot:.5f}  {tag} {clip_path}"
                )
                if still_fail:
                    print("CLIP_FAIL: decode is a still image, not a clip", flush=True)
                if last_fail:
                    print(
                        "LAST_FRAME_FAIL: pred t=-1 does not match GT t=-1 "
                        f"(L1 {last_l1:.5f})",
                        flush=True,
                    )
                if wb is not None:
                    import wandb as wandb_mod

                    row["recon"] = wandb_mod.Video(str(clip_path), fps=24, format="mp4")
            except Exception as exc:
                line += f"  recon_fail {type(exc).__name__}: {exc}"
            finally:
                _park_vae(vae_mod, device_t)
            print(line, flush=True)
            torch.save(
                dict(
                    step=step,
                    wrapper=wrapper.state_dict(),
                    opt=opt.state_dict(),
                    cfg=run_cfg,
                    loss=item,
                ),
                ckpt,
            )
        else:
            extra = f"  rec {parts['rec']:.4f}  dt {parts['dt']:.4f}"
            if "last_mse" in parts:
                extra += (
                    f"  last {parts['last_mse']:.4f}  "
                    f"eKL {parts['energy_kl']:.4f}  dice {parts['occupancy_dice']:.4f}"
                )
                if "last_app" in parts:
                    extra += f"  app {parts['last_app']:.4f}"
                if "last_mot" in parts:
                    extra += f"  mot {parts['last_mot']:.4f}"
                if "occupancy_ce" in parts:
                    extra += f"  occCE {parts['occupancy_ce']:.4f}"
                if "mass_rel" in parts:
                    extra += f"  mass {parts['mass_rel']:.3f}"
                if "fp_energy" in parts:
                    extra += f"  fp {parts['fp_energy']:.4f}"
            if "motion_l2" in parts:
                extra += f"  motion_l2 {parts['motion_l2']:.4f}"
            if "m_mse" in parts:
                extra += f"  m_mse {parts['m_mse']:.4f}  ctx_mse {parts['ctx_mse']:.4f}"
            if "shift_mse" in parts:
                extra += (
                    f"  shift {parts['shift_mse']:.3f} "
                    f"dy {parts['shift_dy']:.2f} dx {parts['shift_dx']:.2f}"
                )
            print(
                f"step {step:5d}  loss {item:.4f}  R={reasoning_steps}  "
                f"grad {grad_norm:.4f}{extra}",
                flush=True,
            )

        if wb is not None:
            wb.log({k: v for k, v in row.items() if k != "z_hat"}, step=step)

    if transfer_library is not None:
        held_action = next(a for a in MOTION_ACTIONS if a != transfer_action)
        held = sample_icq_transfer_task_latents(
            transfer_library,
            seed=seed + 10_000,
            action=held_action,
            query_id=query_id,
            device=device_t,
        )
        print(
            f"held-out transfer query={query_id} action={held_action} "
            f"(train was {transfer_action})",
            flush=True,
        )
    elif aval_library is not None:
        held_items = [
            item
            for item in aval_library
            if aval_holdout and item["name"] in aval_holdout
        ]
        if len(held_items) < 4:
            held_items = aval_library
        held = sample_aval_task_latents(
            held_items, seed=seed + 10_000, holdout=None, device=device_t
        )
    else:
        held = encode_task(
            vae_mod, sample_task(family, seed=seed + 10_000, **clip_kwargs),
            device=device_t, query_cue_frames=query_cue_frames,
        )
    held_m = _eval(wrapper, held, max_reasoning)
    print(
        f"held-out mse {held_m['mse']:.4f}  identity_probe {held_m['identity_probe']}  "
        f"copy_hat {held_m['mse_hat_vs_still']:.4f}  copy_star {held_m['mse_star_vs_still']:.4f}",
        flush=True,
    )
    try:
        vae_mod.to(device_t)
        pred_pix = decode_pixels(vae_mod, held_m["z_hat"])
        gt_pix = decode_pixels(vae_mod, held["query_out"])
        pred_l1 = pixel_temporal_l1(pred_pix)
        still_fail = is_still_clip(pred_pix) and not is_still_clip(gt_pix)
        last_l1 = pixel_last_frame_l1(pred_pix, gt_pix)
        last_mot = pixel_last_frame_motion_l1(pred_pix, gt_pix)
        last_fail = last_frame_mismatch(pred_pix, gt_pix)
        pred_mp4 = Path(recon_dir) / "heldout_pred.mp4"
        pair_mp4 = Path(recon_dir) / "heldout.mp4"
        save_clip_mp4(pred_mp4, pred_pix)
        save_pair_mp4(pair_mp4, pred_pix, gt_pix)
        tag = "CLIP_FAIL still" if still_fail else "clip"
        if last_fail:
            tag = "LAST_FRAME_FAIL"
        print(
            f"held-out {tag} {pair_mp4}  pred {pred_mp4}  tL1 {pred_l1:.5f}  "
            f"lastL1 {last_l1:.5f}  motL1 {last_mot:.5f}",
            flush=True,
        )
        if still_fail:
            print("CLIP_FAIL: held-out decode is a still image, not a clip", flush=True)
        if last_fail:
            print(
                "LAST_FRAME_FAIL: held-out pred t=-1 does not match GT t=-1 "
                f"(L1 {last_l1:.5f})",
                flush=True,
            )
        if wb is not None:
            import wandb as wandb_mod

            wb.log(
                {
                    "held_mse": held_m["mse"],
                    "held_t_l1": pred_l1,
                    "held_clip_fail": int(still_fail),
                    "held_recon": wandb_mod.Video(str(pair_mp4), fps=24, format="mp4"),
                }
            )
    except Exception as exc:
        print(f"held-out recon_fail {type(exc).__name__}: {exc}", flush=True)
    finally:
        _park_vae(vae_mod, device_t)

    if wb is not None:
        wb.finish()
    if len(history) >= 2 and history[-1] >= history[0]:
        print(
            f"warn: loss did not fall ({history[0]:.4f} -> {history[-1]:.4f}); "
            "loop is live, scale/steps may be too small",
            flush=True,
        )
    else:
        print(f"ok: loss {history[0]:.4f} -> {history[-1]:.4f}", flush=True)
    if query_last_fail:
        print(
            "LAST_FRAME_FAIL: ICQ query last frame does not match GT; not a pass",
            flush=True,
        )
        raise SystemExit(2)


if __name__ == "__main__":
    fire.Fire(run)
