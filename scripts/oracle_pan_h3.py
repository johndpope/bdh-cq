"""One-shot oracle test: does the FakeVAE pan warp-floor (lastL1 0.036-0.10)
persist on the real H3 VAE latent? No training loop - just encode, warp by
the TRUE shift, decode, measure. Answers the decode-fidelity-ceiling question
directly and fast, without waiting on a training run.

Run on a CUDA host with the H3 VAE weights (see bdh_cq/video_vae.py
DEFAULT_H3_ROOT / DEFAULT_VAE_PATH, e.g. johndpope@msi.local):

    python scripts/oracle_pan_h3.py

2026-09-11 result (RTX PRO 4000, 6 seeds): mean oracle lastL1 0.173
(range 0.121-0.243) vs FakeVAE's 0.036-0.10 and the 0.031 gate - H3 makes the
warp-decode ceiling *worse*, not better. See docs/HANDOFF_SPRITE_ICQ.md,
"pan" section, for why: warp_still assumes each latent cell is a uniform
pixel block (true by construction for FakeVAE, false for H3's real conv VAE).
"""

from __future__ import annotations

import statistics

import torch

from bdh_cq.video import encode_task, sample_task, warp_still
from bdh_cq.video_probes import (
    LAST_FRAME_L1,
    energy_centroid,
    pixel_last_frame_l1,
    pixel_last_frame_motion_l1,
)
from bdh_cq.video_vae import DEFAULT_H3_ROOT, DEFAULT_VAE_PATH, decode_pixels, load_visual_vae


def main(n_seeds: int = 6) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading H3 VAE on {device}...", flush=True)
    vae = load_visual_vae(DEFAULT_H3_ROOT, DEFAULT_VAE_PATH).to(device)
    vae.eval()
    vae.requires_grad_(False)
    print("loaded.", flush=True)

    results = []
    for seed in range(n_seeds):
        task = encode_task(vae, sample_task("pan", seed=seed), device=device)
        src = energy_centroid(task["query_in"])[:, 0]
        dst = energy_centroid(task["query_out"])[:, -1]
        true_shift = (dst - src).to(task["query_out"].dtype)

        z_pred = warp_still(task["query_in"], true_shift, spatial=16)
        pred = decode_pixels(vae, z_pred)
        gt = decode_pixels(vae, task["query_out"])

        last_l1 = pixel_last_frame_l1(pred, gt)
        mot_l1 = pixel_last_frame_motion_l1(pred, gt)
        results.append(last_l1)
        print(
            f"seed {seed}  true_shift=({float(true_shift[0, 0]):+.2f},"
            f"{float(true_shift[0, 1]):+.2f})  oracle_lastL1={last_l1:.4f}  "
            f"motL1={mot_l1:.4f}  gate({LAST_FRAME_L1:.4f})_pass="
            f"{last_l1 <= LAST_FRAME_L1}",
            flush=True,
        )

    print(
        f"\nmean oracle lastL1 over {len(results)} seeds: "
        f"{statistics.mean(results):.4f}  (FakeVAE was 0.036-0.10; "
        f"gate is {LAST_FRAME_L1:.4f})",
        flush=True,
    )


if __name__ == "__main__":
    main()
