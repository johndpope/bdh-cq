# Handoff: BDH-CQ video ICQ (sprites + talking heads)

**Status:** Sprite **train-query last frame passed**. Held-out (2026-09-10):
**`identity`, `stamp_copy`, `recolor` (`decode=copy`, attention-copy) and
`translate` (`decode=shift` + `--query_cue_frames 9`) all EXIT 0** (recolor's
gate is soft — see below). `pan` / `translate_pan` are decode-fidelity bound
on the FakeVAE latent — the fix (real H3 VAE) needs msi, code is ready. See
"Held-out generalization" below. Talking-head last frame **never passed**
(idle/smear vs laugh). 1B on msi **ran and still failed** last frame.

**Product pass is visual, not pytest.** A still, a gray wash, a flood, or a closed mouth next to GT teeth is a fail even if CLIP_FAIL is clear, tL1 looks fine, or PSNR is ~16–28.

Date of this snapshot: 2026-09-10.

---

## Locked constraints

- Video is ingest → relax → render (not a DiT swap).
- Frozen H3 VisualVAE **G** only. GT = `encode_temporal` + `.mean`. Never `.sample()`, never `MiniMaxH3VideoVAE.from_pretrained`, never train the 33B.
- No SCD/VFM, no op-id / task-family embedding, no Qwen, no audio in v0.
- ICQ: demos write **S**, query still is **H_0**, `canvas_write_s=False` for video.
- Portrait 9:16 for talking heads (512×288, N=4032). Sprites are 128×128 tests-only.
- Legal clip 22 frames @ 24 fps → T′=7. FakeVAE sprites use **spatial=8** so a 32px sprite is 4×4 cells.
- Host: `johndpope@msi.local`, RTX PRO 4000 ~24GB, conda torch 2.13+cu130. Do not train H3 on the Mac.

---

## Pass gate (current code)

`bdh_cq/video_probes.py`:

- `LAST_FRAME_L1 = 8/255`
- Peak: pred max ≥ 0.5 × GT max (kills mean-gray)
- Flood: last-frame occupied frac > 2× GT
- Compact dest (frac ≥ 0.5× GT) **skips** `motL1` — FakeVAE trail already puts oracle `motL1` ~0.047
- Talking heads: `motL1` (L1 weighted by `|GT last − GT first|`) ≤ 0.05. Full-frame lastL1 on a dark 9:16 canvas is ~0.037 for idle vs laugh, so mean L1 **lies**.
- Trainer **exits 2** on train-query last-frame mismatch (`LAST_FRAME_FAIL: ICQ query last frame`).

---

## What worked (sprites)

**Overfit query, family=`translate`, FakeVAE, protocol (~804k), CPU.**

| Clip | lastL1 | occ frac pred/GT | max pred/GT |
|---|---|---|---|
| `logs/recon_sprite_translate/step_00159.mp4` | **0.00551** | 0.076 / 0.078 | 0.957 / 0.961 |
| `logs/recon_sprite_translate/step_00079.mp4` | 0.00494 | 0.078 / 0.078 | 0.90 / 0.90 |

Left = pred, right = GT. t=21 still: `logs/recon_sprite_translate/_preview/t21.jpg`. Oracle stills: `_preview/oracle_t0.jpg`, `oracle_t21.jpg`.

What actually moved the square: **do not repaint the volume**. Predict **(dy, dx)** and **composite** the sprite onto a static background (`composite_shift` in `bdh_cq/video.py`). Shift-head teacher is the **latent energy-centroid** delta, `energy_centroid(query_out)[:,-1] − energy_centroid(query_in)[:,0]`, recomputed inside `train_loss` (`video.py` ~L1060). The pixel-centroid teacher (bright pixels / `vae.spatial`, ~7.875 cells y ≈ 63px for a seed-0 level-3/4 translate) was tried first and **overshoots** the FakeVAE last-time energy dest (~7.42 cells): it trips `motL1` while `lastL1` is already under 8/255, so the square lands past the target. `_bright_centroid_hw` is the leftover pixel helper — currently **unused**. Occupancy as a moving Gaussian at source+(dy,dx). `canvas_write_s=False`.

`encode_task` also stashes `query_shift` (same energy-centroid delta) in the task dict, but that field is **oracle-probe only** — consumed by `tests/test_video_icq.py` to drive `composite_shift` directly, never read by `train_loss` / `reason`. Do not wire training to it.

That is a **2D copy-and-paste head**, not a general motion renderer. It is a valid sprite sanity that ICQ can bind demo displacement into S and apply it to H_0.

### Command that passed

```bash
.venv/bin/python train_video_icq.py \
  --family translate --vae fake --device cpu --scale protocol --overfit True \
  --steps 160 --eval_every 20 --min_reasoning 4 --max_reasoning 4 --motion_rank 0 \
  --wandb False --recon_dir logs/recon_sprite_translate --ckpt logs/sprite_translate.pt
```

Exit 0 + `lastL1` on the **train query** + look at t=21. Held-out fail is expected until generalization is solved.

---

## Sprite challenges / failings

These are why “tests passed” and “loss 0.005, PSNR 16” were not a film-out.

1. **The square is ~6–8% of cells.** Uniform rec (MSE over all latent cells) averages to mean gray. Pred t=21 max ~0.20, `frac>0.35=0`, GT max ~0.90, frac ~0.078. Consecutive tL1 can match because the background is static.

2. **Motion-weighted rec (`latent_mse_motion`, floor=0.05) floods.** Background is ~20× cheaper, so pred paints energy everywhere (`frac>0.35` 0.27 vs GT 0.08). User screenshot: left wash, right compact square.

3. **`CLIP_FAIL` is the wrong gate.** `CLIP_FAIL` = consecutive tL1 < 1/255 (a still). A wash that jitters slightly is CLIP_FAIL-clear and still has no square.

4. **Rank-0 Linear predicting absolute z replaces the sprite.** `cat([z0, volume[:,:,1:]])` locks t=0 appearance then invents later frames. Residual `z0+u` without a dest prior either washes or ghosts. `lock_t0` smears.

5. **Full-frame `warp_still` cannot hit 8/255.** Warping the whole still also translates the checker. Oracle warp lastL1 floors ~**18/255** even with the true shift. **`composite_shift`** punches the source sprite, fills with bg, pastes the warped sprite; oracle then sits under 8/255 (~11/255). Do not go back to full-frame warp and then loosen the gate.

6. **Pixel-centroid teacher overshoots; energy-centroid is the one in the tree.** Bright-pixel centroid delta is ~7.875 cells; the FakeVAE last-time energy dest is ~7.42. Teaching the shift head the pixel value pushes the square ~0.4 cell past the target — `lastL1` stays under 8/255 but `motL1` trips (compact square, slightly wrong place). Current `video.py` teaches `energy_centroid(query_out)[:,-1] − energy_centroid(query_in)[:,0]` and `_bright_centroid_hw` is dead code. Do not revert to the pixel teacher without also loosening `motL1`, which defeats the gate.

7. **Held-out does not work.** Same family, new seed: `logs/recon_sprite_translate/heldout.mp4`, lastL1 **0.045**, motL1 **0.35**, **CLIP_FAIL** (still). `copy_hat 0.0025 < copy_star 0.0052` — pred barely moved. The (dy,dx) head **memorized this clip’s destination**, it did not learn translate. That is the real sprite remaining work.

8. **Tiny (~100k) Linear never made a square.** Protocol (~800k) plus the shift/composite head did. Extra BDH width did not; the **decode head** did.

9. **Overnight agents fought the tree.** Occupancy CE, gaussian logits, `lock_t0` vs `apply_residual`, `LAST_FRAME_L1` 8 vs 20/255 were rewritten in parallel. Trust `composite_shift` + last-frame occupancy/peak + the energy-centroid shift teacher, then re-read the files before editing.

10. **Sprite FakeVAE is not H3.** Identity RGB into ch 0–2, spatial 8, dark bg 8–48, sprite 180–255 (`bdh_cq/video_tasks.py`). A 32px sprite is 4×4 cells on the 16×16 grid. Do not mix spatial-16 random 3→24 into this oracle. **Stale in-tree comments:** `video_tasks.py` (`SPRITE_SIZE`/`SPRITE_ALIGN` say "spatial 16", "2×2 cells") and several `video_probes.py` docstrings ("8×8 latent grid") predate the spatial=8 sprite path — the code paths themselves (`FakeVideoVAE` docstring, `train_video_icq.py` `fake_spatial`, `video.py` `warp_spatial`) are all spatial 8.

---

## Held-out generalization (2026-09-10 pass): decode modes per family

`--overfit False` now trains a fresh task each step; the trainer's end-of-run
held-out eval (`seed + 10_000`) is the gate. `BDHVideoReasoningWrapper` takes
`decode={shift,copy,pan}`, auto-routed by family in `train_video_icq.py`
(`--decode` to override).

**`shift`** (`translate`) — composite the query still by a `(dy, dx)`.

*Direction* was never the problem. A probe (120 random tasks, untrained
protocol) showed the displacement is **not** linearly decodable from
`hidden.mean` (held-out R² −0.4 / −48) but the **demo-average energy-centroid
delta** gives R² 0.91 / 0.93, so `ingest_task` stashes `_demo_shift` and the
`demo_to_shift` `Linear(2,2)` (init `2·I`) gets direction + axis exact on every
task.

*Magnitude* is the demo→query ratio (1.8–4.0×), which tracks the query
**level** (= speed) — **not visible in one still frame**. A per-task gain head
on `hidden.mean` destabilised training (`shift_mse` 2 → 80). **Fix: a K-frame
query cue** (`--query_cue_frames 9`): `encode_task` runs the real query clip
for K frames then freezes it, so the velocity is measurable —
`query_vel = (energy_centroid(query_in)[:, k_lat] − [:, 0]) / k_lat`,
extrapolated to the last latent time in `ingest_task` (`_query_shift_from_vel`,
which then overrides `_demo_shift` and collapses the still to frame 0 for the
composite). With K=9 the extrapolated vector matches the true shift to ~0.1
cell and axis-aligned held-out clears the gate **untrained** (`lastL1`
0.010–0.026); diagonal sits at ~0.04 untrained (the FakeVAE-warp blur is worse
on the diagonal) and the `to_centroid` residual + `shift_mse` loss pull it
under with a short train. **EXIT 0** (protocol, 160 steps, non-overfit):
held-out `lastL1 0.023`, `mse 0.0007`, `identity_probe True`.

This is a task-shape change — the paper's ARC framing is one query input — but a
single still underdetermines *any* speed-carrying family, and a short real
prefix is how video generation is actually conditioned. `--query_cue_frames 0`
keeps the paper split.

**`copy`** (`identity`, `stamp_copy`, `recolor`) — no rigid motion; the answer
edits the still in place. Decode = **attention-copy**: `still_broadcast`, then
paste the pooled source-sprite appearance onto the copy targets, then add a
gated `to_latent(hidden)` residual for detail; `lock_t0` for `t=0`.
  - **Source vs targets** — flood-fill from the sprite centroid within the
    still's support (`energy ≥ OCC_FRAC·peak`) isolates the source component.
    `identity` / `recolor` are one blob → **no targets**, decode falls back to
    `still_broadcast + residual`. `stamp_copy`'s anchors are separate
    components → they are the copy targets, each anchor marker dilated
    **down-right** to the 2×2 stamp footprint (`_copy_tgt` in `ingest_task`).
  - `occ_prior` biases the residual gate toward the still's support (the
    per-cell `to_occupancy` alone stays near-uniform). `res_l2` is dropped for
    copy mode — there the residual *is* the edit.
  - `identity` — **solved** (EXIT 0): held-out `mse 0.0000`, `lastL1 0.002`.
  - `stamp_copy` — **solved** (EXIT 0): held-out `lastL1 0.016`, `mse 0.0011`,
    all step evals `clip`. The attention-copy clears the gate **untrained**
    (`lastL1` 0.0003–0.007 on seeds 0–3); training only firms it up. This
    replaces the earlier per-cell `to_latent` which was flat at `mse 0.003`.
  - `recolor` — **EXIT 0** (held-out `lastL1 0.020`, `mse 0.0004`,
    `identity_probe True` — the residual learned the A→B swap). But the pass is
    **soft**: a color change over ~6% of pixels is a sub-gate mean-L1 move
    either way, and the per-step evals bounce across the 8/255 line
    (0.015–0.030). Wire `recolor_probe` (already in `video_probes.py`) into
    `last_frame_mismatch` before trusting this gate.

**`pan`** (`pan`, `translate_pan`) — `composite_pan` /
`composite_translate_pan` are implemented and oracle-tested; `decode="pan"`
uses full-frame `warp_still` by the scene-centroid shift (the whole scene
translates together for `pan`).

**Decode-fidelity ceiling on FakeVAE, not reasoning:** the oracle full-frame
warp of the FakeVAE latent on a `pan` clip floors at **lastL1 0.036–0.10** vs
the 0.031 gate, *with the true vector* — bilinear warp of a coarse latent +
the periodic checker + the gradient blur on re-pool. `translate_pan` also
can't separate sprite motion from the 4-cell-period checker in the centroid.

**Fix = the real H3 VAE (needs msi — CUDA + H3 weights).** Its trained decoder
can clean up a warped latent; the FakeVAE `repeat_interleave` decoder cannot.
The code is ready: `warp_spatial` is now a wrapper arg, set to `VAE_SPATIAL`
(16) for `--vae h3` in the trainer. Runbook on `johndpope@msi.local`
(RTX PRO 4000, `MINIMAX_H3_ROOT` / `MINIMAX_H3_VAE` defaults in `video_vae.py`):

```bash
# pan — scene translates together, decode is full-frame warp_still
uv run python train_video_icq.py --family pan --vae h3 --device cuda \
  --scale protocol --overfit False --steps 200 --eval_every 25 \
  --min_reasoning 4 --max_reasoning 4 --query_cue_frames 9 --wandb False \
  --recon_dir logs/recon_pan_h3 --ckpt logs/pan_h3.pt
# translate_pan — sprite and bg move by different amounts; wire
# composite_translate_pan into reason() (currently only warp_still is), feed
# it _demo_shift (sprite) + a bg-region pan vector, then:
uv run python train_video_icq.py --family translate_pan --vae h3 --device cuda \
  --scale protocol --overfit False --steps 200 --query_cue_frames 9 \
  --recon_dir logs/recon_tpan_h3 --ckpt logs/tpan_h3.pt --wandb False
```

Check `logs/recon_pan_h3/heldout.mp4` (left = pred, right = GT) and the
trainer exit code. If the H3 latent still blurs on warp, fall back to a
structural gate: `pan_probe` / `translate_pan_probe` (already in
`video_probes.py` — screen centroid + bg phase) instead of the 8/255 L1.

### Gate fixes for the still families

- `is_still_clip(pred)` is only a `CLIP_FAIL` when GT actually moves —
  `still_fail = is_still_clip(pred) and not is_still_clip(gt)` in the trainer.
- `last_frame_mismatch` skips the `motL1` fallback when GT has no motion to
  weight by (`pixel_t0_t21_l1(target) < STILL_L1`). The 8/255 L1 gate is **not**
  relaxed for still GT — a converged `identity` roundtrips well under it, and
  `stamp_copy`/`recolor` must actually apply the edit.

---

## Talking heads (msi) — not a pass

Query: `icq_transfer` id_d **laugh**, demos other ids, 512×288, H3 VAE, R≤2, protocol ~0.8–1.0M.

| Recipe | t=21 visual | lastL1 / motL1 |
|---|---|---|
| Rank-32 IMTalker `m*` | Idle mouth | ~0.037 / 0.07–0.09 (mean L1 lied) |
| Residual `m*` = proj(z−z0), `lambda_ctx=0` | Idle | motL1 ~0.09 |
| Volume `lock_t0` ~1000 steps | Closed mouth, then cheek smear | copy_hat 0.73 vs copy_star 0.87 |
| Volume `still+u` | Ghost / double exposure | copy_hat 1.49 > 0.87 |
| 1B `scale=billion` 40 steps | Closed mouth | lastL1 0.07, loss 9.3→16.4, copy_hat collapsed to still |

**IMTalker idle vs laugh cosine ~0.99** — weak teacher for mouth open. Delta L2 is usable; after RMS norm the direction is still weak.

**Dark canvas:** idle vs laugh full-frame lastL1 ~0.037. Always look at t=21 pred|GT. Pairs: `logs/recon_imtalker_transfer/_preview/t21.jpg` (and `step_00399`, `step_00799`).

Do **not** relaunch the same protocol-scale laugh overfit. It has failed several times.

---

## 1B on the 24GB card

`--scale billion`: `dim=4096`, `dim_qk_heads=81920`, `depth=2` (tied block), `gradient_checkpoint=True`. ~**1.007B** params (the tied block is `3·dim·dim_qk_heads` ≈ 1.0066B; the rest is embeddings / heads / rotary — `sum(p.numel())` prints the exact count at startup).

- `dim=8192` at 1B: weights 4GB, **OOM on first forward**.
- `dim=4096`: forward ~17.6GB bf16. AdamW moments (~8GB) do not fit; trainer uses **SGD**.
- Full ICQ BPTT through ~189 demo chunks OOMs on S (`combine_memories`). Billion path: **demo ingest `no_grad`**, grads from query canvas + reason only.
- 40-step laugh: loss went **up**, `copy_hat` → 0.11 (still). Clip: `logs/recon_billion/step_00039.mp4`.

More parameters did not open the mouth. Sprite pass was a head change, not scale.

---

## Suggested next work (in order)

1. **`recolor` gate** — wire `recolor_probe` into `last_frame_mismatch`; mean
   pixel L1 cannot score a fill A→B swap over ~6% of the frame. Decode already
   runs (`decode=copy`, no copy targets, residual does the recolor).
2. **`pan` / `translate_pan` on the real H3 VAE (needs msi)** — run the two
   commands in the `pan` section above; `warp_spatial` is already threaded to
   16 for `--vae h3`. `reason()`'s `decode="pan"` branch calls `warp_still`
   directly (full-frame); `composite_pan` and `composite_translate_pan` are
   oracle-tested but **not yet wired into `reason()`** — `translate_pan` needs
   `composite_translate_pan` wired in plus a bg-region pan-vector measurement.
3. **`translate` diagonal robustness** — axis-aligned held-out clears the gate
   untrained with the 9-frame cue; diagonal needs the short train it already
   gets. Try `query_cue_frames` 6 vs 12 to see the accuracy/generation-length
   trade (a shorter cue leaves more of the clip to actually generate).
4. **Talking-head decode** analogous to the copy path: keep identity from H_0,
   generate only an expression residual that can make **teeth**. Rank-k /
   IMTalker / 1B Linear volume all failed that.
5. Do not loosen last-frame to pass a closed mouth. `motL1` on the face, or a
   mouth crop, is the talking-head gate.
6. MSI: `johndpope@msi.local`, VAE `/run/media/johndpope/2TB/minimax-h3-nvfp4/vae/minimax_h3_video_vae_fp16.safetensors`, H3 `/home/johndpope/Documents/GitHub/MiniMax-H3`. Transfer latents `data/icq_transfer/latents_h3_512x288_f22_transfer.pt`.

---

## Key paths

| What | Path |
|---|---|
| This report | `docs/HANDOFF_SPRITE_ICQ.md` |
| Sprite pass pair | `logs/recon_sprite_translate/step_00159.mp4` |
| Sprite t=21 still | `logs/recon_sprite_translate/_preview/t21.jpg` |
| Sprite held-out fail | `logs/recon_sprite_translate/heldout.mp4` |
| Earlier shift-head pass | `logs/recon_sprite_shift/step_00199.mp4` |
| Laugh fail t=21 | `logs/recon_imtalker_transfer/_preview/t21.jpg` |
| 1B fail pair | `logs/recon_billion/step_00039.mp4` |
| Composite / decode heads | `bdh_cq/video.py` (`composite_shift`, `composite_pan`, `composite_translate_pan`, `warp_still`, `_copy_logits`, `_shift_from_hidden`) |
| Decode-mode routing | `train_video_icq.py` (`--decode auto|shift|copy|pan`, `COPY_FAMILIES` / `PAN_FAMILIES`) |
| Last-frame gate | `bdh_cq/video_probes.py` |
| Trainer | `train_video_icq.py` |
| Sprite oracle | `bdh_cq/video_tasks.py` (`SPRITE_SIZE=32`, `BG_LO/HI`, `SPRITE_LO/HI`) |
| identity held-out pass | `logs/recon_id_v2/heldout.mp4` |
| stamp_copy held-out fail | `logs/recon_sc_v2/heldout.mp4` |
| translate held-out, pre-cue (direction ok, magnitude off) | `logs/recon_sprite_gen/heldout.mp4` |
| translate held-out, `--query_cue_frames 9` (EXIT 0) | `logs/recon_translate_cue/heldout.mp4` |
