# In-Context Latent Video: Binding Shots in Recurrent Memory

**Status:** working paper (draft), architecture as-built 2026-09-09  
**Date:** 2026-09-09  
**Code:** `bdh-cq` (`video.py`, `aval_clips.py`, FakeVAE, associative `BDHBlock`) + frozen MiniMax-H3 VisualVAE on msi  
**Plan:** [BDH_CQ_VIDEO_PLAN.md](BDH_CQ_VIDEO_PLAN.md)

---

## Abstract

Video diffusion transformers generate by denoising every packed latent token, every layer, every sampling step. BDH-CQ generates by writing a task into a fixed-size associative memory and iterating a compact workspace until a constraint is met, then decoding. This note argues that **shot generation is an in-context binding problem**, not a score-matching problem over 30k–90k tokens, and specifies a first experiment that can fail for the same reason BDH-CQ fails on ARC: composition.

Ground truth for that experiment is not pixels. It is the frozen H3 VisualVAE **posterior mean** of query-output clips, bound from demonstrations with no task-id token and **no text encoder**. Content is sprite oracles *or* AVAL talking-head clips (one actor, 124 expressions; the query still is the cue). Default canvas is **portrait 9:16 at 512×288** (\(N=4032\)), height and width dynamic (16-pixel aligned). Associative omit-self attention is on for \(N>512\). Pixel film-out at 128² is already measured: 28.9 dB overall, 16.9 dB on a 24 px sprite — a moving blob, not a square. Training stays in latent space and runs on **msi CUDA** (RTX PRO 4000 Blackwell). Mac MPS OOM at this \(N\).

---

## 1. The claim

BDH-CQ (Engdahl et al., arXiv:2608.09888) has three objects:

\[
S_t = U_\theta(S_{t-1}, D_t),\qquad
H_0 = E_\theta(x^\star, S_K),\qquad
H_{r+1} = F_\theta(H_r, S_K),\qquad
\hat y = G_\theta(H_R).
\]

\(S\) is written from context and held during reasoning. \(H\) is a silent workspace. \(G\) decodes once. Extra test-time compute is extra \(R\), not extra tokens.

A video DiT (MiniMax-H3 Omni-Transformer: 50 layers, hidden 5376, packed multimodal sequence) spends that compute on softmax attention over the whole clip, every denoise step. That is the wrong place to put “the camera pans **and** the subject walks.” Binding two operators is exactly table 4 of the BDH-CQ paper: rotation∘relocate saturates; reflection∘relocate and color-swap∘relocate collapse.

**Thesis.** Treat a shot as an ARC task whose “grid” is a VAE latent volume. Demonstrations write the shot grammar into \(S\). The query first frame (as a still clip) seeds \(H_0\). \(F\) iterates until the volume matches the bound transform. \(G\) is a frozen VisualVAE decoder — film-out, not the generator.

---

## 2. Ground truth (the load-bearing decision)

Three axes. v0 is the product **H × F × A**.

| Axis | Choice | Why |
|---|---|---|
| Protocol | **H** — in-context demos, **no op id** | An op embedding lets a 6–8M model skip \(S\) and become a labelled multi-task net. ARC `task_prompt` has no family name. |
| Content | **F** — sprite oracles, plus AVAL still→clip ICQ | Sprites: exact GT, `translate_pan` is the cliff. AVAL: 124 same-actor expression clips, 3 demos + 1 query, **no action-name token**. |
| Representation | **A** — frozen VAE posterior **mean** | Lives in \(G\)'s code space. `encode_base` / `encode_videos` always `.sample()`; v0 must call `encode_temporal` + `.mean`. |

Answer tensor:

```text
z* = DiagonalGaussianDistribution(encode_temporal(preprocessed_clip)).mean
   # (B, 24, T'=7, H/16, W/16)
```

Loss is unmasked latent L2 plus a temporal-difference term, **at every reasoning step** including \(R=0\). Pixels are eval-only. The 33B Omni-Transformer is not loaded.

Still inputs (first frame, identity) are encoded as **22-frame still clips on the video path**. Mixing `encode_images` with `encode_videos` teaches an encoder-path mapping, not the oracle.

---

## 3. Canvas: larger, portrait, dynamic

128×128 was a unit-test size. Production v0 is **portrait**.

H3 VisualVAE spatial ratio is 16; `VAEProcessor` crops to multiples of `latent_patch_size * vae_ratio` (default 16). Any \((H,W)\) with \(H \equiv W \equiv 0 \pmod{16}\) is legal. \(T=22\) frames stays production-legal (\(T'=7\)).

\[
N = 7 \cdot (H/16) \cdot (W/16)
\]

| Preset | Pixels \(H\times W\) | Aspect | Latent \(H'\times W'\) | \(N\) |
|---|---|---|---|---|
| **portrait (default)** | 512×288 | 9:16 | 32×18 | 4032 |
| portrait-768 | 768×432 | 9:16 | 48×27 | 9072 |
| square-256 | 256×256 | 1:1 | 16×16 | 1792 |
| landscape | 288×512 | 16:9 | 18×32 | 4032 |
| test | 128×128 | 1:1 | 8×8 | 448 |

Oracles take `height`, `width`. The wrapper must not hardcode 448. Raster order is `(t, h, w)` with \(h\) the vertical axis — portrait means more rows.

At \(N=4032\), materializing \(N\times N\) in `BDHBlock` is no longer “fine.” The causal associative form \(q_i \cdot \sum_{j<i} k_j^\top v_j\) is implemented (`_omit_self_linear_attn`, `seq>512`).

**AVAL (as-built).** Source clips are 720×1280 @ 24 fps, ~6 s, scaled to 512×288 (keep 9:16). The 22-frame window is the **max pixel-Δ** slice, not a fixed start at t=24. If `decode(z*)` is still, the clip is dropped; if too few survive, the next legal length is 39 frames (\(T'=12\)). Names never index `token_embed`. `H_0` is the still volume; the head predicts velocity \(u\) and \(\hat z = z_{\text{still}} + \mathrm{cumsum}(u)\).

**Measured still failure (1000 mixed steps, tiny, ungated 22-frame).** Loss 0.91→~0.11, PSNR 21→31, held-out mse 0.10, **CLIP_FAIL on every eval**. `copy_star ≈ 0.89` (latent moves) vs `gt tL1 ≈ 0.002` (film-out still). Native 1280×720, Qwen, op-id, 33B DiT, and more steps of the same GT do not address this.

**Probe (8 AVAL, 512×288).** Sprite `translate` \(G(z^\star)\) tL1 **0.00512 clip**. AVAL **f=22**: source-moving 6/8, G-moving **6/6**. **f=39**: source-moving 6/8, G-moving **3/6**. Full library admit=decode: **keep 61 / drop 63**. Do not hardcode 39.

**200-step gated tiny.** GT is a clip; pred moves but **does not match the target expression** (held-out: GT eyes open at t=21, pred stays at the t=0 still plus smear). Pixel gate stays a **filter**. Next: `--sprite_mix` so identity/translate remain controls; larger `--scale` for appearance; three-score jsonl (`src_tL1`, `latent_vs_still`, `decode_tL1`). DiT/ConvRot latents stay out.

---

## 4. Measurement: pixel film-out at 128

Host: `msi.local`, RTX PRO 4000 Blackwell. Weights: Comfy-Org `minimax_h3_video_vae_fp16.safetensors` loaded into `AutoencoderKLLegacy` (`missing 0 / unexpected 0`). Encode of a 22×128×128 translating 24 px sprite:

| | |
|---|---|
| \(z\) | `(1, 24, 7, 8, 8)` |
| decode | `(1, 3, 22, 128, 128)` after casting \(z\) to fp16 |
| whole-frame PSNR | **28.9 dB** |
| background | **32.3 dB** |
| sprite pixels | **16.9 dB** |

The recon is a yellow **blob that moves**. Hard edges and inset colors are gone. That is expected: 24 px / 16 = 1.5 latent cells. **Pixels are not the training target.** They remain a coarse “did the blob move” check. Larger portrait canvas buys more latent cells on the subject, not DiT-like texture.

---

## 5. System

```text
demos (still → clip)  --chunk 128-->  S   (fast weights, add-combine)
query still-volume z_in  --one pass--->  H_0 = raster(z_in)  (no empty_cell)
                         R steps
H_{r+1} = BDH(H_r, S frozen, residual list seeded once)
ẑ_r = Linear(dim, 24)(H_r)   reshaped to (B, 24, 7, H/16, W/16)
L = mean_r  [ L2(ẑ_r, z*) + L2(Δẑ_r, Δz*) ]
eval: decode_base(ẑ_R, frame_num=22)  → portrait RGB
```

Text is **demos only**. No Qwen3-VL-32B. No `{translate, pan, …}` embedding.

Audio VAE exists on disk (578M fp32) and is out of v0.

---

## 6. What success looks like

Copied from the ARC protocol, not from FVD:

1. **Atomic ops** (translate, pan, identity) exact on latent-grid centroids at held-out speeds.
2. **Composition** (`translate_pan` at test speeds, opposite-sign pan) is the cliff — report it, do not hide it with an op token.
3. **Effort:** \(R \in \{1,2,4,6,8\}\) monotone in latent L2 / probe pass, matching figure 7.
4. **Copy-detector:** \(\mathrm{MSE}(\hat z, \mathrm{repeat}(z^*_{t=0}))\) worse than \(\mathrm{MSE}(z^*, \mathrm{same})\).
5. Film-out is a blob that moves. That is enough.

---

## 7. What this is not

- Not linear attention inside H3 DiT blocks.
- Not a 33B fine-tune.
- Not next-frame language modeling on rasterized cells.
- Not pixel-space regression through the 36-layer ViT decoder.

---

## 8. Train (as-built)

Laptop pytest: FakeVAE, 128², CPU. Film canvas: 512×288 on **msi**, `--device cuda`.

```bash
PYTHONUNBUFFERED=1 python train_video_icq.py \
  --family aval --steps 80 --scale tiny --overfit=False \
  --device cuda --height 512 --width 288
```

`--family identity` still runs sprite oracles. `--family aval` samples 3+1 AVAL clips. No LLM. Playback of \(G\) is 24 fps; 22 frames ≈ 0.92 s.

---

## References

- Engdahl et al., *BDH-CQ: In-Context Learning with Recurrent Latent Reasoning*, arXiv:2608.09888.
- Kosowski et al., *The Dragon Hatchling*, arXiv:2509.26507.
- MiniMax-H3 VisualVAE: `FL2VA/video_vae/` (`AutoencoderKLLegacy`, f16t4d24, \(T'=7\) at 22 frames).
- This repo: `bdh_cq/bdh_cq.py`, `icq.py`, `tasks.py`, `figure7.py`, `docs/BDH_CQ_VIDEO_PLAN.md`.
