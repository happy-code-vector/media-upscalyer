# Tiled Inference for RRDBNet Models — Design

Date: 2026-09-21
Target: `upscale_video_fast.py` (`upscale_frame_direct`, `upscale_video_fast`, CLI)

## Problem

RRDBNet models (`RealESRGAN_x4plus`, `RealESRGAN_x4plus_anime_6B`) keep dense
feature maps at input resolution: measured ~8.3 KB per input pixel. A 1920x1080
frame needs ~17.1 GiB peak VRAM against this machine's 12 GB RTX 3060, so the
WDDM driver silently pages ~5+ GiB to shared system memory and the pipeline
runs at 18.5-19.2 s/frame in isolation (~30 s/frame in the real pipeline).
Measured on-GPU evidence (bench, fp16):

| input | time/frame | peak VRAM |
|---|---|---|
| 1920x1080 | 18.5-19.2 s | 17.08 GiB (spills) |
| 960x540   | 0.77-0.91 s | 4.28 GiB |
| 480x270   | 0.20 s      | 1.08 GiB |

Pixel-rate extrapolation says a full 1080p frame costs ~3.1 s when memory
fits — the spillover is a ~6x penalty. `realesr-animevideov3` (SRVGG) fits at
1080p and is unaffected.

## Change

1. `upscale_frame_direct(model, frame_bgr, scale=2, model_scale=4, tile=0)` —
   new `tile` parameter. `tile=0` (default) is today's direct path,
   byte-identical.
2. New helper `_upscale_tiled(model, tensor, model_scale, tile, overlap=16)`:
   grid of tiles stepping by `tile` px; each input region expanded by
   `overlap` px into neighbors (clamped at frame edges); model output cropped
   by the same overlap scaled by `model_scale`; cropped tiles written into one
   preallocated `[1,3,H*model_scale,W*model_scale]` fp16 buffer on the GPU.
   Frames where both dims <= tile fall through to the direct call.
3. `upscale_video_fast(..., tile=None)`: resolves `None` to 960 for the two
   RRDBNet model names and 0 for `realesr-animevideov3`; prints the resolved
   value in the startup banner; forwards it to `upscale_frame_direct`.
960^2 x 8.3 KB/px ~ 7.6 GiB peak — fits 12 GB with headroom.
4. CLI: `--tile N` overrides the auto choice (0 = off).

fp16 inference, the antialiased GPU downscale, and everything else stay as-is.

## Verification

1. `tests/test_tiling.py` (6B model, 960x540 smooth frame): forced
   multi-tiling (`tile=320`, 3x2 = 6 tiles) produces the exact output shape;
   tiled vs direct on the same frame > 40 dB PSNR (only seam-context
   differences); a frame smaller than the tile takes the direct path.
   If the PSNR gate fails, raise `overlap` 16 -> 32 and re-run (documented
   fallback, decided by test evidence).
2. Single-frame 1080p timing/VRAM check with tiling: < 6 s/frame and peak
   VRAM < 10 GiB (vs 19 s / 17 GiB un-tiled).
3. 3-second clip end-to-end with `-m RealESRGAN_x4plus_anime_6B`: completes,
   correct duration/resolution/audio; report fps (expect ~3.5-4.5 s/frame
   incl. I/O, i.e. ~0.22-0.29 fps).

## Rejected Alternatives

- Use only `animevideov3` (14x faster) — user chose quality of 6B.
- Smaller fixed tile for everything — slows the SRVGG path for no benefit.
- CPU offload / system-RAM staging — same PCIe thrash being fixed.
- fp32 — doubles memory, worsens the spill.
