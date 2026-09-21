# GPU Downscale for 2x/3x Targets — Design

Date: 2026-09-21
Target: `upscale_video_fast.py` (function `upscale_frame_direct`)

## Problem

`realesr-animevideov3` is x4-native (checkpoint final conv is 48ch = 3×4²; no
x2/x3 weights exist). For a 2x target the pipeline therefore runs the model at
4x, downloads the 33MP output tensor to CPU, and Lanczos-downscales it to 8MP.
Benchmark (RTX 3060, 1080p frame): full path 610 ms/frame, of which ~350 ms is
the 33MP device→host copy + CPU resize. The model pass itself (238 ms) is
GPU-bound; batching (243 ms/frame at batch=4) and cudnn.benchmark give nothing.

## Change

In `upscale_frame_direct`, when `scale != model_scale`, downscale on the GPU
before the device→host copy:

```python
output = model(tensor)
if scale != model_scale:
    h, w = frame_bgr.shape[:2]
    output = torch.nn.functional.interpolate(
        output, size=(h * scale, w * scale),
        mode="bicubic", align_corners=False)
```

The existing CPU `cv2.resize(INTER_LANCZOS4)` block is removed. Explicit
`size=` (not `scale_factor`) guarantees exact output dimensions for any input.
When `scale == model_scale` the code path is unchanged.

## Verification

1. One real frame through old vs new path: PSNR between outputs (expect
   > ~45 dB; difference is the bicubic-vs-Lanczos filter only).
2. 10-second trimmed clip through the full pipeline: frame time drops from
   ~450 ms toward ~260 ms; output duration/resolution/audio verified with
   ffprobe.

## Rejected Alternatives

- Native x2 weights: do not exist in the release (checkpoint inspected).
- Batched inference: no gain, GPU already saturated at batch=1.
- `--cpu-resize` fallback flag: user opted out; git-revert suffices.
