# RRDBNet Tiled Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bound RRDBNet inference VRAM by tiling, taking `RealESRGAN_x4plus_anime_6B` from ~30 s/frame (WDDM spillover) to ~3.5-4.5 s/frame.

**Architecture:** New `_upscale_tiled` helper runs the model on overlapping tiles and stitches into one preallocated GPU buffer; `upscale_frame_direct` gains a `tile` parameter (0 = today's direct path); `upscale_video_fast` auto-enables tile=960 for the two RRDBNet models; `--tile` CLI flag overrides.

**Tech Stack:** Python 3.12 venv at `env/` (torch 2.2.0+cu121, opencv-python), project-local ffmpeg 4.2.2 at `ffmpeg/bin/ffmpeg.exe`.

## Global Constraints

- All python invocations use `./env/Scripts/python.exe` (Git Bash); no activated venv assumed.
- No new pip dependencies (tests are standalone scripts with plain asserts).
- NEVER `git add -A`; add files by exact path only.
- Model weights already exist at `models/RealESRGAN_x4plus_anime_6B.pth` and `models/realesr-animevideov3.pth` — no downloads.
- `tile=0` must keep `upscale_frame_direct` byte-identical to the current direct path (existing `tests/test_gpu_downscale.py` must pass unchanged).
- Binding values from the spec: `overlap=16` (fallback 32 if the PSNR gate fails), auto tile size `960` for exactly the model names `RealESRGAN_x4plus` and `RealESRGAN_x4plus_anime_6B`, `0` for `realesr-animevideov3`.

---

### Task 1: Tiled inference + tests (TDD)

**Files:**
- Create: `tests/test_tiling.py`
- Modify: `upscale_video_fast.py` (new helper + signature changes + CLI)

**Interfaces:**
- Consumes: `load_fast_model(name) -> (model, model_scale)`; existing `upscale_frame_direct(model, frame_bgr, scale=2, model_scale=4)`.
- Produces: `upscale_frame_direct(model, frame_bgr, scale=2, model_scale=4, tile=0)`; `_upscale_tiled(model, tensor, model_scale, tile=960, overlap=16)`; `upscale_video_fast(..., tile=None)`; CLI `--tile`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_tiling.py`:

```python
"""Verification for tiled inference (bounds RRDBNet VRAM).

Run: ./env/Scripts/python.exe tests/test_tiling.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from upscale_video_fast import load_fast_model, upscale_frame_direct


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255**2 / mse)


def main():
    model, model_scale = load_fast_model("RealESRGAN_x4plus_anime_6B")
    rng = np.random.default_rng(0)
    frame = cv2.GaussianBlur(
        rng.integers(0, 256, (540, 960, 3), dtype=np.uint8), (0, 0), 2.0)

    # 1. exact output shape with forced multi-tiling (3 cols x 2 rows = 6 tiles)
    out_tiled = upscale_frame_direct(model, frame, scale=2, model_scale=model_scale, tile=320)
    assert out_tiled.shape == (1080, 1920, 3), f"bad tiled shape {out_tiled.shape}"

    # 2. tiled vs direct on the same frame: only seam-context differences
    out_direct = upscale_frame_direct(model, frame, scale=2, model_scale=model_scale, tile=0)
    p = psnr(out_direct, out_tiled)
    print(f"PSNR tiled-vs-direct: {p:.1f} dB")
    assert p > 40, f"tiling diverges from direct: {p:.1f} dB"

    # 3. frame smaller than tile -> direct path
    small = frame[:270, :480]
    out_small = upscale_frame_direct(model, small, scale=2, model_scale=model_scale, tile=960)
    assert out_small.shape == (540, 960, 3), f"bad small shape {out_small.shape}"

    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./env/Scripts/python.exe tests/test_tiling.py`
Expected: FAIL with `TypeError: upscale_frame_direct() got an unexpected keyword argument 'tile'`.

- [ ] **Step 3: Implement**

In `upscale_video_fast.py`:

3a. Add the helper immediately above `upscale_frame_direct`:

```python
def _upscale_tiled(model, tensor, model_scale, tile=960, overlap=16):
    """Run the model tile-by-tile with overlap, stitching on the GPU.

    Bounds peak VRAM for RRDBNet models whose dense feature maps cost
    ~8.3 KB per input pixel (1080p direct needs ~17 GiB on a 12 GB card).
    """
    _, _, h, w = tensor.shape
    out = torch.empty((1, 3, h * model_scale, w * model_scale),
                      dtype=tensor.dtype, device=tensor.device)
    for y0 in range(0, h, tile):
        for x0 in range(0, w, tile):
            y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
            # grow the input region into neighbours (clamped) so border
            # convolutions see true context
            ye0, xe0 = max(0, y0 - overlap), max(0, x0 - overlap)
            ye1, xe1 = min(h, y1 + overlap), min(w, x1 + overlap)
            t = model(tensor[:, :, ye0:ye1, xe0:xe1])
            # crop the grown margin off the output, scaled to model resolution
            cy0 = (y0 - ye0) * model_scale
            cx0 = (x0 - xe0) * model_scale
            out[:, :, y0 * model_scale:y1 * model_scale,
                x0 * model_scale:x1 * model_scale] = t[
                :, :, cy0:cy0 + (y1 - y0) * model_scale,
                cx0:cx0 + (x1 - x0) * model_scale]
    return out
```

3b. Change the signature line to:
```python
def upscale_frame_direct(model, frame_bgr, scale=2, model_scale=4, tile=0):
```

3c. Replace the `# Upscale` / `output = model(tensor)` two lines with:
```python
        # Upscale (tiled when requested: bounds VRAM for RRDBNet models)
        if tile and (tensor.shape[2] > tile or tensor.shape[3] > tile):
            output = _upscale_tiled(model, tensor, model_scale, tile=tile)
        else:
            output = model(tensor)
```

3d. In `upscale_video_fast`, change the signature to add `tile=None` as the last parameter, and after the `model, model_scale = load_fast_model(model_name)` line add:
```python
    # RRDBNet models need ~8.3KB/input-pixel VRAM: 1080p direct spills to
    # system memory (WDDM) and runs ~6x slower. Tile them by default.
    if tile is None:
        tile = 960 if model_name in ("RealESRGAN_x4plus", "RealESRGAN_x4plus_anime_6B") else 0
    print(f"Tiling: {tile}px" if tile else "Tiling: off")
```

3e. In the main loop's call, forward it:
```python
                    output_frame = upscale_frame_direct(model, frame, scale, model_scale, tile=tile)
```

3f. In `main()`'s argparse, after the `--no-nvenc` line add:
```python
    parser.add_argument("--tile", type=int, default=None,
                        help="Tile size for RRDB models (0=off, default: auto 960)")
```
and pass `tile=args.tile` in the `upscale_video_fast(...)` call.

- [ ] **Step 4: Run tests to verify pass**

Run: `./env/Scripts/python.exe tests/test_tiling.py`
Expected: `PSNR tiled-vs-direct: <value> dB` then `ALL CHECKS PASSED`.
If the PSNR gate fails (< 40 dB): change the helper's default `overlap=16` to `overlap=32` AND the spec fallback applies — re-run; report which value shipped.

Run: `./env/Scripts/python.exe tests/test_gpu_downscale.py`
Expected: unchanged pass (`PSNR old-vs-new` line + `ALL CHECKS PASSED`) — proves tile=0 is byte-identical for existing callers.

- [ ] **Step 5: Commit**

```bash
git add tests/test_tiling.py upscale_video_fast.py
git commit -m "perf: tiled inference for RRDBNet models (bounds VRAM, ~6x faster)

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 2: Timing, VRAM and end-to-end verification

**Files:**
- Create (runtime artifacts only): `.downloads/test3s.mp4`, `output/test3s_6b.mp4`

- [ ] **Step 1: Single-frame 1080p timing/VRAM check**

```bash
./env/Scripts/python.exe -c "
import time, numpy as np, torch, cv2, sys
sys.path.insert(0, '.')
from upscale_video_fast import load_fast_model, upscale_frame_direct
model, ms = load_fast_model('RealESRGAN_x4plus_anime_6B')
frame = (np.random.rand(1080, 1920, 3) * 255).astype(np.uint8)
upscale_frame_direct(model, frame, 2, ms, tile=960)  # warmup
torch.cuda.reset_peak_memory_stats()
t0 = time.perf_counter()
out = upscale_frame_direct(model, frame, 2, ms, tile=960)
torch.cuda.synchronize()
dt = time.perf_counter() - t0
peak = torch.cuda.max_memory_allocated() / 2**30
print(f'tiled 1080p: {dt:.2f} s/frame, peak VRAM {peak:.2f} GiB, out {out.shape}')
assert dt < 6, f'too slow: {dt:.2f} s'
assert peak < 10, f'VRAM not bounded: {peak:.2f} GiB'
print('TIME/VRAM CHECKS PASSED')
"
```
Expected: ~3-4.5 s/frame, ~8 GiB peak, `TIME/VRAM CHECKS PASSED`.

- [ ] **Step 2: 3-second end-to-end run with the 6B model**

```bash
./ffmpeg/bin/ffmpeg.exe -y -hide_banner -loglevel error -i "inputs/video/Six beautiful Princesses.mp4" -t 3 -c copy .downloads/test3s.mp4
./env/Scripts/python.exe upscale_video_fast.py -i .downloads/test3s.mp4 -o output/test3s_6b.mp4 -m RealESRGAN_x4plus_anime_6B
```
Expected: startup banner prints `Tiling: 960px`; completes ~90 frames in ~5-7 min; `DONE!` summary with average speed ~0.2-0.3 fps.

- [ ] **Step 3: Verify output properties**

```bash
./ffmpeg/bin/ffmpeg.exe -hide_banner -i output/test3s_6b.mp4
```
Expected: `Duration: 00:00:02.9x`-`00:00:03.0x`, video `3840x2160 ... 30 fps`, `aac` audio stream (exit code 1 from the probe itself is normal).

- [ ] **Step 4: Report**

Report frames, average fps, time/frame, output specs vs the ~30 s/frame baseline. Nothing to commit (runtime artifacts; `output/` git-ignored).
