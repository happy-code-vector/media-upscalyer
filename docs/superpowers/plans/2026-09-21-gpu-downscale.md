# GPU Downscale Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Downscale the x4 model output to the target scale on the GPU (before the device→host copy) so 2x/3x runs skip the 33MP download + CPU Lanczos pass (~2x end-to-end speedup).

**Architecture:** Single change inside `upscale_frame_direct()` in `upscale_video_fast.py`: after `model(tensor)`, when `scale != model_scale`, call `torch.nn.functional.interpolate(..., mode="bicubic", align_corners=False)` with an explicit `size=(h*scale, w*scale)`, then proceed with the existing squeeze/denormalize/`.cpu()` path. The old CPU `cv2.resize(INTER_LANCZOS4)` block is deleted.

**Tech Stack:** Python 3.12 venv at `env/` (torch 2.2.0+cu121, opencv-python), project-local ffmpeg 4.2.2 at `ffmpeg/bin/ffmpeg.exe`.

## Global Constraints

- All python invocations use `./env/Scripts/python.exe` (Git Bash) or `.\env\Scripts\python.exe` (PowerShell). The venv is already activated in the user's own shell; scripts must not assume it.
- No new pip dependencies (no pytest — tests are standalone scripts with plain asserts).
- The working tree has an unrelated staged-free deletion (`Real-ESRGAN` submodule). NEVER `git add -A`. Add files by exact path only.
- `upscale_video_fast.py` currently contains uncommitted operational fixes from this session (project-local ffmpeg lookup, NVENC `-preset fast`, ffmpeg stderr→log files). Task 0 commits them so the optimization lands as a clean diff.
- Model file already exists at `models/realesr-animevideov3.pth` — no download happens.

---

### Task 0: Commit pending operational fixes (housekeeping)

**Files:**
- Modify (already modified on disk, no edits needed): `upscale_video_fast.py`, `.gitignore`

**Steps:**

- [ ] **Step 1: Verify the pending diff is only the session fixes**

Run: `git diff upscale_video_fast.py .gitignore`
Expected: ffmpeg local-path lookup in `get_ffmpeg_path`, `-preset "fast"` in encode_cmd, three stderr-related edits, `.gitignore` entries `ffmpeg/` and `.downloads/`. No other hunks.

- [ ] **Step 2: Commit both files by exact path**

```bash
git add upscale_video_fast.py .gitignore
git commit -m "fix: project-local ffmpeg, NVENC-compatible preset, stderr-to-file logging

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 1: GPU downscale in `upscale_frame_direct` (TDD)

**Files:**
- Create: `tests/test_gpu_downscale.py`
- Modify: `upscale_video_fast.py:109-146` (function `upscale_frame_direct`)

**Interfaces:**
- Consumes: `load_fast_model(name) -> (model, model_scale)` and `upscale_frame_direct(model, frame_bgr, scale, model_scale) -> np.ndarray (H*scale, W*scale, 3) uint8 BGR` — both already exist in `upscale_video_fast.py`.
- Produces: unchanged public signature; behavior change only.

- [ ] **Step 1: Write the failing test**

Create `tests/test_gpu_downscale.py`:

```python
"""Verification for GPU downscale in upscale_frame_direct.

Run: ./env/Scripts/python.exe tests/test_gpu_downscale.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from upscale_video_fast import load_fast_model, upscale_frame_direct


def cpu_reference(model, frame_bgr, scale, model_scale):
    """The old pipeline: forward -> full-res download -> cv2 Lanczos."""
    with torch.no_grad():
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        chw = rgb.transpose(2, 0, 1) / 255.0
        t = torch.from_numpy(chw).float().cuda().half().unsqueeze(0)
        out = model(t)
    out = out.squeeze(0).float()
    out = (out * 255).clamp(0, 255).byte()
    out_np = out.permute(1, 2, 0).cpu().numpy()
    out_bgr = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)
    h, w = frame_bgr.shape[:2]
    return cv2.resize(out_bgr, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255**2 / mse)


def main():
    model, model_scale = load_fast_model("realesr-animevideov3")
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, (270, 480, 3), dtype=np.uint8)  # small = fast

    # 1. exact output dimensions
    out = upscale_frame_direct(model, frame, scale=2, model_scale=model_scale)
    assert out.shape == (540, 960, 3), f"bad shape {out.shape}"

    # 2. must not call cv2.resize (old CPU path did; new path must not)
    orig_resize = cv2.resize

    def boom(*a, **k):
        raise AssertionError("cv2.resize called - old CPU path still active")

    cv2.resize = boom
    try:
        out2 = upscale_frame_direct(model, frame, scale=2, model_scale=model_scale)
    finally:
        cv2.resize = orig_resize
    assert out2.shape == (540, 960, 3)

    # 3. close to the old Lanczos path (filter difference only)
    ref = cpu_reference(model, frame, 2, model_scale)
    p = psnr(ref, out)
    print(f"PSNR old-vs-new: {p:.1f} dB")
    assert p > 40, f"outputs diverge too much: {p:.1f} dB"

    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./env/Scripts/python.exe tests/test_gpu_downscale.py`
Expected: FAIL with `AssertionError: cv2.resize called - old CPU path still active` (checks 1 passes today; check 2 is the discriminator).

- [ ] **Step 3: Implement the GPU downscale**

In `upscale_video_fast.py`, replace the body of `upscale_frame_direct` from the `# Upscale` line through the final `return output_bgr` with:

```python
        # Upscale
        output = model(tensor)

        # Downscale on GPU when target scale < model scale: avoids the 33MP
        # device->host copy and CPU Lanczos pass (~2x faster end-to-end)
        if scale != model_scale:
            h, w = frame_bgr.shape[:2]
            output = torch.nn.functional.interpolate(
                output, size=(h * scale, w * scale),
                mode="bicubic", align_corners=False)

        # Remove batch dim, CHW -> HWC, denormalize
        output = output.squeeze(0).float()
        output = (output * 255).clamp(0, 255).byte()
        output_np = output.permute(1, 2, 0).cpu().numpy()

        # RGB -> BGR
        output_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)

        return output_bgr
```

(This deletes the old trailing `if scale != model_scale:` cv2.resize block — the GPU interpolate above replaces it.)

- [ ] **Step 4: Run test to verify it passes**

Run: `./env/Scripts/python.exe tests/test_gpu_downscale.py`
Expected: `PSNR old-vs-new: <value> dB` then `ALL CHECKS PASSED`. PSNR is expected in the 40–60 dB range (bicubic-vs-Lanczos on 2:1 downscale). If below 40, STOP and investigate before committing.

- [ ] **Step 5: Commit**

```bash
git add tests/test_gpu_downscale.py upscale_video_fast.py
git commit -m "perf: downscale to target scale on GPU before device->host copy

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 2: End-to-end verification on a 10s clip

**Files:**
- Create (runtime artifacts, git-ignored or temp): `.downloads/test10s.mp4`, `output/test10s_2x.mp4`

**Steps:**

- [ ] **Step 1: Trim a 10-second test clip**

```bash
./ffmpeg/bin/ffmpeg.exe -y -hide_banner -loglevel error -i "inputs/video/Six beautiful Princesses.mp4" -t 10 -c copy .downloads/test10s.mp4
```

- [ ] **Step 2: Run the full pipeline on it**

```bash
./env/Scripts/python.exe upscale_video_fast.py -i .downloads/test10s.mp4 -o output/test10s_2x.mp4
```

Expected: completes with `DONE! ~300 frames` and `Average speed:` — should read **≥ 3 fps** (was 2.2 before the change) and finish in well under 3 minutes. If it hangs, check `output/test10s_2x.mp4.encoder.log`.

- [ ] **Step 3: Verify output properties**

```bash
./ffmpeg/bin/ffmpeg.exe -hide_banner -i output/test10s_2x.mp4
```

Expected (stderr info): `Duration: 00:00:09.9x`–`00:00:10.0x`, video `3840x2160 ... 30 fps`, plus an `aac` audio stream. Exit code 1 from ffmpeg with no output file specified is normal — the stream info above is the check.

- [ ] **Step 4: Report numbers**

Report to the user: frames, average fps (old 2.2 → new ?), output size/duration. Nothing to commit (runtime artifacts only; `output/` is git-ignored).
