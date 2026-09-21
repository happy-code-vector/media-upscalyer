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
    # Realistic (smooth) content, not white noise: on pure noise ANY two
    # downscale filters disagree (~19-33 dB PSNR between them), so the
    # filter-difference check below is only meaningful on low-pass content.
    # Deterministic via the fixed seed.
    base = rng.integers(0, 256, (270, 480, 3), dtype=np.uint8).astype(np.float32)
    frame = np.clip(cv2.GaussianBlur(base, (0, 0), 2.0), 0, 255).astype(np.uint8)  # small = fast

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
