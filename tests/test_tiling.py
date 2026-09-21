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
