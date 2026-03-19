#!/usr/bin/env python3
"""Quick benchmark to test raw upscale speed without I/O overhead."""

import sys
import time
import cv2
import numpy as np

# Fix for torchvision compatibility
import types
import torchvision.transforms.functional as F
_compat_module = types.ModuleType('torchvision.transforms.functional_tensor')
_compat_module.rgb_to_grayscale = F.rgb_to_grayscale
sys.modules['torchvision.transforms.functional_tensor'] = _compat_module

def main():
    import torch
    from basicsr.archs.srvgg_arch import SRVGGNetCompact
    from realesrgan import RealESRGANer
    from basicsr.utils.download_util import load_file_from_url
    from pathlib import Path

    print("="*50)
    print("Real-ESRGAN Benchmark (No I/O overhead)")
    print("="*50)

    # GPU info
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")

    # Download model
    model_path = Path("models/realesr-animevideov3.pth")
    if not model_path.exists():
        print("\nDownloading model...")
        model_path.parent.mkdir(exist_ok=True)
        load_file_from_url(
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
            model_dir=str(model_path.parent),
            file_name=model_path.name
        )

    # Load model
    print("\nLoading model...")
    model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')

    # Test with different tile sizes
    for tile in [0, 800, 512]:
        print(f"\n--- Tile size: {tile} {'(no tiling)' if tile == 0 else ''} ---")

        upsampler = RealESRGANer(
            scale=4,
            model_path=str(model_path),
            model=model,
            tile=tile,
            tile_pad=10,
            pre_pad=0,
            half=True
        )

        # Create test frame (1080p)
        frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

        # Warmup
        _ = upsampler.enhance(frame, outscale=2)
        torch.cuda.synchronize()

        # Benchmark
        times = []
        for i in range(10):
            torch.cuda.synchronize()
            start = time.perf_counter()
            output, _ = upsampler.enhance(frame, outscale=2)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            print(f"  Frame {i+1}: {elapsed*1000:.1f}ms ({1/elapsed:.1f} fps)")

        avg_time = sum(times) / len(times)
        print(f"  Average: {avg_time*1000:.1f}ms ({1/avg_time:.1f} fps)")

        # Memory check
        print(f"  VRAM used: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
        torch.cuda.reset_peak_memory_stats()

        # Cleanup
        del upsampler
        torch.cuda.empty_cache()

    print("\n" + "="*50)
    print("Benchmark complete!")
    print("="*50)

if __name__ == "__main__":
    main()
