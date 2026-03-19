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

def benchmark_model(model_name, model_config, frame, tile_size=0):
    import torch
    from realesrgan import RealESRGANer
    from basicsr.utils.download_util import load_file_from_url
    from pathlib import Path

    model_path = Path("models") / f"{model_name}.pth"
    if not model_path.exists():
        print(f"Downloading {model_name}...")
        model_path.parent.mkdir(exist_ok=True)
        load_file_from_url(url=model_config["url"], model_dir=str(model_path.parent), file_name=f"{model_name}.pth")

    model = model_config["model_class"](**model_config["model_args"])

    upsampler = RealESRGANer(
        scale=model_config["netscale"],
        model_path=str(model_path),
        model=model,
        tile=tile_size,
        tile_pad=10,
        pre_pad=0,
        half=True
    )

    # Warmup
    _ = upsampler.enhance(frame, outscale=2)
    torch.cuda.synchronize()

    # Benchmark
    times = []
    for i in range(5):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _, _ = upsampler.enhance(frame, outscale=2)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    avg_time = sum(times) / len(times)
    vram = torch.cuda.max_memory_allocated() / 1024**3

    del upsampler
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    return avg_time, vram


def main():
    import torch
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from basicsr.archs.srvgg_arch import SRVGGNetCompact

    print("="*60)
    print("Real-ESRGAN Benchmark (No I/O overhead)")
    print("="*60)

    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"PyTorch: {torch.__version__} | CUDA: {torch.version.cuda}")

    # Test frame (1080p)
    frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

    # Models to test
    models = {
        "realesr-animevideov3": {
            "model_class": SRVGGNetCompact,
            "model_args": {"num_in_ch": 3, "num_out_ch": 3, "num_feat": 64, "num_conv": 16, "upscale": 4, "act_type": "prelu"},
            "netscale": 4,
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth"
        },
        "RealESRGAN_x4plus": {
            "model_class": RRDBNet,
            "model_args": {"num_in_ch": 3, "num_out_ch": 3, "num_feat": 64, "num_block": 23, "num_grow_ch": 32, "scale": 4},
            "netscale": 4,
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
        },
        "RealESRGAN_x4plus_anime_6B": {
            "model_class": RRDBNet,
            "model_args": {"num_in_ch": 3, "num_out_ch": 3, "num_feat": 64, "num_block": 6, "num_grow_ch": 32, "scale": 4},
            "netscale": 4,
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
        }
    }

    print("\n" + "-"*60)
    print(f"{'Model':<30} {'Time':>10} {'FPS':>8} {'VRAM':>8}")
    print("-"*60)

    results = []
    for model_name, config in models.items():
        try:
            avg_time, vram = benchmark_model(model_name, config, frame, tile_size=0)
            fps = 1 / avg_time
            print(f"{model_name:<30} {avg_time*1000:>8.0f}ms {fps:>7.1f} {vram:>6.2f}GB")
            results.append((model_name, avg_time, fps, vram))
        except Exception as e:
            print(f"{model_name:<30} ERROR: {e}")

    print("-"*60)

    # Time estimate for 16,449 frames
    print("\nEstimated time for 16,449 frames (1080p → 4K @ 2x):")
    for name, time, fps, vram in results:
        est_hours = (16449 * time) / 3600
        print(f"  {name}: {est_hours:.1f} hours")

    print("\n" + "="*60)
    print("RECOMMENDATION: Use realesr-animevideov3 for fastest speed")
    print("  python upscale_video_pipe.py -i video.mp4 -m realesr-animevideov3")
    print("="*60)

if __name__ == "__main__":
    main()
