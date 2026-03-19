#!/usr/bin/env python3
"""
4K Video Upscaler using Real-ESRGAN - Direct Pipe Mode
Streams video frames directly through AI without intermediate files

FASTER than upscale_video.py but NO RESUME CAPABILITY.

Usage:
    python upscale_video_pipe.py -i input.mp4 -o output.mp4 -m realesr-animevideov3

Based on: https://github.com/xinntao/Real-ESRGAN
"""

# Fix for torchvision 0.17+ compatibility with basicsr
import sys
import types
import torchvision.transforms.functional as F
_compat_module = types.ModuleType('torchvision.transforms.functional_tensor')
_compat_module.rgb_to_grayscale = F.rgb_to_grayscale
sys.modules['torchvision.transforms.functional_tensor'] = _compat_module

import argparse
import cv2
import os
import subprocess
import shutil
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()


def get_optimal_tile_size():
    """Auto-detect optimal tile size based on VRAM."""
    try:
        import torch
        if not torch.cuda.is_available():
            return 0

        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        gpu_name = torch.cuda.get_device_name(0)

        if vram_gb >= 32:
            tile_size = 800
        elif vram_gb >= 24:
            tile_size = 512
        elif vram_gb >= 16:
            tile_size = 400
        elif vram_gb >= 12:
            tile_size = 320
        elif vram_gb >= 8:
            tile_size = 256
        else:
            tile_size = 128

        print(f"GPU: {gpu_name} ({vram_gb:.1f}GB VRAM) → Tile size: {tile_size}")
        return tile_size
    except Exception:
        return 0


def check_gpu():
    """Check if CUDA GPU is available."""
    try:
        import torch
        if not torch.cuda.is_available():
            print("ERROR: GPU not detected. Pipe mode requires CUDA GPU.")
            sys.exit(1)
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        return True
    except ImportError:
        print("ERROR: PyTorch not installed.")
        sys.exit(1)


def get_ffmpeg_path():
    """Get ffmpeg executable path."""
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path

    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass

    print("ERROR: ffmpeg not found.")
    print("  Ubuntu: apt update && apt install -y ffmpeg")
    print("  pip: pip install imageio-ffmpeg")
    sys.exit(1)


def get_video_info(video_path):
    """Get video metadata."""
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return width, height, fps, frame_count


def setup_model(model_name, tile_size=0):
    """Load Real-ESRGAN model."""
    import torch
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from basicsr.utils.download_util import load_file_from_url
    from realesrgan import RealESRGANer
    from realesrgan.archs.srvgg_arch import SRVGGNetCompact

    model_configs = {
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
        },
        "realesr-animevideov3": {
            "model_class": SRVGGNetCompact,
            "model_args": {"num_in_ch": 3, "num_out_ch": 3, "num_feat": 64, "num_conv": 16, "upscale": 4, "act_type": "prelu"},
            "netscale": 4,
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth"
        }
    }

    config = model_configs.get(model_name, model_configs["RealESRGAN_x4plus"])

    model_dir = SCRIPT_DIR / "models"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / f"{model_name}.pth"

    if not model_path.exists():
        print(f"Downloading model {model_name}...")
        model_path = load_file_from_url(url=config["url"], model_dir=str(model_dir), progress=True, file_name=f"{model_name}.pth")

    model_instance = config["model_class"](**config["model_args"])
    upsampler = RealESRGANer(
        scale=config["netscale"],
        model_path=str(model_path),
        model=model_instance,
        tile=tile_size,
        tile_pad=10,
        pre_pad=0,
        half=True  # Always use FP16 for GPU
    )

    return upsampler, config["netscale"]


def upscale_video_pipe(input_path, output_path, model_name, tile_size=0, scale=4, use_nvenc=True):
    """
    Stream video through Real-ESRGAN without intermediate files.

    PIPELINE:
    FFmpeg decode → numpy array → Real-ESRGAN → numpy array → FFmpeg encode
    """
    from tqdm import tqdm
    import torch

    FFMPEG_PATH = get_ffmpeg_path()

    # Get video info
    width, height, fps, frame_count = get_video_info(input_path)
    print(f"Input: {width}x{height} @ {fps:.2f} fps, {frame_count} frames")

    # Calculate output dimensions (ensure even numbers for video encoding)
    out_width = width * scale
    out_height = height * scale
    if out_width % 2 != 0:
        out_width -= 1
    if out_height % 2 != 0:
        out_height -= 1

    print(f"Output: {out_width}x{out_height} ({scale}x upscale)")

    # Auto-detect tile size if not specified
    if tile_size == 0:
        tile_size = get_optimal_tile_size()

    # Load model
    print(f"Loading model: {model_name}")
    upsampler, _ = setup_model(model_name, tile_size)

    # Start FFmpeg decoder (input)
    # Decode video as raw BGR24 frames
    decode_cmd = [
        FFMPEG_PATH, "-i", input_path,
        "-f", "image2pipe",
        "-pix_fmt", "bgr24",
        "-vcodec", "rawvideo",
        "-"
    ]
    decoder = subprocess.Popen(decode_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8)

    # Start FFmpeg encoder (output)
    # Try NVENC first, fallback to libx264
    if use_nvenc and shutil.which("nvidia-smi"):
        encode_cmd = [
            FFMPEG_PATH, "-y",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{out_width}x{out_height}",
            "-r", str(fps),
            "-i", "-",
            "-c:v", "h264_nvenc",
            "-preset", "p4",
            "-rc:v", "vbr_hq",
            "-cq:v", "20",
            "-b:v", "0",
            "-pix_fmt", "yuv420p",
            output_path
        ]
        encoder = subprocess.Popen(encode_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        using_nvenc = True
    else:
        encode_cmd = [
            FFMPEG_PATH, "-y",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{out_width}x{out_height}",
            "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            output_path
        ]
        encoder = subprocess.Popen(encode_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        using_nvenc = False

    print(f"Encoding: {'NVENC (GPU)' if using_nvenc else 'libx264 (CPU)'}")
    print(f"\nProcessing {frame_count} frames...")

    # Processing loop
    frame_size = width * height * 3
    processed = 0

    try:
        with tqdm(total=frame_count, desc="Upscaling", unit="frames") as pbar:
            while True:
                # Read raw frame from decoder
                raw_frame = decoder.stdout.read(frame_size)
                if len(raw_frame) != frame_size:
                    break

                # Convert to numpy array
                frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((height, width, 3))

                # Upscale
                try:
                    output_frame, _ = upsampler.enhance(frame, outscale=scale)

                    # Crop to exact output size if needed
                    if output_frame.shape[1] != out_width or output_frame.shape[0] != out_height:
                        output_frame = output_frame[:out_height, :out_width]

                    # Write to encoder
                    encoder.stdin.write(output_frame.tobytes())

                except Exception as e:
                    print(f"\nError on frame {processed}: {e}")
                    # Write black frame to keep sync
                    black_frame = np.zeros((out_height, out_width, 3), dtype=np.uint8)
                    encoder.stdin.write(black_frame.tobytes())

                processed += 1
                pbar.update(1)

    except KeyboardInterrupt:
        print("\n\nInterrupted! Partial video saved.")
    finally:
        # Cleanup
        decoder.terminate()
        encoder.stdin.close()
        encoder.wait()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n{'='*50}")
    print(f"DONE! Processed {processed} frames")
    print(f"Output: {output_path}")
    print(f"{'='*50}")

    # Now add audio from original video
    add_audio(input_path, output_path)


def add_audio(original_video, video_without_audio):
    """Add audio from original video to upscaled video."""
    FFMPEG_PATH = get_ffmpeg_path()

    temp_output = video_without_audio.replace(".mp4", "_temp.mp4")

    cmd = [
        FFMPEG_PATH, "-y",
        "-i", video_without_audio,
        "-i", original_video,
        "-map", "0:v",
        "-map", "1:a?",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        temp_output
    ]

    result = subprocess.run(cmd, capture_output=True)

    if result.returncode == 0 and os.path.exists(temp_output):
        os.remove(video_without_audio)
        os.rename(temp_output, video_without_audio)
        print("Audio track added from original video.")
    else:
        print("No audio track found or failed to add audio.")


def main():
    parser = argparse.ArgumentParser(
        description="4K Video Upscaler - Pipe Mode (No intermediate files)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python upscale_video_pipe.py -i input.mp4 -o output.mp4

  # Use faster anime model
  python upscale_video_pipe.py -i video.mp4 -o upscaled.mp4 -m realesr-animevideov3

  # Specify tile size manually (for low VRAM)
  python upscale_video_pipe.py -i video.mp4 -o output.mp4 -t 256

NOTE: This mode is FASTER but has NO RESUME capability.
      For long videos, use upscale_video.py instead.
"""
    )

    parser.add_argument("-i", "--input", required=True, help="Input video file")
    parser.add_argument("-o", "--output", default=None, help="Output video file (default: output/<name>_<width>x<height>.mp4)")
    parser.add_argument("-m", "--model", default="realesr-animevideov3",
                        choices=["RealESRGAN_x4plus", "RealESRGAN_x4plus_anime_6B", "realesr-animevideov3"],
                        help="Model to use (default: realesr-animevideov3 - fastest)")
    parser.add_argument("-t", "--tile", type=int, default=0,
                        help="Tile size (0=auto, 256 for low VRAM, 512 for 24GB, 800 for 32GB)")
    parser.add_argument("-s", "--scale", type=int, default=2, choices=[2, 3, 4],
                        help="Scale factor (default: 2)")
    parser.add_argument("--no-nvenc", action="store_true", help="Disable NVENC, use CPU encoding")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    # Generate default output path if not specified
    if args.output is None:
        input_name = Path(args.input).stem
        width, height, _, _ = get_video_info(args.input)
        out_width = width * args.scale
        out_height = height * args.scale
        os.makedirs("output", exist_ok=True)
        args.output = f"output/{input_name}_{out_width}x{out_height}.mp4"

    check_gpu()

    upscale_video_pipe(
        input_path=args.input,
        output_path=args.output,
        model_name=args.model,
        tile_size=args.tile,
        scale=args.scale,
        use_nvenc=not args.no_nvenc
    )


if __name__ == "__main__":
    main()
