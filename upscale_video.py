#!/usr/bin/env python3
"""
4K Video Upscaler using Real-ESRGAN
Standalone script for cloud GPU environments

Based on: https://github.com/yuvraj108c/4k-video-upscaler-colab
Original Real-ESRGAN: https://github.com/xinntao/Real-ESRGAN
"""

# Fix for torchvision 0.17+ compatibility with basicsr
# basicsr tries to import from functional_tensor which was renamed
import torchvision.transforms
import torchvision.transforms.functional
torchvision.transforms.functional_tensor = torchvision.transforms.functional

import argparse
import cv2
import os
import subprocess
import sys
import shutil
from pathlib import Path

# Get script directory for relative paths
SCRIPT_DIR = Path(__file__).parent.resolve()


def check_gpu():
    """Check if CUDA GPU is available."""
    try:
        import torch
        if not torch.cuda.is_available():
            print("ERROR: GPU not detected. This script requires a CUDA-capable GPU.")
            sys.exit(1)
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        return True
    except ImportError:
        print("ERROR: PyTorch not installed. Run with --install-deps first.")
        sys.exit(1)


def setup_environment(install_deps=False):
    """Setup the environment and install dependencies if needed."""
    if install_deps:
        print("Installing dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
            "torch", "torchvision",
            "--extra-index-url", "https://download.pytorch.org/whl/cu121"], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
            "basicsr", "facexlib", "gfpgan", "ffmpeg-python", "tqdm"], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "numpy<2"], check=True)

    realesrgan_dir = SCRIPT_DIR / "Real-ESRGAN"
    if not realesrgan_dir.exists():
        print("Cloning Real-ESRGAN repository...")
        subprocess.run(["git", "clone", "https://github.com/xinntao/Real-ESRGAN.git", str(realesrgan_dir)], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(realesrgan_dir / "requirements.txt")], check=True)
        # setup.py must be run from inside Real-ESRGAN directory
        original_cwd = os.getcwd()
        os.chdir(str(realesrgan_dir))
        try:
            subprocess.run([sys.executable, "setup.py", "develop"], check=True)
        finally:
            os.chdir(original_cwd)

    realesrgan_path = str(realesrgan_dir.resolve())
    if realesrgan_path not in sys.path:
        sys.path.insert(0, realesrgan_path)

    return realesrgan_dir


def get_resolution_params(resolution: str, video_width: int, video_height: int):
    """Calculate output resolution and scale factor."""
    aspect_ratio = video_width / video_height

    resolution_map = {
        "FHD": (1920, 1080), "2k": (2560, 1440), "4k": (3840, 2160),
        "2x": (2 * video_width, 2 * video_height),
        "3x": (3 * video_width, 3 * video_height),
        "4x": (4 * video_width, 4 * video_height),
    }

    final_width, final_height = resolution_map.get(resolution, (2 * video_width, 2 * video_height))

    if aspect_ratio == 1.0 and "x" not in resolution:
        final_height = final_width
    if aspect_ratio < 1.0 and "x" not in resolution:
        final_width, final_height = final_height, final_width

    scale_factor = max(final_width / video_width, final_height / video_height)

    # Ensure even dimensions
    while int(video_width * scale_factor) % 2 != 0 or int(video_height * scale_factor) % 2 != 0:
        scale_factor += 0.01

    return final_width, final_height, scale_factor


def get_ffmpeg_path():
    """Get ffmpeg executable path."""
    # First check if ffmpeg is in PATH
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path

    # Try imageio-ffmpeg
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass

    print("ERROR: ffmpeg not found.")
    print("\nInstall ffmpeg:")
    print("  pip install imageio-ffmpeg")
    print("  Or download from: https://www.gyan.dev/ffmpeg/builds/")
    sys.exit(1)


FFMPEG_PATH = None


def extract_frames(video_path: str, frames_dir: str):
    """Extract frames from video using ffmpeg."""
    global FFMPEG_PATH
    if FFMPEG_PATH is None:
        FFMPEG_PATH = get_ffmpeg_path()

    os.makedirs(frames_dir, exist_ok=True)
    print(f"Extracting frames from {video_path}...")
    cmd = [FFMPEG_PATH, "-y", "-i", video_path, "-qscale:v", "1", "-qmin", "1", "-qmax", "1",
           "-vsync", "0", os.path.join(frames_dir, "frame_%08d.png")]
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"Frames extracted to {frames_dir}")


def get_video_info(video_path: str):
    """Get video dimensions and fps."""
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return width, height, fps, frame_count


def upscale_frames(frames_dir: str, output_dir: str, model: str, scale_factor: float, tile_size: int = 0, use_cpu: bool = False, batch_size: int = 1):
    """Upscale frames using Real-ESRGAN."""
    import torch
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from basicsr.utils.download_util import load_file_from_url
    from realesrgan import RealESRGANer
    from realesrgan.archs.srvgg_arch import SRVGGNetCompact
    from tqdm import tqdm

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

    config = model_configs.get(model, model_configs["RealESRGAN_x4plus"])

    model_dir = SCRIPT_DIR / "models"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / f"{model}.pth"

    if not model_path.exists():
        print(f"Downloading model {model}...")
        model_path = load_file_from_url(url=config["url"], model_dir=str(model_dir), progress=True, file_name=f"{model}.pth")

    model_instance = config["model_class"](**config["model_args"])
    # Use FP16 half precision only when GPU is available
    use_half = not use_cpu and torch.cuda.is_available()
    upsampler = RealESRGANer(scale=config["netscale"], model_path=str(model_path), model=model_instance,
                              tile=tile_size, tile_pad=10, pre_pad=0, half=use_half)

    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    os.makedirs(output_dir, exist_ok=True)
    print(f"Upscaling {len(frame_files)} frames with scale factor {scale_factor}...")
    print(f"Using model: {model} | Half precision: {use_half} | Batch size: {batch_size}")

    if batch_size > 1:
        # Batch processing for better GPU utilization
        for i in tqdm(range(0, len(frame_files), batch_size), desc="Upscaling batches"):
            batch_files = frame_files[i:i + batch_size]
            batch_imgs = []
            for f in batch_files:
                img = cv2.imread(os.path.join(frames_dir, f), cv2.IMREAD_UNCHANGED)
                batch_imgs.append(img)

            # Process batch
            for f, img in zip(batch_files, batch_imgs):
                output, _ = upsampler.enhance(img, outscale=scale_factor)
                cv2.imwrite(os.path.join(output_dir, f), output)
    else:
        # Single frame processing
        for frame_file in tqdm(frame_files, desc="Upscaling"):
            img = cv2.imread(os.path.join(frames_dir, frame_file), cv2.IMREAD_UNCHANGED)
            output, _ = upsampler.enhance(img, outscale=scale_factor)
            cv2.imwrite(os.path.join(output_dir, frame_file), output)

    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Upscaled frames saved to {output_dir}")


def create_video(frames_dir: str, output_path: str, fps: float, use_nvenc: bool = True):
    """Create video from frames using ffmpeg."""
    global FFMPEG_PATH
    if FFMPEG_PATH is None:
        FFMPEG_PATH = get_ffmpeg_path()

    print("Creating video from frames...")
    if use_nvenc and shutil.which("nvidia-smi"):
        cmd = [FFMPEG_PATH, "-y", "-framerate", str(fps), "-i", os.path.join(frames_dir, "frame_%08d.png"),
               "-c:v", "h264_nvenc", "-preset", "p4", "-rc:v", "vbr_hq", "-cq:v", "19", "-b:v", "0", "-pix_fmt", "yuv420p", output_path]
    else:
        cmd = [FFMPEG_PATH, "-y", "-framerate", str(fps), "-i", os.path.join(frames_dir, "frame_%08d.png"),
               "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", output_path]
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"Video created: {output_path}")


def crop_video(input_path: str, output_path: str, width: int, height: int, use_nvenc: bool = True):
    """Crop video to exact target resolution."""
    global FFMPEG_PATH
    if FFMPEG_PATH is None:
        FFMPEG_PATH = get_ffmpeg_path()

    print(f"Cropping video to {width}x{height}...")
    if use_nvenc and shutil.which("nvidia-smi"):
        cmd = [FFMPEG_PATH, "-y", "-hwaccel", "cuda", "-i", input_path,
               "-vf", f"crop={width}:{height}:(in_w-{width})/2:(in_h-{height})/2",
               "-c:v", "h264_nvenc", "-preset", "p4", "-pix_fmt", "yuv420p", output_path]
    else:
        cmd = [FFMPEG_PATH, "-y", "-i", input_path,
               "-vf", f"crop={width}:{height}:(in_w-{width})/2:(in_h-{height})/2",
               "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", output_path]
    subprocess.run(cmd, check=True, capture_output=True)


def add_audio(original_video: str, upscaled_video: str, output_video: str):
    """Add audio from original video to upscaled video."""
    global FFMPEG_PATH
    if FFMPEG_PATH is None:
        FFMPEG_PATH = get_ffmpeg_path()

    print("Adding audio track...")
    cmd = [FFMPEG_PATH, "-y", "-i", upscaled_video, "-i", original_video,
           "-map", "0:v", "-map", "1:a?", "-c:v", "copy", "-c:a", "aac", "-shortest", output_video]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode == 0 and os.path.exists(output_video):
        print(f"Final video with audio: {output_video}")
        return True
    # If no audio or failed, just copy the video
    shutil.copy(upscaled_video, output_video)
    print(f"No audio found. Video saved: {output_video}")
    return False


def upscale_video(video_path: str, output_dir: str, resolution: str = "4k", model: str = "RealESRGAN_x4plus",
                  tile_size: int = 0, keep_frames: bool = False, skip_setup: bool = False, use_cpu: bool = False,
                  batch_size: int = 1, skip_extract: bool = False, frames_dir: str = None):
    """Main video upscaling function."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    os.makedirs(output_dir, exist_ok=True)

    if not skip_setup:
        setup_environment(install_deps=False)

    video_width, video_height, fps, frame_count = get_video_info(video_path)
    print(f"Input: {video_width}x{video_height} @ {fps:.2f} fps, {frame_count} frames")

    final_width, final_height, scale_factor = get_resolution_params(resolution, video_width, video_height)
    print(f"Output: {final_width}x{final_height} (scale: {scale_factor:.2f})")

    video_name = Path(video_path).stem
    temp_frames = frames_dir if frames_dir else os.path.join(output_dir, f"{video_name}_frames")
    temp_upscaled = os.path.join(output_dir, f"{video_name}_upscaled_frames")
    temp_video = os.path.join(output_dir, f"{video_name}_temp.mp4")
    final_video = os.path.join(output_dir, f"{video_name}_{final_width}x{final_height}.mp4")

    try:
        if not skip_extract:
            extract_frames(video_path, temp_frames)
        else:
            print(f"Skipping frame extraction, using existing frames from: {temp_frames}")

        upscale_frames(temp_frames, temp_upscaled, model, scale_factor, tile_size, use_cpu, batch_size)
        create_video(temp_upscaled, temp_video, fps)

        if "x" not in resolution:
            crop_video(temp_video, final_video, final_width, final_height)
            os.remove(temp_video)
        else:
            shutil.move(temp_video, final_video)

        # Add audio from original video
        final_with_audio = os.path.join(output_dir, f"{video_name}_{final_width}x{final_height}_audio.mp4")
        if add_audio(video_path, final_video, final_with_audio):
            os.remove(final_video)
            shutil.move(final_with_audio, final_video)

        print(f"\n{'='*50}\nSUCCESS! Output: {final_video}\n{'='*50}")
    finally:
        if not keep_frames and not frames_dir:
            print("Cleaning up temporary files...")
            for d in [temp_frames, temp_upscaled]:
                if os.path.exists(d):
                    shutil.rmtree(d)


def main():
    parser = argparse.ArgumentParser(description="4K Video Upscaler using Real-ESRGAN")
    parser.add_argument("-i", "--input", type=str, help="Input video file")
    parser.add_argument("-o", "--output", type=str, default="output", help="Output directory")
    parser.add_argument("-r", "--resolution", type=str, default="4k", choices=["FHD", "2k", "4k", "2x", "3x", "4x"])
    parser.add_argument("-m", "--model", type=str, default="RealESRGAN_x4plus",
                        choices=["RealESRGAN_x4plus", "RealESRGAN_x4plus_anime_6B", "realesr-animevideov3"])
    parser.add_argument("-t", "--tile", type=int, default=0, help="Tile size (0=auto, 512 for low VRAM)")
    parser.add_argument("-b", "--batch-size", type=int, default=1, help="Batch size for processing (1=sequential)")
    parser.add_argument("--keep-frames", action="store_true", help="Keep temporary frames")
    parser.add_argument("--skip-extract", action="store_true", help="Skip frame extraction (use existing frames)")
    parser.add_argument("--frames-dir", type=str, help="Use existing frames from this directory")
    parser.add_argument("--install-deps", action="store_true", help="Install dependencies")
    parser.add_argument("--skip-setup", action="store_true", help="Skip environment setup")
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU (slower)")

    args = parser.parse_args()

    if args.install_deps:
        setup_environment(install_deps=True)
        print("Dependencies installed!")
        return

    if not args.input:
        parser.print_help()
        print("\nError: --input is required")
        sys.exit(1)

    if not args.cpu:
        check_gpu()

    upscale_video(args.input, args.output, args.resolution, args.model, args.tile, args.keep_frames,
                  args.skip_setup, args.cpu, args.batch_size, args.skip_extract, args.frames_dir)


if __name__ == "__main__":
    main()
