#!/usr/bin/env python3
"""
Optimized Video Upsscaler - Direct Tensor Pipeline
Reliable sequential I/O with GPU encoding

BENCHMARK:
  RealESRGANer.enhance(): 730ms/frame (1.4 fps)
  Direct tensor call: 60ms/frame (16.8 fps)
"""

import sys
import time
import argparse
import os
import subprocess
import shutil
from pathlib import Path

# Fix for torchvision compatibility
import types
import torchvision.transforms.functional as F
_compat = types.ModuleType('torchvision.transforms.functional_tensor')
_compat.rgb_to_grayscale = F.rgb_to_grayscale
sys.modules['torchvision.transforms.functional_tensor'] = _compat

import cv2
import numpy as np
import torch
from tqdm import tqdm
import threading
import queue


SCRIPT_DIR = Path(__file__).parent.resolve()


def get_ffmpeg_path():
    """Get ffmpeg executable path."""
    # Project-local ffmpeg first (full build with NVENC support)
    local_ffmpeg = SCRIPT_DIR / "ffmpeg" / "bin" / "ffmpeg.exe"
    if local_ffmpeg.exists():
        return str(local_ffmpeg)

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path

    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass

    print("ERROR: ffmpeg not found.")
    print("  Install with: pip install imageio-ffmpeg")
    print("  Or add ffmpeg to PATH")
    sys.exit(1)


def load_fast_model(model_name="realesr-animevideov3"):
    """Load model and return model directly (not wrapped in RealESRGANer)."""
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from basicsr.archs.srvgg_arch import SRVGGNetCompact
    from basicsr.utils.download_util import load_file_from_url

    models = {
        "realesr-animevideov3": {
            "class": SRVGGNetCompact,
            "args": {"num_in_ch": 3, "num_out_ch": 3, "num_feat": 64, "num_conv": 16, "upscale": 4, "act_type": "prelu"},
            "scale": 4,
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth"
        },
        "RealESRGAN_x4plus": {
            "class": RRDBNet,
            "args": {"num_in_ch": 3, "num_out_ch": 3, "num_feat": 64, "num_block": 23, "num_grow_ch": 32, "scale": 4},
            "scale": 4,
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
        },
        "RealESRGAN_x4plus_anime_6B": {
            "class": RRDBNet,
            "args": {"num_in_ch": 3, "num_out_ch": 3, "num_feat": 64, "num_block": 6, "num_grow_ch": 32, "scale": 4},
            "scale": 4,
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
        }
    }

    config = models.get(model_name, models["realesr-animevideov3"])

    model_dir = SCRIPT_DIR / "models"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / f"{model_name}.pth"

    if not model_path.exists():
        print(f"Downloading {model_name}...")
        load_file_from_url(url=config["url"], model_dir=str(model_dir), file_name=f"{model_name}.pth")

    model = config["class"](**config["args"])

    # Load weights - handle different formats
    checkpoint = torch.load(model_path, map_location='cuda', weights_only=False)
    if 'params_ema' in checkpoint:
        state_dict = checkpoint['params_ema']
    elif 'params' in checkpoint:
        state_dict = checkpoint['params']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model = model.cuda().half().eval()

    return model, config["scale"]


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


def upscale_frame_direct(model, frame_bgr, scale=2, model_scale=4, tile=0):
    """
    Upscale frame using direct tensor operations.
    Avoids RealESRGANer overhead.

    Args:
        model: The loaded model
        frame_bgr: numpy array (H, W, 3) in BGR format
        scale: Output scale (2 for 2x)
        model_scale: Model's native scale (4 for these models)
    """
    with torch.no_grad():
        # BGR -> RGB, HWC -> CHW, normalize to [0,1]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_chw = frame_rgb.transpose(2, 0, 1) / 255.0

        # To tensor on GPU
        tensor = torch.from_numpy(frame_chw).float().cuda().half()
        tensor = tensor.unsqueeze(0)  # Add batch dim

        # Upscale (tiled when requested: bounds VRAM for RRDBNet models)
        if tile and (tensor.shape[2] > tile or tensor.shape[3] > tile):
            output = _upscale_tiled(model, tensor, model_scale, tile=tile)
        else:
            output = model(tensor)

        # Downscale on GPU when target scale differs from model scale: avoids the 33MP
        # device->host copy and CPU Lanczos pass (~2x faster end-to-end)
        if scale != model_scale:
            h, w = frame_bgr.shape[:2]
            output = torch.nn.functional.interpolate(
                output, size=(h * scale, w * scale),
                mode="bicubic", align_corners=False, antialias=True)

        # Remove batch dim, CHW -> HWC, denormalize
        output = output.squeeze(0).float()
        output = (output * 255).clamp(0, 255).byte()
        output_np = output.permute(1, 2, 0).cpu().numpy()

        # RGB -> BGR
        output_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)

        return output_bgr


def upscale_video_fast(input_path, output_path, model_name="realesr-animevideov3", scale=2, use_nvenc=True, tile=None):
    """Upscale video using direct tensor pipeline with reliable sequential I/O."""

    # Load model
    print(f"Loading model: {model_name}")
    model, model_scale = load_fast_model(model_name)

    # RRDBNet models need ~8.3KB/input-pixel VRAM: 1080p direct spills to
    # system memory (WDDM) and runs ~6x slower. Tile them by default.
    if tile is None:
        tile = 960 if model_name in ("RealESRGAN_x4plus", "RealESRGAN_x4plus_anime_6B") else 0
    print(f"Tiling: {tile}px" if tile else "Tiling: off")

    # Get video info
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    out_width = width * scale
    out_height = height * scale
    frame_size = width * height * 3

    print(f"Input: {width}x{height} @ {fps:.2f} fps, {frame_count} frames")
    print(f"Output: {out_width}x{out_height} ({scale}x upscale)")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # FFmpeg decoder
    ffmpeg_path = get_ffmpeg_path()
    decode_cmd = [
        ffmpeg_path, "-i", input_path,
        "-f", "image2pipe", "-pix_fmt", "bgr24",
        "-vcodec", "rawvideo", "-"
    ]
    decoder = subprocess.Popen(decode_cmd, stdout=subprocess.PIPE, stderr=open(output_path + ".decoder.log", "w"), bufsize=10**8)

    # FFmpeg encoder - use faster preset for reliability
    if use_nvenc:
        encode_cmd = [
            ffmpeg_path, "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{out_width}x{out_height}", "-r", str(fps),
            "-i", "-", "-c:v", "h264_nvenc", "-preset", "fast",  # p1-p7 presets need newer ffmpeg; 4.2 uses fast
            "-rc", "vbr", "-cq", "23", "-b:v", "8M", "-pix_fmt", "yuv420p",
            output_path
        ]
        print("Encoding: NVENC (GPU) - fast preset")
    else:
        encode_cmd = [
            ffmpeg_path, "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{out_width}x{out_height}", "-r", str(fps),
            "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p", output_path
        ]
        print("Encoding: libx264 (CPU)")

    # stderr -> file: an undrained PIPE fills its OS buffer and deadlocks the encoder
    encoder = subprocess.Popen(encode_cmd, stdin=subprocess.PIPE, stderr=open(output_path + ".encoder.log", "w"))

    # Pre-buffer input frames using a thread (only reader thread, no writer thread)
    input_queue = queue.Queue(maxsize=16)  # Buffer 16 frames
    reader_error = [None]

    def reader_thread():
        """Read frames from decoder and put in queue."""
        try:
            while True:
                raw_frame = decoder.stdout.read(frame_size)
                if len(raw_frame) != frame_size:
                    input_queue.put(None)  # Signal end
                    break
                frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((height, width, 3)).copy()
                input_queue.put(frame)
        except Exception as e:
            reader_error[0] = e
            input_queue.put(None)

    # Start reader thread
    reader = threading.Thread(target=reader_thread, daemon=True)
    reader.start()

    # Main processing loop - write directly to encoder (sequential, reliable)
    processed = 0
    encoder_died = False

    print(f"\nProcessing {frame_count} frames...")
    print("Mode: Reliable sequential I/O (decode || GPU+encode)")
    start_time = time.time()

    try:
        with tqdm(total=frame_count, desc="Upscaling", unit="frames") as pbar:
            while True:
                frame = input_queue.get()
                if frame is None:  # End of input
                    break

                # Check if encoder is still alive
                if encoder.poll() is not None:
                    print(f"\nEncoder died unexpectedly (exit code: {encoder.returncode})")
                    encoder_died = True
                    break

                try:
                    # Upscale
                    output_frame = upscale_frame_direct(model, frame, scale, model_scale, tile=tile)

                    # Write directly to encoder (may block if encoder is slow - this is OK)
                    encoder.stdin.write(output_frame.tobytes())

                except BrokenPipeError:
                    print(f"\nEncoder pipe broken at frame {processed}")
                    encoder_died = True
                    break
                except Exception as e:
                    print(f"\nError on frame {processed}: {e}")
                    # Write black frame to keep sync
                    black = np.zeros((out_height, out_width, 3), dtype=np.uint8)
                    try:
                        encoder.stdin.write(black.tobytes())
                    except:
                        pass

                processed += 1
                pbar.update(1)

    except KeyboardInterrupt:
        print("\n\nInterrupted!")
    finally:
        # Cleanup
        reader.join(timeout=2)
        decoder.terminate()

        try:
            encoder.stdin.close()
        except:
            pass

        # Wait for encoder to finish (can take a while to flush large 4K files)
        try:
            encoder.wait(timeout=300)  # 5 minutes timeout
        except subprocess.TimeoutExpired:
            print("\nEncoder taking long to finalize, terminating...")
            encoder.terminate()
            encoder.wait(timeout=10)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if reader_error[0]:
            print(f"\nReader error: {reader_error[0]}")

    if encoder_died:
        # Get encoder error output from its log file
        try:
            stderr = open(output_path + ".encoder.log").read()
        except OSError:
            stderr = ""
        print(f"\nEncoder stderr:\n{stderr[-1000:]}")  # Last 1000 chars

    elapsed = time.time() - start_time
    avg_fps = processed / elapsed if elapsed > 0 else 0

    # Check if output file exists and has content
    output_exists = os.path.exists(output_path)
    output_size = os.path.getsize(output_path) if output_exists else 0

    print(f"\n{'='*50}")
    print(f"DONE! {processed} frames in {elapsed/60:.1f} min")
    print(f"Average speed: {avg_fps:.1f} fps")
    print(f"Output: {output_path}")
    if output_exists:
        print(f"File size: {output_size / (1024*1024):.1f} MB")
    print(f"{'='*50}")

    # Add audio only if encoding succeeded and file exists
    if output_exists and output_size > 0 and processed > 0:
        add_audio(input_path, output_path)
    else:
        print("Skipping audio - output file missing or empty")


def add_audio(original, video):
    ffmpeg = get_ffmpeg_path()
    temp = video.replace(".mp4", "_temp.mp4")

    cmd = [ffmpeg, "-y", "-i", video, "-i", original,
           "-map", "0:v", "-map", "1:a?", "-c:v", "copy", "-c:a", "aac", "-shortest", temp]

    result = subprocess.run(cmd, capture_output=True)
    if result.returncode == 0 and os.path.exists(temp):
        os.remove(video)
        os.rename(temp, video)
        print("Audio added.")
    else:
        print("No audio or failed.")


def main():
    parser = argparse.ArgumentParser(description="Fast Video Upscaler (Direct Tensor Pipeline)")
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("-m", "--model", default="realesr-animevideov3",
                        choices=["realesr-animevideov3", "RealESRGAN_x4plus_anime_6B", "RealESRGAN_x4plus"])
    parser.add_argument("-s", "--scale", type=int, default=2, choices=[2, 3, 4])
    parser.add_argument("--no-nvenc", action="store_true", help="Use CPU encoding (libx264)")
    parser.add_argument("--tile", type=int, default=None,
                        help="Tile size for RRDB models (0=off, default: auto 960)")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: Not found: {args.input}")
        sys.exit(1)

    if args.output is None:
        name = Path(args.input).stem
        args.output = f"output/{name}_{args.scale}x.mp4"

    os.makedirs("output", exist_ok=True)

    upscale_video_fast(
        input_path=args.input,
        output_path=args.output,
        model_name=args.model,
        scale=args.scale,
        use_nvenc=not args.no_nvenc,
        tile=args.tile
    )


if __name__ == "__main__":
    main()
