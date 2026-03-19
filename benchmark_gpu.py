#!/usr/bin/env python3
"""Check if GPU and FP16 are actually working properly."""

import torch
import numpy as np
import time

print("="*50)
print("GPU & FP16 Diagnostic")
print("="*50)

# Basic GPU info
print(f"\nGPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"PyTorch version: {torch.__version__}")

# Check compute capability (RTX 4090 should be 8.9)
props = torch.cuda.get_device_properties(0)
print(f"Compute capability: {props.major}.{props.minor}")
print(f"Multiprocessors: {props.multi_processor_count}")

# Test FP16 vs FP32 speed
print("\n--- Matrix Multiplication Benchmark ---")
size = 8192

# FP32
a_fp32 = torch.randn(size, size, device='cuda', dtype=torch.float32)
b_fp32 = torch.randn(size, size, device='cuda', dtype=torch.float32)

torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(10):
    c = torch.mm(a_fp32, b_fp32)
torch.cuda.synchronize()
fp32_time = (time.perf_counter() - start) / 10
print(f"FP32: {fp32_time*1000:.2f}ms")

# FP16
a_fp16 = a_fp32.half()
b_fp16 = b_fp32.half()

torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(10):
    c = torch.mm(a_fp16, b_fp16)
torch.cuda.synchronize()
fp16_time = (time.perf_counter() - start) / 10
print(f"FP16: {fp16_time*1000:.2f}ms")
print(f"FP16 speedup: {fp32_time/fp16_time:.2f}x")

if fp32_time / fp16_time < 1.5:
    print("\nWARNING: FP16 is not much faster than FP32!")
    print("This suggests FP16 may not be working properly.")
else:
    print("\nFP16 is working correctly!")

# Test actual model inference precision
print("\n--- Model Precision Test ---")
import sys
import types
import torchvision.transforms.functional as F
_compat = types.ModuleType('torchvision.transforms.functional_tensor')
_compat.rgb_to_grayscale = F.rgb_to_grayscale
sys.modules['torchvision.transforms.functional_tensor'] = _compat

from basicsr.archs.srvgg_arch import SRVGGNetCompact

model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
model = model.cuda().half()
model.eval()

x = torch.randn(1, 3, 1080, 1920, device='cuda', dtype=torch.float16)

# Warmup
with torch.no_grad():
    _ = model(x)
torch.cuda.synchronize()

# Benchmark
times = []
for _ in range(5):
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        _ = model(x)
    torch.cuda.synchronize()
    times.append(time.perf_counter() - start)

avg = sum(times) / len(times)
print(f"Direct model call: {avg*1000:.1f}ms ({1/avg:.1f} fps)")
print(f"Model dtype: {next(model.parameters()).dtype}")
print(f"Input dtype: {x.dtype}")

print("\n" + "="*50)
