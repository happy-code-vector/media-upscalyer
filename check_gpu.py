import torch

if torch.cuda.is_available():
    print("CUDA is available. GPU detected.")
    print("__Number of CUDA Devices:", torch.cuda.device_count())
    print("__CUDA Device Name:", torch.cuda.get_device_name(0))
    print("__CUDA Device Total Memory [GB]:", torch.cuda.get_device_properties(0).total_memory / 1e9)
else:
    print("CUDA is not available. Check GPU, drivers, and PyTorch installation.")

# A common line to set the device for your code
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)