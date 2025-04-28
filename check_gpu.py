# import torch

# # Print PyTorch and CUDA versions
# print("PyTorch version:", torch.__version__)
# print("CUDA version:", torch.version.cuda)

# # Check if CUDA (GPU) is available
# gpu_available = torch.cuda.is_available()
# print("GPU available:", gpu_available)

# # List all GPUs
# if gpu_available:
#     num_gpus = torch.cuda.device_count()
#     print(f"Number of GPUs available: {num_gpus}")
#     for i in range(num_gpus):
#         print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
# else:
#     print("No GPUs available.")

import torch
print("CUDA available:", torch.cuda.is_available())
print("PyTorch built with CUDA:", torch.version.cuda)
print("Torch version:", torch.__version__)
print("Device count:", torch.cuda.device_count())

