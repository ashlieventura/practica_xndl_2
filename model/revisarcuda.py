import torch

print("Torch:", torch.__version__)
print("CUDA disponible:", torch.cuda.is_available())
print("Versión CUDA:", torch.version.cuda)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))