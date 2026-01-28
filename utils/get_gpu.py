import torch

def get_gpu():
    return "mps" if torch.backends.mps.is_available() else "cpu"