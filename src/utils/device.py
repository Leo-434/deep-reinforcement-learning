import torch

def get_device() -> torch.device:
    """
    Automatically detect and return the best available PyTorch device.
    Prioritizes CUDA, then MPS (Apple Silicon), and falls back to CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
