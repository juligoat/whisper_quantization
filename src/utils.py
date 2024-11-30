import torch
import psutil
import os
from pathlib import Path

def setup_device():
    """Configure the device (MPS for M3, CPU otherwise)"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS backend")
    else:
        device = torch.device("cpu")
        print("Using CPU backend")
    return device

def measure_model_size(model):
    """Measure model size in MB"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb

def get_memory_usage():
    """Get current memory usage in MB"""
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2

def ensure_dirs_exist():
    """Ensure all necessary directories exist"""
    dirs = ['data', 'models', 'results']
    for d in dirs:
        Path(d).mkdir(exist_ok=True)
