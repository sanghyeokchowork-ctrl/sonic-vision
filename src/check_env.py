import sys
import torch
import librosa
import numpy as np

def check_environment():
    """
    Checks the installation status of key libraries and hardware availability.
    """
    print("=" * 40)
    print("  Sonic Vision Environment Check")
    print("=" * 40)

    # 1. Check Python Version
    print(f" Python Version   : {sys.version.split()[0]}")

    # 2. Check PyTorch (Deep Learning Core)
    print(f" PyTorch Version  : {torch.__version__}")

    # 3. Check Librosa (Audio Processing Core)
    print(f" Librosa Version  : {librosa.__version__}")

    # 4. Tensor Operation Test
    # Create a random tensor to verify PyTorch functionality
    try:
        x = torch.rand(5, 3)
        print(f" Tensor Test      : Success (Shape: {x.shape})")
    except Exception as e:
        print(f" Tensor Test      : Failed ({e})")

    # 5. Check Device (CPU vs GPU)
    # Checks if CUDA (NVIDIA GPU) or MPS (Mac M-series GPU) is available
    if torch.cuda.is_available():
        device = "cuda"
        print(" Hardware Accel   : NVIDIA GPU (CUDA) Detected!")
    elif torch.backends.mps.is_available():
        device = "mps"
        print(" Hardware Accel   : Apple Silicon GPU (MPS) Detected!")
    else:
        device = "cpu"
        print(" Hardware Accel   : CPU Mode (Training might be slow)")

    print("-" * 40)
    print(f" Active Device    : {device.upper()}")
    print("=" * 40)

if __name__ == "__main__":
    check_environment()