import pkg_resources
import torch
import numpy as np
import sounddevice as sd
import PIL
import soundfile as sf
import tkinter as tk
import transformers
import faster_whisper
import speechbrain
import requests

def get_package_version(package_name):
    try:
        return pkg_resources.get_distribution(package_name).version
    except pkg_resources.DistributionNotFound:
        return "Not installed"

# List of packages to check
packages = {
    "transformers": transformers.__version__,
    "torch": torch.__version__,
    "numpy": np.__version__,
    "Pillow": PIL.__version__,
    "soundfile": sf.__version__,
    "tkinter": tk.TkVersion,
    "faster-whisper": faster_whisper.__version__,
    "speechbrain": speechbrain.__version__,
    "sounddevice": sd.__version__,
    "requests": requests.__version__,
    "num2words": get_package_version("num2words"),
    "bitsandbytes": get_package_version("bitsandbytes"),
}

# Print versions in a formatted way
print("\nInstalled Package Versions:")
print("-" * 40)
for package, version in packages.items():
    print(f"{package:<20} {version}")
print("-" * 40)

# Check CUDA availability
if torch.cuda.is_available():
    print(f"\nCUDA is available!")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name()}")
else:
    print("\nCUDA is not available. Using CPU only.") 