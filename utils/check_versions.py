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
import scipy
import pandas as pd
import torchvision
import torchaudio
import accelerate
import tensorboard
import huggingface_hub
import bitsandbytes
import librosa
import pyaudio
import phonemizer
import nltk
import munch
import resemblyzer
import tqdm
import pesq
import bert_score
import rouge_score
import evaluate
import docker


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
    "scipy": scipy.__version__,
    "pandas": pd.__version__,
    "torchvision": torchvision.__version__,
    "torchaudio": torchaudio.__version__,
    "accelerate": accelerate.__version__,
    "tensorboard": tensorboard.__version__,
    "huggingface_hub": huggingface_hub.__version__,
    "librosa": librosa.__version__,
    "phonemizer": get_package_version("phonemizer"),
    "nltk": nltk.__version__,
    "munch": get_package_version("munch"),
    "resemblyzer": get_package_version("resemblyzer"),
    "tqdm": tqdm.__version__,
    "pesq": get_package_version("pesq"),
    "bert_score": get_package_version("bert_score"),
    "rouge_score": get_package_version("rouge_score"),
    "evaluate": get_package_version("evaluate"),
    "docker": get_package_version("docker"),
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
