import os

import torch
import torchaudio
import subprocess
import sys
import shutil
from pathlib import Path


# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.absolute()
TEMP_DIR = SCRIPT_DIR / "temp"
CACHE_DIR = SCRIPT_DIR / "cache"
DEMUCS_INSTALLED_FLAG = CACHE_DIR / ".demucs_installed"


def is_demucs_installed():
    """Check if demucs is already installed"""
    try:
        import demucs

        return DEMUCS_INSTALLED_FLAG.exists()
    except ImportError:
        return False


def setup_demucs():
    """Setup demucs with pretrained models"""
    if not is_demucs_installed():
        print("First-time setup: Installing demucs...")
        # Install demucs and its dependencies
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "demucs",  # Main package
                "--upgrade",  # Ensure we have the latest version
                "pydub",  # For MP3 handling
            ],
            check=True,
        )
        # Create flag file to indicate installation
        DEMUCS_INSTALLED_FLAG.parent.mkdir(parents=True, exist_ok=True)
        DEMUCS_INSTALLED_FLAG.touch()

    # Clean up any existing temporary directories
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)

    # Create fresh directories
    TEMP_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)

    # Set environment variables to control where demucs creates files
    os.environ["XDG_CACHE_HOME"] = str(CACHE_DIR)


def isolate_vocals(input_file, output_file, target_sr=24000):
    """
    Process audio file to isolate vocals:
    1. Load audio file (MP3 or WAV)
    2. Remove music using demucs
    3. Save the isolated vocals
    """
    try:
        # Setup demucs
        print("Setting up demucs...")
        setup_demucs()

        # Convert paths to Path objects
        input_file = Path(input_file)
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        print("Loading audio file...")
        wav, sr_orig = torchaudio.load(str(input_file))

        # Convert to mono if stereo
        if wav.size(0) > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)

        # Save as temporary file for demucs
        temp_input = TEMP_DIR / "input.mp3"
        torchaudio.save(str(temp_input), wav, sr_orig, format="mp3")

        # Create output directory for demucs
        temp_output_dir = TEMP_DIR / "demucs_output"
        temp_output_dir.mkdir(exist_ok=True, parents=True)

        print("Separating vocals...")
        # Run demucs inference
        subprocess.run(
            [
                "demucs",
                "--two-stems=vocals",  # Only separate vocals
                "-n",
                "htdemucs_ft",  # Use the hybrid transformer model
                "--mp3",  # Output as MP3 to save space
                "-d",
                "cuda" if torch.cuda.is_available() else "cpu",
                "--out",
                str(temp_output_dir),
                str(temp_input),
            ],
            check=True,
        )

        # Load separated vocals
        vocals_path = temp_output_dir / "htdemucs_ft" / temp_input.stem / "vocals.mp3"
        if not vocals_path.exists():
            raise RuntimeError("Demucs failed to generate vocals output")

        print("Processing separated vocals...")
        vocals, loaded_sr = torchaudio.load(str(vocals_path))

        # Resample if needed
        if loaded_sr != target_sr:
            print(f"Resampling from {loaded_sr}Hz to {target_sr}Hz...")
            resampler = torchaudio.transforms.Resample(
                orig_freq=loaded_sr, new_freq=target_sr
            )
            vocals = resampler(vocals)

        # Normalize audio
        max_val = torch.abs(vocals).max()
        if max_val > 0:
            vocals = vocals / max_val * 0.9

        # Save the final output
        print("\nSaving isolated vocals...")
        torchaudio.save(
            str(output_file),
            vocals,
            target_sr,
            encoding="PCM_S",
            bits_per_sample=16,
        )

        print("\nProcessing complete!")
        print(f"Isolated vocals saved to: {output_file}")

    finally:
        # Clean up temp directory
        if TEMP_DIR.exists():
            try:
                shutil.rmtree(TEMP_DIR)
                TEMP_DIR.mkdir(exist_ok=True)  # Recreate empty temp dir
            except Exception as e:
                print(f"Warning: Could not clean up temp directory: {e}")


if __name__ == "__main__":
    try:
        import argparse

        parser = argparse.ArgumentParser(description="Isolate vocals from audio file")
        parser.add_argument(
            "--input", type=str, required=True, help="Input audio file path"
        )
        parser.add_argument(
            "--output", type=str, required=True, help="Output audio file path"
        )
        args = parser.parse_args()

        isolate_vocals(args.input, args.output, target_sr=24000)
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        # Clean up on interrupt
        if TEMP_DIR.exists():
            shutil.rmtree(TEMP_DIR)
            TEMP_DIR.mkdir(exist_ok=True)
    except Exception as e:
        print(f"\nError during processing: {e}")
        # Clean up on error
        if TEMP_DIR.exists():
            shutil.rmtree(TEMP_DIR)
            TEMP_DIR.mkdir(exist_ok=True)
        raise
