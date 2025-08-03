import os
import torch
import torchaudio
import subprocess
import sys
import argparse
import shutil
from pathlib import Path

# Get the root directory (same as in YT_dataset_maker.py)
ROOT_DIR = Path(__file__).parent.parent.parent
RAW_AUDIO_DIR = ROOT_DIR / "Data_prep" / "raw_data" / "full_audio"


def isolate_vocals(input_file, output_file, target_sr=24000):
    """
    Process audio file to isolate vocals:
    1. Load audio file (MP3 or WAV)
    2. Remove music using demucs
    3. Save the isolated vocals
    """
    try:
        # Convert paths to Path objects and resolve to absolute paths
        input_file = Path(input_file).resolve()
        output_file = Path(output_file).resolve()
        output_file.parent.mkdir(parents=True, exist_ok=True)

        print("Loading audio file...")
        wav, sr_orig = torchaudio.load(str(input_file))

        # Convert to mono if stereo
        if wav.size(0) > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)

        # Create temp directory for processing
        temp_dir = ROOT_DIR / "temp"
        temp_dir.mkdir(exist_ok=True)

        # Save as temporary file for demucs
        temp_input = temp_dir / "input.mp3"
        torchaudio.save(str(temp_input), wav, sr_orig, format="mp3")

        print("Separating vocals...")
        # Run demucs inference using Python module to avoid PATH issues
        subprocess.run(
            [
                sys.executable,
                "-m",
                "demucs",
                "--two-stems=vocals",  # Only separate vocals
                "-n",
                "htdemucs_ft",  # Use the hybrid transformer model
                "--mp3",  # Output as MP3 to save space
                "-d",
                "cuda" if torch.cuda.is_available() else "cpu",
                "--out",
                str(temp_dir),  # Specify output directory
                str(temp_input),
            ],
            check=True,
        )

        # Load separated vocals
        vocals_path = temp_dir / "htdemucs_ft" / temp_input.stem / "vocals.mp3"
        if not vocals_path.exists():
            raise RuntimeError(
                f"Demucs failed to generate vocals output at {vocals_path}"
            )

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

        # Clean up temporary files
        if temp_input.exists():
            temp_input.unlink()
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

    except Exception as e:
        print(f"\nError during processing: {e}")
        # Clean up temporary files even if there's an error
        try:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
        except:
            pass
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Isolate vocals from an audio file")
    parser.add_argument("--input", type=str, help="Input audio file path")
    parser.add_argument("--output", type=str, help="Output vocals file path")
    parser.add_argument(
        "--sr", type=int, default=24000, help="Target sample rate (default: 24000)"
    )
    args = parser.parse_args()

    # If no arguments provided, use default paths
    input_file = Path(args.input) if args.input else RAW_AUDIO_DIR / "input.mp3"
    output_file = Path(args.output) if args.output else RAW_AUDIO_DIR / "vocals.wav"

    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")

    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)

    isolate_vocals(input_file, output_file, target_sr=args.sr)
