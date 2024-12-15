import os
import torch
import torchaudio
from tqdm import tqdm
import subprocess
import sys
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.absolute()
TEMP_DIR = SCRIPT_DIR / "temp"
CACHE_DIR = SCRIPT_DIR / "cache"
DEMUCS_OUTPUT_DIR = TEMP_DIR / "demucs_output"


def setup_demucs():
    """Setup demucs with pretrained models"""
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

    # Clean up any existing directories
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)

    # Create fresh directories
    TEMP_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)
    DEMUCS_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    # Set environment variables to control where demucs creates files
    os.environ["XDG_CACHE_HOME"] = str(CACHE_DIR)

    print("Note: Pretrained model will be downloaded on first use")


def process_chunk(chunk_data):
    """Process a single audio chunk using demucs"""
    chunk, sr, chunk_id = chunk_data
    # Save chunk temporarily
    temp_input = TEMP_DIR / f"input_{chunk_id}.mp3"

    # Create a unique output directory for this chunk
    chunk_output_dir = DEMUCS_OUTPUT_DIR / f"chunk_{chunk_id}"
    chunk_output_dir.mkdir(exist_ok=True, parents=True)

    # Save as MP3 using torchaudio, preserving sample rate
    torchaudio.save(str(temp_input), chunk, sr, format="mp3")

    try:
        # Run demucs inference with explicit output path
        subprocess.run(
            [
                "demucs",
                "--two-stems=vocals",  # Only separate vocals
                "-n",
                "htdemucs",  # Use the hybrid transformer model
                "--mp3",  # Output as MP3 to save space
                "-d",
                "cuda" if torch.cuda.is_available() else "cpu",
                "--out",
                str(chunk_output_dir),  # Specify output directory
                str(temp_input),
            ],
            check=True,
        )

        # Demucs creates output in chunk_output_dir/htdemucs/input_{chunk_id}/vocals.mp3
        vocals_path = chunk_output_dir / "htdemucs" / temp_input.stem / "vocals.mp3"
        if vocals_path.exists():
            vocals, loaded_sr = torchaudio.load(str(vocals_path))
            # Ensure sample rate matches
            if loaded_sr != sr:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=loaded_sr, new_freq=sr
                )
                vocals = resampler(vocals)
            return vocals, chunk_id
        else:
            print(f"No vocals file found at {vocals_path}")
            return None, chunk_id
    finally:
        # Cleanup
        if temp_input.exists():
            temp_input.unlink()
        # Remove demucs output directory for this chunk
        if chunk_output_dir.exists():
            shutil.rmtree(chunk_output_dir)


def isolate_vocals(
    input_file,
    output_file,
    target_sr=24000,
    chunk_duration=30,  # in seconds
    max_workers=4,  # Number of parallel workers
):
    """
    Process a long audio file to isolate vocals:
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

        # Process at original sample rate to maintain pitch
        processing_sr = sr_orig

        # Process in chunks with parallel processing
        chunk_size = int(chunk_duration * processing_sr)
        total_chunks = (wav.size(1) + chunk_size - 1) // chunk_size

        # Prepare chunks with overlap
        chunks_to_process = []
        for i in range(total_chunks):
            start = i * chunk_size
            if start > 0:
                start = max(0, start - int(0.1 * processing_sr))
            end = min(start + chunk_size, wav.size(1))
            chunk = wav[:, start:end]
            chunks_to_process.append((chunk, processing_sr, i))

        # Process chunks in parallel
        print("Processing audio chunks...")
        processed_chunks = [None] * total_chunks  # Pre-allocate list
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = list(
                tqdm(
                    executor.map(process_chunk, chunks_to_process),
                    total=len(chunks_to_process),
                    desc="Processing chunks",
                )
            )

            # Collect results in order
            for result, chunk_id in futures:
                if result is not None:
                    if chunk_id > 0:  # Remove overlap for non-first chunks
                        result = result[:, int(0.1 * processing_sr) :]
                    processed_chunks[chunk_id] = result

        # Remove None values and combine chunks
        processed_chunks = [c for c in processed_chunks if c is not None]
        if not processed_chunks:
            raise RuntimeError(
                "No chunks were processed successfully. Check if the input audio contains vocals."
            )

        # Combine all chunks
        processed_audio = torch.cat(processed_chunks, dim=1)

        # Resample to target_sr only at the end if needed
        if processing_sr != target_sr:
            print(f"\nResampling output from {processing_sr}Hz to {target_sr}Hz...")
            resampler = torchaudio.transforms.Resample(
                orig_freq=processing_sr, new_freq=target_sr
            )
            processed_audio = resampler(processed_audio)

        # Normalize audio
        max_val = torch.abs(processed_audio).max()
        if max_val > 0:
            processed_audio = processed_audio / max_val * 0.9

        # Save the final output
        print("\nSaving isolated vocals...")
        torchaudio.save(
            str(output_file),
            processed_audio,
            target_sr,
            encoding="PCM_S",
            bits_per_sample=16,
        )

        print("\nProcessing complete!")
        print(f"Isolated vocals saved to: {output_file}")

    finally:
        # Clean up temp directory and all its contents
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

        isolate_vocals(
            args.input,
            args.output,
            target_sr=24000,
            max_workers=4,  # Adjust based on your CPU/GPU
        )
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
