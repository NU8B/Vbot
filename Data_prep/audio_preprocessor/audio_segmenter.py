import os
import sys
import json
import torch
import gc
import pysrt
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path
import soundfile as sf
from pydub import AudioSegment
import argparse
import ctypes

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def format_time(seconds):
    """Convert seconds to SRT time format"""
    milliseconds = int((seconds - int(seconds)) * 1000)
    time_str = f"{int(seconds // 3600):02}:{int((seconds % 3600) // 60):02}:{int(seconds % 60):02},{milliseconds:03}"
    return time_str


def verify_cuda_setup():
    """Verify CUDA and cuDNN are properly set up"""
    import os
    import torch
    import ctypes
    from pathlib import Path

    if not torch.cuda.is_available():
        logger.error("CUDA is not available. Please check your PyTorch installation.")
        return False

    # Enable TF32 for better performance
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Check CUDA version
    cuda_version = torch.version.cuda
    logger.info(f"CUDA version: {cuda_version}")

    # Try to find cudnn64_8.dll in CUDA 12.1 path
    cuda_path = os.environ.get(
        "CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"
    )
    cudnn_paths = [
        Path(cuda_path) / "bin" / "cudnn64_8.dll",
        Path(cuda_path) / "bin" / "cudnn_ops_infer64_8.dll",
        Path(cuda_path) / "bin" / "cudnn_cnn_infer64_8.dll",
    ]

    missing_dlls = []
    for dll_path in cudnn_paths:
        if not dll_path.exists():
            missing_dlls.append(dll_path.name)

    if missing_dlls:
        logger.error(
            f"The following cuDNN files are missing: {', '.join(missing_dlls)}"
        )
        logger.error(
            f"Please install cuDNN for CUDA 12.1 from: https://developer.nvidia.com/cudnn"
        )
        logger.error(
            f"And copy all files from the cuDNN zip's cuda/bin folder to: {Path(cuda_path) / 'bin'}"
        )
        return False

    try:
        # Try to load cuDNN files
        for dll_path in cudnn_paths:
            ctypes.CDLL(str(dll_path))
        logger.info("All cuDNN files loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to load cuDNN: {e}")
        return False


def transcribe_with_whisperx(audio_file_path, output_dir):
    """
    Transcribe audio using WhisperX CLI for more accurate word-level timestamps.
    Skip if SRT file already exists.
    """
    # Check if SRT file already exists
    base_name = os.path.splitext(os.path.basename(audio_file_path))[0]
    srt_file_path = os.path.join(output_dir, f"{base_name}.srt")

    if os.path.exists(srt_file_path):
        logger.info(
            f"SRT file already exists for {base_name}, skipping transcription..."
        )
        return True

    try:
        import subprocess
        import sys
        import gc  # For manual garbage collection

        # Verify CUDA setup first
        if not verify_cuda_setup():
            logger.error("CUDA setup verification failed")
            return False

        # Get the path to whisperx executable in the current Python environment
        python_dir = os.path.dirname(sys.executable)
        whisperx_path = os.path.join(python_dir, "Scripts", "whisperx.exe")

        if not os.path.exists(whisperx_path):
            logger.error(
                f"WhisperX not found at {whisperx_path}. Please install it using: pip install git+https://github.com/m-bain/whisperx.git"
            )
            return False

        # Clear any existing GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        # Construct the whisperx command with optimized settings
        cmd = [
            whisperx_path,
            str(audio_file_path),
            "--model",
            "large-v3",
            "--language",
            "en",
            "--align_model",
            "WAV2VEC2_ASR_LARGE_LV60K_960H",
            "--output_dir",
            str(output_dir),
            "--compute_type",
            "float16",  # Use float16 for better memory efficiency
            "--batch_size",
            "8",  # Middle ground between memory usage and speed
            "--device",
            "cuda" if torch.cuda.is_available() else "cpu",
            "--verbose",
            "True",  # Fixed the verbose flag
        ]

        print("\nStarting WhisperX transcription...")
        print("This may take several minutes. Progress will be shown below:")
        print("(Note: Progress indicators will appear as transcription proceeds)\n")

        # Run the command and capture output in real-time
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )

        # Track progress through output
        for line in process.stdout:
            line = line.strip()
            if "Detecting language" in line:
                print("Detecting audio language...")
            elif "Transcribing" in line:
                print("Transcribing audio...")
            elif "Aligning" in line:
                print("Aligning timestamps...")
            elif "Writing" in line and ".srt" in line:
                print("Saving transcription to SRT file...")
            elif any(x in line.lower() for x in ["error", "warning", "cuda"]):
                print(f"System message: {line}")

        # Wait for the process to complete
        process.wait()

        if process.returncode != 0:
            print("\nWhisperX transcription failed!")
            return False

        print("\nTranscription completed successfully!")

        # Clear GPU memory again after processing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        return True

    except Exception as e:
        print(f"\nError during transcription: {e}")
        # Ensure GPU memory is cleared even if there's an error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        return False


def parse_srt(srt_file_path):
    """Parse the .srt file and return a list of subtitles with start and end times in seconds."""
    subtitles = pysrt.open(srt_file_path)
    subs = []
    for sub in subtitles:
        start_time = sub.start.ordinal / 1000.0
        end_time = sub.end.ordinal / 1000.0
        text = sub.text.replace("\n", " ").strip()
        subs.append({"start": start_time, "end": end_time, "text": text})
    return subs


def adjust_segments(subs, durations):
    """Combine subtitles into segments up to max duration."""
    MAX_DURATION = 7.8  # Maximum segment duration in seconds
    MIN_DURATION = 2  # Minimum segment duration in seconds
    adjusted_segments = []

    if not subs:
        return []

    current_segment = {
        "start": subs[0]["start"],
        "text": subs[0]["text"],
        "end": subs[0]["end"],
    }

    i = 1
    while i < len(subs):
        # Calculate current segment duration
        current_duration = current_segment["end"] - current_segment["start"]

        # Calculate what duration would be if we add next segment
        potential_duration = subs[i]["end"] - current_segment["start"]

        # Decide whether to combine segments
        should_combine = (
            # Always combine if current segment is too short
            current_duration < MIN_DURATION
            or
            # Or if adding next segment won't exceed max duration
            potential_duration <= MAX_DURATION
        )

        if should_combine:
            # Add to current segment
            current_segment["text"] += " " + subs[i]["text"]
            current_segment["end"] = subs[i]["end"]
            i += 1
        else:
            # Save current segment if it's long enough
            if current_duration >= MIN_DURATION:
                adjusted_segments.append(current_segment)
            # Start new segment
            current_segment = {
                "start": subs[i]["start"],
                "text": subs[i]["text"],
                "end": subs[i]["end"],
            }
            i += 1

    # Handle the last segment
    last_duration = current_segment["end"] - current_segment["start"]
    if last_duration >= MIN_DURATION:
        adjusted_segments.append(current_segment)
    elif adjusted_segments:
        # If last segment is too short, try to combine it with the previous segment
        prev_segment = adjusted_segments[-1]
        total_duration = current_segment["end"] - prev_segment["start"]
        if total_duration <= MAX_DURATION:
            prev_segment["text"] += " " + current_segment["text"]
            prev_segment["end"] = current_segment["end"]
        # If we can't combine it with previous segment and it's not too short, keep it
        elif last_duration >= MIN_DURATION:
            adjusted_segments.append(current_segment)

    return adjusted_segments


def segment_audio(input_file, output_dir):
    """
    Main function to segment audio file using transcription-based segmentation.
    """
    try:
        # Convert paths to Path objects
        input_file = Path(input_file)
        output_dir = Path(output_dir)

        print("\n=== Starting Audio Segmentation Process ===")
        print(f"Input file: {input_file}")
        print(f"Output directory: {output_dir}\n")

        # Create all necessary directories
        wavs_dir = output_dir / "wavs"
        srt_dir = output_dir / "srts"
        wavs_dir.mkdir(parents=True, exist_ok=True)
        srt_dir.mkdir(parents=True, exist_ok=True)

        # Check if input file exists
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        # First generate transcription if SRT doesn't exist
        srt_path = srt_dir / f"{input_file.stem}.srt"
        if not srt_path.exists():
            print("\n=== Generating Transcription ===")
            print("This may take a few minutes depending on the audio length...")
            transcribe_success = transcribe_with_whisperx(str(input_file), str(srt_dir))
            if not transcribe_success:
                raise RuntimeError("Failed to generate transcription")
        else:
            print("\n=== Using Existing Transcription ===")

        # Parse SRT content
        print("\n=== Parsing Transcription ===")
        subs = parse_srt(srt_path)
        print(f"Found {len(subs)} initial segments in transcription")

        if not subs:
            raise ValueError("No segments were generated from SRT")

        # Load audio file
        print("\n=== Loading Audio File ===")
        audio = AudioSegment.from_wav(str(input_file))
        total_duration = len(audio) / 1000.0  # Convert to seconds
        print(f"Input audio duration: {total_duration:.2f} seconds")

        # Create 100ms of silence with same properties as input audio
        silence = AudioSegment.silent(duration=100, frame_rate=audio.frame_rate)

        # Adjust segments
        print("\n=== Adjusting Segments ===")
        print("Combining short segments and splitting long ones...")
        adjusted_segments = adjust_segments(subs, None)
        print(f"Adjusted to {len(adjusted_segments)} segments")

        if not adjusted_segments:
            raise ValueError("No segments were generated after combining")

        # Process segments
        metadata = []
        segment_durations = []
        skipped_too_long = 0
        print("\n=== Processing Audio Segments ===")
        print(f"Processing {len(adjusted_segments)} segments...")

        for idx, segment in enumerate(adjusted_segments, 1):
            start_ms = segment["start"] * 1000
            end_ms = segment["end"] * 1000
            duration = (end_ms - start_ms) / 1000

            # Skip segments that are too long
            if duration > 7.8:
                skipped_too_long += 1
                continue

            # Extract audio segment with 20ms extra at the end
            audio_segment = audio[start_ms : end_ms + 20]

            # Add 100ms silence at both start and end
            audio_segment = silence + audio_segment + silence

            # Update duration to include silence and extra audio
            duration += 0.22  # Add 200ms silence (100ms at each end) + 20ms extra audio
            segment_durations.append(duration)

            output_filename = f"{input_file.stem}_{idx}.wav"
            output_path = wavs_dir / output_filename

            # Export the audio segment
            audio_segment.export(str(output_path), format="wav")

            metadata.append(
                {
                    "segment_id": idx - 1,
                    "filename": output_filename,
                    "text": segment["text"],
                    "duration": duration,  # Updated duration including silences and extra audio
                    "start_time": segment["start"],
                    "end_time": segment["end"] + 0.02,  # Add 20ms to end time
                    "has_leading_silence": True,  # Add flag to indicate silence was added
                    "has_trailing_silence": True,  # Add flag to indicate silence was added
                    "has_trailing_audio": True,  # Add flag to indicate extra audio at end
                }
            )

        if not metadata:
            raise ValueError("No valid segments were generated")

        print(f"\nSegments processed:")
        print(f"  Total segments: {len(adjusted_segments)}")
        print(f"  Successfully processed: {len(metadata)}")
        print(f"  Skipped (too long): {skipped_too_long}")

        # Calculate statistics
        segment_durations = np.array(segment_durations)
        total_segmented_duration = np.sum(segment_durations)
        mean_duration = np.mean(segment_durations)
        std_duration = np.std(segment_durations)
        min_duration = np.min(segment_durations)
        max_duration = np.max(segment_durations)

        # Save metadata
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print("\n=== Processing Summary ===")
        print(f"Total segments saved: {len(metadata)}")
        print(f"Total input duration: {total_duration:.2f} seconds")
        print(f"Total segmented duration: {total_segmented_duration:.2f} seconds")
        print(f"\nDuration distribution (including 100ms silence + 20ms extra):")
        print(f"  Mean: {mean_duration:.2f} seconds")
        print(f"  Std Dev: {std_duration:.2f} seconds")
        print(f"  Min: {min_duration:.2f} seconds")
        print(f"  Max: {max_duration:.2f} seconds")
        print(f"\nTarget duration range: 2-8 seconds")
        print(f"\nOutputs saved to:")
        print(f"  WAV segments: {wavs_dir}")
        print(f"  Metadata: {metadata_path}")

    except Exception as e:
        print(f"\nError during processing: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment audio file into chunks")
    parser.add_argument(
        "--input",
        type=str,
        default="Data_prep/raw_data/full_audio/vocals.wav",
        help="Input audio file path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="Data_prep/raw_data/segments",
        help="Output directory path",
    )
    args = parser.parse_args()

    try:
        segment_audio(args.input, args.output)
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print(f"\nError during processing: {e}")
        raise
