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

        # Construct the whisperx command
        cmd = [
            whisperx_path,
            str(audio_file_path),
            "--model",
            "large-v2",
            "--align_model",
            "WAV2VEC2_ASR_LARGE_LV60K_960H",
            "--output_dir",
            str(output_dir),
            "--compute_type",
            "float16",  # Add compute type for better compatibility
        ]

        # Run the command
        logger.info(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"WhisperX failed with error: {result.stderr}")
            return False

        logger.info(f"Transcription saved to {srt_file_path}")
        return True

    except Exception as e:
        logger.error(f"Error transcribing {audio_file_path}: {e}")
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
    MAX_DURATION = 8  # Maximum segment duration in seconds
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
            logger.info("No SRT file found. Generating transcription...")
            transcribe_success = transcribe_with_whisperx(str(input_file), str(srt_dir))
            if not transcribe_success:
                raise RuntimeError("Failed to generate transcription")

        # Parse SRT content
        logger.info("Parsing SRT content...")
        subs = parse_srt(srt_path)

        if not subs:
            raise ValueError("No segments were generated from SRT")

        # Load audio file
        logger.info("Loading audio file...")
        audio = AudioSegment.from_wav(str(input_file))
        total_duration = len(audio) / 1000.0  # Convert to seconds
        logger.info(f"Input audio duration: {total_duration:.2f} seconds")

        # Create 100ms of silence with same properties as input audio
        silence = AudioSegment.silent(duration=100, frame_rate=audio.frame_rate)

        # Adjust segments
        logger.info("Adjusting segments...")
        adjusted_segments = adjust_segments(subs, None)

        if not adjusted_segments:
            raise ValueError("No segments were generated after combining")

        # Process segments
        metadata = []
        segment_durations = []
        logger.info("Processing segments...")
        for idx, segment in enumerate(tqdm(adjusted_segments)):
            start_ms = segment["start"] * 1000
            end_ms = segment["end"] * 1000
            duration = (end_ms - start_ms) / 1000

            # Skip segments that are too long
            if duration > 8:
                logger.warning(
                    f"Skipping segment {idx} with duration {duration:.2f}s (too long)"
                )
                continue

            # Extract audio segment with 20ms extra at the end and add 100ms silence at the start
            audio_segment = silence + audio[start_ms : end_ms + 20]

            # Update duration to include silence and extra audio
            duration += 0.12  # Add 100ms silence + 20ms extra
            segment_durations.append(duration)

            output_filename = f"{input_file.stem}_{idx+1}.wav"
            output_path = wavs_dir / output_filename

            # Export the audio segment
            audio_segment.export(str(output_path), format="wav")

            metadata.append(
                {
                    "segment_id": idx,
                    "filename": output_filename,
                    "text": segment["text"],
                    "duration": duration,  # Updated duration including silence and extra audio
                    "start_time": segment["start"],
                    "end_time": segment["end"] + 0.02,  # Add 20ms to end time
                    "has_leading_silence": True,  # Add flag to indicate silence was added
                    "has_trailing_audio": True,  # Add flag to indicate extra audio at end
                }
            )

        if not metadata:
            raise ValueError("No valid segments were generated")

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

        logger.info(f"\nProcessing complete!")
        logger.info(f"Total segments saved: {len(metadata)}")
        logger.info(f"Total input duration: {total_duration:.2f} seconds")
        logger.info(f"Total segmented duration: {total_segmented_duration:.2f} seconds")
        logger.info(f"Duration distribution (including 100ms silence + 20ms extra):")
        logger.info(f"  Mean: {mean_duration:.2f} seconds")
        logger.info(f"  Std Dev: {std_duration:.2f} seconds")
        logger.info(f"  Min: {min_duration:.2f} seconds")
        logger.info(f"  Max: {max_duration:.2f} seconds")
        logger.info(f"Target duration range: 2-8 seconds")
        logger.info(f"\nOutputs saved to:")
        logger.info(f"  WAV segments: {wavs_dir}")
        logger.info(f"  Metadata: {metadata_path}")

    except Exception as e:
        logger.error(f"Error during processing: {e}")
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
