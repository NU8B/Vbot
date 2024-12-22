import os
import sys
import json
import torch
import whisper
import pysrt
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path
import soundfile as sf
from pydub import AudioSegment
import argparse

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


def transcribe_with_whisper(audio_file_path, output_dir, num_gpus=None):
    """
    Transcribe audio using local Whisper model.
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
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model("turbo").to(device)
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
        result = model.transcribe(
            audio_file_path,
            language="en",
            word_timestamps=True,
            verbose=True,
            fp16=(device == "cuda"),
        )
        srt_content = ""
        for i, segment in enumerate(result["segments"], 1):
            start_time = format_time(segment["start"])
            end_time = format_time(segment["end"])
            text = segment["text"].strip()
            srt_content += f"{i}\n{start_time} --> {end_time}\n{text}\n\n"

        with open(srt_file_path, "w", encoding="utf-8") as f:
            f.write(srt_content)

        logger.info(f"Transcription saved to {srt_file_path}")

        if device == "cuda":
            torch.cuda.empty_cache()

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


def generate_gaussian_durations(total_duration, min_length=2, max_length=18):
    """Generate segment durations following a truncated Gaussian distribution."""
    mean = (min_length + max_length) / 2
    std_dev = (max_length - min_length) / 6
    durations = []
    accumulated = 0

    while accumulated < total_duration:
        duration = np.random.normal(mean, std_dev)
        duration = max(min(duration, max_length), min_length)

        remaining = total_duration - accumulated

        if remaining < min_length:
            if durations:
                if durations[-1] + remaining <= max_length:
                    durations[-1] += remaining
            break

        if accumulated + duration > total_duration:
            remaining = total_duration - accumulated
            if min_length <= remaining <= max_length:
                durations.append(remaining)
            elif remaining > max_length:
                while remaining > 0:
                    if remaining > max_length:
                        durations.append(max_length)
                        remaining -= max_length
                    else:
                        if remaining >= min_length:
                            durations.append(remaining)
                        elif durations:
                            durations[-1] += remaining
                        break
            break

        durations.append(duration)
        accumulated += duration

    return durations


def adjust_segments(subs, durations):
    """Adjust the segments to match the desired durations."""
    adjusted_segments = []
    i = 0
    num_subs = len(subs)
    END_PADDING = 0.4  # 200ms padding for word completion

    if not subs:
        return []

    start_time = subs[0]["start"]

    while i < num_subs:
        if not durations:
            break

        segment_duration = durations.pop(0)
        target_end_time = start_time + segment_duration

        current_segment = {"start": start_time, "text": "", "end": start_time}

        # Keep accumulating text until we hit our target or would exceed max duration
        while i < num_subs:
            # First add the text and update end time
            current_segment["text"] += " " + subs[i]["text"]
            # Add padding to ensure last word is complete
            current_segment["end"] = subs[i]["end"] + END_PADDING

            # Check if we've reached target duration or would exceed max with next subtitle
            next_duration = current_segment["end"] - current_segment["start"]
            if next_duration >= 18 or subs[i]["end"] >= target_end_time:
                break

            i += 1

        # Now we have a complete segment, check if it's valid
        segment_duration = current_segment["end"] - current_segment["start"]
        if 2 <= segment_duration <= 18:
            current_segment["text"] = current_segment["text"].strip()
            adjusted_segments.append(current_segment)

        # Move to next segment
        i += 1
        if i < num_subs:
            start_time = subs[i]["start"]

    return adjusted_segments


def segment_audio(input_file, output_dir):
    """
    Main function to segment audio file using transcription-based segmentation.
    Follows the exact logic from adm_main.py.
    """
    try:
        # Convert paths to Path objects
        input_file = Path(input_file)
        output_dir = Path(output_dir)
        wavs_dir = output_dir / "wavs"
        srt_dir = output_dir / "srts"  # Add SRT directory

        # Create output directories
        wavs_dir.mkdir(parents=True, exist_ok=True)
        srt_dir.mkdir(parents=True, exist_ok=True)  # Create SRT directory

        # Check if input file exists
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        # First generate transcription if SRT doesn't exist
        srt_path = input_file.with_suffix(".srt")
        if not srt_path.exists():
            logger.info("No SRT file found. Generating transcription...")
            transcribe_success = transcribe_with_whisper(str(input_file), str(srt_dir))
            if not transcribe_success:
                raise RuntimeError("Failed to generate transcription")
            srt_path = srt_dir / f"{input_file.stem}.srt"

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

        # Generate durations and adjust segments
        logger.info("Generating segment durations...")
        durations = generate_gaussian_durations(total_duration)
        adjusted_segments = adjust_segments(subs, durations)

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
            segment_durations.append(duration)
            audio_segment = audio[start_ms:end_ms]

            output_filename = f"{input_file.stem}_{idx+1}.wav"
            output_path = wavs_dir / output_filename

            audio_segment.export(str(output_path), format="wav")

            metadata.append(
                {
                    "segment_id": idx,
                    "filename": output_filename,
                    "text": segment["text"],
                    "duration": duration,
                    "start_time": segment["start"],
                    "end_time": segment["end"],
                }
            )

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
        logger.info(f"Duration distribution:")
        logger.info(f"  Mean: {mean_duration:.2f} seconds")
        logger.info(f"  Std Dev: {std_duration:.2f} seconds")
        logger.info(f"  Min: {min_duration:.2f} seconds")
        logger.info(f"  Max: {max_duration:.2f} seconds")
        logger.info(f"Target duration range: 2-18 seconds")
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
