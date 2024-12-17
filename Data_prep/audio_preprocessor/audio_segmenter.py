import os
import re
import torch
import faster_whisper
import numpy as np
from tqdm import tqdm
from pathlib import Path
from pydub import AudioSegment
import argparse
import shutil

# Get the root directory
ROOT_DIR = Path(__file__).parent.parent.parent


def format_time(seconds):
    """Convert seconds to SRT time format"""
    milliseconds = int((seconds - int(seconds)) * 1000)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"


def transcribe_audio(audio_path, device="cuda" if torch.cuda.is_available() else "cpu"):
    """Transcribe audio using faster_whisper and return segments"""
    try:
        # Clear GPU memory at start
        if device == "cuda":
            torch.cuda.empty_cache()

        print("Loading faster_whisper model...")
        model = faster_whisper.WhisperModel(
            "medium",
            device=device,
            compute_type="float16" if device == "cuda" else "int8",
            num_workers=4,
        )

        print("Transcribing audio...")
        segments, _ = model.transcribe(
            str(audio_path),
            beam_size=5,
            language="en",
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            word_timestamps=True,
        )

        # Convert generator to list to avoid memory issues
        segments_list = list(segments)

        # Clear GPU memory
        if device == "cuda":
            torch.cuda.empty_cache()

        return segments_list

    except Exception as e:
        print(f"Error during transcription: {e}")
        if device == "cuda":
            torch.cuda.empty_cache()
        raise


def combine_segments(segments, max_duration=18.0, min_duration=2.0, batch_size=500):
    """
    Combine segments while respecting duration constraints and aiming for a gaussian distribution
    between min_duration and max_duration.
    Uses word timestamps and silence detection for natural breaks.
    Processes in batches to handle large files efficiently.
    """
    combined = []

    # Process segments in batches to avoid memory issues
    def process_batch(batch_segments):
        total_duration = 0
        all_words = []

        # Calculate total duration for this batch
        for segment in batch_segments:
            if not hasattr(segment, "words") or not segment.words:
                continue
            for word in segment.words:
                all_words.append(word)
                total_duration += word.end - word.start

        if not all_words:
            return []

        # Generate target durations for this batch
        mean = (min_duration + max_duration) / 2
        std_dev = (max_duration - min_duration) / 6
        target_durations = []
        accumulated = 0

        while accumulated < total_duration:
            duration = np.random.normal(mean, std_dev)
            duration = max(min(duration, max_duration), min_duration)

            remaining = total_duration - accumulated

            if remaining < min_duration:
                if (
                    target_durations
                    and target_durations[-1] + remaining <= max_duration
                ):
                    target_durations[-1] += remaining
                break

            if accumulated + duration > total_duration:
                remaining = total_duration - accumulated
                if min_duration <= remaining <= max_duration:
                    target_durations.append(remaining)
                elif remaining > max_duration:
                    while remaining > 0:
                        if remaining > max_duration:
                            target_durations.append(max_duration)
                            remaining -= max_duration
                        else:
                            if remaining >= min_duration:
                                target_durations.append(remaining)
                            elif target_durations:
                                target_durations[-1] += remaining
                            break
                break

            target_durations.append(duration)
            accumulated += duration

        # Create segments for this batch
        batch_combined = []
        current_words = []
        current_duration = 0
        current_start = None
        target_idx = 0

        for word in all_words:
            if target_idx >= len(target_durations):
                break

            word_duration = word.end - word.start

            # Initialize start time if needed
            if current_start is None:
                current_start = word.start

            # If adding this word would exceed target duration
            if current_duration + word_duration > target_durations[target_idx]:
                # Save current group if it meets minimum duration
                if current_duration >= min_duration:
                    batch_combined.append(
                        [
                            {
                                "start": current_start,
                                "end": current_words[-1].end,
                                "text": " ".join(w.word for w in current_words),
                            }
                        ]
                    )
                    # Reset for next group
                    current_words = []
                    current_duration = 0
                    current_start = word.start
                    target_idx += 1

            # Add word to current group
            current_words.append(word)
            current_duration += word_duration

            # Check for natural breaks (silence or punctuation)
            if (
                word.word[-1] in ".!?"
                and min_duration <= current_duration <= max_duration
            ):
                batch_combined.append(
                    [
                        {
                            "start": current_start,
                            "end": word.end,
                            "text": " ".join(w.word for w in current_words),
                        }
                    ]
                )
                current_words = []
                current_duration = 0
                current_start = None
                target_idx += 1

        # Don't forget remaining words
        if current_words and min_duration <= current_duration <= max_duration:
            batch_combined.append(
                [
                    {
                        "start": current_start,
                        "end": current_words[-1].end,
                        "text": " ".join(w.word for w in current_words),
                    }
                ]
            )

        return batch_combined

    # Process segments in batches
    for i in range(0, len(segments), batch_size):
        batch = segments[i : i + batch_size]
        batch_combined = process_batch(batch)
        combined.extend(batch_combined)
        # Clear memory
        del batch_combined
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return combined


def segment_audio(
    input_file,
    output_dir,
    max_segment_duration=18,  # Changed to match reference
    min_segment_duration=2,  # Changed to match reference
    pad_duration=0.5,  # in seconds
    target_sr=24000,
):
    """
    Segment audio file based on faster_whisper transcription:
    1. Transcribe audio using faster_whisper to get word-level timestamps
    2. Combine segments using word boundaries and natural breaks
    3. Export segments with padding
    """
    try:
        # Convert paths to Path objects
        input_file = Path(input_file)
        output_dir = Path(output_dir)

        # Check if input file exists
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load audio
        print("Loading audio file...")
        audio = AudioSegment.from_wav(str(input_file))

        # Convert to mono if stereo
        if audio.channels > 1:
            print("Converting to mono...")
            audio = audio.set_channels(1)

        # Convert to target sample rate
        if audio.frame_rate != target_sr:
            print(f"Resampling from {audio.frame_rate} to {target_sr}...")
            audio = audio.set_frame_rate(target_sr)

        # Get transcription segments with word timestamps
        print("Transcribing audio with word timestamps...")
        segments = transcribe_audio(input_file)

        if not segments:
            raise ValueError("No segments were generated during transcription")

        # Combine segments using word-level timestamps
        print("Combining segments...")
        combined_segments = combine_segments(
            segments, max_segment_duration, min_segment_duration
        )

        if not combined_segments:
            raise ValueError("No segments were generated after combining")

        # Save segments
        print("Saving segments as WAV...")
        segments_saved = 0
        total_duration = 0
        min_found_duration = float("inf")
        max_found_duration = 0
        durations = []  # Keep track of all durations for distribution analysis

        for group in tqdm(combined_segments, desc="Saving segments"):
            try:
                segment_info = group[0]
                duration = segment_info["end"] - segment_info["start"]

                # Strict enforcement of duration limits
                if duration < min_segment_duration or duration > max_segment_duration:
                    print(
                        f"\nSkipping segment: duration {duration:.2f}s outside range [{min_segment_duration}-{max_segment_duration}]s"
                    )
                    continue

                # Extract audio segment
                start_ms = segment_info["start"] * 1000
                end_ms = segment_info["end"] * 1000
                segment = audio[start_ms:end_ms]

                # Add padding
                silence = AudioSegment.silent(duration=pad_duration * 1000)
                padded_segment = silence + segment + silence

                # Calculate final duration
                final_duration = len(padded_segment) / 1000
                total_duration += final_duration
                min_found_duration = min(min_found_duration, final_duration)
                max_found_duration = max(max_found_duration, final_duration)
                durations.append(final_duration)

                # Normalize audio
                normalized_segment = padded_segment.normalize()

                # Save segment
                output_path = output_dir / f"segment_{segments_saved:04d}.wav"
                normalized_segment.export(str(output_path), format="wav")
                segments_saved += 1

            except Exception as e:
                print(f"\nError processing segment {segments_saved}: {e}")
                continue

        if segments_saved == 0:
            raise ValueError("No segments were successfully saved")

        # Calculate distribution statistics
        durations = np.array(durations)
        mean_duration = np.mean(durations)
        std_duration = np.std(durations)

        print("\nProcessing complete!")
        print(f"Total segments saved: {segments_saved}")
        print(f"Total audio duration: {total_duration:.2f} seconds")
        print(f"Duration distribution:")
        print(f"  Mean: {mean_duration:.2f} seconds")
        print(f"  Std Dev: {std_duration:.2f} seconds")
        print(f"  Min: {min_found_duration:.2f} seconds")
        print(f"  Max: {max_found_duration:.2f} seconds")
        print(
            f"Target duration range: {min_segment_duration}-{max_segment_duration} seconds"
        )

    except Exception as e:
        print(f"\nError during processing: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Segment audio file into chunks")
        parser.add_argument(
            "--input", type=str, required=True, help="Input audio file path"
        )
        parser.add_argument(
            "--output", type=str, required=True, help="Output directory path"
        )
        args = parser.parse_args()

        segment_audio(
            args.input,
            args.output,
            max_segment_duration=18,
            min_segment_duration=2,
            pad_duration=0.5,
        )
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print(f"\nError during processing: {e}")
        raise
