import os
import re
import torch
import faster_whisper
import numpy as np
from tqdm import tqdm
from pathlib import Path
from pydub import AudioSegment
import argparse
import soundfile as sf
from typing import Tuple, List, Dict
from g2p import make_g2p

# Get the root directory
ROOT_DIR = Path(__file__).parent.parent.parent

# Speaking rate bin edges from reference implementation
SPEAKING_RATE_BINS = [0.0, 3.83, 7.65, 11.48, 15.30, 19.13, 22.95, 26.78]

# Initialize G2P transducer for phoneme-based speaking rate estimation
try:
    transducer = make_g2p("eng", "eng-ipa")
except ImportError:
    print("Warning: g2p not installed. Speaking rate estimation will be less accurate.")
    transducer = None


def estimate_speaking_rate(text: str, duration: float) -> float:
    """
    Estimate speaking rate using phoneme count per second.
    This matches the reference implementation's approach.
    """
    if transducer is None:
        # Fallback to simple word count if g2p is not available
        return len(text.split()) / duration

    try:
        phonemes = transducer(text).output_string
        return len(phonemes) / duration
    except:
        # Fallback to simple word count if phoneme conversion fails
        return len(text.split()) / duration


def adjust_segments(segments: List[Dict], target_durations: List[float]) -> List[Dict]:
    """
    Adjust segments to match target durations.
    Direct implementation of the reference approach.
    """
    adjusted_segments = []
    i = 0
    num_segments = len(segments)

    if not segments or not target_durations:
        return adjusted_segments

    current_start = segments[0]["start"]

    while i < num_segments and target_durations:
        target_duration = target_durations.pop(0)
        target_end_time = current_start + target_duration

        current_segment = {
            "start": current_start,
            "text": "",
            "end": current_start,
            "words": [],
        }

        # Accumulate segments until we reach target duration
        while i < num_segments:
            segment = segments[i]

            # Add words from this segment
            if "words" in segment and segment["words"]:
                for word in segment["words"]:
                    current_segment["words"].append(word)
                    current_segment["text"] += " " + word["word"]
                    # Force max duration of 24 seconds
                    current_segment["end"] = min(word["end"], current_start + 24.0)

                    # Check if we've reached target duration or max duration
                    if (
                        word["end"] >= target_end_time
                        or current_segment["end"] - current_segment["start"] >= 24.0
                    ):
                        break

            # Check if we've reached target duration or max duration
            if (
                current_segment["end"] >= target_end_time
                or current_segment["end"] - current_segment["start"] >= 24.0
            ):
                break

            i += 1

        # Clean up and save segment if it meets duration constraints
        segment_duration = current_segment["end"] - current_segment["start"]
        if 2.0 <= segment_duration <= 24.0:  # Min 2s, max 24s
            current_segment["text"] = current_segment["text"].strip()
            speaking_rate = estimate_speaking_rate(
                current_segment["text"], segment_duration
            )

            adjusted_segments.append(
                {
                    "start": current_segment["start"],
                    "end": current_segment["end"],
                    "text": current_segment["text"],
                    "speaking_rate": speaking_rate,
                }
            )

        # Move to next segment
        i += 1
        if i < num_segments:
            current_start = segments[i]["start"]

    return adjusted_segments


def combine_segments(segments, max_duration=24.0, min_duration=2.0):
    """
    Combine segments while respecting duration constraints and aiming for a gaussian distribution
    between min_duration and max_duration.
    Uses word timestamps and target durations from a Gaussian distribution.
    """
    if not segments:
        return []

    # Calculate total duration and collect all words
    total_duration = 0
    all_words = []

    for segment in segments:
        if "words" not in segment or not segment["words"]:
            continue
        for word in segment["words"]:
            all_words.append(word)
            total_duration += word["end"] - word["start"]

    if not all_words:
        return []

    # Generate target durations using Gaussian distribution
    mean = (min_duration + max_duration) / 2
    std_dev = (max_duration - min_duration) / 6
    target_durations = []
    accumulated = 0

    while accumulated < total_duration:
        # Generate duration from Gaussian distribution
        duration = np.random.normal(mean, std_dev)
        duration = max(min(duration, max_duration), min_duration)

        remaining = total_duration - accumulated
        if remaining < min_duration:
            if target_durations and target_durations[-1] + remaining <= max_duration:
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

    # Create initial segments based on words
    initial_segments = []
    current_segment = None

    for word in all_words:
        if current_segment is None:
            current_segment = {
                "start": word["start"],
                "end": word["end"],
                "words": [word],
                "text": word["word"],
            }
        else:
            # Check if this word would make the segment too long
            if word["end"] - current_segment["start"] > max_duration:
                initial_segments.append(current_segment)
                current_segment = {
                    "start": word["start"],
                    "end": word["end"],
                    "words": [word],
                    "text": word["word"],
                }
            else:
                current_segment["end"] = word["end"]
                current_segment["words"].append(word)
                current_segment["text"] += " " + word["word"]

    if current_segment:
        initial_segments.append(current_segment)

    # Adjust segments to match target durations
    return adjust_segments(initial_segments, target_durations)


def denoise_audio(audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Denoise audio using basic noise reduction techniques.
    For better results, consider using the 'demucs' library as used in the reference implementation.
    """
    try:
        from scipy import signal

        # Check if audio is too short for filtering
        min_samples = 100  # Increased minimum samples needed for filtering
        if len(audio_data) < min_samples:
            print(
                f"Warning: Audio segment too short ({len(audio_data)} samples) for denoising. Returning original audio."
            )
            return audio_data

        # Apply a high-pass filter to remove low-frequency noise
        nyquist = sample_rate // 2
        cutoff = 50  # Hz
        normalized_cutoff = cutoff / nyquist

        # Calculate required padding based on audio length
        # Use shorter filter order for shorter segments
        if len(audio_data) < 1000:
            filter_order = 2
        elif len(audio_data) < 5000:
            filter_order = 3
        else:
            filter_order = 4

        b, a = signal.butter(filter_order, normalized_cutoff, btype="high")

        # Calculate required padding
        padlen = 3 * max(len(b), len(a))

        # If audio is too short for padding, extend it by mirroring
        if len(audio_data) <= padlen:
            # Mirror the audio data to create enough samples
            pad_needed = padlen - len(audio_data) + 1
            left_pad = pad_needed // 2
            right_pad = pad_needed - left_pad
            audio_data = np.pad(audio_data, (left_pad, right_pad), mode="reflect")

        # Apply filter
        denoised = signal.filtfilt(b, a, audio_data, padlen=padlen)

        # If we padded the audio, trim back to original length
        if len(denoised) > len(audio_data):
            start_idx = (len(denoised) - len(audio_data)) // 2
            denoised = denoised[start_idx : start_idx + len(audio_data)]

        return denoised

    except ImportError:
        print("Warning: scipy not found. Skipping denoising.")
        return audio_data
    except Exception as e:
        print(f"Warning: Error during denoising: {e}. Returning original audio.")
        return audio_data


def normalize_audio(audio_data: np.ndarray, headroom_db: float = 3.0) -> np.ndarray:
    """
    Safely normalize audio while preserving some headroom and preventing clipping.
    """
    if len(audio_data) == 0:
        return audio_data

    # Convert headroom to linear gain
    headroom_linear = 10 ** (-headroom_db / 20)

    # Find the maximum absolute amplitude
    max_amp = np.max(np.abs(audio_data))

    if max_amp > 0:
        # Normalize with headroom
        normalized = audio_data * (headroom_linear / max_amp)
        # Ensure no clipping
        normalized = np.clip(normalized, -1.0, 1.0)
        return normalized
    return audio_data


def detect_speech_segments(
    audio_data: np.ndarray,
    sample_rate: int,
    min_silence_duration_ms: int = 500,
    speech_pad_ms: int = 400,
    vad_threshold: float = 0.5,
) -> List[Tuple[float, float]]:
    """
    Detect speech segments using energy-based VAD.
    For better results, consider using the 'brouhaha' library as used in the reference implementation.
    """
    # Convert parameters to samples
    min_silence_samples = int(min_silence_duration_ms * sample_rate / 1000)
    pad_samples = int(speech_pad_ms * sample_rate / 1000)

    # Calculate frame energy
    frame_length = int(0.025 * sample_rate)  # 25ms frames
    hop_length = int(0.010 * sample_rate)  # 10ms hop

    # Calculate energy for each frame
    energy = np.array(
        [
            np.sum(audio_data[i : i + frame_length] ** 2)
            for i in range(0, len(audio_data) - frame_length, hop_length)
        ]
    )

    # Normalize energy
    if len(energy) > 0:
        energy = energy / np.max(energy)

    # Apply threshold
    speech_frames = energy > vad_threshold

    # Find speech segments
    segments = []
    in_speech = False
    start_frame = 0

    for i in range(len(speech_frames)):
        if speech_frames[i] and not in_speech:
            start_frame = i
            in_speech = True
        elif not speech_frames[i] and in_speech:
            # Check if silence is long enough
            silence_frames = 0
            for j in range(i, min(len(speech_frames), i + min_silence_samples)):
                if not speech_frames[j]:
                    silence_frames += 1
                else:
                    break

            if silence_frames >= min_silence_samples:
                # Convert frames to time
                start_time = (
                    max(0, (start_frame * hop_length - pad_samples)) / sample_rate
                )
                end_time = (
                    min(len(audio_data), (i * hop_length + pad_samples)) / sample_rate
                )
                segments.append((start_time, end_time))
                in_speech = False

    # Handle the case where audio ends during speech
    if in_speech:
        start_time = max(0, (start_frame * hop_length - pad_samples)) / sample_rate
        end_time = len(audio_data) / sample_rate
        segments.append((start_time, end_time))

    return segments


def process_audio_chunk(
    chunk_path: str,
    model: faster_whisper.WhisperModel,
    start_time: float = 0,
) -> List:
    """Process a single audio chunk with VAD and transcription."""
    try:
        # Load audio
        audio_data, sample_rate = sf.read(chunk_path)

        # Check if chunk is too short
        min_chunk_duration = 0.1  # 100ms
        if len(audio_data) / sample_rate < min_chunk_duration:
            print(
                f"Warning: Chunk too short ({len(audio_data) / sample_rate:.3f}s). Skipping."
            )
            return []

        # Just normalize the audio, skip denoising
        normalized_audio = normalize_audio(audio_data)
        sf.write(chunk_path, normalized_audio, sample_rate)

        # Transcribe with more lenient VAD settings
        segments, _ = model.transcribe(
            chunk_path,
            beam_size=5,
            language="en",
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=1000,  # More lenient silence detection
                speech_pad_ms=200,  # Reduced padding to avoid merging unrelated speech
                threshold=0.3,  # More lenient threshold
            ),
            word_timestamps=True,
        )

        # Convert generator to list and create new segment objects
        segments_list = []
        for seg in segments:
            # Create a new dictionary with the segment data
            segment_dict = {
                "start": seg.start + start_time,
                "end": seg.end + start_time,
                "text": seg.text,
                "words": [],
            }

            # Add word data if available
            if hasattr(seg, "words") and seg.words:
                for word in seg.words:
                    word_dict = {
                        "word": word.word,
                        "start": word.start + start_time,
                        "end": word.end + start_time,
                        "probability": word.probability,
                    }
                    segment_dict["words"].append(word_dict)

            segments_list.append(segment_dict)

        return segments_list
    except Exception as e:
        print(f"Warning: Error processing chunk: {e}. Skipping.")
        return []


def transcribe_audio(
    audio_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    chunk_duration: int = 300,
) -> List:
    """Transcribe audio using faster_whisper and return segments with word timestamps"""
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

        # Load audio using pydub to get duration
        audio = AudioSegment.from_file(str(audio_path))
        total_duration = len(audio) / 1000.0  # Convert to seconds

        # Check if audio is too short
        if total_duration < 1.0:  # Less than 1 second
            print(f"Audio file too short: {total_duration:.2f} seconds")
            return []

        all_segments = []
        chunk_overlap = 5  # 5 seconds overlap between chunks

        # Process audio in chunks
        for start_time in tqdm(
            range(0, int(total_duration), chunk_duration - chunk_overlap),
            desc="Processing chunks",
        ):
            end_time = min(start_time + chunk_duration, total_duration)

            # Skip chunks that would be too short
            if end_time - start_time < 1.0:  # Less than 1 second
                continue

            # Extract chunk with overlap
            chunk = audio[start_time * 1000 : end_time * 1000]
            chunk_path = str(Path(audio_path).parent / f"temp_chunk_{start_time}.wav")
            chunk.export(chunk_path, format="wav")

            try:
                print(f"Transcribing chunk {start_time}-{end_time} seconds...")
                segments_list = process_audio_chunk(chunk_path, model, start_time)

                # Only keep segments that start within this chunk's primary region
                # and don't cut words at chunk boundaries
                valid_segments = []
                for seg in segments_list:
                    # Skip segments that start in the overlap region
                    if seg["start"] >= end_time - chunk_overlap:
                        continue

                    # If segment ends in overlap region, keep all its words
                    # This ensures we don't cut words at chunk boundaries
                    valid_segments.append(seg)

                all_segments.extend(valid_segments)

            except Exception as e:
                print(f"Warning: Error processing chunk {start_time}-{end_time}: {e}")
                continue
            finally:
                # Clean up temporary chunk file
                try:
                    os.remove(chunk_path)
                except:
                    pass

            # Clear GPU memory after each chunk
            if device == "cuda":
                torch.cuda.empty_cache()

        if not all_segments:
            print("Warning: No segments were generated during transcription")

        # Sort segments by start time to ensure proper order
        all_segments.sort(key=lambda x: x["start"])

        return all_segments

    except Exception as e:
        print(f"Error during transcription: {e}")
        if device == "cuda":
            torch.cuda.empty_cache()
        raise


def format_time(seconds):
    """Convert seconds to SRT time format"""
    milliseconds = int((seconds - int(seconds)) * 1000)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"


def generate_srt_from_whisper(segments: List[Dict]) -> str:
    """
    Generate SRT content from Whisper segments.
    This creates complete sentence-level segments before duration-based segmentation.
    """
    srt_segments = []
    current_segment = None

    for segment in segments:
        if "words" not in segment or not segment["words"]:
            continue

        for word in segment["words"]:
            if current_segment is None:
                current_segment = {
                    "start": word["start"],
                    "end": word["end"],
                    "text": word["word"],
                }
            else:
                current_segment["end"] = word["end"]
                current_segment["text"] += " " + word["word"]

                # Check for sentence end
                if word["word"][-1] in ".!?":
                    srt_segments.append(current_segment)
                    current_segment = None

    # Add any remaining segment
    if current_segment is not None:
        srt_segments.append(current_segment)

    # Generate SRT content
    srt_content = ""
    for i, segment in enumerate(srt_segments, 1):
        srt_content += f"{i}\n{format_time(segment['start'])} --> {format_time(segment['end'])}\n{segment['text']}\n\n"

    return srt_content


def parse_srt_content(srt_content: str) -> List[Dict]:
    """
    Parse SRT content into a list of segments.
    Matches the reference implementation's approach.
    """
    segments = []
    pattern = re.compile(
        r"(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.*?)\n\n",
        re.DOTALL,
    )

    for match in pattern.finditer(srt_content):
        _, start_time, end_time, text = match.groups()

        # Convert SRT timestamps to seconds
        start_parts = list(map(int, re.split("[:,]", start_time)))
        end_parts = list(map(int, re.split("[:,]", end_time)))

        start_seconds = (
            start_parts[0] * 3600
            + start_parts[1] * 60
            + start_parts[2]
            + start_parts[3] / 1000
        )
        end_seconds = (
            end_parts[0] * 3600 + end_parts[1] * 60 + end_parts[2] + end_parts[3] / 1000
        )

        segments.append(
            {"start": start_seconds, "end": end_seconds, "text": text.strip()}
        )

    return segments


def segment_audio_from_srt(
    audio: AudioSegment,
    srt_segments: List[Dict],
    max_duration: float = 24.0,
    min_duration: float = 2.0,
) -> List[Dict]:
    """
    Segment audio based on SRT segments and target durations.
    This matches the reference implementation's approach.
    """
    total_duration = sum(seg["end"] - seg["start"] for seg in srt_segments)

    # Generate target durations
    mean = (min_duration + max_duration) / 2
    std_dev = (max_duration - min_duration) / 6
    target_durations = []
    accumulated = 0

    while accumulated < total_duration:
        duration = np.random.normal(mean, std_dev)
        duration = max(min(duration, max_duration), min_duration)

        remaining = total_duration - accumulated
        if remaining < min_duration:
            if target_durations and target_durations[-1] + remaining <= max_duration:
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

    # Adjust segments to match target durations
    adjusted_segments = []
    i = 0
    current_start = srt_segments[0]["start"]

    while i < len(srt_segments) and target_durations:
        target_duration = target_durations.pop(0)
        target_end_time = current_start + target_duration

        current_segment = {"start": current_start, "text": "", "end": current_start}

        while i < len(srt_segments):
            current_segment["text"] += " " + srt_segments[i]["text"]
            current_segment["end"] = min(
                srt_segments[i]["end"], current_start + max_duration
            )

            if (
                srt_segments[i]["end"] >= target_end_time
                or current_segment["end"] - current_segment["start"] >= max_duration
            ):
                break
            i += 1

        segment_duration = current_segment["end"] - current_segment["start"]
        if min_duration <= segment_duration <= max_duration:
            current_segment["text"] = current_segment["text"].strip()
            adjusted_segments.append(current_segment)

        i += 1
        if i < len(srt_segments):
            current_start = srt_segments[i]["start"]

    return adjusted_segments


def segment_audio(
    input_file,
    output_dir,
    max_segment_duration=24,
    min_segment_duration=2,
    pad_duration=0.5,
    target_sr=24000,
):
    """
    Segment audio file using the reference implementation's two-step approach:
    1. Create SRT file with complete sentences using Whisper
    2. Segment those sentences using Gaussian duration distribution
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

        # Step 1: Get transcription with word timestamps
        print("Transcribing audio with word timestamps...")
        whisper_segments = transcribe_audio(input_file)

        if not whisper_segments:
            raise ValueError("No segments were generated during transcription")

        # Step 2: Generate SRT content
        print("Generating SRT content...")
        srt_content = generate_srt_from_whisper(whisper_segments)

        # Step 3: Parse SRT into segments
        print("Parsing SRT content...")
        srt_segments = parse_srt_content(srt_content)

        if not srt_segments:
            raise ValueError("No segments were generated from SRT")

        # Step 4: Segment audio based on SRT segments
        print("Segmenting audio based on SRT...")
        combined_segments = segment_audio_from_srt(
            audio, srt_segments, max_segment_duration, min_segment_duration
        )

        if not combined_segments:
            raise ValueError("No segments were generated after combining")

        # Save segments
        print("Saving segments as WAV...")
        segments_saved = 0
        total_duration = 0
        min_found_duration = float("inf")
        max_found_duration = 0
        durations = []

        for segment_info in tqdm(combined_segments, desc="Saving segments"):
            try:
                duration = segment_info["end"] - segment_info["start"]

                if duration < min_segment_duration or duration > max_segment_duration:
                    print(
                        f"\nSkipping segment: duration {duration:.2f}s outside range [{min_segment_duration}-{max_segment_duration}]s"
                    )
                    continue

                start_ms = segment_info["start"] * 1000
                end_ms = segment_info["end"] * 1000
                segment = audio[start_ms:end_ms]

                # Add padding
                silence = AudioSegment.silent(duration=pad_duration * 1000)
                padded_segment = silence + segment + silence

                final_duration = len(padded_segment) / 1000
                total_duration += final_duration
                min_found_duration = min(min_found_duration, final_duration)
                max_found_duration = max(max_found_duration, final_duration)
                durations.append(final_duration)

                # Normalize audio
                normalized_segment = padded_segment.normalize()

                # Save segment and metadata
                output_path = output_dir / f"segment_{segments_saved:04d}.wav"
                normalized_segment.export(str(output_path), format="wav")

                # Save metadata
                metadata_path = output_dir / f"segment_{segments_saved:04d}.txt"
                with open(metadata_path, "w", encoding="utf-8") as f:
                    f.write(f"Text: {segment_info['text']}\n")
                    f.write(f"Duration: {duration:.2f}s\n")

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
            max_segment_duration=24,
            min_segment_duration=2,
            pad_duration=0.5,
        )
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print(f"\nError during processing: {e}")
        raise
