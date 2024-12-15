import os
import numpy as np
import torch
import torchaudio
from tqdm import tqdm
from pathlib import Path
from scipy import signal


def detect_speech(audio, sr, threshold_db=-35, min_silence_duration=0.3):
    """
    Detect speech segments using bandpass filtering for speech frequencies
    and energy detection
    """
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=0)

    # Apply bandpass filter to focus on speech frequencies (80-300 Hz for fundamental frequency)
    nyquist = sr / 2
    low = 80 / nyquist
    high = 300 / nyquist
    b, a = signal.butter(4, [low, high], btype="band")
    filtered_audio = signal.filtfilt(b, a, audio)

    # Calculate RMS energy in small windows
    window_size = int(0.02 * sr)  # 20ms windows
    hop_size = window_size // 2
    n_windows = (len(filtered_audio) - window_size) // hop_size + 1

    # Vectorized energy calculation
    windows = np.lib.stride_tricks.sliding_window_view(filtered_audio, window_size)[
        ::hop_size
    ][:n_windows]
    energy = np.sqrt(np.mean(windows**2, axis=1))

    # Convert to dB
    energy_db = 20 * np.log10(energy + 1e-10)

    # Smooth the energy curve
    window_length = int(0.1 * sr / hop_size)  # 100ms smoothing
    if window_length % 2 == 0:
        window_length += 1
    energy_smooth = signal.savgol_filter(energy_db, window_length, 2)

    # Find speech regions
    is_speech = energy_smooth > threshold_db

    # Convert windows to sample indices
    boundaries = np.where(np.diff(is_speech.astype(int)))[0]
    boundaries = boundaries * hop_size

    if len(boundaries) == 0:
        if is_speech[0]:
            return [(0, len(audio))]
        return []

    segments = []
    start_idx = 0 if is_speech[0] else boundaries[0]

    for idx in boundaries[1:]:
        if idx - start_idx >= min_silence_duration * sr:
            segments.append((start_idx, idx))
        start_idx = idx

    if is_speech[-1]:
        segments.append((start_idx, len(audio)))

    return segments


def add_padding(audio, sr, pad_duration=0.5):
    """Add silence padding to audio"""
    pad_length = int(pad_duration * sr)
    return np.pad(audio, (pad_length, pad_length), mode="constant")


def group_short_segments(
    segments,
    sr,
    min_duration=8.0,
    max_duration=30.0,
    min_valid_duration=0.5,
    pad_duration=0.5,
):
    """
    Group short segments together until they reach minimum duration.
    - min_duration: minimum duration for final segments (default 8 seconds)
    - max_duration: maximum duration for final segments (default 30 seconds)
    - min_valid_duration: minimum duration for a segment to be considered for grouping (default 0.5 seconds)
    - pad_duration: duration of silence padding between combined segments (default 0.5 seconds)
    """
    grouped_segments = []
    current_group = []
    current_duration = 0
    pad_samples = int(pad_duration * sr)

    # First pass: filter out segments that are too short to be valid
    valid_segments = []
    for start, end in segments:
        duration = (end - start) / sr
        if duration >= min_valid_duration:
            valid_segments.append((start, end, duration))

    if not valid_segments:
        return []

    # Sort segments by duration, longer segments first
    valid_segments.sort(key=lambda x: x[2], reverse=True)

    # First, add all segments that are already long enough
    for start, end, duration in valid_segments:
        if duration >= min_duration:
            grouped_segments.append((start, end))

    # Filter out segments that were already added
    remaining_segments = [
        (start, end, duration)
        for start, end, duration in valid_segments
        if duration < min_duration
    ]

    # Now process remaining segments
    current_group = []
    current_duration = 0

    for start, end, duration in remaining_segments:
        # Calculate total duration including padding
        total_padding = pad_duration * (len(current_group))  # Padding between segments
        potential_duration = current_duration + duration + total_padding

        if potential_duration > max_duration and current_group:
            # Save current group if it meets minimum duration
            if current_duration >= min_duration:
                group_start = current_group[0][0]
                group_end = current_group[-1][1]
                grouped_segments.append((group_start, group_end))
            current_group = [(start, end)]
            current_duration = duration
        else:
            current_group.append((start, end))
            current_duration = potential_duration

            # If we've reached minimum duration, save the group
            if current_duration >= min_duration:
                group_start = current_group[0][0]
                group_end = current_group[-1][1]
                grouped_segments.append((group_start, group_end))
                current_group = []
                current_duration = 0

    # Handle remaining group
    if current_group:
        # Try to append to the last group if possible
        if grouped_segments and current_duration < min_duration:
            last_duration = (grouped_segments[-1][1] - grouped_segments[-1][0]) / sr
            total_duration = last_duration + current_duration + pad_duration

            if total_duration <= max_duration:
                # Extend last group
                grouped_segments[-1] = (grouped_segments[-1][0], current_group[-1][1])
            else:
                # Save as separate group even if it's shorter than minimum
                group_start = current_group[0][0]
                group_end = current_group[-1][1]
                grouped_segments.append((group_start, group_end))
        else:
            # Save as separate group
            group_start = current_group[0][0]
            group_end = current_group[-1][1]
            grouped_segments.append((group_start, group_end))

    return grouped_segments


def segment_audio(
    input_file,
    output_dir,
    max_segment_duration=30,  # in seconds
    min_segment_duration=8,  # in seconds
    min_valid_duration=0.5,  # in seconds
    pad_duration=0.5,  # in seconds
    target_sr=24000,
    threshold_db=-30,
):
    """
    Segment an audio file into chunks based on speech detection:
    1. Load audio file
    2. Detect speech segments
    3. Group short segments together with padding
    4. Split long segments and save as WAV files
    """
    try:
        # Convert paths to Path objects
        input_file = Path(input_file)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print("Loading audio file...")
        wav, sr = torchaudio.load(str(input_file))

        # Convert to mono if stereo
        if wav.size(0) > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)

        # Resample if necessary
        if sr != target_sr:
            print(f"Resampling from {sr} to {target_sr}...")
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            wav = resampler(wav)

        # Convert to numpy for processing
        audio = wav.squeeze().numpy()

        # Detect speech segments
        print("Detecting speech segments...")
        raw_segments = detect_speech(audio, target_sr, threshold_db=threshold_db)

        # Group short segments
        print("Grouping segments...")
        segments = group_short_segments(
            raw_segments,
            target_sr,
            min_duration=min_segment_duration,
            max_duration=max_segment_duration,
            min_valid_duration=min_valid_duration,
            pad_duration=pad_duration,
        )

        # Save segments
        print("Saving segments as WAV...")
        segments_saved = 0
        total_duration = 0

        for start_sample, end_sample in tqdm(segments, desc="Saving segments"):
            duration = (end_sample - start_sample) / target_sr
            total_duration += duration

            # Split if too long
            if duration > max_segment_duration:
                n_splits = int(np.ceil(duration / max_segment_duration))
                split_size = int((end_sample - start_sample) / n_splits)

                for i in range(n_splits):
                    split_start = start_sample + i * split_size
                    split_end = min(split_start + split_size, end_sample)

                    # Create segment and ensure it's 2D [channels, samples]
                    segment = torch.FloatTensor(audio[split_start:split_end])
                    if segment.dim() == 1:
                        segment = segment.unsqueeze(0)  # Add channel dimension

                    # Add padding
                    segment_np = segment.numpy().squeeze()
                    padded_segment = add_padding(segment_np, target_sr, pad_duration)
                    segment = torch.FloatTensor(padded_segment).unsqueeze(0)

                    # Normalize audio to prevent clipping
                    if segment.numel() > 0:  # Check if segment is not empty
                        max_val = torch.abs(segment).max()
                        if max_val > 0:
                            segment = segment / max_val * 0.9

                    # Save segment as WAV
                    output_path = output_dir / f"segment_{segments_saved:04d}.wav"
                    torchaudio.save(
                        str(output_path),
                        segment,
                        target_sr,
                        encoding="PCM_S",
                        bits_per_sample=16,
                    )
                    segments_saved += 1
            else:
                # Create segment and ensure it's 2D [channels, samples]
                segment = torch.FloatTensor(audio[start_sample:end_sample])
                if segment.dim() == 1:
                    segment = segment.unsqueeze(0)  # Add channel dimension

                # Add padding
                segment_np = segment.numpy().squeeze()
                padded_segment = add_padding(segment_np, target_sr, pad_duration)
                segment = torch.FloatTensor(padded_segment).unsqueeze(0)

                # Normalize audio to prevent clipping
                if segment.numel() > 0:  # Check if segment is not empty
                    max_val = torch.abs(segment).max()
                    if max_val > 0:
                        segment = segment / max_val * 0.9

                # Save segment as WAV
                output_path = output_dir / f"segment_{segments_saved:04d}.wav"
                torchaudio.save(
                    str(output_path),
                    segment,
                    target_sr,
                    encoding="PCM_S",
                    bits_per_sample=16,
                )
                segments_saved += 1

        print("\nProcessing complete!")
        print(f"Total segments saved: {segments_saved}")
        print(f"Total audio duration: {total_duration:.2f} seconds")
        print(f"Average segment duration: {total_duration/segments_saved:.2f} seconds")
        print(
            f"Segment duration range: {min_segment_duration}-{max_segment_duration} seconds"
        )

    except Exception as e:
        print(f"\nError during processing: {e}")
        raise


if __name__ == "__main__":
    try:
        # Input should be the output from vocal_isolator.py
        input_file = "Data_prep/raw_data/full_audio/amelia_stream_vocals.wav"
        output_dir = "Data_prep/raw_data/2hour_amelia"

        segment_audio(
            input_file,
            output_dir,
            max_segment_duration=30,
            min_segment_duration=8,
            min_valid_duration=0.5,
            pad_duration=0.5,
            threshold_db=-30,  # Adjust this if needed
        )
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print(f"\nError during processing: {e}")
        raise
