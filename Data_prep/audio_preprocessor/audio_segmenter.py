import os
import sys
import json
import torch
import gc
import pysrt
import logging
import numpy as np
from pathlib import Path
from pydub import AudioSegment
import argparse
from pesq import pesq
from pystoi import stoi
import librosa
from pydub import effects
import re
import unicodedata
from scipy import signal

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Audio segmentation settings
SEGMENT_SETTINGS = {
    "max_duration": 7.0,  # Maximum segment duration in seconds
    "min_duration": 2.0,  # Minimum segment duration in seconds
    "discard_threshold": 0.75,  # Discard segments shorter than this duration
    "word_boundary_buffer": 0.015,  # 150ms buffer around word boundaries
    "silence_padding": 0.05,  # 100ms silence padding at start/end
    "fade_duration": 0.015,  # 15ms fade in/out
    "max_gap": 5,  # Maximum allowed gap between segments
}

# Define global thresholds for speech detection
SPEECH_THRESHOLDS = {
    "stoi": 0.94,  # Keep strict for high intelligibility
    "pesq": 3.5,  # Increased since we're getting high quality segments
    "zcr": 0.05,  # Lowered to allow more natural speech variations
    "spec_cent": 1500,  # Lowered to allow more natural voice ranges
    "spectral_flatness": 0.02,  # Lowered - speech is naturally tonal
    "spectral_rolloff": 2000,  # Lowered to allow more voice types
    "rms_energy": 0.01,  # Minimum energy for valid speech
    "mfcc_var": None,  # We'll only use upper bound for MFCC variance
}

# Upper bounds for speech metrics
SPEECH_UPPER_BOUNDS = {
    "zcr": 0.25,  # Stricter to catch high-frequency noise
    "spec_cent": 4000,  # Stricter - most speech energy below 3kHz
    "spectral_flatness": 0.4,  # Stricter to catch white noise and hissing
    "spectral_rolloff": 7000,  # Stricter to catch high-frequency artifacts
    "rms_energy": 0.95,  # Prevent clipping
    "mfcc_var": 20000.0,  # Stricter for more natural speech patterns
    "peak_ratio": 15.0,  # Stricter to catch sudden peaks like mic taps
}


def calculate_snr(signal):
    """Calculate Signal-to-Noise Ratio using RMS-based approach"""
    eps = np.finfo(np.float32).eps

    # Calculate RMS of the signal
    signal_rms = np.sqrt(np.mean(signal**2) + eps)

    # Estimate noise by subtracting the mean (DC component)
    noise = signal - np.mean(signal)
    noise_rms = np.sqrt(np.mean(noise**2) + eps)

    # Calculate SNR
    if noise_rms < eps:
        return 100.0  # Return high SNR for very clean signals

    snr = 20 * np.log10(signal_rms / noise_rms)

    # Clip extreme values
    return np.clip(snr, -100, 100)


def calculate_si_sdr(signal):
    """Calculate Scale-Invariant Signal-to-Distortion Ratio"""
    eps = np.finfo(signal.dtype).eps
    signal = signal.reshape(-1)
    mean_signal = np.mean(signal)
    signal_norm = signal - mean_signal

    # Avoid division by zero
    if np.sum(signal_norm**2) < eps:
        return float("-inf")

    alpha = np.dot(signal_norm, signal_norm) / (np.sum(signal_norm**2) + eps)
    scaled = alpha * signal_norm
    si_sdr = 10 * np.log10(
        np.sum(scaled**2) / (np.sum((scaled - signal_norm) ** 2) + eps)
    )
    return si_sdr


def detect_untranscribed_sounds(audio_array, sample_rate, text):
    """
    Detect if there are speech sounds (like 'um', 'uh', interjections) that aren't in the transcription.
    Returns (bool, dict) indicating if untranscribed sounds were detected and detection details.
    """
    try:
        # Convert text to lowercase and remove punctuation for comparison
        text = text.lower()
        for char in ".,!?":
            text = text.replace(char, "")

        # Get speech activity using VAD-like approach
        frame_length = int(
            0.025 * sample_rate
        )  # 25ms frames (shorter for better precision)
        hop_length = int(0.010 * sample_rate)  # 10ms hop

        # Calculate energy in frames
        frames = librosa.util.frame(
            audio_array, frame_length=frame_length, hop_length=hop_length
        )
        energy = np.sum(frames**2, axis=0)

        # Normalize energy
        energy = (energy - np.min(energy)) / (np.max(energy) - np.min(energy))

        # Find silent gaps (potential locations of untranscribed sounds)
        silence_threshold = 0.15  # Increased from 0.11 to be more sensitive
        is_silence = energy < silence_threshold

        # Count speech segments (continuous non-silent regions)
        speech_segments = 1 + sum(
            1
            for i in range(1, len(is_silence))
            if not is_silence[i] and is_silence[i - 1]
        )

        # Count words in transcription (rough estimate)
        word_count = len(text.split())

        # Make detection more strict:
        # 1. For shorter text (1-3 words), allow no extra segments
        # 2. For medium text (4-8 words), allow only 1 extra segment per 6 words
        # 3. For longer text, allow only 1 extra segment per 8 words
        # This is much stricter than before
        if word_count <= 3:
            allowed_extra_segments = 0
        elif word_count <= 8:
            allowed_extra_segments = word_count // 6
        else:
            allowed_extra_segments = word_count // 8

        # Check for significant energy variations within words
        # This helps catch interjections/sounds within words
        word_boundaries = text.count(" ") + 1
        avg_frames_per_word = len(energy) / word_boundaries
        energy_variations = np.diff(energy)

        # Make variation detection more sensitive for longer segments
        variation_threshold = 0.25 if word_count <= 3 else 0.2
        significant_variations = np.sum(np.abs(energy_variations) > variation_threshold)

        # Adjust expected variations based on word count
        # Stricter for longer segments
        if word_count <= 3:
            expected_variations = (
                word_boundaries * 2
            )  # expect 2 variations per word (start/end)
        else:
            expected_variations = (
                word_boundaries * 1.5
            )  # expect fewer variations for longer text

        # Add duration-based check
        duration = len(audio_array) / sample_rate
        words_per_second = word_count / duration

        # Flag if words per second is too low (indicating possible untranscribed sounds)
        # Normal speech is typically 2-3 words per second
        words_per_second_too_low = words_per_second < 1.5 and duration > 2.0

        has_unexpected_variations = significant_variations > (
            expected_variations * 1.3
        )  # Reduced from 1.5

        # Return detection result and details
        has_untranscribed = (
            (speech_segments > word_count + allowed_extra_segments)
            or has_unexpected_variations
            or words_per_second_too_low
        )

        details = {
            "speech_segments": speech_segments,
            "word_count": word_count,
            "allowed_extra": allowed_extra_segments,
            "energy_variations": significant_variations,
            "expected_variations": expected_variations,
            "has_unexpected_variations": has_unexpected_variations,
            "words_per_second": words_per_second,
            "words_per_second_too_low": words_per_second_too_low,
        }

        if has_untranscribed:
            logger.info(
                f"Detected potential untranscribed sounds: {speech_segments} segments vs {word_count} words "
                f"(allowed extra: {allowed_extra_segments}). "
                f"Energy variations: {significant_variations} vs expected {expected_variations}. "
                f"Words per second: {words_per_second:.2f}"
            )

        return has_untranscribed, details

    except Exception as e:
        logger.warning(f"Error in untranscribed sound detection: {e}")
        return False, {}


def evaluate_audio_quality(audio_array, sample_rate):
    """Evaluate audio quality focusing on speech characteristics"""
    try:
        # Basic speech quality metrics
        # For STOI, create a degraded version of the signal as reference
        # Add some noise and minor distortion to get meaningful STOI values
        noise = np.random.normal(0, 0.01, len(audio_array))
        # Create mild distortion with bandpass filter
        nyquist = sample_rate // 2
        cutoff_low = 100  # Hz
        cutoff_high = 7000  # Hz
        b, a = signal.butter(
            4, [cutoff_low / nyquist, cutoff_high / nyquist], btype="band"
        )
        filtered_signal = signal.filtfilt(b, a, audio_array)
        # Combine filtered signal with noise
        ref_signal = filtered_signal + noise
        # Calculate STOI between original and degraded signal
        stoi_value = stoi(audio_array, ref_signal, sample_rate, extended=False)

        if sample_rate != 16000:
            audio_array_16k = librosa.resample(
                audio_array, orig_sr=sample_rate, target_sr=16000
            )
        else:
            audio_array_16k = audio_array
        pesq_value = pesq(16000, audio_array_16k, audio_array_16k, "nb")

        # Zero crossing rate (helps identify voiced vs unvoiced sounds)
        zcr = np.mean(librosa.feature.zero_crossing_rate(audio_array))

        # Spectral centroid (helps identify speech vs non-speech)
        spec_cent = np.mean(
            librosa.feature.spectral_centroid(y=audio_array, sr=sample_rate)
        )

        # New metrics for better detection
        # Spectral flatness (helps detect noise, hissing)
        spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=audio_array))

        # Spectral rolloff (helps detect sibilance, high-frequency noise)
        spectral_rolloff = np.mean(
            librosa.feature.spectral_rolloff(
                y=audio_array, sr=sample_rate, roll_percent=0.85
            )
        )

        # RMS energy (helps detect silence and loud noises)
        rms_energy = np.sqrt(np.mean(audio_array**2))

        # MFCC variance (helps detect natural speech patterns)
        mfccs = librosa.feature.mfcc(y=audio_array, sr=sample_rate, n_mfcc=13)
        mfcc_var = np.var(mfccs)

        # Peak detection for pops/clicks
        peak_ratio = np.max(np.abs(audio_array)) / (
            np.sqrt(np.mean(audio_array**2)) + 1e-6
        )

        # Add specific check for end-of-segment quality
        end_segment = audio_array[-int(sample_rate * 0.2) :]  # Last 200ms
        end_rms = np.sqrt(np.mean(end_segment**2))

        # Add specific checks for sudden peaks (like mic taps)
        frame_length = int(0.025 * sample_rate)  # 25ms frames
        hop_length = int(0.010 * sample_rate)  # 10ms hop

        # Calculate frame-wise energy
        frames = librosa.util.frame(
            audio_array, frame_length=frame_length, hop_length=hop_length
        )
        frame_energy = np.sum(frames**2, axis=0)

        # Detect sudden energy spikes
        energy_diff = np.diff(frame_energy)
        max_energy_spike = np.max(np.abs(energy_diff)) / (np.mean(frame_energy) + 1e-6)

        # Get harmonic and percussive components with adjusted parameters
        harmonic, percussive = librosa.effects.hpss(
            audio_array,
            margin=3.0,  # Increased margin for less aggressive separation
            kernel_size=31,  # Larger kernel size for smoother separation
        )

        # Calculate percussion ratio with smoothing
        # Take the RMS of both components for a more stable ratio
        harmonic_rms = np.sqrt(np.mean(harmonic**2) + 1e-6)
        percussive_rms = np.sqrt(np.mean(percussive**2) + 1e-6)
        percussion_ratio = percussive_rms / harmonic_rms

        # Calculate the temporal spread of percussive content
        # This helps distinguish between natural speech plosives and actual tapping
        perc_envelope = librosa.feature.rms(
            y=percussive, frame_length=2048, hop_length=512
        )
        perc_peaks = librosa.util.peak_pick(
            perc_envelope[0],
            pre_max=20,
            post_max=20,
            pre_avg=20,
            post_avg=20,
            delta=0.1,
            wait=10,
        )
        percussion_spread = len(perc_peaks) / (
            len(audio_array) / sample_rate
        )  # peaks per second

        # Calculate spectral contrast for better noise detection
        spectral_contrast = np.mean(
            librosa.feature.spectral_contrast(
                y=audio_array, sr=sample_rate, n_bands=6, fmin=200.0
            )
        )

        return {
            "stoi": stoi_value,
            "pesq": pesq_value,
            "zcr": zcr,
            "spec_cent": spec_cent,
            "spectral_flatness": spectral_flatness,
            "spectral_rolloff": spectral_rolloff,
            "rms_energy": rms_energy,
            "mfcc_var": mfcc_var,
            "peak_ratio": peak_ratio,
            "end_segment_rms": end_rms,
            "energy_spike": max_energy_spike,
            "percussion_ratio": percussion_ratio,
            "percussion_spread": percussion_spread,
            "spectral_contrast": spectral_contrast,
        }
    except Exception as e:
        logger.warning(f"Error calculating speech metrics: {e}")
        return None


def get_metric_explanation(metric, value, threshold=None, is_upper_bound=False):
    """Get human-readable explanation for metric failures"""
    explanations = {
        "stoi": {
            "name": "Speech Intelligibility",
            "fail_low": "Speech is not clear or intelligible enough. This might be background noise, mumbling, or non-speech sounds.",
            "range": "0 (unintelligible) to 1 (perfectly clear)",
        },
        "pesq": {
            "name": "Perceptual Speech Quality",
            "fail_low": "Audio quality is too poor. This could be distorted speech, noise, or non-speech sounds.",
            "range": "1 (bad) to 4.5 (excellent)",
        },
        "zcr": {
            "name": "Zero Crossing Rate",
            "fail_low": "Too few frequency changes - might be background noise or humming.",
            "fail_high": "Too many frequency changes - might be consonant sounds, breathing, or white noise.",
            "range": "0.1 to 0.3 (typical speech range)",
        },
        "spec_cent": {
            "name": "Spectral Centroid",
            "fail_low": "Frequency too low - might be background noise or non-speech sounds.",
            "fail_high": "Frequency too high - might be hissing, breathing, or sharp consonants.",
            "range": "2000-4000 Hz (typical speech range)",
        },
        "spectral_flatness": {
            "name": "Spectral Flatness",
            "fail_low": "Audio might contain unwanted tonal sounds.",
            "fail_high": "Too noisy or contains hissing sounds.",
            "range": "0.3-0.6 (typical speech range)",
        },
        "spectral_rolloff": {
            "name": "Spectral Rolloff",
            "fail_low": "Missing high frequencies - might be muffled speech.",
            "fail_high": "Too many high frequencies - likely contains sibilance or hissing.",
            "range": "3000-6000 Hz (typical speech range)",
        },
        "rms_energy": {
            "name": "RMS Energy",
            "fail_low": "Audio too quiet or contains silence.",
            "fail_high": "Audio too loud or contains pops/clicks.",
            "range": "0.05-0.95 (normalized range)",
        },
        "mfcc_var": {
            "name": "MFCC Variance",
            "fail_low": "Speech patterns too uniform - might be synthetic or monotone.",
            "fail_high": "Too much variation - might contain non-speech sounds.",
            "range": "0.5-5.0 (typical speech range)",
        },
        "peak_ratio": {
            "name": "Peak to RMS Ratio",
            "fail_low": "",  # We don't check for low peak ratio
            "fail_high": "Contains sudden loud peaks - likely pops, clicks, or mic hits.",
            "range": "Below 0.8 for clean speech",
        },
    }

    metric_info = explanations[metric]
    if is_upper_bound:
        return f"{metric_info['name']}: {value:.2f} > {threshold} - {metric_info['fail_high']}\nTypical range: {metric_info['range']}"
    else:
        return f"{metric_info['name']}: {value:.2f} < {threshold} - {metric_info['fail_low']}\nTypical range: {metric_info['range']}"


def is_audio_quality_acceptable(metrics, metric_failures):
    """Check if the segment contains clear speech"""
    if metrics is None:
        return False, []

    failed_metrics = []

    # Additional thresholds for new metrics
    if metrics.get("energy_spike", 0) > 5.0:  # Sudden energy spikes (like mic taps)
        explanation = "Detected sudden energy spike - possible mic tap or pop"
        failed_metrics.append(explanation)
        metric_failures["energy_spike"] = metric_failures.get("energy_spike", 0) + 1

    # More nuanced percussion check
    percussion_ratio = metrics.get("percussion_ratio", 0)
    percussion_spread = metrics.get("percussion_spread", 0)

    # Only flag if both ratio is high AND spread indicates non-speech pattern
    if (
        percussion_ratio > 0.5 and percussion_spread > 8.0
    ):  # Increased threshold and added spread check
        explanation = "Excessive percussive content with non-speech pattern"
        failed_metrics.append(explanation)
        metric_failures["percussion_ratio"] = (
            metric_failures.get("percussion_ratio", 0) + 1
        )
    # Add a warning for borderline cases
    elif percussion_ratio > 0.4 and percussion_spread > 6.0:
        logger.info(
            f"Warning: High percussion content but within acceptable range: ratio={percussion_ratio:.2f}, spread={percussion_spread:.2f}"
        )

    if metrics.get("spectral_contrast", 0) < 12:  # Lowered from 15 to be less sensitive
        explanation = "Low spectral contrast - possible background noise or non-speech"
        failed_metrics.append(explanation)
        metric_failures["spectral_contrast"] = (
            metric_failures.get("spectral_contrast", 0) + 1
        )

    # Check lower bounds
    for metric, threshold in SPEECH_THRESHOLDS.items():
        if threshold is None:
            continue
        if metrics[metric] < threshold:
            explanation = get_metric_explanation(metric, metrics[metric], threshold)
            failed_metrics.append(explanation)
            metric_failures[metric] = metric_failures.get(metric, 0) + 1

    # Check upper bounds
    for metric, upper_bound in SPEECH_UPPER_BOUNDS.items():
        if metrics[metric] > upper_bound:
            explanation = get_metric_explanation(
                metric, metrics[metric], upper_bound, True
            )
            failed_metrics.append(explanation)
            metric_failures[metric] = metric_failures.get(metric, 0) + 1

    if failed_metrics:
        logger.info("Segment failed quality checks:")
        for explanation in failed_metrics:
            logger.info(f"  {explanation}")
        return False, failed_metrics

    return True, []


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
            "large-v2",
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
    if not subs:
        return []

    current_segment = {
        "start": max(0, subs[0]["start"] - SEGMENT_SETTINGS["word_boundary_buffer"]),
        "text": subs[0]["text"],
        "end": subs[0]["end"] + SEGMENT_SETTINGS["word_boundary_buffer"],
        "original_timings": [
            (
                max(0, subs[0]["start"] - SEGMENT_SETTINGS["word_boundary_buffer"]),
                subs[0]["end"] + SEGMENT_SETTINGS["word_boundary_buffer"],
            )
        ],  # Track original timings with buffers
    }

    adjusted_segments = []
    i = 1
    while i < len(subs):
        # Calculate current segment duration
        current_duration = current_segment["end"] - current_segment["start"]

        # Calculate gap to next segment, considering word boundary buffers
        next_start_with_buffer = max(
            0, subs[i]["start"] - SEGMENT_SETTINGS["word_boundary_buffer"]
        )
        gap_to_next = next_start_with_buffer - current_segment["end"]

        # If current duration is less than discard_threshold, try to combine regardless
        # If between discard_threshold and min_duration, only combine if conditions are good
        must_combine = current_duration < SEGMENT_SETTINGS["discard_threshold"]
        should_combine = (
            (must_combine or current_duration < SEGMENT_SETTINGS["min_duration"])
            and gap_to_next <= SEGMENT_SETTINGS["max_gap"]
            and (
                subs[i]["end"]
                + SEGMENT_SETTINGS["word_boundary_buffer"]
                - current_segment["start"]
            )
            <= SEGMENT_SETTINGS["max_duration"]
        )

        if should_combine:
            # When combining segments with a gap, adjust the timing
            if gap_to_next > 0.2:  # If gap is larger than 200ms
                # Store original timing of the next segment with buffers
                current_segment["original_timings"].append(
                    (
                        max(
                            0,
                            subs[i]["start"] - SEGMENT_SETTINGS["word_boundary_buffer"],
                        ),
                        subs[i]["end"] + SEGMENT_SETTINGS["word_boundary_buffer"],
                    )
                )

                # Adjust the start time of the next segment to be 200ms after the end of current segment
                adjusted_start = current_segment["end"] + 0.2  # 200ms gap
                duration_of_next = (
                    subs[i]["end"] + SEGMENT_SETTINGS["word_boundary_buffer"]
                ) - (subs[i]["start"] - SEGMENT_SETTINGS["word_boundary_buffer"])

                # Add to current segment with adjusted timing
                current_segment["text"] += " " + subs[i]["text"]
                current_segment["end"] = adjusted_start + duration_of_next
            else:
                # If gap is small enough, just combine normally, keeping word boundary buffers
                current_segment["text"] += " " + subs[i]["text"]
                current_segment["end"] = (
                    subs[i]["end"] + SEGMENT_SETTINGS["word_boundary_buffer"]
                )
                current_segment["original_timings"].append(
                    (
                        max(
                            0,
                            subs[i]["start"] - SEGMENT_SETTINGS["word_boundary_buffer"],
                        ),
                        subs[i]["end"] + SEGMENT_SETTINGS["word_boundary_buffer"],
                    )
                )
            i += 1
        else:
            # If segment is shorter than discard_threshold, discard it
            # If it's between discard_threshold and min_duration, try to combine with next
            # If it's longer than min_duration, save it
            if current_duration < SEGMENT_SETTINGS["discard_threshold"]:
                logger.info(f"Discarding segment (duration: {current_duration:.2f}s)")
            elif current_duration >= SEGMENT_SETTINGS["min_duration"]:
                adjusted_segments.append(current_segment)
            else:
                # If we can't combine and it's too short, discard it
                logger.info(
                    f"Discarding segment that couldn't reach min duration (duration: {current_duration:.2f}s)"
                )

            # Start new segment with buffer
            current_segment = {
                "start": max(
                    0, subs[i]["start"] - SEGMENT_SETTINGS["word_boundary_buffer"]
                ),
                "text": subs[i]["text"],
                "end": subs[i]["end"] + SEGMENT_SETTINGS["word_boundary_buffer"],
                "original_timings": [
                    (
                        max(
                            0,
                            subs[i]["start"] - SEGMENT_SETTINGS["word_boundary_buffer"],
                        ),
                        subs[i]["end"] + SEGMENT_SETTINGS["word_boundary_buffer"],
                    )
                ],
            }
            i += 1

    # Handle the last segment
    last_duration = current_segment["end"] - current_segment["start"]
    if last_duration >= SEGMENT_SETTINGS["min_duration"]:
        adjusted_segments.append(current_segment)
    elif last_duration >= SEGMENT_SETTINGS["discard_threshold"]:
        # Try to combine with the previous segment if possible
        if adjusted_segments:
            prev_segment = adjusted_segments[-1]
            gap_to_prev = current_segment["start"] - prev_segment["end"]

            if gap_to_prev > 0.2:  # If gap is larger than 200ms
                # Adjust the timing to have 200ms gap
                adjusted_start = prev_segment["end"] + 0.2
                duration_of_current = current_segment["end"] - current_segment["start"]
                total_duration = (
                    adjusted_start + duration_of_current - prev_segment["start"]
                )

                if total_duration <= SEGMENT_SETTINGS["max_duration"]:
                    prev_segment["text"] += " " + current_segment["text"]
                    prev_segment["end"] = adjusted_start + duration_of_current
                    prev_segment["original_timings"].append(
                        (
                            max(
                                0,
                                current_segment["start"]
                                - SEGMENT_SETTINGS["word_boundary_buffer"],
                            ),
                            current_segment["end"]
                            + SEGMENT_SETTINGS["word_boundary_buffer"],
                        )
                    )
                else:
                    logger.info(
                        f"Discarding last segment that couldn't reach min duration (duration: {last_duration:.2f}s)"
                    )
            else:
                # If gap is small, combine normally
                total_duration = current_segment["end"] - prev_segment["start"]
                if total_duration <= SEGMENT_SETTINGS["max_duration"]:
                    prev_segment["text"] += " " + current_segment["text"]
                    prev_segment["end"] = current_segment["end"]
                    prev_segment["original_timings"].append(
                        (
                            max(
                                0,
                                current_segment["start"]
                                - SEGMENT_SETTINGS["word_boundary_buffer"],
                            ),
                            current_segment["end"]
                            + SEGMENT_SETTINGS["word_boundary_buffer"],
                        )
                    )
                else:
                    logger.info(
                        f"Discarding last segment that couldn't reach min duration (duration: {last_duration:.2f}s)"
                    )
        else:
            logger.info(
                f"Discarding last segment that couldn't reach min duration (duration: {last_duration:.2f}s)"
            )
    else:
        logger.info(f"Discarding last segment (duration: {last_duration:.2f}s)")

    return adjusted_segments


def save_failed_segment(
    audio_segment,
    segment_idx,
    text,
    start_time,
    end_time,
    failure_reason,
    quality_metrics=None,
    output_dir=None,
):
    """
    Save failed segment with detailed failure information.
    Organizes failures by specific metrics that failed.
    """
    if output_dir is None:
        raise ValueError("output_dir must be provided")

    # Create main failed segments directory if it doesn't exist
    failed_dir = Path(output_dir) / "failed_segments"
    failed_dir.mkdir(exist_ok=True)

    # Determine which metrics failed and save to respective directories
    if quality_metrics:
        failed_metrics = []
        for metric, value in quality_metrics.items():
            threshold = SPEECH_THRESHOLDS.get(metric)
            upper_bound = SPEECH_UPPER_BOUNDS.get(metric)

            if threshold and value < threshold:
                failed_metrics.append(f"{metric}_low")
            elif upper_bound and value > upper_bound:
                failed_metrics.append(f"{metric}_high")
    else:
        failed_metrics = ["unknown"]

    # If no specific metrics failed but we have a failure reason, categorize it
    if not failed_metrics and failure_reason:
        if "untranscribed sounds" in failure_reason.lower():
            failed_metrics = ["untranscribed_sounds"]
        else:
            failed_metrics = ["other_failures"]

    # Save to each relevant failure directory
    for failed_metric in failed_metrics:
        metric_dir = failed_dir / failed_metric
        metric_dir.mkdir(exist_ok=True)

        # Save audio
        output_filename = f"failed_{segment_idx}.wav"
        output_path = metric_dir / output_filename
        audio_segment.export(str(output_path), format="wav")

        # Save detailed metadata
        metadata_path = metric_dir / f"failed_{segment_idx}.txt"
        with open(metadata_path, "w", encoding="utf-8") as f:
            f.write(f"Text: {text}\n")
            f.write(f"Time: {format_time(start_time)} --> {format_time(end_time)}\n")
            f.write(f"Primary Failure Reason: {failure_reason}\n")

            # Add quality metrics if available
            if quality_metrics:
                f.write("\nQuality Metrics:\n")
                for metric, value in quality_metrics.items():
                    threshold = SPEECH_THRESHOLDS.get(metric)
                    upper_bound = SPEECH_UPPER_BOUNDS.get(metric)

                    f.write(f"{metric}: {value:.3f}")
                    if threshold and value < threshold:
                        f.write(f" (Below threshold: {threshold})")
                    elif upper_bound and value > upper_bound:
                        f.write(f" (Above threshold: {upper_bound})")
                    f.write("\n")


def is_valid_text(text):
    """
    Check if the text contains only valid English characters and basic punctuation.
    Returns (bool, str) - (is_valid, cleaned_text)
    """
    # Define allowed characters
    allowed_chars = set(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,!?-'\"() "
    )

    # Remove any leading/trailing whitespace
    text = text.strip()

    # Normalize unicode characters (e.g., convert "'" to "'")
    text = unicodedata.normalize("NFKD", text)

    # Replace smart quotes with regular quotes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(""", "'").replace(""", "'")

    # Replace multiple spaces with single space
    text = re.sub(r"\s+", " ", text)

    # Check for any characters that aren't in our allowed set
    invalid_chars = set(char for char in text if char not in allowed_chars)

    if invalid_chars:
        logger.warning(f"Found invalid characters: {invalid_chars}")
        return False, text

    # Additional checks
    if len(text) < 3:  # Too short to be meaningful
        return False, text

    if text.count('"') % 2 != 0:  # Unmatched quotes
        return False, text

    if text.count("(") != text.count(")"):  # Unmatched parentheses
        return False, text

    return True, text


def process_audio_segment(args):
    """Process a single audio segment in parallel"""
    audio_segment, segment, idx, input_file, output_dir, wavs_dir, full_audio = args
    try:
        # Calculate the actual start and end times with consideration for word boundaries
        start_ms = max(
            0, int((segment["start"] - SEGMENT_SETTINGS["word_boundary_buffer"]) * 1000)
        )
        end_ms = int((segment["end"] + SEGMENT_SETTINGS["word_boundary_buffer"]) * 1000)

        # Process each subsegment according to original timings
        processed_segments = []
        for i, (orig_start, orig_end) in enumerate(segment["original_timings"]):
            # Extract the subsegment with original timing
            subseg_start_ms = int(orig_start * 1000)
            subseg_end_ms = int(orig_end * 1000)
            subsegment = full_audio[subseg_start_ms:subseg_end_ms]

            # If not the first segment, add 200ms gap before it
            if i > 0:
                gap = AudioSegment.silent(
                    duration=200, frame_rate=full_audio.frame_rate
                )
                processed_segments.append(gap)

            processed_segments.append(subsegment)

        # Combine all processed segments
        if processed_segments:
            audio_segment = sum(processed_segments)
        else:
            audio_segment = full_audio[start_ms:end_ms]

        # Create silence with same properties as the segment
        silence_duration = int(
            SEGMENT_SETTINGS["silence_padding"] * 1000
        )  # Convert to ms
        silence = AudioSegment.silent(
            duration=silence_duration, frame_rate=audio_segment.frame_rate
        )

        # Add silence before and after the segment
        audio_segment = silence + audio_segment + silence

        # Apply a gentler fade in/out to prevent abrupt transitions
        fade_ms = int(SEGMENT_SETTINGS["fade_duration"] * 1000)  # Convert to ms
        audio_segment = audio_segment.fade_in(fade_ms).fade_out(fade_ms)

        # Normalize to EBU R128 standard
        target_dBFS = -23.0  # EBU R128 standard
        change_in_dBFS = target_dBFS - audio_segment.dBFS
        audio_segment = audio_segment.apply_gain(change_in_dBFS)

        # Apply compression with gentler settings
        audio_segment = effects.compress_dynamic_range(
            audio_segment,
            threshold=-20.0,
            ratio=3.0,
            attack=10,
            release=50,
        )

        # Convert to numpy array for quality analysis
        samples = np.array(audio_segment.get_array_of_samples())
        if audio_segment.channels == 2:
            samples = samples.reshape((-1, 2)).mean(axis=1)

        # Normalize the audio properly
        if samples.dtype.kind in "iu":  # if integer type
            samples = samples.astype(np.float32) / np.iinfo(samples.dtype).max
        else:  # if already float
            samples = samples.astype(np.float32)
            if np.abs(samples).max() > 1.0:
                samples /= np.abs(samples).max()

        # Evaluate audio quality
        quality_metrics = evaluate_audio_quality(samples, audio_segment.frame_rate)

        # Check for untranscribed sounds
        has_untranscribed, details = detect_untranscribed_sounds(
            samples, audio_segment.frame_rate, segment["text"]
        )

        # Return early if any checks fail
        if has_untranscribed:
            failure_reason = f"Contains untranscribed sounds: {details['speech_segments']} segments vs {details['word_count']} words (allowed extra: {details['allowed_extra']})"
            return None, "untranscribed", failure_reason, quality_metrics

        # Skip if quality is not acceptable
        metric_failures = {}
        is_acceptable, failed_metrics = is_audio_quality_acceptable(
            quality_metrics, metric_failures
        )
        if not is_acceptable:
            failure_reason = "\n".join(failed_metrics)
            return None, "quality", failure_reason, quality_metrics

        # Export the normalized audio segment
        output_filename = f"{Path(input_file).stem}_{idx}.wav"
        output_path = wavs_dir / output_filename
        audio_segment.export(str(output_path), format="wav")

        # Calculate actual duration including silences and buffers
        duration = len(audio_segment) / 1000.0  # Convert to seconds

        # Convert numpy values to Python types for JSON serialization
        quality_metrics_json = {}
        if quality_metrics:
            for metric, value in quality_metrics.items():
                if isinstance(value, (np.float32, np.float64, np.int32, np.int64)):
                    quality_metrics_json[metric] = float(value)
                else:
                    quality_metrics_json[metric] = value

        metadata = {
            "segment_id": idx - 1,
            "filename": output_filename,
            "text": segment["text"] + " $",  # Add stop token
            "duration": float(duration),
            "start_time": float(segment["start"]),
            "end_time": float(segment["end"]),
            "original_timings": segment[
                "original_timings"
            ],  # Include original timings in metadata
            "has_leading_silence": True,
            "has_trailing_silence": True,
            "has_trailing_audio": True,
            "quality_metrics": quality_metrics_json,
            "applied_settings": {
                "word_boundary_buffer": SEGMENT_SETTINGS["word_boundary_buffer"],
                "silence_padding": SEGMENT_SETTINGS["silence_padding"],
                "fade_duration": SEGMENT_SETTINGS["fade_duration"],
            },
        }

        return metadata, None, None, quality_metrics

    except Exception as e:
        return None, "error", str(e), None


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
        failed_dir = output_dir / "failed_segments"
        srt_dir = output_dir / "srts"
        wavs_dir.mkdir(parents=True, exist_ok=True)
        failed_dir.mkdir(parents=True, exist_ok=True)
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
        skipped_untranscribed = 0
        skipped_invalid_text = 0  # New counter for invalid text
        metric_failures = {
            **{metric: 0 for metric in SPEECH_THRESHOLDS.keys()},
            **{metric: 0 for metric in SPEECH_UPPER_BOUNDS.keys()},
        }
        quality_metrics_log = []
        print("\n=== Processing Audio Segments ===")
        print(f"Processing {len(adjusted_segments)} segments...")

        for idx, segment in enumerate(adjusted_segments, 1):
            # Check text validity first
            is_valid, cleaned_text = is_valid_text(segment["text"])
            if not is_valid:
                failure_reason = "Contains invalid characters or malformed text"
                save_failed_segment(
                    audio[int(segment["start"] * 1000) : int(segment["end"] * 1000)],
                    idx,
                    segment["text"],
                    segment["start"],
                    segment["end"],
                    failure_reason,
                    None,
                    output_dir,
                )
                skipped_invalid_text += 1
                continue

            # Update segment text with cleaned version
            segment["text"] = cleaned_text

            start_ms = segment["start"] * 1000
            end_ms = segment["end"] * 1000
            duration = (end_ms - start_ms) / 1000

            # Skip segments that are too long
            if duration > 7:
                skipped_too_long += 1
                continue

            # Process the segment using the improved function
            result = process_audio_segment(
                (None, segment, idx, input_file, output_dir, wavs_dir, audio)
            )
            metadata_entry, failure_type, failure_reason, quality_metrics = result

            if metadata_entry is None:
                if failure_type == "untranscribed":
                    skipped_untranscribed += 1
                save_failed_segment(
                    audio[int(segment["start"] * 1000) : int(segment["end"] * 1000)],
                    idx,
                    segment["text"],
                    segment["start"],
                    segment["end"],
                    failure_reason,
                    quality_metrics,
                    output_dir,
                )
                continue

            metadata.append(metadata_entry)
            segment_durations.append(metadata_entry["duration"])

        if not metadata:
            print("\n=== Quality Metrics Summary ===")
            print(f"All {len(quality_metrics_log)} segments were rejected.")
            print("\nDetailed failures by segment:")
            for log in quality_metrics_log:
                idx = log["segment_idx"]
                text = log["text"]
                start = format_time(log["start_time"])
                end = format_time(log["end_time"])
                print(f"\nSegment {idx}:")
                print(f"  Text: {text}")
                print(f"  Time: {start} --> {end}")
                print("  Metrics with explanations:")

                metrics = log["metrics"]
                for metric, value in metrics.items():
                    threshold = SPEECH_THRESHOLDS.get(metric)
                    upper_bound = SPEECH_UPPER_BOUNDS.get(metric)

                    if threshold and value < threshold:
                        explanation = get_metric_explanation(metric, value, threshold)
                        print(f"    - {explanation}")
                    elif upper_bound and value > upper_bound:
                        explanation = get_metric_explanation(
                            metric, value, upper_bound, True
                        )
                        print(f"    - {explanation}")
                    else:
                        print(f"    - {metric}: {value:.2f} (PASSED)")

            raise ValueError(
                "No valid segments were generated - all segments failed quality checks"
            )

        # Print detailed statistics
        print(f"\nSegments processed:")
        print(f"  Total segments: {len(adjusted_segments)}")
        print(f"  Successfully processed: {len(metadata)}")
        print(f"  Skipped (too long): {skipped_too_long}")
        print(f"  Skipped (untranscribed sounds): {skipped_untranscribed}")
        print(f"  Skipped (invalid text): {skipped_invalid_text}")

        print("\nText Processing:")
        total_invalid = skipped_invalid_text
        if total_invalid > 0:
            print(f"  Total segments with invalid text: {total_invalid}")
            print("  Common issues:")
            print("    - Non-English characters")
            print("    - Unmatched quotes or parentheses")
            print("    - Text too short (< 3 characters)")
            print("  Failed segments saved in:")
            print(f"    {failed_dir}/invalid_text/")

        print("\nFailures by metric:")
        total_metric_failures = 0
        for metric in sorted(metric_failures.keys()):
            if metric_failures[metric] > 0:
                if (
                    metric in SPEECH_THRESHOLDS
                    and SPEECH_THRESHOLDS[metric] is not None
                ):
                    print(
                        f"  - {metric}_low: {metric_failures[metric]} segments (below {SPEECH_THRESHOLDS[metric]})"
                    )
                    total_metric_failures += metric_failures[metric]
                if metric in SPEECH_UPPER_BOUNDS:
                    print(
                        f"  - {metric}_high: {metric_failures[metric]} segments (above {SPEECH_UPPER_BOUNDS[metric]})"
                    )
                    total_metric_failures += metric_failures[metric]

        print(f"\nTotal metric failures: {total_metric_failures}")
        print("Note: Some segments may fail multiple metrics")

        print("\nFailed segments organized by failure type in:")
        print(f"  {failed_dir}/[metric_name]_[low/high]")
        print("Examples of failure directories:")
        print("  - stoi_low: Speech intelligibility too low")
        print("  - spectral_flatness_high: Too much noise/hissing")
        print("  - peak_ratio_high: Contains pops/clicks")
        print("  - untranscribed_sounds: Contains untranscribed sounds")
        print("  - other_failures: Other types of failures")

        # Calculate statistics
        segment_durations = np.array(segment_durations)
        total_segmented_duration = float(
            np.sum(segment_durations)
        )  # Convert to regular float
        mean_duration = float(np.mean(segment_durations))
        std_duration = float(np.std(segment_durations))
        min_duration = float(np.min(segment_durations))
        max_duration = float(np.max(segment_durations))

        # Save metadata
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print("\n=== Processing Summary ===")
        print(f"Total segments saved: {len(metadata)}")
        print(f"Total input duration: {total_duration:.2f} seconds")
        print(f"Total segmented duration: {total_segmented_duration:.2f} seconds")
        print(f"\nDuration distribution (including 100ms silence + 30ms extra):")
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
