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
from tqdm import tqdm
import subprocess
import whisper
import datasets
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global duration settings
DISCARD_THRESHOLD = 1.0  # Discard segments shorter than 1 second
MIN_DURATION = 3.0  # Minimum segment duration in seconds
MAX_DURATION = 7.5  # Maximum segment duration in seconds
COMBINE_SILENCE_GAP = 1.0  # Silence duration between combined segments in seconds

# Audio segmentation settings
SEGMENT_SETTINGS = {
    "silence_padding": 0.1,  # 100ms silence padding at start/end
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


# Initialize Silero VAD model (done once)
def init_silero_vad():
    """Initialize the Silero VAD model"""
    try:
        if not hasattr(init_silero_vad, "model"):
            # Download and load the model
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=False,
            )

            # Store model and get_speech_timestamps function
            init_silero_vad.model = model
            init_silero_vad.get_speech_timestamps = utils[0]

            # Move model to GPU if available
            if torch.cuda.is_available():
                init_silero_vad.model = init_silero_vad.model.cuda()

            # Set model to evaluation mode
            init_silero_vad.model.eval()

            logger.info("Initialized Silero VAD model successfully")
            return True
    except Exception as e:
        logger.error(f"Failed to initialize Silero VAD: {e}")
        return False


def detect_speech_silero(waveform, sample_rate=16000, threshold=0.3):
    """
    Detect speech segments using Silero VAD with confidence scoring

    Args:
        waveform (numpy.ndarray): Audio waveform
        sample_rate (int): Sample rate of the audio
        threshold (float): VAD threshold (0 to 1)

    Returns:
        tuple: (has_speech, speech_timestamps, confidences)
    """
    try:
        # Initialize model if not done yet
        if not hasattr(init_silero_vad, "model"):
            if not init_silero_vad():
                return False, [], []

        # Ensure correct sample rate (Silero VAD expects 16kHz)
        if sample_rate != 16000:
            waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000)

        # Convert to torch tensor
        wav_tensor = torch.FloatTensor(waveform)

        # Move to GPU if available
        if torch.cuda.is_available():
            wav_tensor = wav_tensor.cuda()

        # Get speech timestamps with confidence scores
        speech_timestamps = init_silero_vad.get_speech_timestamps(
            wav_tensor,
            init_silero_vad.model,
            threshold=threshold,
            sampling_rate=16000,
            min_speech_duration_ms=250,
            min_silence_duration_ms=500,
            window_size_samples=512,  # Must be 512 for 16kHz audio
            speech_pad_ms=30,
            return_seconds=True,
        )

        has_speech = len(speech_timestamps) > 0
        if not has_speech:
            return False, [], []

        # Get confidence scores for each segment
        confidences = []
        for ts in speech_timestamps:
            start_frame = int(ts["start"] * 16000)
            end_frame = int(ts["end"] * 16000)
            segment = wav_tensor[start_frame:end_frame]

            # Get confidence score using sliding windows
            window_size = 512  # Must match Silero VAD requirements
            hop_length = 256  # Half window size for 50% overlap
            scores = []

            for i in range(0, len(segment) - window_size, hop_length):
                window = segment[i : i + window_size]
                if len(window) == window_size:  # Only process full windows
                    with torch.no_grad():
                        score = init_silero_vad.model(window, 16000).item()
                        scores.append(score)

            # Use average confidence score for the segment
            avg_confidence = sum(scores) / len(scores) if scores else 0
            confidences.append(avg_confidence)

        return True, speech_timestamps, confidences

    except Exception as e:
        logger.error(f"Error in Silero VAD detection: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return False, [], []


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


def contains_non_english(text):
    """Check if text contains non-English characters"""
    try:
        # Define allowed characters (English letters, numbers, and basic punctuation)
        allowed = set(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?'\"-() "
        )
        # Check if any character is not in allowed set
        return any(c not in allowed for c in text)
    except Exception as e:
        logger.error(f"Error checking text for non-English characters: {e}")
        return True  # Return True to be safe


def transcribe_with_whisperx(audio_file_path, output_dir, batch_size=8):
    """
    Transcribe audio using insanely-fast-whisper with distil-large-v2.
    """
    try:
        # Get base name without extension
        base_name = Path(audio_file_path).stem
        output_txt = Path(output_dir) / f"{base_name}.txt"

        # Skip if already transcribed
        if output_txt.exists():
            with open(output_txt, "r", encoding="utf-8") as f:
                text = f.read().strip()
                if text and not contains_non_english(
                    text
                ):  # Only return True if we have valid English text
                    return True, text

        # Initialize model (only done once)
        if not hasattr(transcribe_with_whisperx, "model"):
            from transformers import pipeline
            from transformers.utils import is_flash_attn_2_available

            transcribe_with_whisperx.model = pipeline(
                "automatic-speech-recognition",
                model="distil-whisper/large-v2",  # Using distilled model for better speed
                torch_dtype=torch.float16,
                device="cuda" if torch.cuda.is_available() else "cpu",
                model_kwargs=(
                    {"attn_implementation": "flash_attention_2"}
                    if is_flash_attn_2_available()
                    else {"attn_implementation": "sdpa"}
                ),
            )
            logger.info("Initialized insanely-fast-whisper model (distil-large-v2)")

        # Load audio file
        audio = librosa.load(audio_file_path, sr=16000)[0]

        # Transcribe audio with optimized settings
        result = transcribe_with_whisperx.model(
            {"array": audio, "sampling_rate": 16000},
            batch_size=1,  # Single batch since we're processing one file at a time
            generate_kwargs={"task": "transcribe", "language": "en"},
        )

        # Get transcription text
        text = result["text"].strip()

        # Check for empty text or non-English characters
        if not text or contains_non_english(text):
            logger.warning(
                f"No valid English transcription detected for {audio_file_path}"
            )
            return False, None

        # Clean up the text
        text = re.sub(r"\s+", " ", text)  # Replace multiple spaces with single space
        text = text.strip()

        # Create output directory if it doesn't exist
        output_txt.parent.mkdir(parents=True, exist_ok=True)

        # Save transcription
        with open(output_txt, "w", encoding="utf-8") as f:
            f.write(text)

        # Clear GPU memory after each file
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        return True, text

    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        return False, None


def process_batch_segments(batch_info):
    """Process a batch of segments to be combined"""
    try:
        segments, output_path = batch_info
        return combine_short_segments(segments, output_path)
    except Exception as e:
        logger.error(f"Error processing batch: {e}")
        return None


def process_segments(input_dir, output_dir):
    """Process audio files in multiple phases"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    try:
        # Create output directories
        wavs_dir = output_dir / "wavs"
        passed_dir = output_dir / "passed_segments"
        failed_dir = output_dir / "failed_segments"
        srt_dir = output_dir / "srts"
        for dir_path in [wavs_dir, passed_dir, failed_dir, srt_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Create directory for segments with no transcription
        no_transcription_dir = failed_dir / "no_transcription_detected"
        no_transcription_dir.mkdir(parents=True, exist_ok=True)

        # Check if we have existing processed files
        existing_segments = list(wavs_dir.glob("*.wav"))
        if not existing_segments:
            # PHASE 1: Initial Processing
            logger.info("\nPhase 1: Initial Processing")
            logger.info("Loading and analyzing segments...")

            # Get list of all wav files
            wav_files = list(input_dir.glob("*.wav"))

            # Process files in parallel
            all_segments = []
            discarded_segments = []

            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                # Submit all files for processing
                future_to_file = {
                    executor.submit(load_and_analyze_segment, wav_file): wav_file
                    for wav_file in wav_files
                }

                # Process results as they complete
                for future in tqdm(
                    as_completed(future_to_file),
                    total=len(wav_files),
                    desc="Loading segments",
                ):
                    wav_file = future_to_file[future]
                    try:
                        segment_info = future.result()
                        if segment_info:
                            if segment_info["duration"] < DISCARD_THRESHOLD:
                                discarded_segments.append(segment_info)
                            else:
                                all_segments.append(segment_info)
                    except Exception as e:
                        logger.error(f"Error processing {wav_file}: {e}")

            if not all_segments:
                raise ValueError("No valid segments found in input directory")

            logger.info(
                f"Discarded {len(discarded_segments)} segments shorter than {DISCARD_THRESHOLD}s"
            )

            # Process valid segments in parallel
            logger.info("Processing valid segments...")
            valid_segments = []
            long_segments = []
            short_segments = []

            # First categorize segments by duration
            for segment in all_segments:
                if segment["duration"] > MAX_DURATION:
                    long_segments.append(segment)
                elif segment["duration"] < MIN_DURATION:
                    short_segments.append(segment)
                else:
                    valid_segments.append(segment)

            # Process valid segments in parallel
            def process_valid_segment(segment):
                """Process a single valid segment"""
                try:
                    output_path = wavs_dir / segment["file_path"].name
                    audio = segment["audio"]

                    # Pre-create silence segments (more efficient than creating for each segment)
                    if not hasattr(process_valid_segment, "silence_padding"):
                        process_valid_segment.silence_padding = AudioSegment.silent(
                            duration=int(SEGMENT_SETTINGS["silence_padding"] * 1000),
                            frame_rate=audio.frame_rate,
                        )

                    # Add silence padding
                    audio = (
                        process_valid_segment.silence_padding
                        + audio
                        + process_valid_segment.silence_padding
                    )

                    # Batch process audio modifications
                    # Normalize and compress in one pass
                    target_dBFS = -23.0
                    change_in_dBFS = target_dBFS - audio.dBFS
                    audio = audio.apply_gain(change_in_dBFS)
                    audio = effects.compress_dynamic_range(
                        audio, threshold=-20.0, ratio=3.0, attack=10, release=50
                    )

                    # Export with optimized settings
                    audio.export(
                        str(output_path),
                        format="wav",
                        parameters=["-ar", str(audio.frame_rate), "-ac", "1"],
                    )

                    return {
                        "audio": audio,
                        "duration": len(audio) / 1000.0,
                        "file_path": output_path,
                        "original_file": segment["file_path"].name,
                    }
                except Exception as e:
                    logger.error(
                        f"Error processing valid segment {segment['file_path'].name}: {e}"
                    )
                    return None

            # Process valid segments in parallel with optimized batch size
            processed_valid_segments = []
            batch_size = os.cpu_count() * 2  # Limit concurrent processing
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                futures = [
                    executor.submit(process_valid_segment, segment)
                    for segment in valid_segments
                ]

                for future in tqdm(
                    as_completed(futures),
                    total=len(valid_segments),
                    desc="Processing valid segments",
                ):
                    result = future.result()
                    if result:
                        processed_valid_segments.append(result)

            logger.info(f"Found {len(long_segments)} long segments (> {MAX_DURATION}s)")
            logger.info(
                f"Found {len(short_segments)} short segments ({DISCARD_THRESHOLD}s - {MIN_DURATION}s)"
            )
            logger.info(f"Found {len(processed_valid_segments)} valid segments")

            # Combine short segments in parallel
            if short_segments:
                logger.info("Combining short segments...")
                batches_to_process = []
                current_batch = []
                combined_duration = 0
                batch_count = 0

                for segment in short_segments:
                    if (
                        combined_duration + segment["duration"] + COMBINE_SILENCE_GAP
                        <= MAX_DURATION
                    ):
                        current_batch.append(segment)
                        combined_duration += segment["duration"] + COMBINE_SILENCE_GAP
                    else:
                        if current_batch:
                            combined_path = wavs_dir / f"combined_{batch_count}.wav"
                            batches_to_process.append((current_batch, combined_path))
                            batch_count += 1
                        current_batch = [segment]
                        combined_duration = segment["duration"]

                # Handle last batch
                if current_batch:
                    combined_path = wavs_dir / f"combined_{batch_count}.wav"
                    batches_to_process.append((current_batch, combined_path))

                # Process batches in parallel
                combined_segments = []
                with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                    futures = [
                        executor.submit(process_batch_segments, batch_info)
                        for batch_info in batches_to_process
                    ]

                    for future, (batch, path) in zip(futures, batches_to_process):
                        combined_audio = future.result()
                        if combined_audio:
                            combined_segments.append(
                                {
                                    "audio": combined_audio,
                                    "duration": len(combined_audio) / 1000.0,
                                    "file_path": path,
                                    "original_segments": [
                                        s["file_path"].name for s in batch
                                    ],
                                }
                            )

                logger.info(f"Created {len(combined_segments)} combined segments")
                valid_segments.extend(combined_segments)

            # Update existing_segments for next phase
            existing_segments = list(wavs_dir.glob("*.wav"))
            logger.info(
                f"Total segments ready for processing: {len(existing_segments)}"
            )

        if (
            existing_segments
        ):  # This will run whether we just created segments or they existed already
            logger.info(f"Found {len(existing_segments)} segments to process")

            # PHASE 2: Transcription
            logger.info("\nPhase 2: Transcription")
            for segment_path in tqdm(existing_segments, desc="Transcribing files"):
                # Skip if already transcribed
                txt_path = srt_dir / f"{segment_path.stem}.txt"
                if not txt_path.exists():
                    success, text = transcribe_with_whisperx(
                        str(segment_path), str(srt_dir)
                    )
                    if not success:
                        logger.warning(f"Failed to transcribe {segment_path.name}")
                        # Move to no transcription directory
                        segment_info = load_and_analyze_segment(segment_path)
                        if segment_info:
                            no_trans_path = no_transcription_dir / segment_path.name
                            segment_info["audio"].export(
                                str(no_trans_path), format="wav"
                            )

                    # Clear GPU memory after each file
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()

            # PHASE 3: Quality Checks and Organization
            logger.info("\nPhase 3: Quality Checks and Organization")
            final_segments = []
            metric_failures = {}
            passed_durations = []  # Track durations of passed segments

            for segment_path in tqdm(
                existing_segments, desc="Processing quality checks"
            ):
                # Load audio for quality check
                segment_info = load_and_analyze_segment(segment_path)
                if not segment_info:
                    continue

                # Get transcription
                txt_path = srt_dir / f"{segment_path.stem}.txt"
                if not txt_path.exists():
                    continue

                with open(txt_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()

                # Check quality metrics
                quality_metrics = evaluate_audio_quality(
                    segment_info["samples"], segment_info["audio"].frame_rate
                )

                is_acceptable, failed_metrics = is_audio_quality_acceptable(
                    quality_metrics, metric_failures
                )

                if not is_acceptable:
                    save_failed_segment(
                        segment_info["audio"],
                        segment_path.name,
                        text,
                        "\n".join(failed_metrics),
                        quality_metrics,
                        failed_dir,
                    )
                    continue

                # Check for untranscribed sounds
                if text:
                    has_untranscribed, details = detect_untranscribed_sounds(
                        segment_info["samples"],
                        segment_info["audio"].frame_rate,
                        text,
                    )

                    if has_untranscribed:
                        failure_reason = f"Contains untranscribed sounds: {details}"
                        save_failed_segment(
                            segment_info["audio"],
                            segment_path.name,
                            text,
                            failure_reason,
                            quality_metrics,
                            failed_dir,
                        )
                        continue

                # If all checks pass:
                # 1. Move to passed directory
                passed_path = passed_dir / segment_path.name
                segment_info["audio"].export(str(passed_path), format="wav")

                # 2. Add to final segments and track duration
                duration = segment_info["duration"]
                passed_durations.append(duration)
                final_segments.append(
                    {
                        "file_path": str(passed_path),
                        "duration": duration,
                        "quality_metrics": quality_metrics,
                        "text": text,
                    }
                )

            # Calculate duration statistics for passed segments
            if passed_durations:
                min_duration = min(passed_durations)
                max_duration = max(passed_durations)
                avg_duration = sum(passed_durations) / len(passed_durations)
                logger.info("\nPassed Segments Duration Statistics:")
                logger.info(f"Minimum duration: {min_duration:.2f} seconds")
                logger.info(f"Maximum duration: {max_duration:.2f} seconds")
                logger.info(f"Average duration: {avg_duration:.2f} seconds")

            # Save metadata
            metadata = [
                {
                    "filename": os.path.basename(segment["file_path"]),
                    "text": segment["text"],
                    "duration": segment["duration"],
                    "quality_metrics": segment["quality_metrics"],
                }
                for segment in final_segments
            ]

            with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            # Save summary separately for debugging
            summary = {
                "processed_segments": len(existing_segments),
                "final_segments": len(final_segments),
                "metric_failures": metric_failures,
                "duration_stats": {
                    "min": min_duration if passed_durations else None,
                    "max": max_duration if passed_durations else None,
                    "avg": avg_duration if passed_durations else None,
                },
            }
            with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            logger.info("\n=== Processing Summary ===")
            logger.info(f"Total segments processed: {len(existing_segments)}")
            logger.info(f"Segments passed quality checks: {len(final_segments)}")
            logger.info(
                f"Segments failed quality checks: {len(existing_segments) - len(final_segments)}"
            )
            logger.info(f"\nPassed segments saved to: {passed_dir}")
            logger.info(f"Failed segments saved to: {failed_dir}")
            logger.info(f"Transcriptions saved to: {srt_dir}")
            logger.info(f"Metadata saved to: {output_dir / 'metadata.json'}")

            return True

        return False

    except Exception as e:
        logger.error(f"Error during processing: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return False


def evaluate_audio_quality(samples, sample_rate):
    """Evaluate audio quality focusing on speech characteristics"""
    try:
        # Ensure samples are normalized between -1 and 1
        if np.abs(samples).max() > 1.0:
            samples = samples / np.abs(samples).max()

        # For STOI, create a degraded version of the signal as reference
        noise = np.random.normal(0, 0.01, len(samples))
        # Create mild distortion with bandpass filter
        nyquist = sample_rate // 2
        cutoff_low = 100  # Hz
        cutoff_high = 7000  # Hz
        b, a = signal.butter(
            4, [cutoff_low / nyquist, cutoff_high / nyquist], btype="band"
        )
        filtered_signal = signal.filtfilt(b, a, samples)
        # Combine filtered signal with noise
        ref_signal = filtered_signal + noise
        # Calculate STOI between original and degraded signal
        stoi_value = stoi(samples, ref_signal, sample_rate, extended=False)

        if sample_rate != 16000:
            samples_16k = librosa.resample(
                samples, orig_sr=sample_rate, target_sr=16000
            )
        else:
            samples_16k = samples
        pesq_value = pesq(16000, samples_16k, samples_16k, "nb")

        # Zero crossing rate (helps identify voiced vs unvoiced sounds)
        zcr = np.mean(librosa.feature.zero_crossing_rate(samples))

        # Spectral centroid (helps identify speech vs non-speech)
        spec_cent = np.mean(
            librosa.feature.spectral_centroid(y=samples, sr=sample_rate)
        )

        # Spectral flatness (helps detect noise, hissing)
        spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=samples))

        # Spectral rolloff (helps detect sibilance, high-frequency noise)
        spectral_rolloff = np.mean(
            librosa.feature.spectral_rolloff(
                y=samples, sr=sample_rate, roll_percent=0.85
            )
        )

        # RMS energy (helps detect silence and loud noises)
        rms_energy = np.sqrt(np.mean(samples**2))

        # MFCC variance (helps detect natural speech patterns)
        mfccs = librosa.feature.mfcc(y=samples, sr=sample_rate, n_mfcc=13)
        mfcc_var = np.var(mfccs) / 10000.0  # Scale down to match expected range

        # Peak detection for pops/clicks
        peak_ratio = np.max(np.abs(samples)) / (np.sqrt(np.mean(samples**2)) + 1e-6)

        # Add specific check for end-of-segment quality
        end_segment = samples[-int(sample_rate * 0.2) :]  # Last 200ms
        end_rms = np.sqrt(np.mean(end_segment**2))

        # Add specific checks for sudden peaks (like mic taps)
        frame_length = int(0.025 * sample_rate)  # 25ms frames
        hop_length = int(0.010 * sample_rate)  # 10ms hop

        # Calculate frame-wise energy
        frames = librosa.util.frame(
            samples, frame_length=frame_length, hop_length=hop_length
        )
        frame_energy = np.sum(frames**2, axis=0)

        # Detect sudden energy spikes
        energy_diff = np.diff(frame_energy)
        max_energy_spike = np.max(np.abs(energy_diff)) / (np.mean(frame_energy) + 1e-6)

        # Get harmonic and percussive components
        harmonic, percussive = librosa.effects.hpss(
            samples,
            margin=3.0,  # Increased margin for less aggressive separation
            kernel_size=31,  # Larger kernel size for smoother separation
        )

        # Calculate percussion ratio with smoothing
        harmonic_rms = np.sqrt(np.mean(harmonic**2) + 1e-6)
        percussive_rms = np.sqrt(np.mean(percussive**2) + 1e-6)
        percussion_ratio = percussive_rms / harmonic_rms

        # Calculate the temporal spread of percussive content
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
            len(samples) / sample_rate
        )  # peaks per second

        # Calculate spectral contrast for better noise detection
        spectral_contrast = np.mean(
            librosa.feature.spectral_contrast(
                y=samples, sr=sample_rate, n_bands=6, fmin=200.0
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


def is_audio_quality_acceptable(metrics, metric_failures):
    """Check if the segment contains clear speech"""
    if metrics is None:
        return False, []

    failed_metrics = []

    # Check for sudden energy spikes (like mic taps)
    if metrics.get("energy_spike", 0) > 5.0:
        explanation = "Detected sudden energy spike - possible mic tap or pop"
        failed_metrics.append(explanation)
        metric_failures["energy_spike"] = metric_failures.get("energy_spike", 0) + 1

    # More nuanced percussion check
    percussion_ratio = metrics.get("percussion_ratio", 0)
    percussion_spread = metrics.get("percussion_spread", 0)

    # Only flag if both ratio is high AND spread indicates non-speech pattern
    if percussion_ratio > 0.5 and percussion_spread > 8.0:
        explanation = "Excessive percussive content with non-speech pattern"
        failed_metrics.append(explanation)
        metric_failures["percussion_ratio"] = (
            metric_failures.get("percussion_ratio", 0) + 1
        )
    elif percussion_ratio > 0.4 and percussion_spread > 6.0:
        logger.info(
            f"Warning: High percussion content but within acceptable range: ratio={percussion_ratio:.2f}, spread={percussion_spread:.2f}"
        )

    # Check spectral contrast for noise detection
    if metrics.get("spectral_contrast", 0) < 12:
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


def detect_non_speech_sounds(samples, sample_rate):
    """
    Detect non-speech sounds (clicks, pops, background noise, etc.)
    Returns: (bool, dict) - has_non_speech and details
    """
    try:
        # Calculate spectral features
        spec = np.abs(librosa.stft(samples))

        # Spectral centroid (helps identify mechanical/electronic sounds)
        cent = librosa.feature.spectral_centroid(S=spec)[0]

        # Spectral bandwidth (helps identify noise)
        bandw = librosa.feature.spectral_bandwidth(S=spec)[0]

        # Spectral rolloff (helps identify high-frequency noise)
        rolloff = librosa.feature.spectral_rolloff(S=spec)[0]

        # Zero crossing rate (helps identify clicks/pops)
        zcr = librosa.feature.zero_crossing_rate(samples)[0]

        # RMS energy
        rms = librosa.feature.rms(S=spec)[0]

        # Calculate statistics
        stats = {
            "centroid_mean": np.mean(cent),
            "centroid_std": np.std(cent),
            "bandwidth_mean": np.mean(bandw),
            "bandwidth_std": np.std(bandw),
            "rolloff_mean": np.mean(rolloff),
            "zcr_mean": np.mean(zcr),
            "zcr_std": np.std(zcr),
            "rms_mean": np.mean(rms),
            "rms_std": np.std(rms),
        }

        # Define thresholds for non-speech sounds
        has_non_speech = (
            # High frequency mechanical noise
            (stats["centroid_mean"] > 4000 and stats["centroid_std"] < 500)
            or
            # Clicks and pops (sudden ZCR changes)
            (stats["zcr_std"] > 0.3 and stats["zcr_mean"] > 0.2)
            or
            # Wide bandwidth noise
            (stats["bandwidth_mean"] > 3000 and stats["bandwidth_std"] < 400)
            or
            # Sudden amplitude changes (clicks/pops)
            (stats["rms_std"] / (stats["rms_mean"] + 1e-6) > 2.0)
        )

        return has_non_speech, stats

    except Exception as e:
        logger.warning(f"Error in non-speech sound detection: {e}")
        return False, {}


def detect_untranscribed_sounds(samples, sample_rate, text=None):
    """Detect if there are speech sounds that aren't in the transcription using Silero VAD"""
    try:
        # If no text provided, we can't check for untranscribed sounds
        if text is None:
            return False, {}

        # First check for non-speech sounds
        has_non_speech, non_speech_stats = detect_non_speech_sounds(
            samples, sample_rate
        )

        # Convert text to lowercase and remove punctuation for comparison
        text = text.lower()
        for char in ".,!?":
            text = text.replace(char, "")

        # Use Silero VAD to detect speech segments with confidence scores
        has_speech, speech_timestamps, confidences = detect_speech_silero(
            samples,
            sample_rate=sample_rate,
            threshold=0.3,
        )

        if not has_speech:
            return has_non_speech, {
                "has_non_speech": has_non_speech,
                "non_speech_stats": non_speech_stats,
            }

        # Count words in transcription
        words = text.split()
        word_count = len(words)

        # Calculate speech duration statistics
        speech_durations = [ts["end"] - ts["start"] for ts in speech_timestamps]
        total_speech_duration = sum(speech_durations)

        # Filter out low-confidence segments
        reliable_segments = []
        reliable_durations = []
        for i, (ts, conf) in enumerate(zip(speech_timestamps, confidences)):
            if conf >= 0.5:  # Only keep segments with high confidence
                reliable_segments.append(ts)
                reliable_durations.append(speech_durations[i])

        # Calculate speech rate only for reliable segments
        total_reliable_duration = sum(reliable_durations)
        words_per_second = (
            word_count / total_reliable_duration if total_reliable_duration > 0 else 0
        )

        # Check for reasonable speech rate (120-180 words per minute is typical)
        too_fast = words_per_second > 4.0  # More than 240 wpm
        too_slow = words_per_second < 1.0  # Less than 60 wpm

        # Analyze gaps between reliable segments
        gaps = []
        for i in range(1, len(reliable_segments)):
            gap = reliable_segments[i]["start"] - reliable_segments[i - 1]["end"]
            if gap > 0.1:  # Only consider gaps longer than 100ms
                # Check for non-speech sounds in the gap
                gap_start = int(reliable_segments[i - 1]["end"] * sample_rate)
                gap_end = int(reliable_segments[i]["start"] * sample_rate)
                gap_audio = samples[gap_start:gap_end]
                if len(gap_audio) > 0:
                    has_gap_noise, _ = detect_non_speech_sounds(gap_audio, sample_rate)
                    if has_gap_noise:
                        gaps.append(gap)

        # Calculate gap statistics
        max_gap = max(gaps) if gaps else 0
        avg_gap = sum(gaps) / len(gaps) if gaps else 0

        # Define gap thresholds based on text content
        base_gap_threshold = 1.2  # More lenient base threshold
        if any(word in ["pause", "wait", "um", "uh", "well", "..."] for word in words):
            base_gap_threshold = (
                1.5  # Even more lenient for texts with pause indicators
            )
        elif "?" in text or "!" in text:
            base_gap_threshold = 1.3  # More lenient for questions/exclamations

        # Check for suspicious gaps with non-speech sounds
        has_suspicious_gaps = (
            len(gaps) > 0
            and max_gap > base_gap_threshold
            and avg_gap > base_gap_threshold * 0.7
            and not any(w in ["pause", "wait", "um", "uh", "well"] for w in words)
        )

        # Calculate average confidence
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0

        # Determine if there are untranscribed sounds - including non-speech detection
        has_untranscribed = (
            has_non_speech  # Non-speech sounds detected
            or (too_slow and avg_confidence > 0.7)
            or (too_fast and avg_confidence > 0.7)
            or (has_suspicious_gaps and avg_confidence > 0.8)
        )

        details = {
            "has_non_speech": has_non_speech,
            "non_speech_stats": non_speech_stats,
            "total_segments": len(speech_timestamps),
            "reliable_segments": len(reliable_segments),
            "word_count": word_count,
            "total_duration": total_speech_duration,
            "reliable_duration": total_reliable_duration,
            "words_per_second": words_per_second,
            "speech_rate_too_slow": too_slow,
            "speech_rate_too_fast": too_fast,
            "max_gap": max_gap,
            "avg_gap": avg_gap,
            "gap_threshold": base_gap_threshold,
            "has_suspicious_gaps": has_suspicious_gaps,
            "avg_confidence": avg_confidence,
            "speech_timestamps": speech_timestamps,
            "confidences": confidences,
        }

        if has_untranscribed:
            reason = []
            if has_non_speech:
                reason.append(
                    f"Non-speech sounds detected (centroid: {non_speech_stats['centroid_mean']:.0f}Hz, "
                    f"zcr: {non_speech_stats['zcr_mean']:.3f})"
                )
            if too_slow and avg_confidence > 0.7:
                reason.append(
                    f"Speech rate too slow ({words_per_second:.1f} wps, conf: {avg_confidence:.2f})"
                )
            if too_fast and avg_confidence > 0.7:
                reason.append(
                    f"Speech rate too fast ({words_per_second:.1f} wps, conf: {avg_confidence:.2f})"
                )
            if has_suspicious_gaps and avg_confidence > 0.8:
                reason.append(
                    f"Suspicious gaps with noise (max: {max_gap:.2f}s, avg: {avg_gap:.2f}s, conf: {avg_confidence:.2f})"
                )

            logger.info(
                f"Detected potential issues: {', '.join(reason)}. "
                f"Reliable speech duration: {total_reliable_duration:.2f}s, "
                f"Words: {word_count}"
            )

        return has_untranscribed, details

    except Exception as e:
        logger.warning(f"Error in untranscribed sound detection: {e}")
        return False, {}


def load_and_analyze_segment(wav_file):
    """Load and analyze a single audio segment"""
    try:
        # Load audio file
        audio = AudioSegment.from_wav(wav_file)

        # Convert to mono if stereo
        if audio.channels == 2:
            audio = audio.set_channels(1)

        # Get samples for analysis
        samples = np.array(audio.get_array_of_samples())
        if audio.channels == 2:
            samples = samples.reshape((-1, 2)).mean(axis=1)

        # Normalize samples
        if np.abs(samples).max() > 0:
            samples = samples / np.abs(samples).max()

        return {
            "audio": audio,
            "samples": samples,
            "duration": len(audio) / 1000.0,  # Duration in seconds
            "file_path": Path(wav_file),
        }
    except Exception as e:
        logger.error(f"Error loading {wav_file}: {e}")
        return None


def save_failed_segment(
    audio, filename, text, failure_reason, quality_metrics, failed_dir
):
    """Save failed segment with detailed information"""
    try:
        # Create subdirectory based on failure type
        failure_type = failure_reason.split(":")[0].lower().replace(" ", "_")
        failure_dir = Path(failed_dir) / failure_type
        failure_dir.mkdir(parents=True, exist_ok=True)

        # Save audio file
        output_path = failure_dir / filename
        audio.export(str(output_path), format="wav")

        # Save failure details
        details = {
            "filename": filename,
            "text": text,
            "failure_reason": failure_reason,
            "quality_metrics": quality_metrics,
        }

        details_path = output_path.with_suffix(".json")
        with open(details_path, "w", encoding="utf-8") as f:
            json.dump(details, f, indent=2, ensure_ascii=False)

    except Exception as e:
        logger.error(f"Error saving failed segment {filename}: {e}")


def combine_short_segments(segments, output_path):
    """Combine short segments with silence gaps"""
    try:
        combined_audio = AudioSegment.empty()
        silence = AudioSegment.silent(
            duration=int(COMBINE_SILENCE_GAP * 1000),
            frame_rate=segments[0]["audio"].frame_rate,
        )
        silence_padding = AudioSegment.silent(
            duration=int(SEGMENT_SETTINGS["silence_padding"] * 1000),
            frame_rate=segments[0]["audio"].frame_rate,
        )

        # Calculate total duration including all silences
        total_duration = 0
        for i, segment in enumerate(segments):
            # Add silence gap duration (except for first segment)
            if i > 0:
                total_duration += COMBINE_SILENCE_GAP
            # Add segment duration
            total_duration += segment["duration"]
            # Add padding duration (for both start and end)
            total_duration += 2 * SEGMENT_SETTINGS["silence_padding"]

        # Check if total duration would exceed MAX_DURATION
        if total_duration > MAX_DURATION:
            logger.warning(
                f"Combined duration {total_duration:.2f}s would exceed MAX_DURATION {MAX_DURATION}s, skipping combination"
            )
            return None

        # If duration is acceptable, proceed with combination
        for i, segment in enumerate(segments):
            # Add silence before segment (except for first segment)
            if i > 0:
                combined_audio += silence

            # Add silence padding at start and end
            processed_segment = silence_padding + segment["audio"] + silence_padding

            # Normalize to EBU R128 standard
            target_dBFS = -23.0
            change_in_dBFS = target_dBFS - processed_segment.dBFS
            processed_segment = processed_segment.apply_gain(change_in_dBFS)

            # Apply compression
            processed_segment = effects.compress_dynamic_range(
                processed_segment, threshold=-20.0, ratio=3.0, attack=10, release=50
            )

            combined_audio += processed_segment

        # Export combined audio
        combined_audio.export(str(output_path), format="wav")
        return combined_audio

    except Exception as e:
        logger.error(f"Error combining segments: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process pre-segmented audio files")
    parser.add_argument(
        "--input",
        type=str,
        default="Data_prep/raw_data/raw_segments",
        help="Input directory containing pre-segmented WAV files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="Data_prep/raw_data/segments",
        help="Output directory path",
    )
    args = parser.parse_args()

    try:
        # Check if input directory exists
        input_dir = Path(args.input)
        if not input_dir.exists():
            print(f"\nError: Input directory '{args.input}' does not exist")
            sys.exit(1)

        # Create output directory if it doesn't exist
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Process segments
        process_segments(args.input, args.output)
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print(f"\nError during processing: {e}")
        raise
