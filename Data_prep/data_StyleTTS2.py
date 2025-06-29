import os
import sys
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torchaudio
import numpy as np
import torch
from tqdm import tqdm
from StyleTTS2.text_utils import TextCleaner, symbols
from phonemizer.backend import EspeakBackend
from transformers import AlbertTokenizer
import torchaudio.transforms as T

# Default paths relative to script location
SCRIPT_DIR = Path(__file__).parent
DEFAULT_SEGMENTS_DIR = SCRIPT_DIR / "raw_data" / "segments"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "Data"
DEFAULT_SR = 24000
DEFAULT_MAX_TOKENS = 377


def clean_text(text):
    """Clean text to only include allowed characters"""
    allowed_chars = set(symbols)
    return "".join(c for c in text if c in allowed_chars)


def phonemize_text(text):
    """Convert text to phonemes using espeak with stress marks"""
    try:
        # Initialize backend directly for more control
        backend = EspeakBackend(
            "en-us", with_stress=True, punctuation_marks=';:,.!?¡¿—…"«»"" '
        )
        phonemes = backend.phonemize([text])[0]
        return clean_text(phonemes)
    except Exception as e:
        print(f"Error phonemizing text: {e}")
        return None


def parallel_phonemize(texts, max_workers=4):
    """Phonemize texts in parallel"""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(phonemize_text, text) for text in texts]
        return [future.result() for future in as_completed(futures)]


def prefetch_audio(wav_path, sr, target_sr):
    """Prefetch and preprocess audio files"""
    try:
        wav, sr_orig = torchaudio.load(wav_path)
        if sr_orig != target_sr:
            resampler = T.Resample(orig_freq=sr_orig, new_freq=target_sr)
            wav = resampler(wav)
        # Convert to mono if stereo
        if wav.size(0) > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        return wav, None
    except Exception as e:
        return None, str(e)


def parallel_audio_processing(data_dir, audio_files, target_sr, max_workers=8):
    """Process audio files in parallel"""
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for audio_file in audio_files:
            wav_path = os.path.join(data_dir, audio_file)
            future = executor.submit(prefetch_audio, wav_path, None, target_sr)
            futures[audio_file] = future

        for audio_file, future in tqdm(futures.items(), desc="Loading audio files"):
            wav, error = future.result()
            if wav is not None:
                results[audio_file] = wav
            else:
                print(f"\nError loading {audio_file}: {error}")
    return results


def find_segments_dir():
    """Find the segments directory by searching common locations"""
    possible_paths = [
        DEFAULT_SEGMENTS_DIR,
        SCRIPT_DIR.parent / "raw_data" / "segments",
        Path.cwd() / "raw_data" / "segments",
        Path.cwd() / "Data_prep" / "raw_data" / "segments",
    ]

    for path in possible_paths:
        if (path / "metadata.json").exists():
            # Check if we have reviewed segments or original passed segments
            if (path / "reviewed_approved").exists() and list(
                (path / "reviewed_approved").glob("*.wav")
            ):
                print(f"Found reviewed segments in: {path}")
                return path
            elif (path / "passed_segments").exists() and list(
                (path / "passed_segments").glob("*.wav")
            ):
                print(f"Found unreviewed segments in: {path}")
                return path

    raise FileNotFoundError(
        "Could not find segments directory with metadata.json and segments. "
        "Please run the audio segmenter first, or complete the review process."
    )


def prepare_data(
    data_dir=None, output_dir=None, sr=DEFAULT_SR, max_tokens=DEFAULT_MAX_TOKENS
):
    """Prepare data for StyleTTS2 training"""
    print("\nInitializing StyleTTS2 data preparation...")

    # Find input directory if not specified
    if data_dir is None:
        segments_dir = find_segments_dir()

        # Check if we have reviewed approved segments
        approved_dir = segments_dir / "reviewed_approved"
        passed_dir = segments_dir / "passed_segments"

        if approved_dir.exists() and list(approved_dir.glob("*.wav")):
            data_dir = approved_dir
            print(f"Using reviewed approved segments: {approved_dir}")

            # Check if we have approved segments metadata
            approved_metadata_file = segments_dir / "approved_segments_metadata.json"
            if approved_metadata_file.exists():
                metadata_path = approved_metadata_file
                print("Using approved segments metadata")
            else:
                metadata_path = segments_dir / "metadata.json"
                print("Using original metadata (filtering for approved files)")
        else:
            data_dir = passed_dir
            metadata_path = segments_dir / "metadata.json"
            print(f"Using unreviewed passed segments: {passed_dir}")

        print(f"Found segments directory: {segments_dir}")
    else:
        data_dir = Path(data_dir)
        segments_dir = data_dir.parent
        metadata_path = segments_dir / "metadata.json"

    # Use default output directory if not specified
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
        print(f"Using default output directory: {output_dir}")
    output_dir = Path(output_dir)

    # Initialize models
    text_cleaner = TextCleaner()
    tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")

    # Create only the necessary output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    wavs_dir = output_dir / "wavs"  # This is where we'll store the final wav files
    wavs_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata from segmentation
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata_list = json.load(f)

    print(f"Loaded metadata with {len(metadata_list)} segments")

    # If using reviewed approved segments, filter metadata to only include files that exist
    if "reviewed_approved" in str(data_dir):
        available_files = {f.name for f in data_dir.glob("*.wav")}
        original_count = len(metadata_list)
        metadata_list = [
            item for item in metadata_list if item["filename"] in available_files
        ]
        print(
            f"Filtered to {len(metadata_list)} approved segments (from {original_count} total)"
        )

    # Get all wav files and their corresponding texts
    audio_files = [(item["filename"], item["text"]) for item in metadata_list]

    # Verify that audio files exist
    missing_files = []
    for filename, _ in audio_files:
        if not (data_dir / filename).exists():
            missing_files.append(filename)

    if missing_files:
        print(f"Warning: {len(missing_files)} audio files not found:")
        for f in missing_files[:5]:  # Show first 5
            print(f"  - {f}")
        if len(missing_files) > 5:
            print(f"  ... and {len(missing_files) - 5} more")

        # Filter out missing files
        audio_files = [(f, t) for f, t in audio_files if f not in missing_files]
        print(f"Proceeding with {len(audio_files)} available files")

    if not audio_files:
        raise ValueError("No audio files found to process!")

    # Prefetch audio files in parallel
    print("\nPrefetching audio files...")
    audio_data = parallel_audio_processing(data_dir, [f[0] for f in audio_files], sr)

    # Parallel phonemization
    print("\nPhonemizing texts in parallel...")
    texts_to_phonemize = [text for _, text in audio_files]
    phonemes_list = parallel_phonemize(texts_to_phonemize, max_workers=8)

    train_list = []
    val_list = []
    skipped_count = 0
    token_skipped = 0

    print("\nProcessing results...")
    for i, ((audio_file, _), phonemes) in enumerate(zip(audio_files, phonemes_list)):
        try:
            if not phonemes or not phonemes.strip():
                print(f"Skipping {audio_file}: Empty or invalid phonemes")
                skipped_count += 1
                continue

            # Get preprocessed audio
            wav = audio_data.get(audio_file)
            if wav is None:
                print(f"\nSkipping {audio_file} - audio preprocessing failed")
                skipped_count += 1
                continue

            # Check minimum duration (0.5 seconds)
            min_samples = int(0.5 * sr)
            if wav.size(1) < min_samples:
                if wav.size(1) < min_samples * 0.5:  # If less than 0.25 seconds
                    print(f"\nSkipping {audio_file} - too short")
                    skipped_count += 1
                    continue
                padding = min_samples - wav.size(1)
                wav = torch.nn.functional.pad(wav, (0, padding))

            # Verify with TextCleaner
            try:
                cleaned_phonemes = text_cleaner(phonemes)
            except:
                print(f"\nSkipping {audio_file} - invalid phonemes for TextCleaner")
                skipped_count += 1
                continue

            # Direct tokenization check
            tokens = tokenizer.encode(phonemes, add_special_tokens=True)
            if len(tokens) > max_tokens:
                print(
                    f"\nSkipping {audio_file} - sequence too long (tokens: {len(tokens)} > {max_tokens})"
                )
                token_skipped += 1
                continue

            # Create new filename without spaces
            new_filename = f"{i:04d}.wav"

            # Save processed wav
            output_wav_path = os.path.join(output_dir, "wavs", new_filename)
            torchaudio.save(output_wav_path, wav, sr)

            # Create metadata entry
            metadata_entry = f"{new_filename}|{phonemes}|0"

            # Split into train/val (90/10)
            if np.random.random() < 0.9:
                train_list.append(metadata_entry)
            else:
                val_list.append(metadata_entry)

        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            skipped_count += 1
            continue

    # Clear audio data from memory
    del audio_data

    print(f"\nProcessing complete:")
    print(f"Total files processed: {len(audio_files)}")
    print(f"Files skipped due to token length: {token_skipped}")
    print(f"Total files skipped: {skipped_count}")
    print(f"Files included: {len(train_list) + len(val_list)}")

    # Print length statistics using direct tokenization
    if train_list:
        print("\nLength statistics for included files:")
        direct_lengths = [
            len(tokenizer.encode(entry.split("|")[1], add_special_tokens=True))
            for entry in train_list
        ]
        print(f"Max direct token length: {max(direct_lengths)}")
        print(f"Average token length: {sum(direct_lengths)/len(direct_lengths):.2f}")

    # Save metadata files
    with open(os.path.join(output_dir, "train_list.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(train_list))
    with open(os.path.join(output_dir, "val_list.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(val_list))

    print(f"\nStyleTTS2 dataset prepared successfully!")
    print(f"Output directory: {output_dir}")
    print(f"Training samples: {len(train_list)}")
    print(f"Validation samples: {len(val_list)}")

    # If we used reviewed segments, provide feedback about the review process
    if "reviewed_approved" in str(data_dir):
        print(f"\n✅ Used manually reviewed and approved segments")
        print(f"   This should result in higher quality training data!")
    else:
        print(f"\n⚠️  Using unreviewed segments")
        print(f"   Consider using the segment reviewer for better quality:")
        print(f"   python Data_prep/segment_reviewer/segment_reviewer.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for StyleTTS2 training")
    parser.add_argument(
        "--input",
        type=str,
        help="Input directory containing segmented audio files (optional, will auto-detect if not specified)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help=f"Output directory for prepared dataset (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=DEFAULT_SR,
        help=f"Target sample rate (default: {DEFAULT_SR})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Maximum number of tokens per sample (default: {DEFAULT_MAX_TOKENS})",
    )
    args = parser.parse_args()

    try:
        prepare_data(args.input, args.output, args.sr, args.max_tokens)
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        raise
