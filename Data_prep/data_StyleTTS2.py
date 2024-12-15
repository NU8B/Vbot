import os
import sys
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import hashlib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torchaudio
import numpy as np
import torch
from tqdm import tqdm
from faster_whisper import WhisperModel
from StyleTTS2.text_utils import TextCleaner, symbols
from phonemizer.backend import EspeakBackend
from transformers import AlbertTokenizer
import torchaudio.transforms as T


def clean_text(text):
    """Clean text to only include allowed characters"""
    allowed_chars = set(symbols)
    return "".join(c for c in text if c in allowed_chars)


def get_file_hash(file_path):
    """Get MD5 hash of file for caching"""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        buf = f.read(65536)  # Read in 64kb chunks
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()


def load_cache(cache_file):
    """Load transcription cache from file"""
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache_file, cache):
    """Save transcription cache to file"""
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


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


def process_single(model, audio_path):
    """Process a single audio file with Whisper"""
    try:
        segments, _ = model.transcribe(
            audio_path,
            beam_size=5,
            language="en",
            condition_on_previous_text=False,
        )
        return " ".join(segment.text.strip() for segment in segments)
    except Exception as e:
        print(f"\nError transcribing {audio_path}: {e}")
        return None


def process_batch(model, audio_paths, max_workers=4):
    """Process audio files in parallel"""
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a list to store futures
        futures = []

        # Submit all tasks
        for audio_path in audio_paths:
            future = executor.submit(process_single, model, audio_path)
            futures.append(future)

        # Process results as they complete
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Transcribing"
        ):
            result = future.result()
            results.append(result)

    return results


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


def prepare_data(data_dir, output_dir, sr, max_tokens):
    """Prepare data for StyleTTS2 training"""
    # Initialize models
    print("Loading Whisper model...")
    model = WhisperModel("distil-large-v3", device="cuda", compute_type="float16")
    text_cleaner = TextCleaner()
    tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")

    # Create all necessary directories
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create centralized cache directory in root
    root_dir = Path(__file__).parent.parent  # Get root directory
    cache_dir = root_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "whisper_transcriptions.json"

    # Create wavs directory
    wavs_dir = output_dir / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)

    # Load cache
    cache = load_cache(cache_file)

    train_list = []
    val_list = []
    skipped_count = 0
    token_skipped = 0

    # Get all wav files
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    audio_files = [f for f in os.listdir(data_dir) if f.endswith(".wav")]
    audio_files.sort()

    # Prepare files for processing
    files_to_process = []
    cached_results = []

    print("Checking cache and preparing files...")
    for audio_file in tqdm(audio_files):
        wav_path = os.path.join(data_dir, audio_file)
        file_hash = get_file_hash(wav_path)

        if file_hash in cache:
            cached_results.append((audio_file, cache[file_hash]))
        else:
            files_to_process.append((audio_file, wav_path, file_hash))

    # Prefetch audio files in parallel
    print("\nPrefetching audio files...")
    audio_data = parallel_audio_processing(
        data_dir, [f[0] for f in files_to_process], sr
    )

    # Process uncached files in parallel
    if files_to_process:
        print(f"\nProcessing {len(files_to_process)} uncached files...")
        batch_paths = [f[1] for f in files_to_process]

        # Use more workers for transcription since it's GPU-bound
        batch_results = process_batch(model, batch_paths, max_workers=8)

        # Update cache with new results
        for (audio_file, wav_path, file_hash), text in zip(
            files_to_process, batch_results
        ):
            if text is not None:  # Only cache successful transcriptions
                cache[file_hash] = text

        # Save updated cache
        save_cache(cache_file, cache)

    # Combine cached and new results
    all_results = cached_results + [
        (f[0], cache[f[2]]) for f in files_to_process if f[2] in cache
    ]

    # Parallel phonemization
    print("\nPhonemizing texts in parallel...")
    texts_to_phonemize = [result[1] for result in all_results]
    # Use more workers for phonemization since it's CPU-bound
    phonemes_list = parallel_phonemize(texts_to_phonemize, max_workers=8)

    print("\nProcessing results...")
    for i, ((audio_file, _), phonemes) in enumerate(zip(all_results, phonemes_list)):
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


if __name__ == "__main__":
    data_dir = "Data_prep/raw_data/2hour_amelia"
    output_dir = "Data_prep/Data"
    sr = 24000
    max_tokens = 377

    prepare_data(data_dir, output_dir, sr, max_tokens)
