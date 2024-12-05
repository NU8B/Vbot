import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torchaudio
import numpy as np
import torch
from tqdm import tqdm
from faster_whisper import WhisperModel
from StyleTTS2.text_utils import TextCleaner, symbols
from phonemizer.backend import EspeakBackend
from transformers import AlbertTokenizer


def clean_text(text):
    """Clean text to only include allowed characters"""
    allowed_chars = set(symbols)
    return "".join(c for c in text if c in allowed_chars)


def phonemize_text(text):
    """Convert text to phonemes using espeak with stress marks"""
    try:
        # Initialize backend directly for more control
        backend = EspeakBackend(
            "en-us", with_stress=True, punctuation_marks=';:,.!?¡¿—…"«»“” '
        )
        phonemes = backend.phonemize([text])[0]
        return clean_text(phonemes)
    except Exception as e:
        print(f"Error phonemizing text: {e}")
        return None


def transcribe_audio(model, audio_path):
    """Transcribe audio using Whisper"""
    segments, _ = model.transcribe(
        audio_path,
        beam_size=5,
        language="en",
        condition_on_previous_text=False,
    )
    text = " ".join(segment.text.strip() for segment in segments)
    # Convert to phonemes
    return phonemize_text(text)


def prepare_data(data_dir, output_dir, sr, max_tokens):
    """Prepare data for StyleTTS2 training"""
    # Initialize models
    model = WhisperModel("distil-large-v3", device="cuda", compute_type="float16")
    text_cleaner = TextCleaner()
    tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")

    # Create output directory structure
    os.makedirs(os.path.join(output_dir, "wavs"), exist_ok=True)

    train_list = []
    val_list = []
    skipped_count = 0
    token_skipped = 0

    # Get all wav files
    audio_files = [f for f in os.listdir(data_dir) if f.endswith(".wav")]
    audio_files.sort()

    for i, audio_file in enumerate(tqdm(audio_files, desc="Processing files")):
        try:
            # Load and process audio
            wav_path = os.path.join(data_dir, audio_file)
            wav, sr_orig = torchaudio.load(wav_path)

            # Check minimum duration (0.5 seconds)
            min_samples = int(0.5 * sr_orig)
            if wav.size(1) < min_samples:
                if wav.size(1) < min_samples * 0.5:  # If less than 0.25 seconds
                    print(f"\nSkipping {audio_file} - too short")
                    skipped_count += 1
                    continue
                padding = min_samples - wav.size(1)
                wav = torch.nn.functional.pad(wav, (0, padding))

            # Transcribe audio and convert to phonemes
            phonemes = transcribe_audio(model, wav_path)
            if not phonemes or not phonemes.strip():
                print(f"Skipping {audio_file}: Empty or invalid phonemes")
                skipped_count += 1
                continue

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
    data_dir = "Data_prep/raw_data/raw_audio"
    output_dir = "Data_prep/Data"
    sr = 24000
    max_tokens = 377

    prepare_data(data_dir, output_dir, sr, max_tokens)
