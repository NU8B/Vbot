import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torchaudio
import numpy as np
from tqdm import tqdm
from faster_whisper import WhisperModel
from StyleTTS2.text_utils import TextCleaner, symbols
from phonemizer.backend import EspeakBackend


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


def prepare_data(data_dir, output_dir, sr=24000):
    """Prepare data for StyleTTS2 training"""
    # Initialize Whisper and TextCleaner
    model = WhisperModel("distil-large-v3", device="cuda", compute_type="float16")
    text_cleaner = TextCleaner()

    # Create output directory structure
    os.makedirs(os.path.join(output_dir, "wavs"), exist_ok=True)

    train_list = []
    val_list = []

    # Get all wav files
    audio_files = [f for f in os.listdir(data_dir) if f.endswith(".wav")]

    # Sort files to ensure consistent ordering
    audio_files.sort()

    for i, audio_file in enumerate(tqdm(audio_files, desc="Processing files")):
        try:
            # Load and process audio
            wav_path = os.path.join(data_dir, audio_file)
            wav, sr_orig = torchaudio.load(wav_path)

            # Convert to mono if necessary
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)

            # Resample if needed
            if sr_orig != sr:
                wav = torchaudio.transforms.Resample(sr_orig, sr)(wav)

            # Transcribe audio and convert to phonemes
            phonemes = transcribe_audio(model, wav_path)

            # Skip if phonemes is None or empty
            if not phonemes or not phonemes.strip():
                print(f"Skipping {audio_file}: Empty or invalid phonemes")
                continue

            # Verify phonemes can be processed by TextCleaner
            _ = text_cleaner(phonemes)

            # Create new filename without spaces
            new_filename = f"{i:04d}.wav"

            # Save processed wav in wavs subdirectory
            output_wav_path = os.path.join(output_dir, "wavs", new_filename)
            torchaudio.save(output_wav_path, wav, sr)

            # Create metadata entry (filename|phonemes|speaker_id) without 'wavs/' prefix
            metadata_entry = f"{new_filename}|{phonemes}|0"

            # Split into train/val (90/10)
            if np.random.random() < 0.9:
                train_list.append(metadata_entry)
            else:
                val_list.append(metadata_entry)

        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            continue

    # Save metadata files
    with open(os.path.join(output_dir, "train_list.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(train_list))
    with open(os.path.join(output_dir, "val_list.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(val_list))


if __name__ == "__main__":
    data_dir = "Data_prep/raw_data/raw_audio"
    output_dir = "Data_prep/Data"
    sr = 24000

    prepare_data(data_dir, output_dir, sr)
