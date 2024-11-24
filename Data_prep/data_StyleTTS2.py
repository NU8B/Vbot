import os
import torchaudio
from tqdm import tqdm
import random
from faster_whisper import WhisperModel
import eng_to_ipa as ipa
import torch
from transformers import AlbertTokenizer

# Initialize models
model = WhisperModel("distil-large-v3", device="cuda", compute_type="float16")
tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")


def count_tokens(text):
    """Accurate token count using BERT tokenizer"""
    tokens = tokenizer.encode(text, add_special_tokens=True)
    return len(tokens)


def clean_text(text):
    """Clean text before conversion"""
    if not isinstance(text, str):
        text = str(text)

    # Remove repetitive sentences
    sentences = text.split(".")
    unique_sentences = []
    for sent in sentences:
        sent = sent.strip()
        if sent and sent not in unique_sentences:
            unique_sentences.append(sent)
    text = ". ".join(unique_sentences)

    # Basic cleaning
    text = (
        text.replace(".", "").replace(",", "").replace("?", "").replace("!", "").strip()
    )

    # Remove repetitive words or phrases
    words = text.split()
    unique_words = []
    prev_words = []
    for word in words:
        # Check for repetition within a window of 3 words
        if len(prev_words) < 3 or word not in prev_words[-3:]:
            unique_words.append(word)
            prev_words.append(word)

    return " ".join(unique_words)


def add_stress_marks(text):
    """Add stress marks to IPA text"""
    words = text.split()
    marked_words = []
    for word in words:
        if len(word) > 1 and not any(c in word for c in "ˈˌ"):
            word = "ˈ" + word
        marked_words.append(word)
    return " ".join(marked_words)


def text_to_ipa(text):
    """Convert text to IPA using eng_to_ipa"""
    try:
        # Clean the text
        text = clean_text(text)

        # Convert to IPA
        ipa_text = ipa.convert(text)

        # Remove the period at the end if present
        if ipa_text.endswith("."):
            ipa_text = ipa_text[:-1]

        return ipa_text.strip()

    except Exception as e:
        print(f"Error converting to IPA: {text} - {str(e)}")
        return text


def get_transcription(audio_path):
    """Get transcription using Whisper"""
    segments, _ = model.transcribe(
        audio_path,
        beam_size=5,
        language="en",
        condition_on_previous_text=False,
    )
    return " ".join(segment.text.strip() for segment in segments)


def prepare_styletts2_data(raw_audio_dir, output_dir, max_tokens=510):
    """Prepare data in StyleTTS2 format with token limit"""
    # Create necessary directories
    os.makedirs(os.path.join(output_dir, "wavs"), exist_ok=True)

    # Process audio files and create metadata
    metadata = []
    wav_files = [f for f in os.listdir(raw_audio_dir) if f.endswith(".wav")]
    skipped_count = 0
    token_skipped = 0

    for idx, filename in enumerate(tqdm(wav_files, desc="Processing files")):
        try:
            wav_path = os.path.join(raw_audio_dir, filename)

            # Get transcription
            transcription = get_transcription(wav_path)

            # Clean text
            cleaned_text = clean_text(transcription)

            # Check token count using BERT tokenizer
            token_count = count_tokens(cleaned_text)
            if token_count > max_tokens:
                print(
                    f"\nSkipping {filename} - too many tokens ({token_count} > {max_tokens})"
                )
                token_skipped += 1
                skipped_count += 1
                continue

            # Load and check audio
            waveform, sample_rate = torchaudio.load(wav_path)

            # Check minimum duration and pad if possible
            min_samples = int(0.5 * sample_rate)
            if waveform.size(1) < min_samples:
                if waveform.size(1) < min_samples * 0.5:  # If less than 0.25 seconds
                    print(f"\nSkipping {filename} - too short")
                    skipped_count += 1
                    continue
                padding = min_samples - waveform.size(1)
                waveform = torch.nn.functional.pad(waveform, (0, padding))

            # Convert transcription to IPA
            ipa_text = text_to_ipa(cleaned_text)

            if not ipa_text:
                print(f"\nSkipping {filename} - empty IPA conversion")
                skipped_count += 1
                continue

            # Process audio
            if sample_rate != 24000:
                resampler = torchaudio.transforms.Resample(sample_rate, 24000)
                waveform = resampler(waveform)

            # Save processed audio
            new_filename = f"{idx+1:04d}.wav"
            output_path = os.path.join(output_dir, "wavs", new_filename)
            torchaudio.save(output_path, waveform, 24000)

            metadata.append((new_filename, ipa_text))

        except Exception as e:
            print(f"\nError processing {filename}: {e}")
            skipped_count += 1
            continue

    print(f"\nProcessing complete:")
    print(f"Total files processed: {len(wav_files)}")
    print(f"Files skipped due to token length: {token_skipped}")
    print(f"Total files skipped: {skipped_count}")
    print(f"Files included: {len(metadata)}")

    if not metadata:
        raise ValueError("No valid files to process!")

    # Split into train/val sets (90/10 split)
    random.shuffle(metadata)
    split_idx = int(len(metadata) * 0.9)
    train_data = metadata[:split_idx]
    val_data = metadata[split_idx:]

    # Write train_list.txt
    with open(os.path.join(output_dir, "train_list.txt"), "w", encoding="utf-8") as f:
        for filename, phonemes in train_data:
            f.write(f"{filename}|{phonemes}|0\n")

    # Write val_list.txt
    with open(os.path.join(output_dir, "val_list.txt"), "w", encoding="utf-8") as f:
        for filename, phonemes in val_data:
            f.write(f"{filename}|{phonemes}|0\n")


if __name__ == "__main__":
    try:
        raw_audio_dir = "raw_data/raw_audio"
        output_dir = "Data"

        print(f"Processing audio files from: {raw_audio_dir}")
        prepare_styletts2_data(raw_audio_dir, output_dir, max_tokens=510)
        print("Processing completed successfully!")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
