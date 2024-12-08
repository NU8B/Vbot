from faster_whisper import WhisperModel
import os
import torchaudio
from datasets import Dataset, Audio
import pandas as pd
from huggingface_hub import login
from pathlib import Path

# Login to Hugging Face
login("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")

# Model setup
model_size = "distil-large-v3"
model = WhisperModel(model_size, device="cuda", compute_type="float16")


def process_audio(audio_path):
    # Load and resample audio to 24kHz
    wav, sr = torchaudio.load(audio_path)
    if sr != 24000:
        wav = torchaudio.functional.resample(wav, sr, 24000)

    # Convert to mono if stereo
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # Normalize audio
    wav = wav / wav.abs().max()
    return wav


def create_dataset():
    data = []
    os.makedirs("raw_data/raw_audio", exist_ok=True)
    os.makedirs("raw_data/processed", exist_ok=True)

    for filename in os.listdir("data_train/raw_audio"):
        if filename.endswith((".wav", ".mp3", ".m4a")):
            try:
                # Process audio
                input_path = os.path.join("raw_data/raw_audio", filename)
                base_filename = Path(filename).stem
                output_path = f"raw_data/processed/{base_filename}.wav"

                # Process and save audio
                wav = process_audio(input_path)
                torchaudio.save(output_path, wav, 24000)

                # Transcribe
                segments, _ = model.transcribe(
                    output_path,
                    beam_size=5,
                    language="en",
                    condition_on_previous_text=False,
                )
                transcript = " ".join(segment.text.strip() for segment in segments)

                # Add to dataset
                data.append(
                    {
                        "audio": output_path,
                        "text": transcript,
                    }
                )
                print(f"Processed: {filename}")

            except Exception as e:
                print(f"Error processing {filename}: {e}")

    # Create DataFrame and Dataset
    if not data:
        raise ValueError(
            "No audio files were processed successfully. Check your raw_audio directory."
        )

    df = pd.DataFrame(data)
    dataset = Dataset.from_pandas(df)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=24000))

    return dataset


def main():
    dataset = create_dataset()
    dataset.push_to_hub("nonoJDWAOIDAWKDA/test1", private=True)


if __name__ == "__main__":
    main()
