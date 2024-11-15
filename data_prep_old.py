from faster_whisper import WhisperModel
import os


# model_size = "large-v3"
model_size = "distil-large-v3"
# Define directory containing audio files
audio_dir = "data_train/wavs"
model = WhisperModel(model_size, device="cuda", compute_type="float16")


def transcribe_and_save(audio_file):

    base_filename, _ = os.path.splitext(os.path.basename(audio_file))
    try:
        segments, _ = model.transcribe(
            audio_file, beam_size=5, language="en", condition_on_previous_text=False
        )
        transcript = " ".join(segment.text.lstrip() for segment in segments)
        print(f"Transcribed and saved: {audio_file}")
        with open(
            os.path.join("data_train", "metadata.csv"), "a", encoding="utf-8"
        ) as csvfile:
            csvfile.write(f"{base_filename}|{transcript}|{transcript}\n")
    except Exception as e:
        print(f"Error transcribing {audio_file}: {e}")


for filename in os.listdir(audio_dir):
    if filename.endswith(".wav"):
        audio_file = os.path.join(audio_dir, filename)
        transcribe_and_save(audio_file)
