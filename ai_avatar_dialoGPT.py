from transformers import (
    SpeechT5Processor,
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from faster_whisper import WhisperModel
import tkinter as tk
from tkinter import scrolledtext, ttk
import threading
import torch
import pyaudio
import wave
from PIL import Image, ImageTk
import time
import soundfile as sf
import os
import numpy as np
from speechbrain.inference import EncoderClassifier
import sounddevice as sd
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

os.makedirs("./cache", exist_ok=True)


class AnimatedAvatar:
    def __init__(self, canvas, width, height):
        self.canvas = canvas
        self.width = width
        self.height = height

        # Load avatar images (you'll need to create these)
        self.mouth_closed = Image.open("./asset/pic-close.png")
        self.mouth_open = Image.open("./asset/pic-open.png")

        # Resize images
        self.mouth_closed = self.mouth_closed.resize((width, height))
        self.mouth_open = self.mouth_open.resize((width, height))

        # Convert to PhotoImage
        self.mouth_closed_photo = ImageTk.PhotoImage(self.mouth_closed)
        self.mouth_open_photo = ImageTk.PhotoImage(self.mouth_open)

        # Create image on canvas
        self.image_on_canvas = self.canvas.create_image(
            0, 0, anchor=tk.NW, image=self.mouth_closed_photo
        )

    def animate_mouth(self, duration):
        frames = int(duration * 10)  # 10 frames per second
        for i in range(frames):
            if i % 2 == 0:
                self.canvas.itemconfig(
                    self.image_on_canvas, image=self.mouth_open_photo
                )
            else:
                self.canvas.itemconfig(
                    self.image_on_canvas, image=self.mouth_closed_photo
                )
            self.canvas.update()
            time.sleep(0.1)
        self.canvas.itemconfig(self.image_on_canvas, image=self.mouth_closed_photo)


class AIAssistant:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Assistant")
        self.root.geometry("700x400")

        # Initialize Whisper model
        self.whisper_model = WhisperModel(
            "small", device="cuda" if torch.cuda.is_available() else "cpu"
        )

        # Initialize SpeechT5
        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        self.model = SpeechT5ForTextToSpeech.from_pretrained(
            "nonoJDWAOIDAWKDA/speecht5_finetuned_nono"
        )
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

        # Create speaker embeddings
        self.speaker_embeddings = self.create_speaker_embeddings()

        # Initialize chat models
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        self.chat_model = AutoModelForCausalLM.from_pretrained(
            "microsoft/DialoGPT-medium"
        )

        self.chat_history_ids = None
        self.is_listening = False

        self.setup_gui()

    def create_speaker_embeddings(self):
        # Initialize the speaker encoder with updated parameters
        speaker_encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-xvect-voxceleb",
            savedir=os.path.join("./cache", "spkrec-xvect-voxceleb"),
            run_opts={
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "local_strategy": "copy",
            },
        )

        # Load reference audio
        audio_path = "data_train/raw_audio/Sound 121.wav"
        waveform, sample_rate = sf.read(audio_path)

        # Convert to mono if stereo
        if len(waveform.shape) > 1:
            waveform = waveform.mean(axis=1)

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            from scipy import signal

            samples = len(waveform)
            waveform = signal.resample(waveform, int(samples * 16000 / sample_rate))

        # Normalize audio
        waveform = waveform / np.abs(waveform).max()

        # Convert to tensor
        with torch.no_grad():
            # Get speaker embeddings using SpeechBrain
            speaker_embeddings = speaker_encoder.encode_batch(
                torch.tensor(waveform).unsqueeze(0)
            )
            speaker_embeddings = torch.nn.functional.normalize(
                speaker_embeddings, dim=2
            )
            speaker_embeddings = speaker_embeddings.squeeze().cpu()

            return speaker_embeddings

    def setup_gui(self):
        # Create a frame for the avatar
        self.avatar_frame = ttk.Frame(self.root)
        self.avatar_frame.pack(side=tk.LEFT, padx=10, pady=10)

        # Create a canvas for the avatar
        self.avatar_canvas = tk.Canvas(self.avatar_frame, width=200, height=200)
        self.avatar_canvas.pack()

        # Create the animated avatar
        self.avatar = AnimatedAvatar(self.avatar_canvas, 200, 200)

        # Create a frame for the chat interface
        self.chat_frame = ttk.Frame(self.root)
        self.chat_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=10, pady=10)

        self.text_area = scrolledtext.ScrolledText(
            self.chat_frame, wrap=tk.WORD, width=50, height=20
        )
        self.text_area.pack(expand=True, fill=tk.BOTH)

        self.input_frame = ttk.Frame(self.chat_frame)
        self.input_frame.pack(pady=5)

        self.input_entry = ttk.Entry(self.input_frame, width=40)
        self.input_entry.pack(side=tk.LEFT, padx=5)

        self.send_button = ttk.Button(
            self.input_frame, text="Send", command=self.on_send
        )
        self.send_button.pack(side=tk.LEFT)

        self.voice_button = ttk.Button(
            self.input_frame, text="Voice Input", command=self.toggle_listening
        )
        self.voice_button.pack(side=tk.LEFT, padx=5)

    def listen(self):
        self.text_area.insert(tk.END, "Listening...\n")
        self.text_area.see(tk.END)
        try:
            audio_data = self.record_audio()
            self.text_area.insert(tk.END, "Processing audio...\n")
            self.text_area.see(tk.END)
            self.process_audio(audio_data)
        except Exception as e:
            self.text_area.insert(tk.END, f"Error capturing audio: {str(e)}\n")
            self.text_area.see(tk.END)
        finally:
            self.is_listening = False
            self.voice_button.config(text="Voice Input")

    def record_audio(self):
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        RECORD_SECONDS = 5

        p = pyaudio.PyAudio()

        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )

        frames = []

        for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()

        audio_data = b"".join(frames)

        # Save the audio as a WAV file
        with wave.open("output.wav", "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(audio_data)

        return "output.wav"

    def process_audio(self, audio_file):
        try:
            # Use faster-whisper for transcription
            segments, _ = self.whisper_model.transcribe(
                audio_file,
                beam_size=5,
                language="en",
                condition_on_previous_text=False,
            )

            # Join all segments
            text = " ".join(segment.text.strip() for segment in segments)

            self.text_area.insert(tk.END, f"You said: {text}\n")
            self.text_area.see(tk.END)
            self.process_input(text)

        except Exception as e:
            self.text_area.insert(tk.END, f"Error processing audio: {str(e)}\n")
            self.text_area.see(tk.END)

    def process(self, text):
        if text:
            # Remove clean_up_tokenization_spaces from encode
            new_user_input_ids = self.tokenizer.encode(
                text + self.tokenizer.eos_token, return_tensors="pt"
            )

            bot_input_ids = (
                torch.cat([self.chat_history_ids, new_user_input_ids], dim=-1)
                if self.chat_history_ids is not None
                else new_user_input_ids
            )

            self.chat_history_ids = self.chat_model.generate(
                bot_input_ids,
                max_length=1000,
                pad_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
                do_sample=True,
                top_k=100,
                top_p=0.7,
                temperature=0.8,
            )

            # Keep clean_up_tokenization_spaces only for decode
            response = self.tokenizer.decode(
                self.chat_history_ids[:, bot_input_ids.shape[-1] :][0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            return response
        return "I'm sorry, I couldn't process that."

    def update_ui_and_speak(self, text):
        self.text_area.insert(tk.END, f"AI: {text}\n")
        self.text_area.see(tk.END)
        threading.Thread(target=self.speak, args=(text,), daemon=True).start()

    def speak(self, text):
        # Process text through SpeechT5 with attention mask
        inputs = self.processor(
            text=text, return_tensors="pt", padding=True, return_attention_mask=True
        )

        # Generate speech with attention mask
        speech = self.model.generate_speech(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            speaker_embeddings=self.speaker_embeddings.unsqueeze(0),
            vocoder=self.vocoder,
        )

        audio_data = speech.numpy()

        # Calculate actual duration from audio length
        duration = len(audio_data) / 16000  # audio length / sample rate

        # Start mouth animation with actual duration
        threading.Thread(
            target=self.avatar.animate_mouth, args=(duration,), daemon=True
        ).start()

        # Play audio
        sd.play(audio_data, samplerate=16000)

        # Add a small buffer to ensure audio completes
        time.sleep(duration + 0.1)  # Add 100ms buffer
        sd.stop()

    def on_send(self):
        user_input = self.input_entry.get()
        if user_input:
            self.text_area.insert(tk.END, f"You: {user_input}\n")
            self.text_area.see(tk.END)
            self.input_entry.delete(0, tk.END)
            threading.Thread(
                target=self.process_input, args=(user_input,), daemon=True
            ).start()

    def process_input(self, user_input):
        response = self.process(user_input)
        self.root.after(0, self.update_ui_and_speak, response)

    def toggle_listening(self):
        if not self.is_listening:
            self.is_listening = True
            self.voice_button.config(text="Stop Listening")
            threading.Thread(target=self.listen, daemon=True).start()
        else:
            self.is_listening = False
            self.voice_button.config(text="Voice Input")


def main():
    root = tk.Tk()
    app = AIAssistant(root)
    root.mainloop()


if __name__ == "__main__":
    main()
