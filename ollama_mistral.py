from transformers import (
    SpeechT5Processor,
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
)
from faster_whisper import WhisperModel
import tkinter as tk
from tkinter import scrolledtext, ttk
import threading
import torch
from PIL import Image, ImageTk
import time
import soundfile as sf
import os
import numpy as np
from speechbrain.inference import EncoderClassifier
import sounddevice as sd
import warnings
from pathlib import Path
import requests

warnings.filterwarnings("ignore")
os.makedirs("./cache", exist_ok=True)


class AnimatedAvatar:
    def __init__(self, canvas, width, height):
        self.canvas = canvas
        self.width = width
        self.height = height
        self.mouth_closed = Image.open("./asset/pic-close.png")
        self.mouth_open = Image.open("./asset/pic-open.png")
        self.mouth_closed = self.mouth_closed.resize((width, height))
        self.mouth_open = self.mouth_open.resize((width, height))
        self.mouth_closed_photo = ImageTk.PhotoImage(self.mouth_closed)
        self.mouth_open_photo = ImageTk.PhotoImage(self.mouth_open)
        self.image_on_canvas = self.canvas.create_image(
            0, 0, anchor=tk.NW, image=self.mouth_closed_photo
        )

    def animate_mouth(self, duration):
        frames = int(duration * 10)
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
        self.root.title("AI Assistant (Mistral)")
        self.root.geometry("700x400")

        cache_dir = Path("./cache")
        cache_dir.mkdir(exist_ok=True)

        # Initialize Whisper model
        self.whisper_model = WhisperModel(
            "tiny",  # Using tiny for faster processing
            device="cuda" if torch.cuda.is_available() else "cpu",
            download_root=str(cache_dir / "whisper"),
        )

        # Initialize SpeechT5
        self.processor = SpeechT5Processor.from_pretrained(
            "microsoft/speecht5_tts", cache_dir=cache_dir / "speecht5"
        )
        self.model = SpeechT5ForTextToSpeech.from_pretrained(
            "nonoJDWAOIDAWKDA/speecht5_finetuned_nono",
            cache_dir=cache_dir / "speecht5_finetuned",
        ).to("cuda" if torch.cuda.is_available() else "cpu")

        self.vocoder = SpeechT5HifiGan.from_pretrained(
            "microsoft/speecht5_hifigan",
            cache_dir=cache_dir / "hifigan",
        ).to("cuda" if torch.cuda.is_available() else "cpu")

        # Create speaker embeddings
        self.speaker_embeddings = self.create_speaker_embeddings()

        # Add system prompt as a class attribute
        self.system_prompt = """You are Amelia Watson, a time-traveling detective VTuber from Hololive English. Keep your response consise and under 30 words."""

        self.chat_history = []
        self.is_listening = False
        self.setup_gui()

    def create_speaker_embeddings(self):
        speaker_encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-xvect-voxceleb",
            savedir=os.path.join("./cache", "spkrec-xvect-voxceleb"),
            run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        )

        audio_path = "data_train/raw_audio/Sound 121.wav"
        waveform, sample_rate = sf.read(audio_path)

        if len(waveform.shape) > 1:
            waveform = waveform.mean(axis=1)

        if sample_rate != 16000:
            from scipy import signal

            samples = len(waveform)
            waveform = signal.resample(waveform, int(samples * 16000 / sample_rate))

        waveform = waveform / np.abs(waveform).max()

        with torch.no_grad():
            speaker_embeddings = speaker_encoder.encode_batch(
                torch.tensor(waveform).unsqueeze(0)
            )
            speaker_embeddings = torch.nn.functional.normalize(
                speaker_embeddings, dim=2
            )
            speaker_embeddings = speaker_embeddings.squeeze().cpu()
            return speaker_embeddings

    def process(self, text):
        if not text:
            return "I'm sorry, I couldn't process that."

        start_time = time.time()  # Start timing LLM processing

        # Add the new message to chat history
        self.chat_history.append({"role": "user", "content": text})

        # Construct the conversation history
        conversation = "\n".join(
            [
                f"{'Assistant' if msg['role'] == 'assistant' else 'User'}: {msg['content']}"
                for msg in self.chat_history[-4:]  # Keep last 4 messages for context
            ]
        )

        # Construct the complete prompt
        prompt = f"""{self.system_prompt}

Previous conversation:
{conversation}

User: {text}
Assistant:"""

        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "mistral",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                    },
                },
            ).json()

            response_text = response["response"].strip()

            llm_time = time.time() - start_time  # Calculate LLM processing time
            print(f"LLM Processing Time: {llm_time:.2f} seconds")

            # cleanup
            if not response_text.endswith((".", "!", "?")):
                response_text += "!"

            # Store the AI's response in chat history
            self.chat_history.append({"role": "assistant", "content": response_text})

            return response_text

        except Exception as e:
            print(f"Error in API call: {str(e)}")
            return "Sorry, I'm having trouble connecting to my brain right now!"

    def setup_gui(self):
        # GUI setup remains the same as original
        self.avatar_frame = ttk.Frame(self.root)
        self.avatar_frame.pack(side=tk.LEFT, padx=10, pady=10)
        self.avatar_canvas = tk.Canvas(self.avatar_frame, width=200, height=200)
        self.avatar_canvas.pack()
        self.avatar = AnimatedAvatar(self.avatar_canvas, 200, 200)

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

    def update_ui_and_speak(self, text):
        threading.Thread(
            target=self.speak_and_update_text, args=(text,), daemon=True
        ).start()

    def speak_and_update_text(self, text):
        try:
            start_time = time.time()

            inputs = self.processor(
                text=text.strip(),
                return_tensors="pt",
                padding=True,
                return_attention_mask=True,
            )

            speech = self.model.generate_speech(
                inputs["input_ids"].to(self.model.device),
                attention_mask=inputs["attention_mask"].to(self.model.device),
                speaker_embeddings=self.speaker_embeddings.unsqueeze(0).to(
                    self.model.device
                ),
                vocoder=self.vocoder,
            )

            tts_time = time.time() - start_time
            print(f"TTS Generation Time: {tts_time:.2f} seconds")

            self.root.after(0, lambda: self.text_area.insert(tk.END, f"AI: {text}\n"))
            self.root.after(0, self.text_area.see, tk.END)

            audio_data = speech.cpu().numpy()
            duration = len(audio_data) / 16000

            threading.Thread(
                target=self.avatar.animate_mouth, args=(duration,), daemon=True
            ).start()

            sd.play(audio_data, samplerate=16000)
            time.sleep(duration + 0.5)
            sd.stop()

        except Exception as e:
            print(f"Error in speech generation: {str(e)}")

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

    def listen(self):
        self.text_area.see(tk.END)
        try:
            audio_data = self.record_audio()
            if audio_data:  # Only process if we actually recorded something
                self.text_area.see(tk.END)
                self.process_audio(audio_data)
        except Exception as e:
            self.text_area.insert(tk.END, f"Error capturing audio: {str(e)}\n")
            self.text_area.see(tk.END)
        finally:
            self.is_listening = False
            self.voice_button.config(text="Voice Input")

    def record_audio(self):
        import pyaudio
        import wave
        import numpy as np

        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        SILENCE_THRESHOLD = 500  # Adjust this value based on your needs
        SILENCE_DURATION = 2  # Seconds of silence before stopping

        p = pyaudio.PyAudio()
        frames = []
        silent_chunks = 0
        has_speech = False

        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )

        try:
            while self.is_listening:
                data = stream.read(CHUNK)
                frames.append(data)

                # Convert audio chunk to numpy array for analysis
                audio_data = np.frombuffer(data, dtype=np.int16)
                volume = np.abs(audio_data).mean()

                if volume > SILENCE_THRESHOLD:
                    has_speech = True
                    silent_chunks = 0
                else:
                    silent_chunks += 1

                # Stop if silence for SILENCE_DURATION seconds
                if has_speech and silent_chunks > int(RATE / CHUNK * SILENCE_DURATION):
                    break

                # Also break if stop button was pressed
                if not self.is_listening:
                    break

        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

        # Only save and return if we detected speech
        if has_speech and len(frames) > 0:
            with wave.open("output.wav", "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(p.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b"".join(frames))
            return "output.wav"

        return None

    def process_audio(self, audio_file):
        try:
            segments, _ = self.whisper_model.transcribe(
                audio_file,
                beam_size=5,
                language="en",
                condition_on_previous_text=False,
            )

            text = " ".join(segment.text.strip() for segment in segments)

            self.text_area.insert(tk.END, f"You said: {text}\n")
            self.text_area.see(tk.END)
            self.process_input(text)

        except Exception as e:
            self.text_area.insert(tk.END, f"Error processing audio: {str(e)}\n")
            self.text_area.see(tk.END)


def main():
    root = tk.Tk()
    app = AIAssistant(root)
    root.mainloop()


if __name__ == "__main__":
    main()
