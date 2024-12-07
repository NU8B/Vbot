import tkinter as tk
from tkinter import scrolledtext, ttk
from pathlib import Path
import threading
import torch
import time
import sounddevice as sd
import warnings
import numpy as np
from faster_whisper import WhisperModel
from ollama_speecht5 import AnimatedAvatar
import os
import requests
import re
from num2words import num2words

warnings.filterwarnings("ignore")
os.makedirs("./cache", exist_ok=True)


class AIStyleTTS2:
    def __init__(self, root):
        # Initialize GUI
        self.root = root
        self.root.title("AI Chatbot")
        self.root.geometry("700x400")

        cache_dir = Path("./cache/style_tts2_ft")
        cache_dir.mkdir(exist_ok=True)

        # Initialize Whisper model
        self.whisper_model = WhisperModel(
            "small",
            device="cuda" if torch.cuda.is_available() else "cpu",
            compute_type="int8",
            download_root=str(cache_dir / "whisper"),
        )

        # Initialize StyleTTS2
        from inference_styleTTS2 import StyleTTS2Inference

        self.tts_model = StyleTTS2Inference()

        # Load reference audio for style
        ref_path = "ref.wav"  # Using the same reference audio
        self.ref_style = self.tts_model.compute_style(ref_path)

        # Add system prompt as a class attribute
        self.system_prompt = """You are Amelia Watson, a time-traveling detective VTuber from Hololive English. Keep your response consise and under 30 words. Only use string text in your response. NO EMOJIS"""

        self.chat_history = []
        self.is_listening = False
        self.is_speaking = False
        self.is_processing = False
        self.setup_gui()

    def setup_gui(self):
        # GUI setup
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

    def process(self, text):
        if not text:
            return "I'm sorry, I couldn't process that."

        llm_start_time = time.time()

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
                    "model": "stheno",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                    },
                },
            ).json()

            response_text = response["response"].strip()

            self.llm_time = time.time() - llm_start_time

            # cleanup
            if not response_text.endswith((".", "!", "?")):
                response_text += "!"

            # Store the AI's response in chat history
            self.chat_history.append({"role": "assistant", "content": response_text})

            return response_text

        except Exception as e:
            print(f"Error in API call: {str(e)}")
            self.llm_time = time.time() - llm_start_time
            return "Sorry, I'm having trouble connecting to my brain right now!"

    def _process_text(self, text):
        try:
            processing_start = self.timings.get("processing_start", time.time())

            # LLM Processing
            llm_start = time.time()
            response = self.process(text)
            self.timings["llm"] = time.time() - llm_start

            # Clean the response text for TTS
            cleaned_response = self.clean_text_for_tts(response.strip())

            # Text to Speech
            tts_start = time.time()
            speech = self.tts_model.inference(
                text=cleaned_response,
                ref_s=self.ref_style,
                alpha=0.3,
                beta=0.7,
                diffusion_steps=5,
                embedding_scale=1,
            )
            self.timings["tts"] = time.time() - tts_start

            # Calculate and print timing metrics
            total_time = time.time() - processing_start

            print(f"Total Time: {total_time:.2f}s")
            if "stt" in self.timings:
                print(f"├─ STT: {self.timings['stt']:.2f}s")
            print(f"├─ LLM: {self.timings['llm']:.2f}s")
            print(f"├─ TTS: {self.timings['tts']:.2f}s")

            # Prepare audio playback
            duration = len(speech) / 24000  # StyleTTS2 uses 24kHz sampling rate

            # Update UI and start playback
            def update_and_play():
                self.text_area.insert(tk.END, f"AI: {response}\n")
                self.text_area.see(tk.END)
                threading.Thread(
                    target=self.avatar.animate_mouth, args=(duration,), daemon=True
                ).start()
                sd.play(speech, samplerate=24000)  # StyleTTS2 uses 24kHz
                time.sleep(duration + 0.5)  # Wait for audio to finish
                sd.stop()

                # Cleanup after audio is done
                self.is_processing = False
                self.is_speaking = False
                self.enable_input_controls()

            # Run audio playback in a separate thread
            threading.Thread(target=update_and_play, daemon=True).start()

        except Exception as e:
            print(f"Error in text processing: {str(e)}")
            self.is_processing = False
            self.is_speaking = False
            self.enable_input_controls()

    def on_send(self):
        if self.is_processing or self.is_speaking:
            return

        text = self.input_entry.get().strip()
        if text:
            self.is_processing = True
            self.disable_input_controls()
            self.timings = {}  # Initialize timings dictionary

            self.text_area.insert(tk.END, f"You: {text}\n")
            self.text_area.see(tk.END)
            self.input_entry.delete(0, tk.END)

            threading.Thread(
                target=self._process_text, args=(text,), daemon=True
            ).start()

    def disable_input_controls(self):
        self.root.after(
            0,
            lambda: [
                self.input_entry.config(state="disabled"),
                self.send_button.config(state="disabled"),
                self.voice_button.config(state="disabled"),
            ],
        )

    def enable_input_controls(self):
        self.root.after(
            0,
            lambda: [
                self.input_entry.config(state="normal"),
                self.send_button.config(state="normal"),
                self.voice_button.config(state="normal"),
            ],
        )

    def toggle_listening(self):
        if self.is_processing or self.is_speaking:
            return
        if not self.is_listening:
            self.is_listening = True
            self.voice_button.config(text="Stop Listening")
            threading.Thread(target=self.listen, daemon=True).start()
        else:
            self.is_listening = False
            self.voice_button.config(text="Voice Input")

    def listen(self):
        try:
            audio_data = self.record_audio()
            if audio_data:
                self.is_processing = True
                self.disable_input_controls()
                self.total_start_time = time.time()
                self.process_audio(audio_data)
        except Exception as e:
            self.text_area.insert(tk.END, f"Error capturing audio: {str(e)}\n")
            self.text_area.see(tk.END)
            self.is_processing = False
            self.enable_input_controls()
        finally:
            self.is_listening = False
            self.voice_button.config(text="Voice Input")

    def record_audio(self):
        import pyaudio
        import wave

        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        SILENCE_THRESHOLD = 500
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

        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

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
            processing_start = time.time()

            # Speech to Text
            stt_start = time.time()
            segments, _ = self.whisper_model.transcribe(
                audio_file,
                beam_size=5,
                language="en",
                condition_on_previous_text=False,
            )
            text = " ".join(segment.text.strip() for segment in segments)
            self.timings = {
                "stt": time.time() - stt_start,
                "processing_start": processing_start,
            }

            # Batch UI updates
            self.root.after(
                0, lambda: self.text_area.insert(tk.END, f"You said: {text}\n")
            )

            self._process_text(text)

        except Exception as e:
            print(f"Error in audio processing: {str(e)}")

    def normalize_text(self, text):
        """Normalize text by converting to lowercase and cleaning punctuation."""
        # Convert to lowercase
        text = text.lower()

        # Only remove specific punctuation, keeping important ones
        # Keep: periods, commas, question marks, exclamation marks
        text = re.sub(r"[^\w\s\',.!?]", "", text)

        # Ensure single spaces between words and punctuation
        text = " ".join(text.split())

        return text

    def clean_text_for_tts(self, text):
        """Process text through all cleaning steps."""
        print(f"Original text: {text}")

        # Remove speaker labels and colons
        text = re.sub(r"^.*?:", "", text).strip()

        # Convert numbers to words
        def replace_num(match):
            try:
                num = int(match.group())
                # Use 'year' for 4-digit numbers
                if len(str(num)) == 4:
                    return num2words(num, to="year")
                # Use 'ordinal' for dates, 'cardinal' for other numbers
                return num2words(num, to="cardinal")
            except ValueError:
                return match.group()

        # Convert numbers including years (e.g., 1920s)
        text = re.sub(r"\b\d+s?\b", lambda m: replace_num(m).replace("-", " "), text)

        # Preserve only allowed punctuation (.,!?) and handle spacing
        text = re.sub(r"[^a-zA-Z0-9\s,.!?]", "", text)
        text = re.sub(r"\s+([,.!?])", r"\1", text)
        text = re.sub(r"([,.!?])\s+", r"\1 ", text)

        # Ensure proper word separation
        text = " ".join(text.split())

        print(f"Clean text: {text}\n")
        return text

    def warm_up(self):
        """Warm up the LLM with a simple prompt to reduce initial latency."""
        try:
            print("Warming up LLM...")
            warm_up_prompt = f"""{self.system_prompt}

User: Hi there!
Assistant:"""

            requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "stheno",
                    "prompt": warm_up_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                    },
                },
            )
            print("LLM warm-up complete!")
        except Exception as e:
            print(f"LLM warm-up error (this is not critical): {str(e)}")


def main():
    root = tk.Tk()
    app = AIStyleTTS2(root)
    root.mainloop()


if __name__ == "__main__":
    main()
