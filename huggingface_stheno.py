from transformers import (
    SpeechT5Processor,
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig
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
import re
from num2words import num2words

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


from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig

class AIAssistant:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Assistant (Mistral)")
        self.root.geometry("700x400")

        cache_dir = Path("./cache")
        cache_dir.mkdir(exist_ok=True)

        # Initialize Whisper model
        self.whisper_model = WhisperModel(
            "small",
            device="cuda" if torch.cuda.is_available() else "cpu",
            compute_type="int8",
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

        # Configure 4-bit quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        # Initialize LLM model and tokenizer
        self.model_id = "bluuwhale/L3-SthenoMaidBlackroot-8B-V1"
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            cache_dir=cache_dir / "llm",
            trust_remote_code=True
        )
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            cache_dir=cache_dir / "llm",
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )

        # Create speaker embeddings
        self.speaker_embeddings = self.create_speaker_embeddings()

        # Add system prompt as a class attribute
        self.system_prompt = """You are Amelia Watson, a time-traveling detective VTuber from Hololive English. Keep your response consise and under 30 words. Only use string text in your response. NO EMOJIS"""

        self.chat_history = []
        self.is_listening = False
        self.is_speaking = False
        self.is_processing = False
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

        llm_start_time = time.time()

        # Add the new message to chat history
        self.chat_history.append({"role": "user", "content": text})

        # Construct the conversation history
        conversation = "\n".join(
            [
                f"{'Assistant' if msg['role'] == 'assistant' else 'User'}: {msg['content']}"
                for msg in self.chat_history[-4:]
            ]
        )

        # Construct the complete prompt
        prompt = f"""{self.system_prompt}

Previous conversation:
{conversation}

User: {text}
Assistant:"""

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.llm_model.device)
            outputs = self.llm_model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
            response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the assistant's response
            response_text = response_text.split("Assistant:")[-1].strip()

            self.llm_time = time.time() - llm_start_time

            # cleanup
            if not response_text.endswith((".", "!", "?")):
                response_text += "!"

            # Store the AI's response in chat history
            self.chat_history.append({"role": "assistant", "content": response_text})

            return response_text

        except Exception as e:
            print(f"Error in model inference: {str(e)}")
            self.llm_time = time.time() - llm_start_time
            return "Sorry, I'm having trouble thinking right now!"

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

    def process_audio(self, audio_file):
        try:
            processing_start = time.time()  # Start total timing here

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
                "processing_start": processing_start,  # Store the start time
            }

            # Batch UI updates
            self.root.after(
                0, lambda: self.text_area.insert(tk.END, f"You said: {text}\n")
            )

            self._process_text(text)

        except Exception as e:
            print(f"Error in audio processing: {str(e)}")

    def _process_text(self, text):
        try:
            processing_start = self.timings.get("processing_start", time.time())

            # LLM Processing
            llm_start = time.time()
            response = self.process(text)
            self.timings["llm"] = time.time() - llm_start

            # Clean the response text for TTS
            cleaned_response = self.clean_text_for_tts(response.strip())

            # Text to Speech - prepare inputs
            inputs = self.processor(
                text=cleaned_response,
                return_tensors="pt",
                padding=True,
                return_attention_mask=True,
            ).to(self.model.device)

            # Generate speech
            tts_start = time.time()
            with torch.inference_mode():
                speech = self.model.generate_speech(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    speaker_embeddings=self.speaker_embeddings.unsqueeze(0).to(
                        self.model.device
                    ),
                    vocoder=self.vocoder,
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
            audio_data = speech.cpu().numpy()
            duration = len(audio_data) / 16000

            # Update UI and start playback
            def update_and_play():
                self.text_area.insert(tk.END, f"AI: {response}\n")
                self.text_area.see(tk.END)
                threading.Thread(
                    target=self.avatar.animate_mouth, args=(duration,), daemon=True
                ).start()
                sd.play(audio_data, samplerate=16000)
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
                self.total_start_time = (
                    time.time()
                )  # Move timing start to here, after recording
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
        import numpy as np

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

        # Convert numbers to words using num2words
        def replace_num(match):
            try:
                return num2words(int(match.group()))
            except ValueError:
                return match.group()

        converted_text = re.sub(r"\b\d+\b", replace_num, text)

        final_text = self.normalize_text(converted_text)
        print(f"Clean text: {final_text}\n")

        return final_text


def main():
    root = tk.Tk()
    app = AIAssistant(root)
    root.mainloop()


if __name__ == "__main__":
    main()
