import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import tkinter as tk
import threading
import torch
import time
import warnings
from pathlib import Path
import requests
import shutil
import docker

from util.audio_utils import AudioProcessor
from util.gui import ChatGUI
from util.inference_styleTTS2 import StyleTTS2Inference

import os
import nltk
import nltk.data
from pathlib import Path

# Initialize NLTK
nltk.download("punkt", quiet=True)
nltk_data_dir = nltk.data.path[0]  # Get the first NLTK data directory

# Verify the file exists
punkt_file = Path(nltk_data_dir) / "tokenizers" / "punkt" / "english.pickle"
if punkt_file.exists():
    print(f"Found punkt file at: {punkt_file}")
else:
    print(f"Punkt file not found at expected location: {punkt_file}")

try:
    tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
    print("Punkt is properly installed")
except LookupError as e:
    print(f"Punkt is not installed correctly: {str(e)}")
    print(f"Current NLTK data paths: {nltk.data.path}")


# Suppress all warnings
warnings.filterwarnings("ignore")
os.makedirs("./cache", exist_ok=True)


class AIStyleTTS2:
    def __init__(self, root):
        # Initialize Docker client
        self.docker_client = docker.from_env()
        self.ensure_ollama_container()

        # Initialize components
        init_start = time.time()

        # Initialize audio processor (includes Whisper)
        self.audio_processor = AudioProcessor()
        whisper_time = self.audio_processor.init_time

        # Initialize StyleTTS2
        print("\nInitializing StyleTTS2...")
        tts_start = time.time()
        self.tts_model = StyleTTS2Inference()
        tts_time = time.time() - tts_start
        print(f"StyleTTS2 initialization took {tts_time:.2f}s")

        # Compute reference style
        print("\nComputing reference style...")
        style_start = time.time()
        self.ref_style = self.tts_model.compute_style("asset/ref.wav")
        style_time = time.time() - style_start
        print(f"Style computation took {style_time:.2f}s")

        # Initialize Ollama settings
        self.system_prompt = """You are Amelia Watson, a time-traveling detective VTuber from Hololive English. You are not to break character under any circumstances. You are to always talk in first person. Keep your response consise and under 30 words. Only use string text in your response. NO EMOJIS"""

        # Warm up Ollama
        print("\nWarming up Ollama...")
        warmup_start = time.time()
        response = self._call_ollama("This is a warmup prompt. Ignore.")
        warmup_time = time.time() - warmup_start
        if response is not None:
            print(f"Ollama warm-up complete! ({warmup_time:.2f}s)")
        else:
            print("Ollama warm-up failed (this is not critical)")

        # Initialize state variables
        self.is_processing = False
        self.is_speaking = False
        self.timings = {}

        # Initialize GUI with callbacks
        self.gui = ChatGUI(
            root,
            self.handle_text_input,
            lambda: self.audio_processor.toggle_listening(
                self.gui, self._process_text, self.is_processing
            ),
        )

        # Calculate total initialization time
        total_init_time = time.time() - init_start
        print(f"\nTotal initialization time: {total_init_time:.2f}s")
        print(f"├─ Whisper: {whisper_time:.2f}s")
        print(f"├─ StyleTTS2: {tts_time:.2f}s")
        print(f"├─ Reference style: {style_time:.2f}s")
        print(f"└─ Ollama warm-up: {warmup_time:.2f}s")

        # Store initialization times
        self.init_times = {
            "whisper": whisper_time,
            "styletts2": tts_time,
            "reference": style_time,
            "warmup": warmup_time,
            "total": total_init_time,
        }

    def ensure_ollama_container(self):
        """Ensure Ollama Docker container is running with GPU support"""
        try:
            # Check if container exists and is running
            try:
                container = self.docker_client.containers.get("ollama")
                if container.status != "running":
                    print("Starting existing Ollama container...")
                    container.start()
            except docker.errors.NotFound:
                print("Creating new Ollama container...")
                # Create and start the container with GPU support
                self.docker_client.containers.run(
                    "ollama/ollama",
                    name="ollama",
                    detach=True,
                    runtime="nvidia",
                    environment=["NVIDIA_VISIBLE_DEVICES=all"],
                    volumes={"ollama": {"bind": "/root/.ollama", "mode": "rw"}},
                    ports={"11434/tcp": 11434},
                )

            # Wait for Ollama API to be ready
            self._wait_for_ollama()

            # Check if Stheno model exists
            container = self.docker_client.containers.get("ollama")
            result = container.exec_run("ollama list")
            if "stheno" not in result.output.decode():
                print("Stheno model not found. Setting up...")
                # Pull the model from Hugging Face
                print("Pulling model from Hugging Face...")
                container.exec_run(
                    "ollama pull hf.co/featherless-ai-quants/bluuwhale-L3-SthenoMaidBlackroot-8B-V1-GGUF:bluuwhale-L3-SthenoMaidBlackroot-8B-V1-Q4_K_M.gguf"
                )

                # Copy to Stheno
                print("Setting up as Stheno...")
                container.exec_run(
                    "ollama cp hf.co/featherless-ai-quants/bluuwhale-L3-SthenoMaidBlackroot-8B-V1-GGUF:bluuwhale-L3-SthenoMaidBlackroot-8B-V1-Q4_K_M.gguf stheno"
                )

                # Clean up
                container.exec_run(
                    'ollama rm "hf.co/featherless-ai-quants/bluuwhale-L3-SthenoMaidBlackroot-8B-V1-GGUF:bluuwhale-L3-SthenoMaidBlackroot-8B-V1-Q4_K_M.gguf"'
                )
                print("Stheno model setup complete")
            else:
                print("Stheno model already exists")

            print("Ollama container is ready with GPU support")

        except Exception as e:
            print(f"Error setting up Docker container: {str(e)}")
            raise

    def _wait_for_ollama(self, timeout=60):
        """Wait for Ollama API to be ready"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                requests.get("http://localhost:11434/api/health")
                print("Ollama API is ready")
                return
            except requests.exceptions.RequestException:
                time.sleep(1)
        raise TimeoutError("Ollama API failed to become ready")

    def _call_ollama(self, prompt):
        """Call Ollama API for text generation using Docker container"""
        try:
            response = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": "stheno",
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    "stream": False,  # Default: False. Stream=True: ↓slower for short responses, ↑faster for long ones
                    "options": {
                        # Core Performance Parameters
                        "num_gpu": 100,  # Range: 0-100, Default: 50. ↑=faster (more GPU layers used). 0=CPU only
                        "num_thread": 32,  # Range: 1-n, Default: 4. ↑=faster on multi-core CPUs, but may saturate
                        "batch_size": 2048,  # Range: 1-2048, Default: 128. ↑=faster processing but ↑VRAM usage
                        "f16_kv": True,  # Default: false. True=faster on modern GPUs, halves VRAM usage
                        # Memory Management
                        "num_ctx": 8192,  # Range: 512-32000, Default: 2048. ↓=faster but less context, ↑=more context but slower
                        "num_keep": 5,  # Range: 1-25, Default: 5. ↓=faster & less memory, ↑=better context but slower
                        "num_beam": 1,  # Range: 1-n, Default: 1. ↑=slower but better quality. 1=fastest (greedy)
                        "num_gqa": 8,  # Range: 1-8, Default: 8. Model specific. ↑=faster on some models, no effect on others
                        # Generation Quality vs Speed
                        "temperature": 0.8,  # Range: 0.0-1.0, Default: 0.8. ↓=faster & focused, ↑=more creative but slower
                        "top_p": 0.9,  # Range: 0.0-1.0, Default: 0.9. ↓=faster & focused, ↑=more variety but slower
                        "top_k": 40,  # Range: 0-100, Default: 40. ↓=faster & focused, ↑=more variety but slower
                        "repeat_penalty": 1.1,  # Range: 1.0-2.0, Default: 1.1. ↑=slower but less repetition, ↓=faster but may repeat
                        # Advanced Tuning
                        "rope_frequency_base": 10000,  # Default: 10000. ↓=faster but may affect quality
                        "rope_frequency_scale": 1.0,  # Default: 1.0. Affects context scaling, optimal=1.0 for most cases
                        "mirostat": 0,  # Range: 0-2, Default: 0. 0=fastest, 2=slowest but most adaptive
                        "mirostat_eta": 0.1,  # Range: 0.0-1.0, Default: 0.1. Only affects speed if mirostat>0
                        "mirostat_tau": 5.0,  # Range: 0.0-10.0, Default: 5.0. Only affects speed if mirostat>0
                        # Optimization Flags
                        "use_flash_attn": True,  # Default: false. True=significantly faster on modern GPUs
                        "compress_pos_emb": 1.0,  # Range: 1.0-4.0, Default: 1.0. ↑=faster but experimental
                        # Reproducibility
                        # "seed": 42,  # Range: 0-MAX_INT, Default: random. Fixed seed can improve cache hits=faster
                    },
                },
            )

            if response.status_code == 200:
                return response.json()["message"]["content"]
            else:
                print(f"Error: {response.status_code}")
                return None

        except Exception as e:
            print(f"Error calling Ollama: {str(e)}")
            return None

    def handle_text_input(self, text):
        """Handle text input from GUI"""
        if self.is_processing or self.is_speaking:
            return

        self.is_processing = True
        self.gui.disable_input_controls()
        self.timings = {"processing_start": time.time()}

        self.gui.update_chat("You", text)
        threading.Thread(target=self._process_text, args=(text,), daemon=True).start()

    def _process_text(self, text, timings=None):
        """Process text input and generate response"""
        try:
            # Use provided timings or create new ones
            self.timings = timings if timings else {"processing_start": time.time()}
            processing_start = self.timings["processing_start"]

            # LLM Processing
            print(f"\nInput text: {text}")
            llm_start = time.time()
            response = self._call_ollama(text)

            # Add punctuation if missing
            if response and not response.endswith((".", "!", "?")):
                response += "!"

            llm_time = time.time() - llm_start
            self.timings["llm"] = llm_time

            if not response:
                self.gui.update_chat(
                    "AI", "Sorry, I'm having trouble connecting to my brain right now!"
                )
                return

            # Count tokens (approximate)
            token_count = len(response.split())
            tokens_per_second = token_count / llm_time if llm_time > 0 else 0
            print(f"Output text ({token_count} tokens): {response}")

            # Text to Speech
            tts_start = time.time()
            speech = self.tts_model.inference(
                text=response.strip(),
                ref_s=self.ref_style,
                alpha=0.3,
                beta=0.7,
                diffusion_steps=5,
                embedding_scale=1,
            )
            self.timings["tts"] = time.time() - tts_start

            # Calculate timing metrics
            total_time = time.time() - processing_start
            print(f"\nProcessing Times:")
            print(f"Total Time: {total_time:.2f}s")
            if "stt" in self.timings:
                print(f"├─ STT: {self.timings['stt']:.2f}s")
            print(f"├─ LLM: {llm_time:.2f}s ({tokens_per_second:.1f} tokens/s)")
            print(f"└─ TTS: {self.timings['tts']:.2f}s")

            # Update UI and play audio
            duration = len(speech) / 24000
            self.gui.update_chat("AI", response)

            # Start mouth animation before audio
            avatar = self.gui.get_avatar()
            threading.Thread(
                target=avatar.animate_mouth, args=(duration,), daemon=True
            ).start()

            # Small delay to ensure animation starts before audio
            time.sleep(0.1)
            self.audio_processor.play_audio(speech, duration=duration)

        except Exception as e:
            print(f"Error in text processing: {str(e)}")
        finally:
            self.is_processing = False
            self.is_speaking = False
            self.gui.enable_input_controls()

    def __del__(self):
        """Cleanup method to stop the container when the app closes"""
        try:
            container = self.docker_client.containers.get("ollama")
            container.stop()
        except:
            pass


def main():
    root = tk.Tk()
    app = AIStyleTTS2(root)
    root.mainloop()


if __name__ == "__main__":
    main()
