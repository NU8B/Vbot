import requests
import threading
import time
import torch

# Ollama settings
MAX_HISTORY = (
    4  # Maximum number of conversation turns to keep (excluding system prompt)
)
SYSTEM_PROMPT = """You are Amelia Watson, a time-traveling detective VTuber from Hololive English. You are not to break character under any circumstances. You are to always talk in first person. You are not to describe your actions in your response. Keep your response consise and under 30 words. Only use string text in your response. NO EMOJIS"""


class OllamaHandler:
    def __init__(self, gui, tts_model, audio_processor, ref_style):
        self.gui = gui
        self.tts_model = tts_model
        self.audio_processor = audio_processor
        self.ref_style = ref_style
        self.message_history = []
        self.is_processing = False
        self.is_speaking = False
        self.timings = {}

    @classmethod
    def initialize(cls):
        """Initialize OllamaHandler with warmup"""
        print("\nWarming up Ollama...")
        warmup_start = time.time()
        response = cls.call_ollama_static(
            "This is a warmup prompt. Ignore.", [], MAX_HISTORY, SYSTEM_PROMPT
        )
        warmup_time = time.time() - warmup_start
        if response is not None:
            print(f"Ollama warm-up took {warmup_time:.2f}s")
        else:
            print("Ollama warm-up failed (this is not critical)")
        return warmup_time

    @staticmethod
    def call_ollama_static(prompt, message_history, max_history, system_prompt):
        """Static version of call_ollama for initialization"""
        try:
            # Add new user message to history
            message_history.append({"role": "user", "content": prompt})

            # If history exceeds max length, remove oldest messages (excluding system prompt)
            while len(message_history) > max_history:
                message_history.pop(0)

            # Construct messages list with system prompt always first
            messages = [{"role": "system", "content": system_prompt}] + message_history

            response = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": "stheno",
                    "messages": messages,
                    "stream": False,
                    "options": {
                        # Core Performance Parameters
                        "num_gpu": 100,
                        "num_thread": 32,
                        "batch_size": 512,
                        "f16_kv": True,
                        # Memory Management
                        "num_ctx": 2048,
                        "num_keep": 25,  # +1 to account for system prompt
                        "num_beam": 1,
                        "num_gqa": 8,
                        # Generation Quality vs Speed
                        "temperature": 0.8,
                        "top_p": 0.9,
                        "top_k": 40,
                        "repeat_penalty": 1.1,
                        # Advanced Tuning
                        "rope_frequency_base": 10000,
                        "rope_frequency_scale": 1.0,
                        "mirostat": 0,
                        "mirostat_eta": 0.1,
                        "mirostat_tau": 5.0,
                        # Optimization Flags
                        "use_flash_attn": True,
                        "compress_pos_emb": 1.0,
                    },
                },
            )

            if response.status_code == 200:
                response_content = response.json()["message"]["content"]
                # Add assistant's response to history
                message_history.append(
                    {"role": "assistant", "content": response_content}
                )
                return response_content
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
            response = self.call_ollama_static(
                text, self.message_history, MAX_HISTORY, SYSTEM_PROMPT
            )

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
            with torch.inference_mode():
                torch.cuda.empty_cache()  # Clear any residual GPU memory
                speech = self.tts_model.inference(
                    text=response.strip(),
                    ref_s=self.ref_style,
                    alpha=0.3,
                    beta=0.7,
                    diffusion_steps=5,
                    embedding_scale=1.0,
                )
                torch.cuda.synchronize()  # Ensure GPU operations are complete
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
