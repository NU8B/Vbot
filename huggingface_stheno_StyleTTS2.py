import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import tkinter as tk
import threading
import torch
import time
import warnings
from pathlib import Path

from util.audio_utils import AudioProcessor
from util.gui import ChatGUI
from util.inference_styleTTS2 import StyleTTS2Inference
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

# Suppress all warnings
warnings.filterwarnings("ignore")
os.makedirs("./cache", exist_ok=True)


class AIAssistant:
    def __init__(self, root):
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

        # Configure 4-bit quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        # Initialize LLM model and tokenizer
        cache_dir = Path("./cache")
        self.model_id = "bluuwhale/L3-SthenoMaidBlackroot-8B-V1"
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, cache_dir=cache_dir / "llm", trust_remote_code=True
        )
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            cache_dir=cache_dir / "llm",
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
        )

        # Add system prompt as a class attribute
        self.system_prompt = """You are Amelia Watson, a time-traveling detective VTuber from Hololive English. Keep your response consise and under 30 words. Only use string text in your response. NO EMOJIS"""

        # Initialize state variables
        self.is_processing = False
        self.is_speaking = False
        self.timings = {}
        self.chat_history = []

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

        # Store initialization times
        self.init_times = {
            "whisper": whisper_time,
            "styletts2": tts_time,
            "reference": style_time,
            "total": total_init_time,
        }

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
            response = self.process(text)

            # Add punctuation if missing
            if response and not response.endswith((".", "!", "?")):
                response += "!"

            llm_time = time.time() - llm_start
            self.timings["llm"] = llm_time

            if not response:
                self.gui.update_chat(
                    "AI", "Sorry, I'm having trouble thinking right now!"
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
            self.gui.update_chat("System", "Error processing response")
        finally:
            self.is_processing = False
            self.is_speaking = False
            self.gui.enable_input_controls()

    def process(self, text):
        """Process text through the LLM"""
        if not text:
            return "I'm sorry, I couldn't process that."

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
            inputs = self.tokenizer(prompt, return_tensors="pt").to(
                self.llm_model.device
            )
            outputs = self.llm_model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract only the assistant's response
            response_text = response_text.split("Assistant:")[-1].strip()

            # Store the AI's response in chat history
            self.chat_history.append({"role": "assistant", "content": response_text})

            return response_text

        except Exception as e:
            print(f"Error in model inference: {str(e)}")
            return None


def main():
    root = tk.Tk()
    app = AIAssistant(root)
    root.mainloop()


if __name__ == "__main__":
    main()
