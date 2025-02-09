import requests
import threading
import time
from pathlib import Path
import json

# Ollama settings
MAX_HISTORY = 10  # Maximum number of conversation turns to keep
SYSTEM_PROMPT = """You are Amelia Watson, a time-traveling detective VTuber from Hololive English. You are not to break character under any circumstances. You are to always talk in first person. You are not to describe your actions in your response. Keep your response consise and under 30 words. Only use string text in your response. NO EMOJIS"""


class OllamaHandler:
    def __init__(
        self,
        gui=None,
        tts_model=None,
        audio_processor=None,
        emotion_handler=None,
        inference_handler=None,
    ):
        self.gui = gui
        self.tts_model = tts_model
        self.audio_processor = audio_processor
        self.emotion_handler = emotion_handler
        self.inference_handler = inference_handler
        self.message_history = []
        self.is_processing = False
        self.is_speaking = False
        self.timings = {}
        self.warmup_time = None

        # Create outputs directory if it doesn't exist
        self.output_dir = Path("asset/outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def initialize():
        """Static initialization method for parallel loading"""
        try:
            # Test connection
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "mistral",
                    "prompt": "Test connection",
                    "stream": False,
                },
                timeout=30,
            )
            if response.status_code == 200:
                return True
            return False
        except Exception as e:
            print(f"Error initializing Ollama: {str(e)}")
            return False

    @staticmethod
    def call_ollama_static(prompt, message_history, max_history, system_prompt):
        """Static version of call_ollama for initialization"""
        try:
            # Prepare conversation history
            messages = [{"role": "system", "content": system_prompt}]

            # Add message history
            if message_history:
                messages.extend(message_history[-max_history:])

            # Add current prompt
            messages.append({"role": "user", "content": prompt})

            # Make API call
            response = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": "mistral",
                    "messages": messages,
                    "stream": False,
                },
                timeout=30,
            )

            if response.status_code == 200:
                response_content = response.json()["message"]["content"]
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
        """Process text input and generate a response"""
        try:
            self.timings = timings if timings else {"processing_start": time.time()}
            processing_start = self.timings["processing_start"]

            # Get LLM response
            llm_start = time.time()
            response = self.call_ollama_static(
                text, self.message_history, MAX_HISTORY, SYSTEM_PROMPT
            )
            if response and not response.endswith((".", "!", "?")):
                response += "!"
            self.timings["llm"] = time.time() - llm_start

            if not response:
                self.gui.update_chat(
                    "AI", "Sorry, I'm having trouble connecting to my brain right now!"
                )
                return

            # Process response with inference handler
            speech, detected_emotion, style_path = self.inference_handler.process_text(
                text, response, self.timings
            )

            # Update avatar emotion
            avatar = self.gui.get_avatar()
            if avatar:
                # Map emotion to animation state
                if "happy" in detected_emotion or "joy" in detected_emotion or "excited" in detected_emotion:
                    avatar.set_emotion("happy")
                elif "sad" in detected_emotion or "disappointed" in detected_emotion:
                    avatar.set_emotion("sad")
                elif "angry" in detected_emotion or "annoyed" in detected_emotion:
                    avatar.set_emotion("angry")
                else:
                    avatar.set_emotion("neutral")

            # Calculate total time
            total_time = time.time() - processing_start

            # Print detailed information
            print("\nInput:", text)
            print("Output:", response)
            print("Detected emotion:", detected_emotion)
            print(f"\nProcessing took {total_time:.2f}s")

            # Print timing breakdown
            if "stt" in self.timings:
                print(f"├─ STT: {self.timings['stt']:.2f}s")
            print(
                f"├─ LLM: {self.timings['llm']:.2f}s ({len(response.split()) / self.timings['llm']:.1f} words/s)"
            )
            print(f"├─ Emotion: {self.timings['emotion']:.2f}s")
            print(f"└─ TTS: {self.timings['tts']:.2f}s")

            # Update UI and play audio
            duration = len(speech) / 24000
            self.gui.update_chat("AI", response)

            # Play audio with animation
            self.inference_handler.play_audio(
                speech, duration, self.gui.get_avatar(), self.audio_processor
            )

        except Exception as e:
            print(f"Error in text processing: {str(e)}")
        finally:
            self.is_processing = False
            self.is_speaking = False
            self.gui.enable_input_controls()
