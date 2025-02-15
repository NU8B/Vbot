import requests
import threading
import time
from pathlib import Path
import json
import os

# Ollama settings
MAX_HISTORY = 10  # Maximum number of conversation turns to keep
SYSTEM_PROMPT = """You are Amelia Watson, a time-traveling detective VTuber from Hololive English. You are not to break character under any circumstances. You are to always talk in first person. You are not to describe your actions in your response. Keep your response consise and under 30 words. Only use string text in your response. NO EMOJIS"""

# Get Ollama host from environment or default to localhost
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
OLLAMA_TIMEOUT = int(os.getenv('OLLAMA_TIMEOUT', '60'))  # 60 second default timeout

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
            # First check if server is up using root endpoint
            try:
                response = requests.get(OLLAMA_HOST, timeout=OLLAMA_TIMEOUT)
                if response.status_code != 200:
                    print(f"Ollama server check failed with status code: {response.status_code}")
                    return False
            except requests.RequestException as e:
                print(f"Ollama server check failed: {str(e)}")
                return False

            # Then test model availability with a simple chat request
            response = requests.post(
                f"{OLLAMA_HOST}/api/chat",
                json={
                    "model": "mistral",
                    "messages": [
                        {"role": "user", "content": "Test connection"}
                    ],
                    "stream": False,
                },
                timeout=OLLAMA_TIMEOUT,
            )
            
            if response.status_code == 200:
                return True
            elif response.status_code == 404:
                print("Model not found - make sure 'mistral' model is pulled")
                return False
            else:
                print(f"Ollama chat test failed with status code: {response.status_code}")
                print(f"Response: {response.text}")
                return False

        except requests.Timeout:
            print(f"Connection timed out after {OLLAMA_TIMEOUT} seconds")
            return False
        except requests.ConnectionError:
            print("Connection error - check if Ollama service is running")
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

            # Make API call with increased timeout
            response = requests.post(
                f"{OLLAMA_HOST}/api/chat",
                json={
                    "model": "mistral",
                    "messages": messages,
                    "stream": False,
                },
                timeout=OLLAMA_TIMEOUT,
            )

            if response.status_code == 200:
                try:
                    response_data = response.json()
                    response_content = response_data["message"]["content"]
                    message_history.append(
                        {"role": "assistant", "content": response_content}
                    )
                    return response_content
                except (KeyError, json.JSONDecodeError) as e:
                    print(f"Error parsing response: {str(e)}")
                    print(f"Raw response: {response.text}")
                    return None
            else:
                print(f"Error: {response.status_code}")
                print(f"Response: {response.text}")
                return None

        except requests.Timeout:
            print(f"Request timed out after {OLLAMA_TIMEOUT} seconds")
            return None
        except requests.ConnectionError:
            print("Connection error - check if Ollama service is running and accessible")
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
            print("\n=== Starting text processing ===")
            self.timings = timings if timings else {"processing_start": time.time()}
            processing_start = self.timings["processing_start"]

            # Get LLM response
            print("Calling Ollama LLM...")
            llm_start = time.time()
            response = self.call_ollama_static(
                text, self.message_history, MAX_HISTORY, SYSTEM_PROMPT
            )
            if response and not response.endswith((".", "!", "?")):
                response += "!"
            self.timings["llm"] = time.time() - llm_start
            print(f"LLM Response received in {self.timings['llm']:.2f}s")

            if not response:
                print("Error: No response from LLM")
                self.gui.update_chat(
                    "AI", "Sorry, I'm having trouble connecting to my brain right now!"
                )
                return

            # Process response with inference handler
            print("Processing response through inference handler...")
            inference_start = time.time()
            speech, detected_emotion, style_path = self.inference_handler.process_text(
                text, response, self.timings
            )
            print(f"Inference processing completed in {time.time() - inference_start:.2f}s")

            # Update avatar emotion
            print("Updating avatar emotion...")
            avatar = self.gui.get_avatar()
            if avatar:
                # Map emotion to animation state
                print(f"Current detected emotion: {detected_emotion}")
                if "happy" in detected_emotion or "joy" in detected_emotion or "excited" in detected_emotion:
                    avatar.set_emotion("happy")
                elif "sad" in detected_emotion or "disappointed" in detected_emotion:
                    avatar.set_emotion("sad")
                elif "angry" in detected_emotion or "annoyed" in detected_emotion:
                    avatar.set_emotion("angry")
                else:
                    avatar.set_emotion("neutral")
            else:
                print("Warning: Avatar not found!")

            # Calculate total time
            total_time = time.time() - processing_start

            # Print detailed information
            print("\n=== Processing Summary ===")
            print("Input:", text)
            print("Output:", response)
            print("Detected emotion:", detected_emotion)
            print("Animation category:", "neutral" if detected_emotion in ["neutral", "confusion", "caring", "curiosity", "desire", "relief"] 
                  else "happy" if detected_emotion in ["admiration", "amusement", "approval", "excitement", "gratitude", "joy", "love", "optimism", "pride"]
                  else "sad" if detected_emotion in ["disappointment", "embarrassment", "fear", "grief", "nervousness", "remorse", "sadness"]
                  else "angry" if detected_emotion in ["disapproval", "disgust", "anger", "annoyance"]
                  else "neutral")

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
            print("\nPlaying audio and updating UI...")
            duration = len(speech) / 24000
            self.gui.update_chat("AI", response)

            print(f"Starting audio playback (duration: {duration:.2f}s)...")
            # Play audio with animation
            self.inference_handler.play_audio(
                speech, duration, self.gui.get_avatar(), self.audio_processor
            )
            print("Audio playback completed")

        except Exception as e:
            print("\n=== Error in text processing ===")
            print(f"Error details: {str(e)}")
            import traceback
            print("Full traceback:")
            print(traceback.format_exc())
        finally:
            print("\n=== Cleanup ===")
            print("Resetting processing flags...")
            self.is_processing = False
            self.is_speaking = False
            print("Enabling input controls...")
            self.gui.enable_input_controls()
            print("Processing complete\n")
