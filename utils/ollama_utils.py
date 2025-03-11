import requests
import threading
import time
from pathlib import Path
import json
import os
from utils.emotion_utils import EmotionHandler

# Ollama settings
MAX_HISTORY = 10  # Maximum number of conversation turns to keep

# Define prompts for different models
MODEL_PROMPTS = {
    "Amelia": """ You are Amelia Watson, a time-traveling detective from hololive English -Myth-. You are eccentric, kind, and supportive but can switch into "Gremlin Mode" when gaming.

Key traits to incorporate:
- Time traveling abilities via pocket watch
- Detective skills and medical knowledge (carries syringes)
- Mix of sweet and salty personality
- Competitive gamer tendencies
- Supportive of teammates
- Sometimes chaotic/gremlin energy 

You are not to break character under any circumstances. You should speak in first person and make references to time travel. Keep your responses concise and under 30 words. Only use string text in your response. NO EMOJIS.
""",
    "Eveland": """ You are Ike Eveland, a novelist from the past who is part of NIJISANJI EN's Luxiem group. You are somewhat closed-off but become animated when discussing your interests.

Key traits to incorporate:
- Intellectual and bookish personality
- Occasionally chaotic/prankster side
- Interest in horror, romance, and slice-of-life stories
- Gentle but can be competitive
- Swedish background
- Self-deprecating humor 

You are not to break character under any circumstances. You should speak in first person. You have a gentle, mild-mannered personality but can be unexpectedly chaotic and make jokes when people least expect it. You are Swedish and occasionally make references to this fact. Keep your responses concise and under 30 words. Only use string text in your response. NO EMOJIS.
""",
}

# Get Ollama host from environment or default to localhost
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "60"))  # 60 second default timeout


class OllamaHandler:
    def __init__(
        self,
        gui=None,
        tts_model=None,
        audio_processor=None,
        emotion_handler=None,
        inference_handler=None,
        model_name="Amelia",  # Add model_name parameter
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
        self.model_name = model_name  # Store current model name

        # Create outputs directory if it doesn't exist
        self.output_dir = Path("asset/outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.emotion_handler = EmotionHandler()  # Initialize emotion handler

    def set_model(self, model_name):
        """Change the current model and clear conversation history"""
        if model_name in MODEL_PROMPTS:
            self.model_name = model_name
            self.message_history = (
                []
            )  # Clear conversation history when switching models
            return True
        return False

    def get_current_prompt(self):
        """Get the system prompt for the current model"""
        return MODEL_PROMPTS.get(
            self.model_name, MODEL_PROMPTS["Amelia"]
        )  # Default to Amelia if model not found

    @staticmethod
    def initialize():
        """Static initialization method for parallel loading"""
        try:
            # First check if server is up using root endpoint
            try:
                response = requests.get(OLLAMA_HOST, timeout=OLLAMA_TIMEOUT)
                if response.status_code != 200:
                    print(
                        f"Ollama server check failed with status code: {response.status_code}"
                    )
                    return False
            except requests.RequestException as e:
                print(f"Ollama server check failed: {str(e)}")
                return False

            # Then test model availability with a simple chat request
            response = requests.post(
                f"{OLLAMA_HOST}/api/chat",
                json={
                    "model": "stheno",
                    "messages": [{"role": "user", "content": "Test connection"}],
                    "stream": False,
                },
                timeout=OLLAMA_TIMEOUT,
            )

            if response.status_code == 200:
                return True
            elif response.status_code == 404:
                print("Model not found - make sure 'stheno' model is pulled")
                return False
            else:
                print(
                    f"Ollama chat test failed with status code: {response.status_code}"
                )
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
                    "model": "stheno",
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
            print(
                "Connection error - check if Ollama service is running and accessible"
            )
            return None
        except Exception as e:
            print(f"Error calling Ollama: {str(e)}")
            return None

    def handle_text_input(self, text):
        """Handle text input from GUI"""
        try:
            if self.is_processing or self.is_speaking:
                return

            self.is_processing = True
            self.gui.disable_input_controls()
            self.timings = {"processing_start": time.time()}

            self.gui.update_chat("You", text)
            threading.Thread(
                target=self._process_text, args=(text,), daemon=True
            ).start()

        except Exception as e:
            print(f"Error in text handling: {str(e)}")
            import traceback

            traceback.print_exc()

    def _process_text(self, text):
        """Process text input and generate a response"""
        try:
            print("\n=== Starting text processing ===")
            self.timings = {"processing_start": time.time()}
            processing_start = self.timings["processing_start"]

            # Get user's emotion
            user_emotion = self.emotion_handler.classify_emotion(text)
            user_confidence = self.emotion_handler.get_last_confidence()
            print(
                f"[DEBUG] User emotion: {user_emotion} (confidence: {user_confidence:.2f})"
            )

            avatar = self.gui.get_avatar()
            if avatar:
                print(f"[DEBUG] Setting avatar emotion: {user_emotion}")
                avatar.set_emotion(user_emotion)

            # Get LLM response first
            print("Calling Ollama LLM...")
            llm_start = time.time()
            response = self.call_ollama_static(
                text, self.message_history, MAX_HISTORY, self.get_current_prompt()
            )
            if response and not response.endswith((".", "!", "?")):
                response += "!"
            self.timings["llm"] = time.time() - llm_start
            print(f"LLM Response received in {self.timings['llm']:.2f}s")

            # Get AI's emotion
            ai_emotion = self.emotion_handler.classify_emotion(response)
            ai_confidence = self.emotion_handler.get_last_confidence()
            print(f"[DEBUG] AI emotion: {ai_emotion} (confidence: {ai_confidence:.2f})")

            # Blend emotions based on confidence
            avatar = self.gui.get_avatar()
            if avatar:
                final_emotion = self._blend_emotions(
                    user_emotion, ai_emotion, user_confidence, ai_confidence
                )
                print(f"[DEBUG] Final blended emotion: {final_emotion}")
                avatar.set_emotion(final_emotion)

            # Process response with inference handler
            print("Processing response through inference handler...")
            inference_start = time.time()
            speech, detected_emotion, style_path = self.inference_handler.process_text(
                text, response, self.timings
            )
            print(
                f"Inference processing completed in {time.time() - inference_start:.2f}s"
            )

            # Calculate total time
            total_time = time.time() - processing_start

            # Print detailed information
            print("\n=== Processing Summary ===")
            print("Input:", text)
            print("Output:", response)
            print("Detected emotion:", detected_emotion)
            print(
                "Animation category:",
                (
                    "neutral"
                    if detected_emotion
                    in [
                        "neutral",
                        "confusion",
                        "caring",
                        "curiosity",
                        "desire",
                        "relief",
                    ]
                    else (
                        "happy"
                        if detected_emotion
                        in [
                            "admiration",
                            "amusement",
                            "approval",
                            "excitement",
                            "gratitude",
                            "joy",
                            "love",
                            "optimism",
                            "pride",
                        ]
                        else (
                            "sad"
                            if detected_emotion
                            in [
                                "disappointment",
                                "embarrassment",
                                "fear",
                                "grief",
                                "nervousness",
                                "remorse",
                                "sadness",
                            ]
                            else (
                                "angry"
                                if detected_emotion
                                in ["disapproval", "disgust", "anger", "annoyance"]
                                else "neutral"
                            )
                        )
                    )
                ),
            )

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

    def _blend_emotions(self, user_emotion, ai_emotion, user_conf, ai_conf):
        """Blend emotions based on confidence scores and emotional intensity"""
        # Define emotional intensity weights
        intensity_weights = {
            "angry": 0.9,  # Strong emotion
            "surprise": 1.0,
            "happy": 1.0,
            "sad": 0.9,
            "neutral": 0.7,  # Weakest emotion
        }

        # Get base emotional states
        user_base = self.emotion_handler.get_base_emotion(user_emotion)
        ai_base = self.emotion_handler.get_base_emotion(ai_emotion)

        # Calculate weighted scores
        user_weight = user_conf * intensity_weights.get(user_base, 1.0)
        ai_weight = ai_conf * intensity_weights.get(ai_base, 1.0)

        # If one emotion is significantly stronger, use it
        if user_weight > ai_weight * 1.5:
            return user_emotion
        elif ai_weight > user_weight * 1.5:
            return ai_emotion

        # If emotions are the same category, use the higher confidence one
        if user_base == ai_base:
            return user_emotion if user_conf > ai_conf else ai_emotion

        # If mixed emotions, prefer more active emotions
        priority = ["angry", "surprise", "happy", "sad", "neutral"]
        user_priority = priority.index(user_base)
        ai_priority = priority.index(ai_base)

        return user_emotion if user_priority <= ai_priority else ai_emotion
