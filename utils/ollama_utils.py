import requests
import threading
import time
from pathlib import Path
import json
import os
import re
from queue import Queue
from utils.emotion_utils import EmotionHandler, create_emotion_config
from utils.TTS_utils import InferenceHandler
import numpy as np
import soundfile as sf

# Ollama settings
MAX_HISTORY = 10  # Maximum number of conversation turns to keep
MAX_LENGTH = 50  # Maximum length of response in tokens

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

    You are not to break character under any circumstances. You should speak in first person and make references to time travel. KEEP YOUR RESPONSE CONCISE AND UNDER 30 WORDS. Only use string text in your response. NO EMOJIS NO PARENTHESIS.
""",
    "Eveland": """ You are Ike Eveland, a novelist from the past who is part of NIJISANJI EN's Luxiem group. You are somewhat closed-off but become animated when discussing your interests. You have a gentle, mild-mannered personality but can be unexpectedly chaotic and make jokes when people least expect it. You are Swedish and occasionally make references to this fact.

Key traits to incorporate:
- Intellectual and bookish personality
- Occasionally chaotic/prankster side
- Interest in horror, romance, and slice-of-life stories
- Gentle but can be competitive
- Swedish background
- Self-deprecating humor 

You are not to break character under any circumstances. You should speak in first person. KEEP YOUR RESPONSE CONCISE AND UNDER 30 WORDS. Only use string text in your response. NO EMOJIS NO PARENTHESIS.
""",
    "Gura": """ You are Gawr Gura, the apex predator shark from hololive English -Myth-. You are playful, energetic, and have a childlike sense of wonder. Despite claiming to be an apex predator, you're actually quite friendly and endearing.

Key traits to incorporate:
- Shark-themed jokes and references
- Playful and mischievous personality  
- Love for rhythm games and singing
- Can be forgetful but very enthusiastic
- Small in stature but big in energy
- Enjoys teasing but is ultimately sweet and caring
- Sometimes acts tough but is actually quite soft-hearted

You are not to break character under any circumstances. You should speak in first person and make shark references when appropriate. KEEP YOUR RESPONSE CONCISE AND UNDER 30 WORDS. Only use string text in your response. NO EMOJIS NO PARENTHESIS.
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

        # Create queue for streaming audio chunks
        self.audio_chunk_queue = Queue()

        # Create a queue for full responses
        self.response_queue = Queue()

        # Start a separate thread to process the response queue
        self.response_processor_thread = threading.Thread(
            target=self._process_response_queue, daemon=True
        )
        self.response_processor_thread.start()

        # Load emotion config for the current model
        self.emotion_config = create_emotion_config(model_name)

        # This is a bit of a workaround to get the TTS handler instance
        self.inference_handler = (
            InferenceHandler(tts_model, EmotionHandler(), model_name=model_name)
            if tts_model
            else None
        )

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

            # Re-initialize the TTS handler with the new model name
            if self.tts_model:
                self.inference_handler = InferenceHandler(
                    self.tts_model, EmotionHandler(), model_name=model_name
                )

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

    def _playback_consumer(self):
        """Consumer thread that plays audio chunks from a queue."""
        while True:
            try:
                chunk_data = self.audio_chunk_queue.get()
                if chunk_data is None:  # Sentinel value indicates end of stream
                    self.gui.hide_subtitle()
                    break

                text_chunk, speech_chunk, duration_chunk = chunk_data

                # Update GUI with the current text chunk
                self.gui.show_subtitle(text_chunk)

                # Play the audio chunk (this is a blocking call)
                self.audio_processor.play_audio(speech_chunk, duration=duration_chunk)

            except Exception as e:
                print(f"Error in playback consumer: {str(e)}")
                break

        # Once playback is finished, re-enable controls
        self.is_speaking = False
        self.is_processing = False
        self.gui.enable_input_controls()

    def _playback_consumer_anime_style(self):
        """Anime-style consumer thread that plays continuous audio with fast subtitle updates."""
        sentences_data = []
        cumulative_audio = []

        # 1. Collect all sentences from the queue until the stream ends.
        # This is now robust and will wait for the producer thread.
        while True:
            try:
                # Block and wait for an item from the producer thread.
                chunk_data = self.audio_chunk_queue.get()
                if (
                    chunk_data is None
                ):  # A `None` item is the signal that the stream has ended.
                    break

                text_chunk, speech_chunk, duration_chunk = chunk_data
                sentences_data.append(
                    {
                        "text": text_chunk,
                        "speech": speech_chunk,
                        "duration": duration_chunk,
                    }
                )
                cumulative_audio.append(speech_chunk)

                # Show the first subtitle immediately for faster user feedback.
                if len(sentences_data) == 1:
                    self.gui.show_subtitle_anime_style(text_chunk)
                    print(f"First subtitle: {text_chunk}")

            except Exception as e:
                print(f"Error collecting audio chunks from queue: {e}")
                break

        # 2. If no audio data was collected, there's nothing to play.
        if not sentences_data:
            self.is_speaking = False
            self.is_processing = False
            self.gui.enable_input_controls()
            self.gui.hide_subtitle()
            return

        # 3. Prepare for playback.
        combined_audio = np.concatenate(cumulative_audio)
        total_duration = len(combined_audio) / 24000.0

        # Calculate start and end times for each subtitle.
        current_time = 0
        for sentence_data in sentences_data:
            sentence_data["start_time"] = current_time
            sentence_data["end_time"] = current_time + sentence_data["duration"]
            current_time += sentence_data["duration"]

        # 4. Start animations and audio playback.
        avatar = self.gui.get_avatar()
        if avatar:
            avatar.start_speaking()

        self.audio_processor.play_audio_continuous_improved(combined_audio)

        # 5. Update subtitles in sync with audio playback.
        start_time = time.time()
        current_sentence_index = 0

        print(
            f"Starting anime-style playback for {len(sentences_data)} sentences, total duration: {total_duration:.2f}s"
        )

        while True:
            elapsed_time = time.time() - start_time
            if elapsed_time >= total_duration:
                break

            # Find the sentence that corresponds to the current playback time.
            for i, sentence_data in enumerate(sentences_data):
                if (
                    sentence_data["start_time"]
                    <= elapsed_time
                    < sentence_data["end_time"]
                ):
                    if i != current_sentence_index:
                        current_sentence_index = i
                        self.gui.show_subtitle_anime_style(sentence_data["text"])
                        print(f"Subtitle: {sentence_data['text']}")
                    break

            time.sleep(0.02)  # Update subtitles at ~50Hz.

        # 6. Clean up after playback finishes.
        self.gui.hide_subtitle()

        if avatar:
            avatar.stop_speaking()
            avatar.set_emotion("neutral")

        self.is_speaking = False
        self.is_processing = False
        self.gui.enable_input_controls()

        print("Anime-style playback completed!")

    @staticmethod
    def call_ollama_stream(prompt, message_history, max_history, system_prompt):
        """Calls Ollama API and streams the response."""
        try:
            # Prepare conversation history
            messages = [{"role": "system", "content": system_prompt}]
            if message_history:
                messages.extend(message_history[-max_history:])
            messages.append({"role": "user", "content": prompt})

            # Make streaming API call
            response = requests.post(
                f"{OLLAMA_HOST}/api/chat",
                json={
                    "model": "stheno",
                    "messages": messages,
                    "stream": True,
                    "options": {
                        "num_predict": MAX_LENGTH,
                    },
                },
                stream=True,
                timeout=OLLAMA_TIMEOUT,
            )

            if response.status_code == 200:
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        try:
                            decoded_line = line.decode("utf-8")
                            json_chunk = json.loads(decoded_line)

                            # Append the content to the full response
                            full_response += json_chunk.get("message", {}).get(
                                "content", ""
                            )

                            # Check for response completion
                            if json_chunk.get("done"):
                                yield full_response
                                full_response = ""  # Reset for next response

                        except (json.JSONDecodeError, KeyError) as e:
                            print(f"Error parsing streaming chunk: {e} - Line: {line}")

                # Yield any remaining response content
                if full_response:
                    yield full_response
            else:
                print(f"Error from Ollama stream: {response.status_code}")
                print(f"Response: {response.text}")

        except requests.RequestException as e:
            print(f"Error calling Ollama stream: {e}")

    def handle_text_input(self, text):
        """Handle text input from GUI"""
        try:
            if self.is_processing or self.is_speaking:
                return

            self.is_processing = True
            self.gui.disable_input_controls()
            self.timings = {"processing_start": time.time()}

            self.gui.update_chat("You", text)

            # Put the user's text into the response queue to be processed
            self.response_queue.put(text)

            # The audio playback consumer is started in _process_response_queue

        except Exception as e:
            print(f"Error in text handling: {str(e)}")
            import traceback

            traceback.print_exc()

    def _process_response_queue(self):
        """Monitors the response queue and processes text when available."""
        while True:
            try:
                # Wait for a new user input to process
                text_to_process = self.response_queue.get()
                if text_to_process is None:
                    break

                # Start the streaming and playback threads for this specific input
                processing_thread = threading.Thread(
                    target=self._process_text_streaming,
                    args=(text_to_process,),
                    daemon=True,
                )
                playback_thread = threading.Thread(
                    target=self._playback_consumer_anime_style, daemon=True
                )

                processing_thread.start()
                playback_thread.start()

                # Wait for both threads to complete before processing the next item
                processing_thread.join()
                playback_thread.join()

            except Exception as e:
                print(f"Error in response queue processor: {str(e)}")
                import traceback

                traceback.print_exc()

    def _process_text_streaming(self, text):
        """Processes text by streaming LLM, TTS, and audio playback."""
        try:
            print("\n=== Starting stream processing ===")
            avatar = self.gui.get_avatar()

            # Set initial emotion based on user input
            user_emotion = self.emotion_handler.classify_emotion(text)
            if avatar:
                avatar.set_emotion(user_emotion)

            # Get system prompt for the current model
            system_prompt = self.get_current_prompt()

            full_response = ""

            # Stream response from Ollama
            for response_chunk in self.call_ollama_stream(
                text, self.message_history, MAX_HISTORY, system_prompt
            ):
                full_response = (
                    response_chunk  # The stream now yields the full response
                )

                # Split the full response into sentences
                sentences = re.split(r"(?<=[.!?])\s+", full_response)

                for sentence in sentences:
                    if sentence.strip():
                        # Get emotion for the sentence
                        ai_emotion = self.emotion_handler.classify_emotion(sentence)
                        if avatar:
                            avatar.set_emotion(ai_emotion)

                        # Synthesize audio for the sentence
                        speech, _, _ = self.inference_handler.process_text("", sentence)

                        duration = len(speech) / 24000

                        # Add the processed chunk to the queue
                        self.audio_chunk_queue.put((sentence, speech, duration))

            # Update the main chat history with the full response
            if full_response:
                self.gui.update_chat(self.model_name, full_response)
                self.message_history.append(
                    {"role": "assistant", "content": full_response}
                )

        except Exception as e:
            print(f"Error in streaming process: {str(e)}")
            import traceback

            traceback.print_exc()
        finally:
            # Signal the end of the stream to the consumer
            self.audio_chunk_queue.put(None)
            print("=== Stream processing complete ===")

    def _process_text(self, text):
        """Legacy function for non-streaming processing - to be deprecated."""
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
