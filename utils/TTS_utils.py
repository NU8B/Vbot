import threading
import time
import soundfile as sf
from pathlib import Path
from .emotion_utils import EMOTION_CONFIG, DIFFUSION_STEPS


class InferenceHandler:
    def __init__(self, tts_model, emotion_handler):
        self.tts_model = tts_model
        self.emotion_handler = emotion_handler
        self.timings = {}
        self.last_style = None

        # Create outputs directory if it doesn't exist
        self.output_dir = Path("asset/outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_text(self, text, response, timings=None):
        """Process text and generate speech with emotion."""
        self.timings = timings if timings else {}

        # Classify emotion
        emotion_start = time.time()
        detected_emotion = self.emotion_handler.classify_emotion(response)
        self.timings["emotion"] = time.time() - emotion_start

        # Get style file and parameters for emotion
        emotion_params = EMOTION_CONFIG[detected_emotion]
        style_path = f"asset/ref_sound/{emotion_params['file']}"

        # Get cached style
        current_ref_style = self.tts_model.get_cached_style(style_path)

        # Text to Speech with emotion-specific parameters
        tts_start = time.time()
        speech = self.tts_model.inference(
            text=response.strip(),
            ref_s=current_ref_style,
            alpha=emotion_params["alpha"],
            beta=emotion_params["beta"],
            diffusion_steps=DIFFUSION_STEPS,
            embedding_scale=emotion_params["embedding_scale"],
        )
        self.timings["tts"] = time.time() - tts_start

        return speech, detected_emotion, style_path

    def play_audio(self, speech, duration, avatar, audio_processor):
        """Play audio with avatar animation."""
        try:
            print("\n=== Starting audio playback ===")

            # Save audio to outputs directory
            print("Saving audio file...")
            output_path = self.output_dir / "output.wav"
            sf.write(str(output_path), speech, 24000)

            # Start avatar animation
            print("Starting avatar speaking animation...")
            avatar.start_speaking()

            # Small delay to ensure animation starts before audio
            time.sleep(0.1)

            # Play audio
            print(f"Playing audio (duration: {duration:.2f}s)...")
            audio_processor.play_audio(speech, duration=duration)
            print("Audio playback completed")

            # Stop avatar animation and reset to idle
            print("Stopping avatar speaking animation and resetting to idle...")
            avatar.stop_speaking()
            avatar.set_emotion("neutral")  # Reset to neutral/idle state
            print("=== Audio playback complete ===\n")

        except Exception as e:
            print(f"Error in audio playback: {str(e)}")
            import traceback

            print(traceback.format_exc())
            # Make sure to stop speaking animation and reset state even if there's an error
            if avatar:
                avatar.stop_speaking()
                avatar.set_emotion("neutral")
