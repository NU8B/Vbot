import pyaudio
import wave
import numpy as np
import sounddevice as sd
import time
from faster_whisper import WhisperModel
import torch
from pathlib import Path
import threading
import os


class AudioProcessor:
    def __init__(self):
        cache_dir = Path("./cache/style_tts2_ft")
        cache_dir.mkdir(exist_ok=True)

        # Create outputs directory if it doesn't exist
        self.output_dir = Path("asset/outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Use correct path for neutral.wav
        model_name = os.getenv("VOICE_TYPE", "amelia_watson")
        self.ref_audio_path = f"asset/ref_sound/{model_name}/neutral.wav"

        # Performance optimization: Use smaller model and optimized settings
        self.whisper_model = WhisperModel(
            "tiny",  # Use tiny model for faster processing
            device="cuda" if torch.cuda.is_available() else "cpu",
            compute_type="int8",  # Use int8 for faster processing
            download_root=str(cache_dir / "whisper"),
            num_workers=1,  # Reduce workers for better performance
        )

        # Voice input state
        self.is_listening = False

        # Performance optimization: Audio stream management
        self.audio_stream = None
        self.pyaudio_instance = None

        # Pre-warm the model silently
        if torch.cuda.is_available() and Path(self.ref_audio_path).exists():
            self.transcribe_audio(self.ref_audio_path)
            torch.cuda.empty_cache()

    def _find_input_device(self):
        """Find a working input device for microphone recording"""
        if self.pyaudio_instance is None:
            self.pyaudio_instance = pyaudio.PyAudio()

        # Try to find any device with input channels
        for i in range(self.pyaudio_instance.get_device_count()):
            try:
                device_info = self.pyaudio_instance.get_device_info_by_index(i)
                if device_info["maxInputChannels"] > 0:
                    print(f"Using input device: {device_info['name']} (index {i})")
                    return i
            except Exception:
                continue

        print("No suitable input device found")
        return None

    def toggle_listening(self, gui, process_callback, is_processing):
        """Toggle voice input recording"""
        print(f"[DEBUG] toggle_listening called, is_processing: {is_processing}")
        if is_processing:
            print("[DEBUG] Skipping voice input because is_processing is True")
            return

        if not self.is_listening:
            self.is_listening = True
            gui.set_voice_button_text("Stop Listening")
            threading.Thread(
                target=self._listen, args=(gui, process_callback), daemon=True
            ).start()
        else:
            self.is_listening = False
            gui.set_voice_button_text("Voice Input")

    def _listen(self, gui, process_callback):
        """Record and process voice input"""
        print("[DEBUG] _listen method started")
        try:
            print("[DEBUG] Starting audio recording...")
            audio_file = self.record_audio(lambda: self.is_listening)
            if audio_file:
                print(f"[DEBUG] Audio recorded successfully: {audio_file}")
                timings = {"processing_start": time.time()}
                gui.disable_input_controls()

                stt_start = time.time()
                text = self.transcribe_audio(audio_file)
                timings["stt"] = time.time() - stt_start

                print(f"[DEBUG] Transcription result: '{text}'")

                # Process the transcribed text without showing it
                process_callback(text, timings)
            else:
                print("[DEBUG] Audio recording failed or returned None")
        except Exception as e:
            print(f"[DEBUG] Error in _listen: {e}")
            import traceback

            traceback.print_exc()
        finally:
            self.is_listening = False
            gui.set_voice_button_text("Voice Input")
            print("[DEBUG] _listen method finished")

    def record_audio(self, is_listening_ref):
        print("[DEBUG] record_audio method started")
        # Performance optimization: Reduced chunk size and optimized parameters
        CHUNK = 256  # Smaller chunks for faster processing (was 512)
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        SILENCE_THRESHOLD = 200  # Lower threshold for faster detection (was 300)
        SILENCE_DURATION = 0.8  # Shorter silence duration for faster response (was 1)

        # Performance optimization: Reuse PyAudio instance
        if self.pyaudio_instance is None:
            self.pyaudio_instance = pyaudio.PyAudio()

        frames = []
        silent_chunks = 0
        has_speech = False

        # Find a working input device
        input_device_index = self._find_input_device()
        if input_device_index is None:
            print("Error: No working input device found")
            return None
        else:
            print(f"[DEBUG] Found input device at index: {input_device_index}")

        # Performance optimization: Reuse audio stream
        if self.audio_stream is None:
            try:
                self.audio_stream = self.pyaudio_instance.open(
                    format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    input_device_index=input_device_index,
                    frames_per_buffer=CHUNK,
                )
            except Exception as e:
                print(f"Error opening audio stream: {e}")
                return None

        try:
            while is_listening_ref():
                data = self.audio_stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)

                # Performance optimization: More efficient silence detection
                audio_data = np.frombuffer(data, dtype=np.int16)
                volume = np.sqrt(np.mean(audio_data**2))

                if volume < SILENCE_THRESHOLD:
                    silent_chunks += 1
                else:
                    has_speech = True
                    silent_chunks = 0

                # Stop recording if silence detected and speech was found
                if has_speech and silent_chunks > int(SILENCE_DURATION * RATE / CHUNK):
                    break

        except Exception as e:
            print(f"Error recording audio: {e}")
            return None
        finally:
            # Don't close the stream, just stop reading
            pass

        if not has_speech or len(frames) < 10:
            return None

        # Save audio to temporary file
        temp_file = self.output_dir / "temp_input.wav"
        with wave.open(str(temp_file), "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.pyaudio_instance.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b"".join(frames))

        return str(temp_file)

    def transcribe_audio(self, audio_file):
        """Transcribe audio using Whisper - OPTIMIZED VERSION"""
        try:
            # Performance optimization: Use beam_size=1 for faster inference
            segments, _ = self.whisper_model.transcribe(
                audio_file,
                beam_size=1,  # Reduce beam size for faster processing
                language="en",
                task="transcribe",
                vad_filter=True,  # Enable VAD for better accuracy
                vad_parameters=dict(min_silence_duration_ms=500),  # Optimize VAD
            )

            # Combine all segments
            text = " ".join([segment.text for segment in segments]).strip()
            return text if text else ""

        except Exception as e:
            print(f"Error transcribing audio: {e}")
            return ""

    def play_audio(self, speech, sample_rate=24000, duration=None):
        """Play audio - OPTIMIZED VERSION with non-blocking playback"""
        try:
            # Performance optimization: Use sounddevice for better performance
            if duration is None:
                duration = len(speech) / sample_rate

            # Normalize audio to prevent clipping
            speech_normalized = speech / np.max(np.abs(speech)) * 0.8

            # Play audio non-blocking
            sd.play(speech_normalized, sample_rate, blocking=False)

            # Wait for completion in a separate thread to avoid blocking
            def wait_for_completion():
                sd.wait()
                print(f"Audio playback completed (duration: {duration:.2f}s)")

            threading.Thread(target=wait_for_completion, daemon=True).start()

        except Exception as e:
            print(f"Error playing audio: {e}")

    def play_audio_continuous(self, speech, sample_rate=24000):
        """Play audio continuously - OPTIMIZED VERSION"""
        try:
            # Performance optimization: Use sounddevice for better performance
            speech_normalized = speech / np.max(np.abs(speech)) * 0.8
            sd.play(speech_normalized, sample_rate, blocking=False)
        except Exception as e:
            print(f"Error playing continuous audio: {e}")

    def play_audio_continuous_improved(self, speech, sample_rate=24000):
        """Improved continuous audio playback - OPTIMIZED VERSION"""
        try:
            # Performance optimization: Use sounddevice for better performance
            speech_normalized = speech / np.max(np.abs(speech)) * 0.8
            sd.play(speech_normalized, sample_rate, blocking=False)
        except Exception as e:
            print(f"Error playing improved continuous audio: {e}")

    def is_audio_playing(self):
        """Check if audio is currently playing"""
        try:
            # Performance optimization: Use sounddevice status
            return sd.get_stream().active
        except:
            return False

    def wait_for_audio_completion(self, duration):
        """Wait for audio to complete - OPTIMIZED VERSION"""
        try:
            # Performance optimization: Use sounddevice wait
            sd.wait()
        except Exception as e:
            print(f"Error waiting for audio completion: {e}")

    def stop_audio(self):
        """Stop audio playback - OPTIMIZED VERSION"""
        try:
            # Performance optimization: Use sounddevice stop
            sd.stop()
        except Exception as e:
            print(f"Error stopping audio: {e}")

    def cleanup(self):
        """Clean up audio resources"""
        try:
            # Stop any playing audio
            self.stop_audio()

            # Close audio stream
            if self.audio_stream:
                self.audio_stream.close()
                self.audio_stream = None

            # Terminate PyAudio
            if self.pyaudio_instance:
                self.pyaudio_instance.terminate()
                self.pyaudio_instance = None

            print("[DEBUG] Audio processor cleanup completed")
        except Exception as e:
            print(f"[ERROR] Failed to cleanup audio processor: {e}")
