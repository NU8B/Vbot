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
import sys


def get_base_path():
    """Get the base path for the application, handling PyInstaller bundled exe"""
    if getattr(sys, "frozen", False):
        # Running as compiled executable
        return sys._MEIPASS
    else:
        # Running as script
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class AudioProcessor:
    def __init__(self, device_index=None):
        # Get base path for PyInstaller compatibility
        base_path = get_base_path()

        cache_dir = Path(base_path) / "cache" / "style_tts2_ft"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Create outputs directory if it doesn't exist
        self.output_dir = Path(base_path) / "asset" / "outputs"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Use correct path for neutral.wav
        model_name = os.getenv("VOICE_TYPE", "amelia_watson")
        self.ref_audio_path = os.path.join(
            base_path, f"asset/ref_sound/{model_name}/neutral.wav"
        )

        # Performance optimization: Use smallest model and optimized settings
        # Smart device selection with cuDNN fallback
        whisper_device = self._select_whisper_device()

        self.whisper_model = WhisperModel(
            "tiny",  # Use tiny model for maximum speed
            device=whisper_device,
            compute_type="int8",  # Use int8 for maximum speed
            download_root=str(cache_dir / "whisper"),
            num_workers=1,  # Single worker for better performance
        )

        # Voice input state
        self.is_listening = False

        # Performance optimization: Audio stream management
        self.audio_stream = None
        self.pyaudio_instance = None

        # Microphone selection
        self.selected_device_index = device_index
        self.selected_device_name = None

        # Pre-warm the model silently
        if torch.cuda.is_available() and Path(self.ref_audio_path).exists():
            torch.cuda.empty_cache()

    def _select_whisper_device(self):
        """Device selection for Whisper STT.

        Defaults to CPU: ctranslate2's CUDA initialization crashes the
        process (access violation in WhisperModel.__init__) when it races
        torch's CUDA context during startup — observed 2026-07-06 with
        ctranslate2 4.7.1 + torch 2.7.1+cu128 while StyleTTS2 loads
        concurrently. The tiny int8 model transcribes faster than realtime
        on CPU, and keeping STT off the GPU avoids contention with TTS
        inference and avatar rendering. Set VBOT_WHISPER_DEVICE=cuda to
        opt back in for experimentation.
        """
        requested = os.getenv("VBOT_WHISPER_DEVICE", "cpu").lower()
        if requested == "cuda" and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def set_microphone(self, device_index, device_name):
        """Set the microphone to use for recording"""
        self.selected_device_index = device_index
        self.selected_device_name = device_name

        # Close existing audio stream to force recreation with new device
        if self.audio_stream:
            try:
                self.audio_stream.close()
            except:
                pass
            self.audio_stream = None

    def _find_input_device(self):
        """Find a working input device for microphone recording"""
        if self.pyaudio_instance is None:
            self.pyaudio_instance = pyaudio.PyAudio()

        # If a specific microphone is selected, try to use it
        if self.selected_device_index is not None:
            try:
                device_info = self.pyaudio_instance.get_device_info_by_index(
                    self.selected_device_index
                )
                if device_info["maxInputChannels"] > 0:
                    print(
                        f"[DEBUG] Using selected microphone: {device_info['name']} (index {self.selected_device_index})"
                    )
                    print(
                        f"[DEBUG] Device details: channels={device_info['maxInputChannels']}, sample_rate={device_info.get('defaultSampleRate', 'N/A')}"
                    )
                    return self.selected_device_index
                else:
                    print(
                        f"[DEBUG] Selected device {self.selected_device_index} has no input channels, falling back to auto-detection"
                    )
            except Exception:
                pass

        # Fall back to auto-detection if no specific device is selected or if it fails
        for i in range(self.pyaudio_instance.get_device_count()):
            try:
                device_info = self.pyaudio_instance.get_device_info_by_index(i)
                if device_info["maxInputChannels"] > 0:
                    return i
            except Exception:
                continue

        return None

    def toggle_listening(self, gui, process_callback, is_processing):
        """Toggle voice input recording"""
        if is_processing:
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
        try:
            audio_file = self.record_audio(lambda: self.is_listening)
            if audio_file:
                timings = {"processing_start": time.time()}
                gui.disable_input_controls()

                stt_start = time.time()
                text = self.transcribe_audio(audio_file)
                timings["stt"] = time.time() - stt_start

                # Process the transcribed text without showing it
                process_callback(text, timings)
        except Exception as e:
            import traceback

            traceback.print_exc()
        finally:
            self.is_listening = False
            gui.set_voice_button_text("Voice Input")

    def record_audio(self, is_listening_ref):
        # Performance optimization: Ultra-fast parameters for seamless experience
        CHUNK = 128  # Smaller chunks for faster processing (was 256)
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        SILENCE_THRESHOLD = 150  # Lower threshold for faster detection (was 200)
        SILENCE_DURATION = 0.6  # Shorter silence duration for faster response (was 0.8)

        # Performance optimization: Reuse PyAudio instance
        if self.pyaudio_instance is None:
            self.pyaudio_instance = pyaudio.PyAudio()

        frames = []
        silent_chunks = 0
        has_speech = False

        # Find a working input device
        input_device_index = self._find_input_device()
        if input_device_index is None:
            return None

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
            except Exception:
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

        except Exception:
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
        """Transcribe audio using Whisper - ULTRA OPTIMIZED VERSION"""
        try:
            # Performance optimization: Minimal beam size for maximum speed
            segments, _ = self.whisper_model.transcribe(
                audio_file,
                beam_size=1,  # Minimal beam size for maximum speed
                language="en",
                task="transcribe",
                vad_filter=False,  # Disable VAD for faster processing
                # Remove VAD parameters for speed
            )

            # Combine all segments quickly
            text = " ".join([segment.text for segment in segments]).strip()
            return text if text else ""

        except Exception:
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

            threading.Thread(target=wait_for_completion, daemon=True).start()

        except Exception:
            pass

    def play_audio_continuous(self, speech, sample_rate=24000):
        """Play audio continuously - OPTIMIZED VERSION"""
        try:
            # Performance optimization: Use sounddevice for better performance
            speech_normalized = speech / np.max(np.abs(speech)) * 0.8
            sd.play(speech_normalized, sample_rate, blocking=False)
        except Exception:
            pass

    def play_audio_continuous_improved(self, speech, sample_rate=24000):
        """Improved continuous audio playback - OPTIMIZED VERSION"""
        try:
            # Performance optimization: Use sounddevice for better performance
            speech_normalized = speech / np.max(np.abs(speech)) * 0.8
            sd.play(speech_normalized, sample_rate, blocking=False)
        except Exception:
            pass

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
        except Exception:
            pass

    def stop_audio(self):
        """Stop audio playback - OPTIMIZED VERSION"""
        try:
            # Performance optimization: Use sounddevice stop
            sd.stop()
        except Exception:
            pass

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

        except Exception:
            pass
