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
        model_name = os.getenv('VOICE_TYPE', 'amelia_watson')
        self.ref_audio_path = f"asset/ref_sound/{model_name}/neutral.wav"

        self.whisper_model = WhisperModel(
            "small",
            device="cuda" if torch.cuda.is_available() else "cpu",
            compute_type="float16" if torch.cuda.is_available() else "int8",
            download_root=str(cache_dir / "whisper"),
            num_workers=4,
        )

        # Voice input state
        self.is_listening = False

        # Pre-warm the model silently
        if torch.cuda.is_available() and Path(self.ref_audio_path).exists():
            self.transcribe_audio(self.ref_audio_path)
            torch.cuda.empty_cache()

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
        finally:
            self.is_listening = False
            gui.set_voice_button_text("Voice Input")

    def record_audio(self, is_listening_ref):
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        SILENCE_THRESHOLD = 350
        SILENCE_DURATION = 2

        p = pyaudio.PyAudio()
        frames = []
        silent_chunks = 0
        has_speech = False

        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )

        try:
            while is_listening_ref():
                data = stream.read(CHUNK)
                frames.append(data)

                # Convert audio chunk to numpy array for analysis
                audio_data = np.frombuffer(data, dtype=np.int16)
                volume = np.abs(audio_data).mean()

                if volume > SILENCE_THRESHOLD:
                    has_speech = True
                    silent_chunks = 0
                else:
                    silent_chunks += 1

                # Stop if silence for SILENCE_DURATION seconds
                if has_speech and silent_chunks > int(RATE / CHUNK * SILENCE_DURATION):
                    break

        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

        if has_speech and len(frames) > 0:
            output_path = self.output_dir / "recorded_input.wav"
            with wave.open(str(output_path), "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(p.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b"".join(frames))
            return str(output_path)

        return None

    def transcribe_audio(self, audio_file):
        """Transcribe audio file to text using Whisper"""
        segments, _ = self.whisper_model.transcribe(
            audio_file,
            beam_size=5,
            language="en",
            condition_on_previous_text=False,
            initial_prompt="Transcribe the following audio:",
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        return " ".join(segment.text.strip() for segment in segments)

    def play_audio(self, speech, sample_rate=24000, duration=None):
        """Play audio with optional duration"""
        if duration is None:
            duration = len(speech) / sample_rate

        sd.play(speech, samplerate=sample_rate)
        time.sleep(duration + 0.5)
        sd.stop()

    def play_audio_continuous(self, speech, sample_rate=24000):
        """Play audio continuously without blocking for anime-style playback"""
        sd.play(speech, samplerate=sample_rate)
        # Return immediately without blocking - caller manages timing
        return len(speech) / sample_rate

    def play_audio_continuous_improved(self, speech, sample_rate=24000):
        """Improved continuous audio playback with better session management"""
        # Stop any existing audio first
        sd.stop()
        
        # Start playback with proper configuration
        duration = len(speech) / sample_rate
        
        # Use blocking=False for non-blocking playback but ensure it stays active
        sd.play(speech, samplerate=sample_rate, blocking=False)
        
        print(f"Started audio playback: {duration:.2f}s, samples: {len(speech)}")
        
        return duration

    def is_audio_playing(self):
        """Check if audio is currently playing"""
        return sd.query_devices() is not None and hasattr(sd, '_last_callback') and sd._last_callback is not None

    def wait_for_audio_completion(self, duration):
        """Wait for audio to complete with proper monitoring"""
        import time
        start_time = time.time()
        
        while time.time() - start_time < duration + 0.5:  # Small buffer
            if not sd.get_stream().active:
                break
            time.sleep(0.1)

    def stop_audio(self):
        """Stop any currently playing audio"""
        sd.stop()
