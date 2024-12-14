import time
import threading
from pathlib import Path
from .audio_utils import AudioProcessor
from .inference_styleTTS2 import StyleTTS2Inference
from .docker_utils import DockerHandler
from .ollama_utils import OllamaHandler
from .emotion_utils import EmotionHandler, EMOTION_MAPPING
from .StyleTTS_utils import InferenceHandler


class InitializationHandler:
    def __init__(self):
        self.init_start = time.time()
        self.docker_handler = None
        self.tts_model = None
        self.audio_processor = None
        self.emotion_handler = None
        self.inference_handler = None
        self.ollama_handler = None
        self.warmup_time = None

        # Timing information
        self.group1_time = None
        self.group2_time = None
        self.results = {}

    def _run_parallel_tasks(self, tasks):
        """Run tasks in parallel and collect results"""
        threads = []
        results = {}

        for name, (func, args, kwargs) in tasks.items():

            def wrapper(n=name, f=func, a=args, k=kwargs):
                start = time.time()
                result = f(*a, **k)
                results[n] = {"result": result, "time": time.time() - start}

            thread = threading.Thread(target=wrapper, daemon=True)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        return results

    def initialize_all(self):
        """Initialize all components in the correct order"""
        # Create cache directory
        Path("./cache").mkdir(exist_ok=True)

        # Initialize Docker first
        docker_start = time.time()
        self.docker_handler = DockerHandler()
        self.docker_time = time.time() - docker_start

        # Group 1: Ollama warmup and StyleTTS2 init
        self._initialize_group1()

        # Group 2: Whisper, reference style, and emotion classifier
        self._initialize_group2()

        # Create inference handler
        self.inference_handler = InferenceHandler(self.tts_model, self.emotion_handler)

        return self._get_initialization_results()

    def _initialize_group1(self):
        """Initialize Ollama and StyleTTS2"""
        group1_start = time.time()
        print("\nWarming up Ollama...")
        print("Initializing StyleTTS2...")

        tasks = {
            "ollama": (OllamaHandler.initialize, [], {}),
            "tts": (StyleTTS2Inference, [], {}),
        }

        self.results = self._run_parallel_tasks(tasks)
        self.warmup_time = self.results["ollama"]["time"]
        self.tts_model = self.results["tts"]["result"]
        self.group1_time = time.time() - group1_start

        print(f"\nOllama warm-up took {self.warmup_time:.2f}s")
        print(f"StyleTTS2 initialization took {self.group1_time:.2f}s")

    def _initialize_group2(self):
        """Initialize Whisper, cache styles, and emotion classifier"""
        group2_start = time.time()
        print("\nInitializing Whisper model...")
        print("Initializing emotion classifier...")

        # Check if all styles are cached
        all_styles_cached = True
        unique_styles = set(EMOTION_MAPPING.values())
        for style_file in unique_styles:
            style_path = f"asset/ref_sound/{style_file}"
            if not self.tts_model.is_style_cached(style_path):
                all_styles_cached = False
                break

        if all_styles_cached:
            print("Using cached styles\n")
        else:
            print("Computing reference styles...\n")

        tasks = {
            "whisper": (AudioProcessor, [], {}),
            "ref_style": (self._cache_all_styles, [], {}),
            "emotion": (self._init_emotion_classifier, [], {}),
        }

        self.results.update(self._run_parallel_tasks(tasks))
        self.audio_processor = self.results["whisper"]["result"]
        ref_style = self.results["ref_style"]["result"]
        self.emotion_handler = self.results["emotion"]["result"]
        self.group2_time = time.time() - group2_start

        print(
            f"Emotion classifier initialization took {self.results['emotion']['time']:.2f}s"
        )
        print(f"Whisper initialization took {self.results['whisper']['time']:.2f}s")

    def _cache_all_styles(self):
        """Cache all unique style files"""
        unique_styles = set(EMOTION_MAPPING.values())

        # Quick check if all styles are cached
        all_cached = True
        for style_file in unique_styles:
            style_path = f"asset/ref_sound/{style_file}"
            if not self.tts_model.is_style_cached(style_path):
                all_cached = True
                break

        if all_cached:
            # Just load and return neutral style without computation
            return self.tts_model.get_cached_style("asset/ref_sound/neutral.wav")

        # If not all cached, compute missing styles
        style_start = time.time()
        for style_file in unique_styles:
            style_path = f"asset/ref_sound/{style_file}"
            if not self.tts_model.is_style_cached(style_path):
                self.tts_model.compute_style(style_path)

        style_time = time.time() - style_start
        print(f"Styles computation took {style_time:.2f}s")

        return self.tts_model.get_cached_style("asset/ref_sound/neutral.wav")

    def _init_emotion_classifier(self):
        """Initialize the emotion classifier"""
        return EmotionHandler()

    def create_ollama_handler(self, gui=None):
        """Create and return an OllamaHandler instance"""
        self.ollama_handler = OllamaHandler(
            gui,
            self.tts_model,
            self.audio_processor,
            self.emotion_handler,
            self.inference_handler,
        )
        self.ollama_handler.warmup_time = self.warmup_time
        return self.ollama_handler

    def _get_initialization_results(self):
        """Return all initialized components and timing information"""
        total_init_time = time.time() - self.init_start

        # Calculate actual group times including overlap
        print(f"\nTotal initialization time: {total_init_time:.2f}s")
        print(f"├─ Docker setup: {self.docker_time:.2f}s")
        print(f"├─ Group 1: {self.group1_time:.2f}s")
        print(f"│  ├─ Ollama warm-up")
        print(f"│  └─ StyleTTS2")
        print(f"└─ Group 2: {self.group2_time:.2f}s")
        print(f"   ├─ Whisper")
        print(f"   ├─ Reference styles")
        print(f"   └─ Emotion classifier")

        return {
            "docker_handler": self.docker_handler,
            "tts_model": self.tts_model,
            "audio_processor": self.audio_processor,
            "emotion_handler": self.emotion_handler,
            "inference_handler": self.inference_handler,
            "warmup_time": self.warmup_time,
        }
