import time
import threading
from pathlib import Path
from .audio_utils import AudioProcessor
from .inference_styleTTS2 import StyleTTS2Inference
from .docker_utils import DockerHandler
from .ollama_utils import OllamaHandler
from .emotion_utils import EmotionHandler, EMOTION_MAPPING
from .TTS_utils import InferenceHandler


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
        self.styles_were_cached = None  # Store initial cache state

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
        self.styles_were_cached = True
        unique_styles = set(EMOTION_MAPPING.values())
        for style_file in unique_styles:
            style_path = f"asset/ref_sound/{style_file}"
            if not self.tts_model.is_style_cached(style_path):
                self.styles_were_cached = False
                break

        if self.styles_were_cached:
            print("Using cached styles")
            print("Warming up inference...")
        else:
            print("Computing reference styles...")

        tasks = {
            "whisper": (AudioProcessor, [], {}),
            "emotion": (self._init_emotion_classifier, [], {}),
        }

        if self.styles_were_cached:
            # When styles are cached, we can load and use them directly
            cached_style = self.tts_model.get_cached_style(
                "asset/ref_sound/neutral.wav"
            )

            def warmup_task():
                return self.tts_model.inference(
                    text="This is a warm-up inference.",
                    ref_s=cached_style,
                    alpha=0.3,
                    beta=0.7,
                    diffusion_steps=5,
                    embedding_scale=1.0,
                )

            tasks["warmup"] = (warmup_task, [], {})
            tasks["ref_style"] = (lambda: cached_style, [], {})
        else:
            tasks["ref_style"] = (self._cache_all_styles, [], {})

        self.results.update(self._run_parallel_tasks(tasks))
        self.audio_processor = self.results["whisper"]["result"]
        ref_style = self.results["ref_style"]["result"]
        self.emotion_handler = self.results["emotion"]["result"]

        # Print timings after operations
        print(
            f"\nEmotion classifier initialization took {self.results['emotion']['time']:.2f}s"
        )
        print(f"Whisper initialization took {self.results['whisper']['time']:.2f}s")
        if "style_computation" in self.results:
            print(
                f"Styles computation took {self.results['style_computation']['time']:.2f}s"
            )

        # Calculate group2 time before warmup for non-cached case
        self.group2_time = time.time() - group2_start

        # Handle warmup for non-cached case
        if not self.styles_were_cached:
            print("\nWarming up inference...")
            warmup_start = time.time()
            _ = self.tts_model.inference(
                text="This is a warm-up inference.",
                ref_s=ref_style,
                alpha=0.3,
                beta=0.7,
                diffusion_steps=5,
                embedding_scale=1.0,
            )
            self.results["warmup"] = {"time": time.time() - warmup_start}
            print(f"Warm-up took {self.results['warmup']['time']:.2f}s")
        else:
            print(f"Warm-up took {self.results['warmup']['time']:.2f}s")

    def _cache_all_styles(self):
        """Cache all unique style files"""
        unique_styles = set(EMOTION_MAPPING.values())

        # Quick check if all styles are cached
        all_cached = True
        for style_file in unique_styles:
            style_path = f"asset/ref_sound/{style_file}"
            if not self.tts_model.is_style_cached(style_path):
                all_cached = False
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

        self.results["style_computation"] = {"time": time.time() - style_start}
        return self.tts_model.compute_style("asset/ref_sound/neutral.wav")

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
        # Calculate actual group times including overlap
        print(f"\nTotal initialization time: {time.time() - self.init_start:.2f}s")
        print(f"├─ Docker setup: {self.docker_time:.2f}s")
        print(f"├─ Group 1: {self.group1_time:.2f}s")
        print(f"│  ├─ Ollama warm-up")
        print(f"│  └─ StyleTTS2")

        if self.styles_were_cached:
            # When styles were initially cached, show warm-up as part of Group 2
            print(f"└─ Group 2: {self.group2_time:.2f}s")
            print(f"   ├─ Whisper")
            print(f"   ├─ Reference styles")
            print(f"   ├─ Emotion classifier")
            if "warmup" in self.results:
                print(f"   └─ Inference warm-up")
        else:
            # When styles needed computation, show warm-up separately
            print(f"├─ Group 2: {self.group2_time:.2f}s")
            print(f"│  ├─ Whisper")
            print(f"│  ├─ Reference styles")
            print(f"│  └─ Emotion classifier")
            if "warmup" in self.results:
                print(f"└─ Inference warm-up: {self.results['warmup']['time']:.2f}s")

        return {
            "docker_handler": self.docker_handler,
            "tts_model": self.tts_model,
            "audio_processor": self.audio_processor,
            "emotion_handler": self.emotion_handler,
            "inference_handler": self.inference_handler,
            "warmup_time": self.warmup_time,
        }
