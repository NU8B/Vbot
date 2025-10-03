import time
import threading
from pathlib import Path
from .audio_utils import AudioProcessor
from .inference_styleTTS2 import StyleTTS2Inference
from .docker_utils import DockerHandler
from .ollama_utils import OllamaHandler
from .emotion_utils import EmotionHandler, EMOTION_MAPPING
from .TTS_utils import InferenceHandler
from .performance_boost import (
    memory_manager,
    startup_optimizer,
    model_cache,
    performance_monitor,
    LazyLoader,
    MemoryManager,
)


class InitializationHandler:
    def __init__(self, model_name="Amelia", device_index=None):
        self.init_start = time.time()
        self.model_name = model_name
        self.docker_handler = None
        self.tts_model = None
        self.audio_processor = None
        self.emotion_handler = None
        self.inference_handler = None
        self.ollama_handler = None
        self.warmup_time = None
        self.styles_were_cached = None  # Store initial cache state
        self.device_index = device_index

        # Timing information
        self.group1_time = None
        self.group2_time = None
        self.results = {}

        # Performance optimizations
        self.lazy_loader = LazyLoader()

        # Clear memory before starting and optimize VRAM
        memory_manager.clear_cache()
        MemoryManager.optimize_pytorch()  # Set VRAM limits
        MemoryManager.aggressive_vram_cleanup()  # Clean VRAM aggressively

        # Mark critical vs optional components
        startup_optimizer.mark_critical("docker")
        startup_optimizer.mark_critical("tts")
        startup_optimizer.mark_optional("ollama")
        startup_optimizer.mark_optional("whisper")
        startup_optimizer.mark_optional("emotion")
        startup_optimizer.mark_optional("ref_style")

    def _initialize_docker(self):
        """Initialize Docker handler"""
        start_time = time.time()
        docker_handler = DockerHandler()
        self.docker_time = time.time() - start_time  # Set docker_time for reporting
        return docker_handler

    def _initialize_tts(self):
        """Initialize TTS model - NO CACHING, FRESH INSTANCE EVERY TIME"""
        
        print(f"🔊 Creating COMPLETELY FRESH StyleTTS2Inference for model: {self.model_name}")
        print(f"🧵 Thread: {threading.current_thread().name}")
        print(f"🚫 NO CACHING - Fresh instance guaranteed")
        
        # Create completely fresh TTS model - no caching whatsoever
        tts_model = StyleTTS2Inference(model_name=self.model_name)
        
        print(f"✅ TTS model created - Name: {tts_model.model_name}, Repo: {tts_model.repo_id}")
        print(f"🆔 TTS model object ID: {id(tts_model)}")
        print(f"🔑 TTS unique ID: {getattr(tts_model, '_unique_id', 'N/A')}")

        return tts_model

    def _initialize_ollama(self):
        """Initialize Ollama with lazy loading"""
        return OllamaHandler.initialize()

    def _initialize_ref_style(self):
        """Initialize reference style with caching"""
        if not self.tts_model:
            # TTS model should already be created directly - this shouldn't happen
            print(f"⚠️ WARNING: TTS model not found for {self.model_name} during ref_style init")
            return None

        # Check if all styles are cached for this model
        self.styles_were_cached = True
        unique_styles = set(EMOTION_MAPPING.values())
        print(f"🔍 Checking reference styles for {self.model_name}...")
        
        for style_file in unique_styles:
            style_path = f"asset/ref_sound/{self.model_name}/{style_file}.wav"
            is_cached = self.tts_model.is_style_cached(style_path)
            print(f"   {style_file}.wav: {'✅ Cached' if is_cached else '❌ Not cached'}")
            if not is_cached:
                self.styles_were_cached = False

        if self.styles_were_cached:
            print("🚀 Using cached reference styles")
            neutral_path = f"asset/ref_sound/{self.model_name}/neutral.wav"
            return self.tts_model.get_cached_style(neutral_path)
        else:
            print("⏳ Computing reference styles...")
            return self._cache_all_styles()

    def get_audio_processor_when_ready(self):
        """Get audio processor when it's ready (non-blocking)"""
        if not hasattr(self, "_audio_processor_future"):
            # Start loading if not already started
            self._audio_processor_future = startup_optimizer.lazy_loader.load_async(
                "whisper", AudioProcessor, device_index=self.device_index
            )

        if self._audio_processor_future.done():
            try:
                return self._audio_processor_future.result()
            except Exception as e:
                print(f"Error loading audio processor: {e}")
                return None
        return None

    def get_ollama_when_ready(self):
        """Get Ollama handler when it's ready (non-blocking)"""
        return startup_optimizer.lazy_loader.get("ollama", timeout=0.1)

    def create_ollama_handler(self):
        """Create OllamaHandler with all required components"""
        # Wait for Ollama to be ready
        ollama = self.get_ollama_when_ready()
        if not ollama:
            print("⏳ Ollama not ready, waiting...")
            ollama = startup_optimizer.lazy_loader.get("ollama", timeout=30)

        # Get audio processor (may still be loading)
        audio_processor = self.get_audio_processor_when_ready()
        if not audio_processor:
            print("⏳ Audio processor not ready yet, will set later...")

        # Import OllamaHandler
        from .ollama_utils import OllamaHandler

        # Create the handler with all available components
        handler = OllamaHandler(
            tts_model=self.tts_model,
            audio_processor=audio_processor,  # May be None initially
            emotion_handler=self.emotion_handler,
            inference_handler=self.inference_handler,
            model_name=self.model_name,
        )
        
        # Explicitly set the model personality to ensure it's properly initialized
        print(f"🎭 Initializing {self.model_name} personality in OllamaHandler...")
        handler.set_model(self.model_name)

        return handler

    def update_ollama_handler_audio_processor(self, ollama_handler):
        """Update the OllamaHandler with audio processor when it's ready"""
        if not ollama_handler.audio_processor:
            audio_processor = self.get_audio_processor_when_ready()
            if audio_processor:
                print("✅ Audio processor ready, updating OllamaHandler...")
                ollama_handler.audio_processor = audio_processor
                return True
        return False

    def cleanup(self):
        """Clean up resources"""
        try:
            if hasattr(self, "lazy_loader"):
                self.lazy_loader.cleanup()
            memory_manager.clear_cache()
        except Exception as e:
            print(f"Error during cleanup: {e}")

    def _run_parallel_tasks(self, tasks):
        """Run tasks in parallel and collect results (legacy method)"""
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
        """Initialize all components with optimized startup sequence"""
        performance_monitor.start_timer("total_initialization")

        # Create cache directory
        Path("./cache").mkdir(exist_ok=True)

        # Phase 1: Critical components first (blocking)
        performance_monitor.start_timer("critical_components")
        
        # REMOVE TTS FROM SHARED LOADER - Create fresh instance for each character
        critical_loaders = {
            "docker": (self._initialize_docker, [], {}),
            # "tts": (self._initialize_tts, [], {}),  # REMOVED - no shared TTS
        }

        print("🚀 Loading critical components...")
        startup_optimizer.preload_critical(critical_loaders)

        # Get critical components
        self.docker_handler = startup_optimizer.lazy_loader.get("docker")
        
        # ALWAYS create a fresh TTS model for each character - NO SHARING
        print(f"🔄 Creating FRESH TTS model for {self.model_name} (NO CACHING)")
        self.tts_model = self._initialize_tts()  # Always call directly
        critical_time = performance_monitor.end_timer("critical_components")

        # Set group1_time for compatibility with old reporting
        self.group1_time = critical_time

        # Phase 2: Start optional components (non-blocking)
        performance_monitor.start_timer("optional_components_start")
        optional_loaders = {
            "ollama": (self._initialize_ollama, [], {}),
            "whisper": (AudioProcessor, [], {"device_index": self.device_index}),
            "emotion": (self._init_emotion_classifier, [], {}),
            "ref_style": (self._initialize_ref_style, [], {}),
        }

        print("⏳ Starting optional components in background...")
        startup_optimizer.start_optional_loading(optional_loaders)
        performance_monitor.end_timer("optional_components_start")

        # UI can now show while background loading continues
        print("✅ Critical components ready - UI can start!")

        # Phase 3: Wait for essential optional components
        performance_monitor.start_timer("essential_optional")
        essential_components = ["emotion", "ref_style"]

        for component in essential_components:
            print(f"⏳ Waiting for {component}...")
            result = startup_optimizer.lazy_loader.get(component, timeout=15)
            if component == "emotion":
                self.emotion_handler = result
            elif component == "ref_style":
                self.ref_style = result

        performance_monitor.end_timer("essential_optional")

        # Set group2_time for compatibility with old reporting
        self.group2_time = performance_monitor.get_metrics().get(
            "essential_optional", 0
        )

        # Create inference handler
        self.inference_handler = InferenceHandler(
            self.tts_model, self.emotion_handler, model_name=self.model_name
        )

        # Audio processor and Ollama can load in background
        performance_monitor.end_timer("total_initialization")
        performance_monitor.print_metrics()

        return self._get_initialization_results()

    def initialize_for_character_switch(self):
        """Initialize only components needed for character switching (no AudioProcessor)"""
        # Create cache directory
        Path("./cache").mkdir(exist_ok=True)

        # Group 1: Ollama warmup and StyleTTS2 init
        self._initialize_group1()

        # Group 2: Only emotion classifier and styles (no AudioProcessor)
        self._initialize_group2_for_switch()

        # Create inference handler
        self.inference_handler = InferenceHandler(
            self.tts_model, self.emotion_handler, model_name=self.model_name
        )

        return self._get_initialization_results()

    def _initialize_group1(self):
        """Initialize Ollama and StyleTTS2"""
        group1_start = time.time()
        print(f"Initializing StyleTTS2 ({self.model_name} model)...")

        tasks = {
            "ollama": (OllamaHandler.initialize, [], {}),
            "tts": (StyleTTS2Inference, [], {"model_name": self.model_name}),
        }

        self.results = self._run_parallel_tasks(tasks)
        self.warmup_time = self.results["ollama"]["time"]
        self.tts_model = self.results["tts"]["result"]
        self.group1_time = time.time() - group1_start

        print(f"\nOllama warm-up took {self.warmup_time:.2f}s")
        print(f"StyleTTS2 initialization took {self.group1_time:.2f}s")

    def _initialize_group2_for_switch(self):
        """Initialize only emotion classifier and styles (no AudioProcessor)"""
        group2_start = time.time()
        print("\nInitializing emotion classifier...")

        # Check if all styles are cached for this model
        self.styles_were_cached = True
        unique_styles = set(EMOTION_MAPPING.values())
        for style_file in unique_styles:
            style_path = f"asset/ref_sound/{self.model_name}/{style_file}.wav"
            if not self.tts_model.is_style_cached(style_path):
                self.styles_were_cached = False
                break

        if self.styles_were_cached:
            print("Using cached styles")
            print("Warming up inference...")
        else:
            print("Computing reference styles...")

        tasks = {
            "emotion": (self._init_emotion_classifier, [], {}),
        }

        if self.styles_were_cached:
            # When styles are cached, we can load and use them directly
            neutral_path = f"asset/ref_sound/{self.model_name}/neutral.wav"
            cached_style = self.tts_model.get_cached_style(neutral_path)

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

        # Handle ref_style result with error checking
        if "ref_style" in self.results and "result" in self.results["ref_style"]:
            ref_style = self.results["ref_style"]["result"]
        else:
            print("⚠️ Reference style computation failed, using empty cache")
            ref_style = {}

        self.emotion_handler = self.results["emotion"]["result"]

        # Print timings after operations
        print(
            f"\nEmotion classifier initialization took {self.results['emotion']['time']:.2f}s"
        )
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

    def _initialize_group2(self):
        """Initialize Whisper, cache styles, and emotion classifier"""
        group2_start = time.time()
        print("\nInitializing Whisper model...")
        print("Initializing emotion classifier...")

        # Check if all styles are cached for this model
        self.styles_were_cached = True
        unique_styles = set(EMOTION_MAPPING.values())
        for style_file in unique_styles:
            style_path = f"asset/ref_sound/{self.model_name}/{style_file}.wav"
            if not self.tts_model.is_style_cached(style_path):
                self.styles_were_cached = False
                break

        if self.styles_were_cached:
            print("Using cached styles")
            print("Warming up inference...")
        else:
            print("Computing reference styles...")

        tasks = {
            "whisper": (AudioProcessor, [], {"device_index": self.device_index}),
            "emotion": (self._init_emotion_classifier, [], {}),
        }

        if self.styles_were_cached:
            # When styles are cached, we can load and use them directly
            neutral_path = f"asset/ref_sound/{self.model_name}/neutral.wav"
            cached_style = self.tts_model.get_cached_style(neutral_path)

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
        """Cache all unique style files with error handling"""
        try:
            unique_styles = set(EMOTION_MAPPING.values())

            # Quick check if all styles are cached
            all_cached = True
            for style_file in unique_styles:
                # Add .wav extension to the style file
                style_path = f"asset/ref_sound/{self.model_name}/{style_file}.wav"
                if not self.tts_model.is_style_cached(style_path):
                    all_cached = False
                    break

            if all_cached:
                # Just load and return neutral style without computation
                neutral_path = f"asset/ref_sound/{self.model_name}/neutral.wav"
                return self.tts_model.get_cached_style(neutral_path)

            # If not all cached, compute missing styles
            style_start = time.time()
            for style_file in unique_styles:
                try:
                    # Add .wav extension to the style file
                    style_path = f"asset/ref_sound/{self.model_name}/{style_file}.wav"
                    if not self.tts_model.is_style_cached(style_path):
                        self.tts_model.compute_style(style_path)
                except Exception as e:
                    print(f"⚠️ Failed to compute style for {style_file}: {e}")
                    continue

            self.results["style_computation"] = {"time": time.time() - style_start}
            neutral_path = f"asset/ref_sound/{self.model_name}/neutral.wav"
            return self.tts_model.compute_style(neutral_path)
        except Exception as e:
            print(f"⚠️ Style computation failed: {e}")
            return {}

    def _init_emotion_classifier(self):
        """Initialize the emotion classifier"""
        return EmotionHandler(model_name=self.model_name)

    def create_ollama_handler(self, gui=None):
        """Create and return an OllamaHandler instance"""
        self.ollama_handler = OllamaHandler(
            gui,
            self.tts_model,
            self.audio_processor,
            self.emotion_handler,
            self.inference_handler,
            model_name=self.model_name,  # CRITICAL: Pass the correct model name!
        )
        self.ollama_handler.warmup_time = self.warmup_time
        return self.ollama_handler

    def _get_initialization_results(self):
        """Return all initialized components and timing information"""
        # Calculate actual group times including overlap
        print(f"\nTotal initialization time: {time.time() - self.init_start:.2f}s")

        # Only print docker time if it exists (for character switching)
        if hasattr(self, "docker_time") and self.docker_time is not None:
            print(f"├─ Docker setup: {self.docker_time:.2f}s")

        # Safe formatting for group times - handle None values
        group1_time = getattr(self, "group1_time", None)
        group2_time = getattr(self, "group2_time", None)

        if group1_time is not None:
            print(f"├─ Group 1: {group1_time:.2f}s")
            print(f"│  ├─ Ollama warm-up")
            print(f"│  └─ StyleTTS2")
        else:
            print("├─ Group 1: N/A (optimized loading)")

        if (
            group2_time is not None
            and getattr(self, "styles_were_cached", None) is not None
        ):
            if self.styles_were_cached:
                # When styles were initially cached, show warm-up as part of Group 2
                print(f"└─ Group 2: {group2_time:.2f}s")
                if (
                    hasattr(self, "audio_processor")
                    and self.audio_processor is not None
                ):
                    print(f"   ├─ Whisper")
                print(f"   ├─ Reference styles")
                print(f"   ├─ Emotion classifier")
                if hasattr(self, "results") and "warmup" in self.results:
                    print(f"   └─ Inference warm-up")
            else:
                # When styles needed computation, show warm-up separately
                print(f"├─ Group 2: {group2_time:.2f}s")
                if (
                    hasattr(self, "audio_processor")
                    and self.audio_processor is not None
                ):
                    print(f"│  ├─ Whisper")
                print(f"│  ├─ Reference styles")
                print(f"│  └─ Emotion classifier")
                if hasattr(self, "results") and "warmup" in self.results:
                    print(
                        f"└─ Inference warm-up: {self.results['warmup']['time']:.2f}s"
                    )
        else:
            print("└─ Group 2: N/A (optimized loading)")

        result = {
            "tts_model": self.tts_model,
            "emotion_handler": self.emotion_handler,
            "inference_handler": self.inference_handler,
            "warmup_time": self.warmup_time if hasattr(self, "warmup_time") else 0,
        }

        # Only include components that exist (for character switching)
        if hasattr(self, "docker_handler") and self.docker_handler is not None:
            result["docker_handler"] = self.docker_handler

        # Audio processor might still be loading
        audio_processor = self.get_audio_processor_when_ready()
        if audio_processor is not None:
            result["audio_processor"] = audio_processor
        else:
            # Create a lazy audio processor getter
            result["get_audio_processor"] = self.get_audio_processor_when_ready

        return result

    def cleanup(self):
        """Clean up resources"""
        try:
            if hasattr(self, "lazy_loader"):
                self.lazy_loader.cleanup()
            memory_manager.clear_cache()
        except Exception as e:
            print(f"Error during cleanup: {e}")

    def _safe_format_time(self, time_value, default="N/A"):
        """Safely format time values, handling None"""
        if time_value is None:
            return default
        try:
            return f"{time_value:.2f}s"
        except (TypeError, ValueError):
            return default
