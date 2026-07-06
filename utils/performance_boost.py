"""
Performance optimization utilities for Vbot.
This module provides optimizations for faster boot times and reduced resource usage.
"""

import os
import gc
import torch
import threading
import time
from contextlib import contextmanager
from typing import Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
import psutil


class MemoryManager:
    """Manages GPU and system memory efficiently"""

    @staticmethod
    def optimize_pytorch():
        """Apply PyTorch optimizations for better performance"""
        if torch.cuda.is_available():
            # Enable memory efficient attention
            torch.backends.cuda.enable_flash_sdp(True)

            # Optimize memory allocation - Use only 40% of GPU memory to reduce VRAM usage
            torch.cuda.set_per_process_memory_fraction(0.4)  # Use max 40% of GPU memory

            # Enable memory pool for better allocation with smaller chunks
            os.environ.update(
                {
                    "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128,roundup_power2_divisions:8,garbage_collection_threshold:0.6",
                    "CUDA_LAUNCH_BLOCKING": "0",
                    "TOKENIZERS_PARALLELISM": "false",
                }
            )

    @staticmethod
    def clear_cache():
        """Clear GPU and system memory cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

    @staticmethod
    def aggressive_vram_cleanup():
        """Perform aggressive VRAM cleanup to minimize usage"""
        if torch.cuda.is_available():
            # Clear all caches multiple times
            for _ in range(3):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                gc.collect()

            # Reset memory stats
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()

    @staticmethod
    def set_low_vram_mode():
        """Configure PyTorch for minimal VRAM usage"""
        if torch.cuda.is_available():
            # Use only 25% of GPU memory for ultra low VRAM mode
            torch.cuda.set_per_process_memory_fraction(0.25)

            # Force immediate cleanup
            os.environ.update(
                {
                    "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:64,roundup_power2_divisions:4,garbage_collection_threshold:0.4,expandable_segments:False",
                }
            )

    @staticmethod
    def get_memory_info():
        """Get current memory usage information"""
        info = {"system_ram": psutil.virtual_memory().percent}

        if torch.cuda.is_available():
            info["gpu_memory"] = {
                "allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
                "cached": torch.cuda.memory_reserved() / 1024**3,  # GB
                "total": torch.cuda.get_device_properties(0).total_memory
                / 1024**3,  # GB
            }
        return info

    @staticmethod
    def log_memory(label):
        """Print a compact one-line memory snapshot for runtime diagnostics.

        Used around character switching and startup phases so memory
        behavior shows up in the console logs users already read.
        """
        info = MemoryManager.get_memory_info()
        gpu = info.get("gpu_memory")
        if gpu:
            peak = torch.cuda.max_memory_allocated() / 1024**3
            print(
                f"🧠 [{label}] RAM {info['system_ram']:.0f}% | "
                f"CUDA {gpu['allocated']:.2f}GB alloc, {gpu['cached']:.2f}GB "
                f"reserved, {peak:.2f}GB peak"
            )
        else:
            print(f"🧠 [{label}] RAM {info['system_ram']:.0f}% | CUDA n/a")
        return info


@contextmanager
def track_cuda_peak(label):
    """Measure the CUDA footprint of a component load.

    Reports retained allocation (still held after loading) and transient
    peak (high-water mark during loading). Only meaningful around
    sequential loads — parallel loaders would attribute each other's
    allocations. No-op without CUDA.
    """
    if not torch.cuda.is_available():
        yield
        return

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    yield
    torch.cuda.synchronize()
    retained = (torch.cuda.memory_allocated() - before) / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"🧠 [{label}] CUDA load cost: +{retained:.2f}GB retained, {peak:.2f}GB peak during load")


class LazyLoader:
    """Lazy loading implementation for heavy components"""

    def __init__(self):
        self._cache = {}
        self._loading_futures = {}
        self._thread_pool = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="LazyLoader"
        )

    def load_async(self, key: str, loader_func, *args, **kwargs) -> Future:
        """Start loading a component asynchronously"""
        if key in self._cache:
            # Already loaded, return completed future
            future = Future()
            future.set_result(self._cache[key])
            return future

        if key in self._loading_futures:
            # Already loading
            return self._loading_futures[key]

        # Start loading
        future = self._thread_pool.submit(loader_func, *args, **kwargs)
        self._loading_futures[key] = future

        # Cache result when done
        def cache_result(fut):
            try:
                result = fut.result()
                self._cache[key] = result
            except Exception as e:
                print(f"Error loading {key}: {e}")
            finally:
                self._loading_futures.pop(key, None)

        future.add_done_callback(cache_result)
        return future

    def get(self, key: str, timeout: float = None):
        """Get a loaded component, waiting if necessary"""
        if key in self._cache:
            return self._cache[key]

        if key in self._loading_futures:
            return self._loading_futures[key].result(timeout=timeout)

        return None

    def is_loaded(self, key: str) -> bool:
        """Check if a component is loaded"""
        return key in self._cache

    def is_loading(self, key: str) -> bool:
        """Check if a component is currently loading"""
        return key in self._loading_futures

    def cleanup(self):
        """Clean up the lazy loader"""
        self._thread_pool.shutdown(wait=False)


class StartupOptimizer:
    """Optimizes application startup sequence"""

    def __init__(self):
        self.lazy_loader = LazyLoader()
        self.critical_components = set()
        self.optional_components = set()

    def mark_critical(self, component_name: str):
        """Mark a component as critical (must load before UI shows)"""
        self.critical_components.add(component_name)

    def mark_optional(self, component_name: str):
        """Mark a component as optional (can load in background)"""
        self.optional_components.add(component_name)

    def preload_critical(self, loaders: Dict[str, tuple]):
        """Preload critical components in parallel"""
        futures = {}

        for name, (loader_func, args, kwargs) in loaders.items():
            if name in self.critical_components:
                futures[name] = self.lazy_loader.load_async(
                    name, loader_func, *args, **kwargs
                )

        # Wait for all critical components
        for name, future in futures.items():
            try:
                future.result(timeout=30)  # 30 second timeout
                print(f"✓ Critical component '{name}' loaded")
            except Exception as e:
                print(f"✗ Failed to load critical component '{name}': {e}")

    def start_optional_loading(self, loaders: Dict[str, tuple]):
        """Start loading optional components in background"""
        for name, (loader_func, args, kwargs) in loaders.items():
            if name in self.optional_components:
                self.lazy_loader.load_async(name, loader_func, *args, **kwargs)
                print(f"⏳ Started loading optional component '{name}'")


class ModelCache:
    """Caches model states and components for faster switching"""

    def __init__(self, cache_dir: str = "cache/models"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache = {}
        self._disk_cache_map = {}

    def cache_model_state(self, model_name: str, component_name: str, state_dict: dict):
        """Cache a model's state dictionary"""
        cache_key = f"{model_name}_{component_name}"

        # Save to memory for fast access
        self._memory_cache[cache_key] = state_dict

        # Save to disk for persistence
        cache_file = self.cache_dir / f"{cache_key}.pt"
        torch.save(state_dict, cache_file)
        self._disk_cache_map[cache_key] = cache_file

    def load_model_state(self, model_name: str, component_name: str) -> Optional[dict]:
        """Load a cached model state"""
        cache_key = f"{model_name}_{component_name}"

        # Try memory first
        if cache_key in self._memory_cache:
            return self._memory_cache[cache_key]

        # Try disk cache
        if cache_key in self._disk_cache_map:
            cache_file = self._disk_cache_map[cache_key]
            if cache_file.exists():
                state_dict = torch.load(cache_file, map_location="cpu")
                self._memory_cache[cache_key] = state_dict
                return state_dict

        return None

    def has_cached_state(self, model_name: str, component_name: str) -> bool:
        """Check if a model state is cached"""
        cache_key = f"{model_name}_{component_name}"
        return (
            cache_key in self._memory_cache
            or cache_key in self._disk_cache_map
            and self._disk_cache_map[cache_key].exists()
        )


class PerformanceMonitor:
    """Monitors performance metrics during runtime"""

    def __init__(self):
        self.metrics = {}
        self.start_times = {}

    def start_timer(self, operation: str):
        """Start timing an operation"""
        self.start_times[operation] = time.time()

    def end_timer(self, operation: str) -> float:
        """End timing an operation and return duration"""
        if operation in self.start_times:
            duration = time.time() - self.start_times[operation]
            self.metrics[operation] = duration
            del self.start_times[operation]
            return duration
        return 0.0

    def get_metrics(self) -> Dict[str, float]:
        """Get all recorded metrics"""
        return self.metrics.copy()

    def print_metrics(self):
        """Print performance metrics"""
        print("\n=== Performance Metrics ===")
        for operation, duration in self.metrics.items():
            print(f"{operation}: {duration:.2f}s")
        print("=" * 28)


# Global instances
memory_manager = MemoryManager()
startup_optimizer = StartupOptimizer()
model_cache = ModelCache()
performance_monitor = PerformanceMonitor()

# Apply initial optimizations
memory_manager.optimize_pytorch()
