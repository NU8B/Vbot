"""
Model Preloader Module
Handles loading all avatar models during startup for seamless switching
"""

import os
import sys
import threading
import time
from typing import Dict, Any, Callable, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from utils.initialization_utils import InitializationHandler
from utils.performance_boost import performance_monitor, memory_manager


class ModelPreloader:
    """Handles preloading all avatar models for seamless switching"""
    
    def __init__(self, device_index: int = None):
        self.device_index = device_index
        self.available_models = ["Amelia", "Eveland", "Gura", "Shiori", "Wilson"]
        self.loaded_models = {}
        self.loading_progress = {}
        self.is_loading = False
        self.loading_complete = False
        self.progress_callback = None
        
    def set_progress_callback(self, callback: Callable[[str, float, str], None]):
        """Set callback for progress updates: callback(model_name, progress, status)"""
        self.progress_callback = callback
    
    def _update_progress(self, model_name: str, progress: float, status: str):
        """Update loading progress"""
        self.loading_progress[model_name] = {"progress": progress, "status": status}
        if self.progress_callback:
            self.progress_callback(model_name, progress, status)
    
    def _load_single_model(self, model_name: str) -> Dict[str, Any]:
        """Load a single model and return its components"""
        try:
            self._update_progress(model_name, 0.1, f"Initializing {model_name}...")
            
            # Create initialization handler for this model
            init_handler = InitializationHandler(
                model_name=model_name, 
                device_index=self.device_index
            )
            
            self._update_progress(model_name, 0.3, f"Loading {model_name} components...")
            
            # Initialize components
            components = init_handler.initialize_all()
            
            self._update_progress(model_name, 0.8, f"Finalizing {model_name}...")
            
            # Create Ollama handler with correct model
            ollama_handler = init_handler.create_ollama_handler()
            
            # Ensure the ollama handler is using the correct model
            if hasattr(ollama_handler, 'model_name'):
                ollama_handler.model_name = model_name
            
            # Store reference to init_handler for later audio processor setup
            ollama_handler._init_handler = init_handler
            
            # Try to set up audio processor if it's ready
            if not ollama_handler.audio_processor:
                print(f"🔍 {model_name}: Audio processor not ready during preload, will set up later")
            
            # Debug TTS model in components
            tts_model = components.get("tts_model")
            if tts_model:
                print(f"🔍 {model_name} TTS model in components:")
                print(f"   Model Name: {getattr(tts_model, 'model_name', 'Unknown')}")
                print(f"   Repo ID: {getattr(tts_model, 'repo_id', 'Unknown')}")
                print(f"   Object ID: {id(tts_model)}")
            else:
                print(f"⚠️ {model_name} has NO TTS model in components!")
            
            # Store the model data
            model_data = {
                "init_handler": init_handler,
                "components": components,
                "ollama_handler": ollama_handler,
                "model_name": model_name
            }
            
            print(f"✅ {model_name} model data created with model_name: {model_data['model_name']}")
            
            self._update_progress(model_name, 1.0, f"{model_name} loaded successfully!")
            
            return model_data
            
        except Exception as e:
            self._update_progress(model_name, 0.0, f"Failed to load {model_name}: {str(e)}")
            print(f"Error loading {model_name}: {e}")
            return None
    
    def preload_all_models(self, max_workers: int = 1) -> bool:
        """
        Preload all models using parallel processing
        
        Args:
            max_workers: Maximum number of models to load simultaneously
            
        Returns:
            bool: True if all models loaded successfully
        """
        if self.is_loading:
            return False
        
        self.is_loading = True
        self.loading_complete = False
        
        print("🚀 Starting model preloading...")
        performance_monitor.start_timer("preload_all_models")
        
        # Apply memory optimizations
        memory_manager.optimize_pytorch()
        
        # Initialize progress tracking
        for model in self.available_models:
            self.loading_progress[model] = {"progress": 0.0, "status": "Waiting..."}
        
        success_count = 0
        
        # Load models sequentially to avoid GPU memory issues
        print("🧠 Loading models sequentially to manage GPU memory...")
        
        if max_workers == 1:
            # Sequential loading for better memory management
            for model in self.available_models:
                print(f"🔄 Loading {model}...")
                try:
                    # Clear GPU memory before each model
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    model_data = self._load_single_model(model)
                    if model_data:
                        self.loaded_models[model] = model_data
                        success_count += 1
                        print(f"✅ {model} loaded successfully ({success_count}/{len(self.available_models)})")
                    else:
                        print(f"❌ Failed to load {model}")
                except Exception as e:
                    print(f"❌ Error loading {model}: {e}")
        else:
            # Parallel loading (original code)
            with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ModelLoader") as executor:
                # Submit all loading tasks
                future_to_model = {
                    executor.submit(self._load_single_model, model): model 
                    for model in self.available_models
                }
                
                # Process completed tasks
                for future in as_completed(future_to_model):
                    model_name = future_to_model[future]
                    try:
                        model_data = future.result()
                        if model_data:
                            self.loaded_models[model_name] = model_data
                            success_count += 1
                            print(f"✅ {model_name} loaded successfully")
                        else:
                            print(f"❌ Failed to load {model_name}")
                    except Exception as e:
                        print(f"❌ Exception loading {model_name}: {e}")
                        self._update_progress(model_name, 0.0, f"Error: {str(e)}")
        
        performance_monitor.end_timer("preload_all_models")
        
        self.is_loading = False
        self.loading_complete = True
        
        print(f"🎉 Model preloading complete! {success_count}/{len(self.available_models)} models loaded")
        
        return success_count == len(self.available_models)
    
    def get_model_data(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get preloaded model data"""
        return self.loaded_models.get(model_name)
    
    def is_model_loaded(self, model_name: str) -> bool:
        """Check if a specific model is loaded"""
        return model_name in self.loaded_models
    
    def get_loading_progress(self) -> Dict[str, Dict[str, Any]]:
        """Get current loading progress for all models"""
        return self.loading_progress.copy()
    
    def get_overall_progress(self) -> float:
        """Get overall loading progress (0.0 to 1.0)"""
        if not self.loading_progress:
            return 0.0
        
        total_progress = sum(model["progress"] for model in self.loading_progress.values())
        return total_progress / len(self.loading_progress)
    
    def cleanup_unused_models(self, keep_model: str):
        """Clean up unused models to free memory (keep only the specified model)"""
        for model_name, model_data in list(self.loaded_models.items()):
            if model_name != keep_model:
                try:
                    # Cleanup the model
                    if "init_handler" in model_data:
                        model_data["init_handler"].cleanup()
                    print(f"🧹 Cleaned up {model_name}")
                except Exception as e:
                    print(f"Error cleaning up {model_name}: {e}")
                
                # Remove from loaded models
                del self.loaded_models[model_name]
    
    def cleanup_all(self):
        """Clean up all loaded models"""
        for model_name, model_data in self.loaded_models.items():
            try:
                if "init_handler" in model_data:
                    model_data["init_handler"].cleanup()
                print(f"🧹 Cleaned up {model_name}")
            except Exception as e:
                print(f"Error cleaning up {model_name}: {e}")
        
        self.loaded_models.clear()
        memory_manager.clear_cache()


class LoadingScreen:
    """Loading screen to show model preloading progress"""
    
    def __init__(self, root, preloader: ModelPreloader):
        self.root = root
        self.preloader = preloader
        self.progress_bars = {}
        self.status_labels = {}
        self.overall_progress_bar = None
        self.loading_frame = None
        
    def create_loading_ui(self):
        """Create the loading screen UI"""
        # Create loading frame
        self.loading_frame = tk.Frame(self.root, bg="#1a1a2e")
        self.loading_frame.pack(expand=True, fill=tk.BOTH)
        
        # Title
        title_label = tk.Label(
            self.loading_frame,
            text="Loading Vbot",
            font=("Arial", 24, "bold"),
            fg="#ffffff",
            bg="#1a1a2e"
        )
        title_label.pack(pady=(50, 10))
        
        # Subtitle
        subtitle_label = tk.Label(
            self.loading_frame,
            text="Preparing your AI companions...",
            font=("Arial", 14),
            fg="#b0b0b0",
            bg="#1a1a2e"
        )
        subtitle_label.pack(pady=(0, 30))
        
        # Overall progress
        overall_frame = tk.Frame(self.loading_frame, bg="#1a1a2e")
        overall_frame.pack(pady=(0, 20), padx=50, fill=tk.X)
        
        tk.Label(
            overall_frame,
            text="Overall Progress:",
            font=("Arial", 12, "bold"),
            fg="#ffffff",
            bg="#1a1a2e"
        ).pack(anchor="w")
        
        self.overall_progress_bar = ttk.Progressbar(
            overall_frame,
            length=400,
            mode='determinate'
        )
        self.overall_progress_bar.pack(fill=tk.X, pady=(5, 0))
        
        # Individual model progress
        models_frame = tk.Frame(self.loading_frame, bg="#1a1a2e")
        models_frame.pack(expand=True, fill=tk.BOTH, padx=50, pady=20)
        
        for i, model in enumerate(self.preloader.available_models):
            model_frame = tk.Frame(models_frame, bg="#2b2b3b", relief=tk.RAISED, borderwidth=1)
            model_frame.pack(fill=tk.X, pady=5, padx=10)
            
            # Model name
            name_label = tk.Label(
                model_frame,
                text=model,
                font=("Arial", 11, "bold"),
                fg="#ffffff",
                bg="#2b2b3b"
            )
            name_label.pack(anchor="w", padx=10, pady=(5, 0))
            
            # Progress bar
            progress_bar = ttk.Progressbar(
                model_frame,
                length=300,
                mode='determinate'
            )
            progress_bar.pack(fill=tk.X, padx=10, pady=2)
            self.progress_bars[model] = progress_bar
            
            # Status label
            status_label = tk.Label(
                model_frame,
                text="Waiting...",
                font=("Arial", 9),
                fg="#b0b0b0",
                bg="#2b2b3b"
            )
            status_label.pack(anchor="w", padx=10, pady=(0, 5))
            self.status_labels[model] = status_label
    
    def update_progress(self, model_name: str, progress: float, status: str):
        """Update progress for a specific model"""
        if model_name in self.progress_bars:
            self.progress_bars[model_name]['value'] = progress * 100
            self.status_labels[model_name].config(text=status)
        
        # Update overall progress
        overall_progress = self.preloader.get_overall_progress()
        if self.overall_progress_bar:
            self.overall_progress_bar['value'] = overall_progress * 100
        
        # Update UI
        self.root.update_idletasks()
    
    def hide_loading_screen(self):
        """Hide the loading screen"""
        if self.loading_frame:
            self.loading_frame.destroy()
            self.loading_frame = None


# Import tkinter here to avoid issues when this module is imported
try:
    import tkinter as tk
    from tkinter import ttk
except ImportError:
    print("Warning: tkinter not available for loading screen UI")


# Example usage and testing
if __name__ == "__main__":
    def test_preloader():
        preloader = ModelPreloader()
        
        def progress_callback(model_name, progress, status):
            print(f"{model_name}: {progress*100:.1f}% - {status}")
        
        preloader.set_progress_callback(progress_callback)
        success = preloader.preload_all_models(max_workers=2)
        
        print(f"Preloading {'successful' if success else 'failed'}")
        print(f"Loaded models: {list(preloader.loaded_models.keys())}")
    
    test_preloader()
