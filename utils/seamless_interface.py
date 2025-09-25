"""
Seamless Interface Module
Creates a single window that transitions from welcome screen to chat interface
"""

import tkinter as tk
from tkinter import ttk
import os
import sys
from typing import Dict, Any, Callable, Optional
import threading
import time

# Import PIL for image handling
try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️ PIL/Pillow not available - using placeholder images")

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# from utils.preloader import ModelPreloader, LoadingScreen  # REMOVED - No more preloading
from utils.welcome_screen import AvatarRecommender, RecommendationDialog
from utils.user_preferences import get_user_preferences, record_avatar_selection
from utils.performance_boost import performance_monitor


class SeamlessVbotInterface:
    """Main interface that seamlessly transitions from loading to welcome to chat"""
    
    def __init__(self, device_index: int = None):
        self.device_index = device_index
        self.root = None
        # self.preloader = None  # REMOVED - No more preloading
        self.current_screen = "welcome"  # welcome -> chat (skip loading)
        self.selected_avatar = None
        self.chat_gui = None
        self.avatar_recommender = AvatarRecommender()
        self.user_prefs = get_user_preferences()
        
        # Window size management
        self.original_window_size = "1280x720"
        self.selection_window_size = "1280x900"  # Taller for avatar selection
        
        # UI components
        self.main_container = None
        self.loading_screen = None
        self.welcome_container = None
        self.chat_container = None
        
        # Avatar selection UI
        self.avatar_frames = {}
        self.avatar_images = {}  # Store loaded images
        self.continue_btn = None
        
        # Avatar image mapping
        self.avatar_image_files = {
            "Amelia": "ame.jpg",
            "Gura": "gura.jpg",
            "Eveland": "ike.jpg",  # Eveland maps to ike.jpg
            "Shiori": "shiori.jpg",
            "Wilson": "wilson.jpg"
        }
        
    def _load_avatar_image(self, avatar_name: str, size: tuple = (120, 120)):
        """Load and resize avatar image"""
        if not PIL_AVAILABLE:
            return None
            
        # Check if already loaded
        cache_key = f"{avatar_name}_{size[0]}x{size[1]}"
        if cache_key in self.avatar_images:
            return self.avatar_images[cache_key]
            
        try:
            # Get image filename
            image_filename = self.avatar_image_files.get(avatar_name)
            if not image_filename:
                print(f"⚠️ No image file mapped for avatar: {avatar_name}")
                return None
                
            # Build full path
            image_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "recommender-avatar-profile",
                image_filename
            )
            
            if not os.path.exists(image_path):
                print(f"⚠️ Image file not found: {image_path}")
                return None
                
            # Load and resize image
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                    
                # Resize to fit the specified size while maintaining aspect ratio
                img.thumbnail(size, Image.Resampling.LANCZOS)
                
                # Create a new image with the exact size and center the resized image
                final_img = Image.new('RGB', size, (59, 59, 75))  # Dark background color
                
                # Calculate position to center the image
                x = (size[0] - img.width) // 2
                y = (size[1] - img.height) // 2
                final_img.paste(img, (x, y))
                
                # Convert to PhotoImage
                photo = ImageTk.PhotoImage(final_img)
                
                # Cache the image
                self.avatar_images[cache_key] = photo
                return photo
                
        except Exception as e:
            print(f"❌ Error loading image for {avatar_name}: {e}")
            return None
            
        return None
        
    def initialize(self):
        """Initialize the main window and start the loading process"""
        # Create main window with taller size for avatar selection
        self.root = tk.Tk()
        self.root.title("Vbot - AI Companion")
        self.root.geometry(self.selection_window_size)
        self.root.configure(bg="#1a1a2e")
        self.root.resizable(True, True)
        
        # Center window with selection size
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (1280 // 2)
        y = (self.root.winfo_screenheight() // 2) - (900 // 2)  # Adjusted for taller window
        self.root.geometry(f"{self.selection_window_size}+{x}+{y}")
        
        # Create main container
        self.main_container = tk.Frame(self.root, bg="#1a1a2e")
        self.main_container.pack(expand=True, fill=tk.BOTH)
        
        # REMOVED: No more preloading - go straight to welcome screen
        # Start with welcome screen (skip loading)
        self._show_welcome_screen()
        
        return self.root
    
    # REMOVED: _preload_models_async - No more preloading
    
    def _on_loading_progress(self, model_name: str, progress: float, status: str):
        """Handle loading progress updates"""
        if self.loading_screen:
            self.loading_screen.update_progress(model_name, progress, status)
    
    def _show_loading_screen(self):
        """Show the loading screen"""
        self.current_screen = "loading"
        
        # Clear container
        for widget in self.main_container.winfo_children():
            widget.destroy()
        
        # Create loading screen
        self.loading_screen = LoadingScreen(self.main_container, self.preloader)
        self.loading_screen.create_loading_ui()
    
    def _transition_to_welcome(self, preload_success: bool):
        """Transition from loading to welcome screen"""
        if not preload_success:
            self._show_error_screen()
            return
        
        # Check if we should skip welcome screen
        if not self.user_prefs.should_show_welcome_screen():
            last_avatar = self.user_prefs.get_last_selected_avatar()
            if last_avatar and self.preloader.is_model_loaded(last_avatar):
                self.selected_avatar = last_avatar
                self._transition_to_chat()
                return
        
        self._show_welcome_screen()
    
    def _show_welcome_screen(self):
        """Show the welcome/avatar selection screen"""
        self.current_screen = "welcome"
        
        # Clear container
        for widget in self.main_container.winfo_children():
            widget.destroy()
        
        # Create welcome screen
        self._create_welcome_ui()
    
    def _create_welcome_ui(self):
        """Create the welcome screen UI"""
        # Header
        header_frame = tk.Frame(self.main_container, bg="#1a1a2e", height=120)
        header_frame.pack(fill=tk.X, padx=20, pady=(20, 0))
        header_frame.pack_propagate(False)
        
        # Title
        title_label = tk.Label(
            header_frame,
            text="Choose Your AI Companion",
            font=("Arial", 28, "bold"),
            fg="#ffffff",
            bg="#1a1a2e"
        )
        title_label.pack(pady=(20, 5))
        
        # Subtitle
        subtitle_label = tk.Label(
            header_frame,
            text="All models are ready! Select your preferred avatar to continue.",
            font=("Arial", 14),
            fg="#4CAF50",  # Green to indicate ready status
            bg="#1a1a2e"
        )
        subtitle_label.pack()
        
        # Avatar selection area
        content_frame = tk.Frame(self.main_container, bg="#1a1a2e")
        content_frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)
        
        # Create scrollable frame for avatars
        canvas = tk.Canvas(content_frame, bg="#1a1a2e", highlightthickness=0)
        scrollbar = ttk.Scrollbar(content_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#1a1a2e")
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Create avatar cards
        avatars = self.avatar_recommender.get_all_avatars()
        cols = 3
        
        for i, avatar_name in enumerate(avatars):
            row = i // cols
            col = i % cols
            self._create_avatar_card(scrollable_frame, avatar_name, row, col)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Mousewheel support (bind to canvas specifically, not globally)
        def _on_mousewheel(event):
            try:
                canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            except tk.TclError:
                # Canvas might be destroyed, ignore the error
                pass
        canvas.bind("<MouseWheel>", _on_mousewheel)
        
        # Store canvas reference for cleanup
        self.welcome_canvas = canvas
        
        # Footer with buttons
        self._create_welcome_footer()
        
        # Pre-select last used avatar if available
        last_avatar = self.user_prefs.get_last_selected_avatar()
        if last_avatar and last_avatar in avatars:
            self._select_avatar(last_avatar)
    
    def _create_avatar_card(self, parent, avatar_name: str, row: int, col: int):
        """Create an avatar selection card"""
        avatar_info = self.avatar_recommender.get_avatar_info(avatar_name)
        
        # REMOVED: No more preloader - all models are available for selection
        is_loaded = True  # All avatars are now selectable
        
        # Main card frame
        card_frame = tk.Frame(
            parent,
            bg="#2b2b3b" if is_loaded else "#3d3d3d",
            relief=tk.RAISED,
            borderwidth=2,
            padx=15,
            pady=15
        )
        card_frame.grid(row=row, column=col, padx=15, pady=15, sticky="nsew")
        
        # Configure grid weights
        parent.grid_columnconfigure(col, weight=1)
        parent.grid_rowconfigure(row, weight=1)
        
        # Store frame reference
        self.avatar_frames[avatar_name] = card_frame
        
        # Status indicator
        status_color = "#4CAF50" if is_loaded else "#FF5722"
        status_text = "✓ Ready" if is_loaded else "✗ Not Loaded"
        
        status_label = tk.Label(
            card_frame,
            text=status_text,
            font=("Arial", 9, "bold"),
            fg=status_color,
            bg=card_frame["bg"]
        )
        status_label.pack(anchor="ne")
        
        # Avatar image
        image_frame = tk.Frame(card_frame, bg="#3b3b4b", width=120, height=120)
        image_frame.pack(pady=(0, 10))
        image_frame.pack_propagate(False)
        
        # Try to load real avatar image
        avatar_photo = self._load_avatar_image(avatar_name, (120, 120))
        
        if avatar_photo:
            # Use real image
            image_label = tk.Label(
                image_frame,
                image=avatar_photo,
                bg="#3b3b4b"
            )
            image_label.pack(expand=True)
            # Keep a reference to prevent garbage collection
            image_label.image = avatar_photo
        else:
            # Fallback to placeholder emoji
            placeholder_label = tk.Label(
                image_frame,
                text="👤",
                font=("Arial", 36),
                fg=avatar_info.get("color", "#ffffff"),
                bg="#3b3b4b"
            )
            placeholder_label.pack(expand=True)
        
        # Avatar name
        name_label = tk.Label(
            card_frame,
            text=avatar_info.get("name", avatar_name),
            font=("Arial", 14, "bold"),
            fg="#ffffff",
            bg=card_frame["bg"]
        )
        name_label.pack(pady=(0, 5))
        
        # Personality
        personality_label = tk.Label(
            card_frame,
            text=avatar_info.get("personality", ""),
            font=("Arial", 10),
            fg=avatar_info.get("color", "#b0b0b0"),
            bg=card_frame["bg"]
        )
        personality_label.pack(pady=(0, 10))
        
        # Select button (only enabled if model is loaded)
        select_btn = tk.Button(
            card_frame,
            text="Select" if is_loaded else "Not Available",
            font=("Arial", 11, "bold"),
            bg=avatar_info.get("color", "#4a90e2") if is_loaded else "#666666",
            fg="black" if is_loaded else "#999999",
            relief=tk.FLAT,
            padx=15,
            pady=6,
            state="normal" if is_loaded else "disabled",
            command=lambda name=avatar_name: self._select_avatar(name) if is_loaded else None
        )
        select_btn.pack()
        
        # Hover effects (only for loaded models)
        if is_loaded:
            def on_enter(e, frame=card_frame, btn=select_btn, color=avatar_info.get("color", "#4a90e2")):
                frame.configure(bg="#3b3b4b")
                darker_color = self._darken_color(color)
                btn.configure(bg=darker_color)
            
            def on_leave(e, frame=card_frame, btn=select_btn, color=avatar_info.get("color", "#4a90e2")):
                if avatar_name != self.selected_avatar:
                    frame.configure(bg="#2b2b3b")
                btn.configure(bg=color)
            
            card_frame.bind("<Enter>", on_enter)
            card_frame.bind("<Leave>", on_leave)
            select_btn.bind("<Enter>", on_enter)
            select_btn.bind("<Leave>", on_leave)
            
            # Make card clickable
            def on_card_click(e, name=avatar_name):
                self._select_avatar(name)
            
            card_frame.bind("<Button-1>", on_card_click)
            
            # Make image clickable too if it exists
            if 'image_label' in locals():
                image_label.bind("<Button-1>", on_card_click)
    
    def _create_welcome_footer(self):
        """Create footer with action buttons"""
        footer_frame = tk.Frame(self.main_container, bg="#1a1a2e", height=80)
        footer_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        footer_frame.pack_propagate(False)
        
        # Continue button
        self.continue_btn = tk.Button(
            footer_frame,
            text="Continue",
            font=("Arial", 14, "bold"),
            bg="#4a90e2",
            fg="white",
            relief=tk.FLAT,
            padx=30,
            pady=12,
            state="disabled",
            command=self._continue_to_chat
        )
        self.continue_btn.pack(side=tk.RIGHT, pady=20)
        
        # Recommendation button
        recommend_btn = tk.Button(
            footer_frame,
            text="Get Recommendation",
            font=("Arial", 12),
            bg="#2d2d44",
            fg="white",
            relief=tk.FLAT,
            padx=20,
            pady=10,
            command=self._show_recommendation_dialog
        )
        recommend_btn.pack(side=tk.LEFT, pady=20)
    
    def _select_avatar(self, avatar_name: str):
        """Handle avatar selection"""
        # REMOVED: No more preloader check - all avatars are selectable
        
        # Reset previous selection
        if self.selected_avatar and self.selected_avatar in self.avatar_frames:
            self.avatar_frames[self.selected_avatar].configure(bg="#2b2b3b")
        
        # Set new selection
        self.selected_avatar = avatar_name
        if avatar_name in self.avatar_frames:
            self.avatar_frames[avatar_name].configure(bg="#3b3b4b")
        
        # Enable continue button
        if self.continue_btn:
            self.continue_btn.configure(state="normal")
    
    def _continue_to_chat(self):
        """Continue to chat interface - Load model on-demand"""
        if self.selected_avatar:
            # Record selection
            record_avatar_selection(self.selected_avatar)
            
            # Show loading message
            self._show_loading_message(f"Loading {self.selected_avatar}...")
            
            # Load model in background thread
            threading.Thread(
                target=self._load_selected_model,
                daemon=True,
                name="ModelLoader"
            ).start()
    
    def _show_loading_message(self, message: str):
        """Show loading message while model loads"""
        # Clear container
        for widget in self.main_container.winfo_children():
            widget.destroy()
        
        # Loading frame
        loading_frame = tk.Frame(self.main_container, bg="#1a1a2e")
        loading_frame.pack(expand=True, fill=tk.BOTH)
        
        # Loading message
        tk.Label(
            loading_frame,
            text="🚀 Loading...",
            font=("Arial", 24, "bold"),
            fg="#4CAF50",
            bg="#1a1a2e"
        ).pack(pady=(200, 20))
        
        tk.Label(
            loading_frame,
            text=message,
            font=("Arial", 16),
            fg="#ffffff",
            bg="#1a1a2e"
        ).pack(pady=(0, 20))
        
        # Update display
        self.root.update()
    
    def _load_selected_model(self):
        """Load the selected model on-demand"""
        try:
            print(f"🚀 Loading {self.selected_avatar} model on-demand...")
            
            # Import here to avoid circular imports
            from utils.initialization_utils import InitializationHandler
            
            # Create initialization handler for the selected model
            init_handler = InitializationHandler(
                model_name=self.selected_avatar, 
                device_index=self.device_index
            )
            
            # Initialize all components
            components = init_handler.initialize_all()
            
            # Create Ollama handler
            ollama_handler = init_handler.create_ollama_handler()
            
            # Store model data
            model_data = {
                "init_handler": init_handler,
                "components": components,
                "ollama_handler": ollama_handler,
                "model_name": self.selected_avatar
            }
            
            # Schedule transition to chat on main thread
            self.root.after(100, lambda: self._transition_to_chat_with_data(model_data))
            
        except Exception as e:
            print(f"❌ Error loading {self.selected_avatar}: {e}")
            # Schedule error display on main thread
            self.root.after(100, lambda: self._show_error_screen(f"Failed to load {self.selected_avatar}: {str(e)}"))
    
    def _transition_to_chat_with_data(self, model_data: Dict[str, Any]):
        """Transition to chat with loaded model data"""
        self.current_screen = "chat"
        
        # Store model data for switching
        self.model_data = {self.selected_avatar: model_data}
        
        # Clear container
        for widget in self.main_container.winfo_children():
            widget.destroy()
        
        # Create chat interface
        self._create_chat_interface(model_data)
    
    # REMOVED: Old _transition_to_chat function - replaced with _transition_to_chat_with_data
    
    def _fade_transition(self, callback):
        """Create a smooth fade transition effect"""
        # Simple transition - clear and rebuild
        # In a more advanced implementation, you could add actual fade effects
        for widget in self.main_container.winfo_children():
            widget.destroy()
        
        # Brief pause for smooth transition
        self.root.after(100, callback)
    
    def _create_chat_interface(self, model_data: Dict[str, Any]):
        """Create the main chat interface"""
        # Import here to avoid circular imports
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from Vbot import ModernChatGUI
        
        # Clean up welcome screen canvas binding
        if hasattr(self, 'welcome_canvas') and self.welcome_canvas:
            try:
                self.welcome_canvas.unbind_all("<MouseWheel>")
            except:
                pass
        
        # Update window title
        self.root.title(f"Vbot - {self.selected_avatar}")
        
        # Use the provided model data directly (no preloader)
        correct_model_data = model_data
        
        print(f"🎭 Creating chat interface for {self.selected_avatar}")
        print(f"📋 Using model: {correct_model_data['model_name']}")
        print(f"🔍 Model data keys: {list(correct_model_data.keys())}")
        
        # Debug: Check if the init_handler has the correct model
        if 'init_handler' in correct_model_data:
            init_handler = correct_model_data['init_handler']
            if hasattr(init_handler, 'model_name'):
                print(f"🔍 InitHandler model_name: {init_handler.model_name}")
            if hasattr(init_handler, 'character_model_path'):
                print(f"🔍 Character model path: {init_handler.character_model_path}")
        
        # Set environment variable for the selected avatar (this is what AnimatedCharacter uses)
        os.environ["VOICE_TYPE"] = self.selected_avatar
        print(f"🔧 Set VOICE_TYPE environment variable to: {self.selected_avatar}")
        print(f"🔍 Current environment VOICE_TYPE: {os.getenv('VOICE_TYPE')}")
        
        # Resize window back to original size for chat interface
        self.root.geometry(self.original_window_size)
        
        # Re-center window with original size
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (1280 // 2)
        y = (self.root.winfo_screenheight() // 2) - (720 // 2)
        self.root.geometry(f"{self.original_window_size}+{x}+{y}")
        
        # Clear the main container completely
        for widget in self.main_container.winfo_children():
            widget.destroy()
        
        # Remove the main container to give full control to ModernChatGUI
        self.main_container.destroy()
        
        # Create chat GUI in the root window (it will create its own layout)
        # Get the actual voice toggle function (not a lambda that returns it)
        voice_toggle_func = self._get_voice_toggle_callback(correct_model_data)
        
        self.chat_gui = ModernChatGUI(
            self.root,
            correct_model_data["ollama_handler"].handle_text_input_simple,
            voice_toggle_func,  # Pass the actual function, not a lambda
            model_name=self.selected_avatar
        )
        
        # Set up the chat system
        correct_model_data["ollama_handler"].gui = self.chat_gui
        
        # Ensure the TTS model is correct for the selected avatar
        correct_tts_model = correct_model_data["components"].get("tts_model")
        if correct_tts_model:
            print(f"🔊 Setting TTS model for {self.selected_avatar} voice")
            print(f"🔍 TTS model type: {type(correct_tts_model).__name__}")
            print(f"🔍 Initial TTS model name: {getattr(correct_tts_model, 'model_name', 'Unknown')}")
            print(f"🔍 Initial TTS repo_id: {getattr(correct_tts_model, 'repo_id', 'Unknown')}")
            
            # Check if the TTS model matches the selected avatar
            if hasattr(correct_tts_model, 'model_name') and correct_tts_model.model_name != self.selected_avatar:
                print(f"⚠️ Initial TTS model mismatch! Expected {self.selected_avatar}, got {correct_tts_model.model_name}")
                print("🔄 This is why all voices sound like Amelia!")
                
            correct_model_data["ollama_handler"].tts_model = correct_tts_model
        else:
            print(f"⚠️ No TTS model found for {self.selected_avatar}")
            print(f"🔍 Available components: {list(correct_model_data['components'].keys())}")
        
        # Set up audio processor if available
        self._setup_audio_processor(correct_model_data)
        
        # Set up model switching callback
        self.chat_gui.set_model_switch_callback(self._handle_model_switch)
        
        print(f"🎉 Vbot is ready with {self.selected_avatar}!")
    
    def _setup_audio_processor(self, model_data: Dict[str, Any]):
        """Set up audio processor for voice input and TTS output"""
        retry_count = 0
        max_retries = 10
        
        def check_audio_processor():
            nonlocal retry_count
            ollama_handler = model_data["ollama_handler"]
            init_handler = model_data["init_handler"]
            
            if not ollama_handler.audio_processor:
                retry_count += 1
                print(f"🔍 Checking audio processor availability... (attempt {retry_count}/{max_retries})")
                
                # Try to get audio processor directly
                audio_processor = init_handler.get_audio_processor_when_ready()
                if audio_processor:
                    print("✅ Audio processor found, updating OllamaHandler...")
                    ollama_handler.audio_processor = audio_processor
                    print("🎤 Voice input ready")
                    print("🔊 TTS audio output ready")
                    print(f"✅ Audio processor set: {ollama_handler.audio_processor is not None}")
                    return
                
                # Try the update method as fallback
                if init_handler.update_ollama_handler_audio_processor(ollama_handler):
                    print("🎤 Voice input ready")
                    print("🔊 TTS audio output ready")
                    print(f"✅ Audio processor set: {ollama_handler.audio_processor is not None}")
                    return
                
                if retry_count < max_retries:
                    print(f"⏳ Audio processor not ready yet, retrying in 1s... ({retry_count}/{max_retries})")
                    self.root.after(1000, check_audio_processor)
                else:
                    print("❌ Audio processor setup failed after maximum retries")
                    print("🔧 TTS audio output may not work properly")
            else:
                print("✅ Audio processor already available")
        
        self.root.after(100, check_audio_processor)
    
    def _get_voice_toggle_callback(self, model_data: Dict[str, Any]):
        """Get voice toggle callback"""
        def handle_voice_input(text, timings):
            model_data["ollama_handler"].handle_text_input_simple(text)
        
        def get_voice_toggle():
            components = model_data["components"]
            init_handler = model_data["init_handler"]
            
            if "audio_processor" in components:
                audio_processor = components["audio_processor"]
            elif "get_audio_processor" in components:
                audio_processor = components["get_audio_processor"]()
            else:
                audio_processor = init_handler.get_audio_processor_when_ready()
            
            if audio_processor:
                return audio_processor.toggle_listening(
                    self.chat_gui, handle_voice_input, 
                    model_data["ollama_handler"].is_processing
                )
            else:
                print("Audio processor not ready yet")
                return None
        
        return get_voice_toggle
    
    def _handle_model_switch(self, new_model: str):
        """Handle switching to a different model - Load on-demand"""
        print(f"🔄 Switching to {new_model}...")
        
        # CRITICAL: Stop any current processing immediately
        if self.chat_gui and hasattr(self.chat_gui, 'ollama_handler'):
            current_handler = self.chat_gui.ollama_handler
            if hasattr(current_handler, 'is_processing'):
                current_handler.is_processing = False
                print(f"🛑 Stopped current processing for immediate switch")
        
        # CRITICAL: Update selected avatar immediately to prevent race conditions
        old_avatar = self.selected_avatar
        self.selected_avatar = new_model
        print(f"🔄 Avatar switch: {old_avatar} → {new_model}")
        
        # Check if we already have this model loaded
        if hasattr(self, 'model_data') and new_model in self.model_data:
            print(f"✅ Using cached model data for {new_model}")
            new_model_data = self.model_data[new_model]
            
            # IMMEDIATE: Switch Ollama handler right away for cached models
            self._apply_model_switch(new_model_data, new_model)
            return
        else:
            print(f"🚀 Loading {new_model} model on-demand for switching...")
            # Load the new model on-demand
            try:
                from utils.initialization_utils import InitializationHandler
                
                # Create initialization handler for the new model
                init_handler = InitializationHandler(
                    model_name=new_model, 
                    device_index=self.device_index
                )
                
                # Initialize all components
                components = init_handler.initialize_all()
                
                # Create Ollama handler
                ollama_handler = init_handler.create_ollama_handler()
                
                # Store model data
                new_model_data = {
                    "init_handler": init_handler,
                    "components": components,
                    "ollama_handler": ollama_handler,
                    "model_name": new_model
                }
                
                # Cache the model data
                if not hasattr(self, 'model_data'):
                    self.model_data = {}
                self.model_data[new_model] = new_model_data
                
            except Exception as e:
                print(f"❌ Error loading {new_model}: {e}")
                return
        
        # Apply the model switch
        self._apply_model_switch(new_model_data, new_model)
    
    def _apply_model_switch(self, new_model_data, new_model):
        """Apply the actual model switch with new data"""
        
        # Record selection
        record_avatar_selection(new_model)
        
        # Set environment variable for the new avatar
        os.environ["VOICE_TYPE"] = new_model
        print(f"🔧 Updated VOICE_TYPE environment variable to: {new_model}")
        
        # Update chat GUI and recreate avatar with correct model
        self.chat_gui._select_character(new_model)
        
        # CRITICAL: Immediately switch the Ollama handler to stop old responses
        new_ollama_handler = new_model_data["ollama_handler"]
        new_ollama_handler.gui = self.chat_gui
        self.chat_gui.on_send_callback = new_ollama_handler.handle_text_input_simple
        
        # CRITICAL: Set the correct model for the Ollama handler IMMEDIATELY
        if hasattr(new_ollama_handler, 'set_model'):
            print(f"🎭 Setting Ollama handler to {new_model} personality")
            new_ollama_handler.set_model(new_model)
        
        # CRITICAL: Update the chat GUI's ollama handler reference IMMEDIATELY
        if hasattr(self.chat_gui, 'ollama_handler'):
            self.chat_gui.ollama_handler = new_ollama_handler
            print(f"✅ Chat GUI now using {new_model} Ollama handler")
        
        # Ensure the TTS model is correct for this avatar
        new_tts_model = new_model_data["components"].get("tts_model")
        if new_tts_model:
            print(f"🔊 Updating TTS model to {new_model} voice")
            print(f"🔍 TTS model type: {type(new_tts_model).__name__}")
            print(f"🔍 Current TTS model name: {getattr(new_tts_model, 'model_name', 'Unknown')}")
            print(f"🔍 Current TTS repo_id: {getattr(new_tts_model, 'repo_id', 'Unknown')}")
            
            # Force update the TTS model to ensure it's using the correct voice
            if hasattr(new_tts_model, 'model_name') and new_tts_model.model_name != new_model:
                print(f"⚠️ TTS model mismatch! Expected {new_model}, got {new_tts_model.model_name}")
                print("🔄 This explains why all voices sound like Amelia!")
                
            new_ollama_handler.tts_model = new_tts_model
        else:
            print(f"⚠️ No TTS model found for {new_model}")
            print(f"🔍 Available components: {list(new_model_data['components'].keys())}")
        
        # Update the chat GUI's ollama handler reference
        if hasattr(self.chat_gui, 'ollama_handler'):
            self.chat_gui.ollama_handler = new_ollama_handler
        
        # CRITICAL: Set up audio processor for the new Ollama handler
        audio_processor = new_model_data["components"].get("audio_processor")
        if audio_processor:
            print(f"🎤 Setting up audio processor for {new_model}")
            new_ollama_handler.audio_processor = audio_processor
            print(f"✅ Audio processor set for {new_model}")
        else:
            print(f"⚠️ No audio processor found for {new_model}")
            print(f"🔍 Available components: {list(new_model_data['components'].keys())}")
        
        # Update window title
        self.root.title(f"Vbot - {new_model}")
        
        print(f"✅ Switched to {new_model}")
    
    def _show_recommendation_dialog(self):
        """Show recommendation dialog"""
        dialog = RecommendationDialog(self.root, self.avatar_recommender, self._select_avatar)
        dialog.show()
    
    def _show_error_screen(self, message: str = "Failed to load models"):
        """Show error screen"""
        # Clear container
        for widget in self.main_container.winfo_children():
            widget.destroy()
        
        # Error message
        error_frame = tk.Frame(self.main_container, bg="#1a1a2e")
        error_frame.pack(expand=True, fill=tk.BOTH)
        
        tk.Label(
            error_frame,
            text="⚠️ Error",
            font=("Arial", 24, "bold"),
            fg="#FF5722",
            bg="#1a1a2e"
        ).pack(pady=(100, 20))
        
        tk.Label(
            error_frame,
            text=message,
            font=("Arial", 14),
            fg="#ffffff",
            bg="#1a1a2e"
        ).pack(pady=(0, 20))
        
        tk.Button(
            error_frame,
            text="Retry",
            font=("Arial", 12),
            bg="#4a90e2",
            fg="white",
            relief=tk.FLAT,
            padx=20,
            pady=10,
            command=self._retry_initialization
        ).pack()
    
    def _retry_initialization(self):
        """Retry the initialization process"""
        self._show_loading_screen()
        threading.Thread(
            target=self._preload_models_async,
            daemon=True,
            name="ModelPreloader"
        ).start()
    
    def _darken_color(self, hex_color: str) -> str:
        """Darken a hex color by 20%"""
        try:
            hex_color = hex_color.lstrip('#')
            rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            darkened = tuple(max(0, int(c * 0.8)) for c in rgb)
            return f"#{darkened[0]:02x}{darkened[1]:02x}{darkened[2]:02x}"
        except:
            return "#333333"
    
    def cleanup(self):
        """Clean up resources"""
        # REMOVED: No more preloader to cleanup
        
        if self.chat_gui:
            self.chat_gui.cleanup()
    
    def run(self):
        """Run the seamless interface"""
        root = self.initialize()
        
        # Set up cleanup on window close
        def on_closing():
            self.cleanup()
            root.destroy()
        
        root.protocol("WM_DELETE_WINDOW", on_closing)
        
        # Start the main loop
        root.mainloop()


# Main function to launch the seamless interface
def launch_seamless_vbot(device_index: int = None):
    """Launch the seamless Vbot interface"""
    interface = SeamlessVbotInterface(device_index)
    interface.run()


if __name__ == "__main__":
    launch_seamless_vbot()
