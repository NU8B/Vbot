import os
import sys
import tkinter as tk
import warnings
from pathlib import Path
import wx
from datetime import datetime
import re
import threading
import time
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from utils.gui import ChatGUI
from utils.initialization_utils import InitializationHandler
from utils.avatar import AnimatedCharacter

# Suppress all warnings and optimize for performance
warnings.filterwarnings("ignore")

# Performance optimizations
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable tokenizer parallelism warnings
os.environ["CUDA_LAUNCH_BLOCKING"] = (
    "0"  # Disable CUDA launch blocking for better performance
)

# Enhanced performance optimizations
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
    "max_split_size_mb:128"  # Optimize CUDA memory allocation
)
os.environ["OMP_NUM_THREADS"] = "4"  # Limit OpenMP threads
os.environ["MKL_NUM_THREADS"] = "4"  # Limit MKL threads

# Performance constants
WX_UPDATE_INTERVAL = 33  # ~30 FPS instead of 60 FPS for better performance
UI_UPDATE_INTERVAL = 100  # UI updates every 100ms
CHAT_HISTORY_UPDATE_INTERVAL = 500  # Chat history updates every 500ms
SUBTITLE_UPDATE_INTERVAL = 50  # Subtitle updates every 50ms

# Enhanced performance optimizations
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
    "max_split_size_mb:128"  # Optimize CUDA memory allocation
)
os.environ["OMP_NUM_THREADS"] = "4"  # Limit OpenMP threads
os.environ["MKL_NUM_THREADS"] = "4"  # Limit MKL threads

# Window and avatar sizing constants
WINDOW_WIDTH = 1280  # 720p width
WINDOW_HEIGHT = 720  # 720p height
AVATAR_SIZE = 512  # Fixed avatar resolution

# Performance constants
WX_UPDATE_INTERVAL = 33  # ~30 FPS instead of 60 FPS for better performance
UI_UPDATE_INTERVAL = 100  # UI updates every 100ms
CHAT_HISTORY_UPDATE_INTERVAL = 500  # Chat history updates every 500ms
SUBTITLE_UPDATE_INTERVAL = 50  # Subtitle updates every 50ms


class ModernChatGUI(ChatGUI):
    def __init__(self, root, on_send_callback, on_voice_toggle_callback):
        # Initialize wx.App first (needed for avatar)
        self.wx_app = wx.App()

        # Get model name from environment
        self.model_name = os.getenv("VOICE_TYPE", "Amelia")

        # Use a neutral background color
        self.bg_color = "#2b2b3b"

        # Initialize parent class but don't create its widgets
        self.root = root
        self.on_send_callback = on_send_callback
        self.on_voice_toggle_callback = on_voice_toggle_callback
        self.is_processing = False

        # Performance optimization: Thread pool for background tasks
        self.thread_pool = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="VbotWorker"
        )

        # Performance optimization: Queues for async operations
        self.ui_update_queue = queue.Queue()
        self.chat_update_queue = queue.Queue()
        self.subtitle_update_queue = queue.Queue()

        # Available models
        self.available_models = ["Amelia", "Eveland", "Gura"]
        self.current_model_index = (
            0
            if self.model_name == "Amelia"
            else (1 if self.model_name == "Eveland" else 2)
        )

        # Main container - use neutral background color
        self.main_container = tk.Frame(root, bg="#1a1a2e")
        self.main_container.pack(expand=True, fill=tk.BOTH)

        # Avatar container - neutral background, centered
        self.avatar_container = tk.Frame(
            self.main_container,
            bg="#1a1a2e",
            width=WINDOW_WIDTH,
            height=WINDOW_HEIGHT,
        )
        self.avatar_container.pack(expand=True, fill=tk.BOTH)

        # Create and initialize avatar
        print("Creating avatar frame...")
        self.wx_frame = wx.Frame(
            None,
            style=wx.BORDER_NONE
            | wx.FRAME_NO_TASKBAR,  # Remove window decorations and taskbar entry
            size=(WINDOW_WIDTH, WINDOW_HEIGHT),
        )
        self.wx_frame.SetBackgroundColour("#2b2b3b")

        print("Initializing avatar...")
        self.avatar = AnimatedCharacter(self.wx_frame, WINDOW_WIDTH, WINDOW_HEIGHT)
        self.wx_window_id = self.wx_frame.GetHandle()

        # Create a tkinter window to embed the wx.Frame
        self.avatar_embed = tk.Frame(self.avatar_container, bg="#1a1a2e")
        self.avatar_embed.pack(expand=True, fill=tk.BOTH)

        # Bottom bar container - darker purple theme
        self.bottom_bar = tk.Frame(self.main_container, bg="#1a1a2e", height=60)
        self.bottom_bar.pack(fill=tk.X, side=tk.BOTTOM, padx=20, pady=20)
        self.bottom_bar.pack_propagate(False)

        # Input box - darker theme
        self.input_box = tk.Entry(
            self.bottom_bar,
            bg="#5b5b6b",
            fg="white",
            insertbackground="white",
            font=("Arial", 12),
            relief=tk.FLAT,
            highlightthickness=0,  # Remove the focus border
        )
        self.input_box.pack(
            side=tk.LEFT, expand=True, fill=tk.BOTH, padx=(20, 10), pady=10
        )

        # Button container - match bottom bar
        button_container = tk.Frame(self.bottom_bar, bg="#1a1a2e")
        button_container.pack(side=tk.RIGHT, padx=(0, 20))

        # Background selection button - darker purple theme, auto size
        self.background_btn = tk.Button(
            button_container,
            text="🖼️",
            font=("Arial", 12),
            bg="#2d2d44",
            fg="white",
            relief=tk.FLAT,
            command=self._show_background_menu,
            height=1,  # Fixed height to make button smaller
        )
        self.background_btn.pack(side=tk.RIGHT, padx=3, pady=8)

        # Character selection button - keep model-specific colors, auto size
        self.character_btn = tk.Button(
            button_container,
            text="👤",
            font=("Arial", 12),
            bg=(
                "#ffd05c"
                if self.model_name == "Amelia"
                else ("#318fc5" if self.model_name == "Eveland" else "#4a90e2")
            ),
            fg="black",
            relief=tk.FLAT,
            command=self._show_character_menu,
            height=1,  # Fixed height to make button smaller
        )
        self.character_btn.pack(side=tk.RIGHT, padx=3, pady=8)

        # History button - darker purple theme, auto size
        self.history_btn = tk.Button(
            button_container,
            text="💭",
            font=("Arial", 12),
            bg="#2d2d44",
            fg="white",
            relief=tk.FLAT,
            command=self.show_history,
            height=1,  # Fixed height to make button smaller
        )
        self.history_btn.pack(side=tk.RIGHT, padx=3, pady=8)

        # Voice button - darker purple theme, auto size
        self.voice_btn = tk.Button(
            button_container,
            text="🎤",
            font=("Arial", 12),
            bg="#2d2d44",
            fg="white",
            relief=tk.FLAT,
            command=self._handle_voice,
            height=1,  # Fixed height to make button smaller
        )
        self.voice_btn.pack(side=tk.RIGHT, padx=3, pady=8)

        # Send button - darker purple theme, auto size
        self.send_btn = tk.Button(
            button_container,
            text="➤",
            font=("Arial", 12),
            bg="#2d2d44",
            fg="white",
            relief=tk.FLAT,
            command=self._handle_send,
            height=1,  # Fixed height to make button smaller
        )
        self.send_btn.pack(side=tk.RIGHT, padx=3, pady=8)

        # Bind enter key to send
        self.input_box.bind("<Return>", lambda e: self._handle_send())

        # Chat history storage
        self.chat_history = []
        self.history_window = None
        self.pending_chat_updates = []  # Buffer for chat updates

        # Subtitle system
        self.subtitle_window = None
        self.subtitle_text = None
        self.subtitle_queue = []
        self.subtitle_timer = None
        self.is_showing_subtitle = False
        self.pending_subtitle_updates = []  # Buffer for subtitle updates

        # Performance optimization: UI update timers
        self.ui_update_timer = None
        self.chat_update_timer = None
        self.subtitle_update_timer = None

        # Style buttons on hover - darker purple theme
        for btn in [
            self.send_btn,
            self.voice_btn,
            self.history_btn,
            self.background_btn,
        ]:
            btn.bind("<Enter>", lambda e, b=btn: b.configure(bg="#3d3d5c"))
            btn.bind("<Leave>", lambda e, b=btn: b.configure(bg="#2d2d44"))

        # Special hover effect for character button based on current model
        def character_hover_enter(e):
            if self.model_name == "Amelia":
                self.character_btn.configure(bg="#e6bb53")
            elif self.model_name == "Eveland":
                self.character_btn.configure(bg="#2b7eb3")
            else:  # Gura
                self.character_btn.configure(bg="#3a7bc8")

        def character_hover_leave(e):
            if self.model_name == "Amelia":
                self.character_btn.configure(bg="#ffd05c")
            elif self.model_name == "Eveland":
                self.character_btn.configure(bg="#318fc5")
            else:  # Gura
                self.character_btn.configure(bg="#4a90e2")

        self.character_btn.bind("<Enter>", character_hover_enter)
        self.character_btn.bind("<Leave>", character_hover_leave)

        # Embed wx.Frame into tkinter with a slight delay
        self.root.after(100, self._embed_wx_frame)

        # Start optimized wx update timer - REDUCED FREQUENCY FOR BETTER PERFORMANCE
        self.root.after(
            WX_UPDATE_INTERVAL, self._update_wx
        )  # ~30 FPS instead of 60 FPS

        # Start UI update timers
        self.root.after(UI_UPDATE_INTERVAL, self._process_ui_updates)
        self.root.after(CHAT_HISTORY_UPDATE_INTERVAL, self._process_chat_updates)
        self.root.after(SUBTITLE_UPDATE_INTERVAL, self._process_subtitle_updates)
        self.root.after(UI_UPDATE_INTERVAL, self._process_ui_updates)
        self.root.after(CHAT_HISTORY_UPDATE_INTERVAL, self._process_chat_updates)
        self.root.after(SUBTITLE_UPDATE_INTERVAL, self._process_subtitle_updates)

        # Bind window movement to update subtitle position
        self.root.bind("<Configure>", self._on_window_configure)

    def _embed_wx_frame(self):
        """Embed wx.Frame into tkinter window"""
        try:
            print("Embedding avatar frame...")
            import win32gui
            import win32con

            # Get the window handle of the tkinter frame
            tk_handle = self.avatar_embed.winfo_id()

            # Set the wx.Frame as a child of the tkinter frame
            win32gui.SetParent(self.wx_window_id, tk_handle)

            # Remove window styles that make it look like a separate window
            style = win32gui.GetWindowLong(self.wx_window_id, win32con.GWL_STYLE)
            style = style & ~(
                win32con.WS_CAPTION | win32con.WS_THICKFRAME | win32con.WS_SYSMENU
            )
            win32gui.SetWindowLong(self.wx_window_id, win32con.GWL_STYLE, style)

            # Show the wx.Frame
            self.wx_frame.Show()

            # Position the wx.Frame to fill the entire container
            win32gui.SetWindowPos(
                self.wx_window_id,
                win32con.HWND_TOP,
                0,
                0,
                WINDOW_WIDTH,
                WINDOW_HEIGHT,
                win32con.SWP_SHOWWINDOW
                | win32con.SWP_FRAMECHANGED,  # Add SWP_FRAMECHANGED to apply style changes
            )

            print("Avatar frame embedded successfully")

        except Exception as e:
            print(f"Error embedding wx.Frame: {str(e)}")
            import traceback

            traceback.print_exc()

    def _update_wx(self):
        """Update wx.App - OPTIMIZED FOR PERFORMANCE"""
        try:
            self.wx_app.Yield()
            self.root.after(
                WX_UPDATE_INTERVAL, self._update_wx
            )  # ~30 FPS instead of 60 FPS
        except Exception as e:
            print(f"Error updating wx.App: {str(e)}")

    def _process_ui_updates(self):
        """Process UI updates from queue to reduce blocking"""
        try:
            while not self.ui_update_queue.empty():
                update_func = self.ui_update_queue.get_nowait()
                update_func()
        except Exception as e:
            print(f"Error processing UI updates: {e}")
        finally:
            self.root.after(UI_UPDATE_INTERVAL, self._process_ui_updates)

    def _process_chat_updates(self):
        """Process chat updates from queue to reduce blocking"""
        try:
            if self.pending_chat_updates:
                # Process all pending updates at once
                updates = self.pending_chat_updates.copy()
                self.pending_chat_updates.clear()

                for speaker, message in updates:
                    self.chat_history.append(
                        (speaker, message, datetime.now().strftime("%I:%M %p"))
                    )
        except Exception as e:
            print(f"Error processing chat updates: {e}")
        finally:
            self.root.after(CHAT_HISTORY_UPDATE_INTERVAL, self._process_chat_updates)

    def _process_subtitle_updates(self):
        """Process subtitle updates from queue to reduce blocking"""
        try:
            while not self.subtitle_update_queue.empty():
                update_func = self.subtitle_update_queue.get_nowait()
                update_func()
        except Exception as e:
            print(f"Error processing subtitle updates: {e}")
        finally:
            self.root.after(SUBTITLE_UPDATE_INTERVAL, self._process_subtitle_updates)

    def _on_window_configure(self, event):
        """Handle window configuration changes (move, resize) to update subtitle position."""
        try:
            if (
                self.subtitle_window
                and self.subtitle_window.winfo_exists()
                and self.is_showing_subtitle
            ):
                # Update subtitle window position
                main_x = self.root.winfo_x()
                main_y = self.root.winfo_y()
                main_width = self.root.winfo_width()
                main_height = self.root.winfo_height()

                subtitle_x = main_x + (main_width - 800) // 2
                subtitle_y = main_y + main_height - 200  # 50px from bottom

                self.subtitle_window.geometry(f"800x150+{subtitle_x}+{subtitle_y}")
        except Exception as e:
            print(f"[ERROR] Failed to update subtitle position: {e}")

    def show_history(self):
        """Show chat history popup - OPTIMIZED VERSION"""
        if self.history_window is None or not self.history_window.winfo_exists():
            self.history_window = tk.Toplevel(self.root)
            self.history_window.title("Chat History")
            self.history_window.geometry("600x800")
            self.history_window.configure(bg="#1e1e2d")

            # Create a frame to hold the text widget and scrollbar
            history_frame = tk.Frame(self.history_window, bg="#1e1e2d")
            history_frame.pack(expand=True, fill=tk.BOTH, padx=15, pady=15)

            # Add scrollbar
            scrollbar = tk.Scrollbar(history_frame)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

            # Create text widget for history with improved styles
            history_text = tk.Text(
                history_frame,
                bg="#1e1e2d",
                fg="white",
                font=("Arial", 11),
                wrap=tk.WORD,
                relief=tk.FLAT,
                padx=15,
                pady=15,
                spacing1=8,  # Add space before each line
                spacing3=8,  # Add space after each line
                yscrollcommand=scrollbar.set,
            )
            history_text.pack(expand=True, fill=tk.BOTH)
            scrollbar.config(command=history_text.yview)

            # Configure tags for different message types
            history_text.tag_configure(
                "user",
                foreground="#2196F3",
                font=("Arial", 11, "bold"),  # Blue for "You"
            )
            history_text.tag_configure(
                "ai",
                foreground="#4CAF50",
                font=("Arial", 11, "bold"),  # Green for AI characters
            )
            history_text.tag_configure(
                "system", foreground="#FF9800", font=("Arial", 11, "bold")
            )
            history_text.tag_configure(
                "timestamp", foreground="#9E9E9E", font=("Arial", 9, "italic")
            )
            history_text.tag_configure(
                "message", foreground="white", font=("Arial", 11)
            )

            # Display chat history with improved formatting and colors
            print(f"[DEBUG] Show History: Displaying {len(self.chat_history)} messages")

            if not self.chat_history:
                history_text.insert(tk.END, "No chat history yet.\n", "message")
            else:
                # Display messages with proper colors
                for i, message in enumerate(self.chat_history):
                    if isinstance(message, str):  # Handle old format messages
                        try:
                            speaker, content = message.split(": ", 1)
                            # Apply color based on speaker
                            tag = "user" if speaker == "You" else "ai"
                            history_text.insert(tk.END, f"{speaker}: ", tag)
                            history_text.insert(tk.END, f"{content}\n\n", "message")
                        except ValueError:
                            history_text.insert(tk.END, f"{message}\n\n", "message")
                    else:  # Handle new format messages with timestamp
                        try:
                            speaker, content, timestamp = message
                            # Insert timestamp
                            history_text.insert(tk.END, f"{timestamp}\n", "timestamp")
                            # Apply color based on speaker
                            tag = "user" if speaker == "You" else "ai"
                            history_text.insert(tk.END, f"{speaker}: ", tag)
                            history_text.insert(tk.END, f"{content}\n\n", "message")
                        except (ValueError, TypeError) as e:
                            print(
                                f"[ERROR] Show History: Failed to parse message {i}: {e}"
                            )
                            history_text.insert(
                                tk.END,
                                f"Error displaying message: {message}\n\n",
                                "message",
                            )

            # Auto-scroll to bottom
            history_text.see(tk.END)
            history_text.configure(state="disabled")

    def create_subtitle_window(self):
        """Create the subtitle window that overlays the main window."""
        if self.subtitle_window is None or not self.subtitle_window.winfo_exists():
            self.subtitle_window = tk.Toplevel(self.root)
            self.subtitle_window.title("Subtitles")
            self.subtitle_window.geometry("800x150")
            self.subtitle_window.configure(bg="#000000")

            # Make window stay on top and remove decorations
            self.subtitle_window.attributes("-topmost", True)
            self.subtitle_window.overrideredirect(True)

            # Set 50% opacity for the background
            self.subtitle_window.attributes("-alpha", 0.7)

            # Position at bottom center of main window
            main_x = self.root.winfo_x()
            main_y = self.root.winfo_y()
            main_width = self.root.winfo_width()
            main_height = self.root.winfo_height()

            subtitle_x = main_x + (main_width - 800) // 2
            subtitle_y = main_y + main_height - 200  # 50px from bottom

            self.subtitle_window.geometry(f"800x150+{subtitle_x}+{subtitle_y}")

            # Create subtitle text widget
            self.subtitle_text = tk.Text(
                self.subtitle_window,
                bg="#000000",
                fg="#FFFFFF",
                font=("Arial", 16, "bold"),
                wrap=tk.WORD,
                relief=tk.FLAT,
                padx=20,
                pady=20,
                height=4,
                state="disabled",
            )
            self.subtitle_text.pack(expand=True, fill=tk.BOTH)

            # Configure text tags for different styles
            self.subtitle_text.tag_configure(
                "current", foreground="#00FF00", font=("Arial", 16, "bold")
            )
            self.subtitle_text.tag_configure(
                "completed", foreground="#888888", font=("Arial", 16)
            )

            print("[DEBUG] Subtitle window created successfully")

    def show_subtitle(self, text, duration=None):
        """Show subtitle text with optional duration - DISPLAY ALL AT ONCE VERSION."""
        try:
            if not self.subtitle_window or not self.subtitle_window.winfo_exists():
                self.create_subtitle_window()

            # Clear current subtitle
            self.subtitle_text.configure(state="normal")
            self.subtitle_text.delete(1.0, tk.END)

            # Display all text at once
            self.subtitle_text.insert(tk.END, text, "current")
            self.subtitle_text.configure(state="disabled")
            self.subtitle_text.see(tk.END)

            # Show subtitle window
            self.subtitle_window.deiconify()
            self.is_showing_subtitle = True

            # Schedule hiding the subtitle after audio duration (if provided) or after 5 seconds
            if duration:
                self.subtitle_timer = self.root.after(
                    int(duration * 1000) + 2000, self.hide_subtitle
                )
            else:
                self.subtitle_timer = self.root.after(5000, self.hide_subtitle)

        except Exception as e:
            print(f"[ERROR] Failed to show subtitle: {e}")

    def _split_into_sentences(self, text):
        """Split text into sentences for better subtitle timing - OPTIMIZED VERSION."""
        import re

        # Performance optimization: Compile regex once
        if not hasattr(self, "_sentence_pattern"):
            self._sentence_pattern = re.compile(r"(?<=[.!?])\s+")
            self._comma_pattern = re.compile(r"(?<=[,;:])\s+")

        # Split by sentence endings, but keep punctuation
        sentences = self._sentence_pattern.split(text.strip())

        # Filter out empty sentences and clean up
        sentences = [s.strip() for s in sentences if s.strip()]

        # If no sentences found, split by commas or other punctuation
        if not sentences:
            sentences = self._comma_pattern.split(text.strip())
            sentences = [s.strip() for s in sentences if s.strip()]

        # If still no splits, use the whole text
        if not sentences:
            sentences = [text.strip()]

        return sentences

    def _display_sentences(self, sentences, total_duration=None):
        """Display sentences one by one with timing - DEPRECATED (now shows all at once)."""
        # This method is kept for compatibility but not used anymore
        pass

    def _show_next_sentence(self, time_per_sentence):
        """Show the next sentence in the queue - DEPRECATED (now shows all at once)."""
        # This method is kept for compatibility but not used anymore
        pass

    def hide_subtitle(self):
        """Hide the subtitle window."""
        try:
            if self.subtitle_timer:
                self.root.after_cancel(self.subtitle_timer)
                self.subtitle_timer = None

            if self.subtitle_window and self.subtitle_window.winfo_exists():
                self.subtitle_window.withdraw()

            self.is_showing_subtitle = False
            print("[DEBUG] Subtitle hidden")

        except Exception as e:
            print(f"[ERROR] Failed to hide subtitle: {e}")

    def update_subtitle(self, text):
        """Update the current subtitle text."""
        try:
            if self.subtitle_text and self.is_showing_subtitle:
                self.subtitle_text.configure(state="normal")
                self.subtitle_text.delete(1.0, tk.END)
                self.subtitle_text.insert(tk.END, text, "current")
                self.subtitle_text.configure(state="disabled")
                self.subtitle_text.see(tk.END)
        except Exception as e:
            print(f"[ERROR] Failed to update subtitle: {e}")

    def cleanup_subtitles(self):
        """Clean up subtitle resources."""
        try:
            if self.subtitle_timer:
                self.root.after_cancel(self.subtitle_timer)
                self.subtitle_timer = None

            if self.subtitle_window and self.subtitle_window.winfo_exists():
                self.subtitle_window.destroy()
                self.subtitle_window = None

            self.is_showing_subtitle = False
            print("[DEBUG] Subtitle cleanup completed")
        except Exception as e:
            print(f"[ERROR] Failed to cleanup subtitles: {e}")

    def update_chat(self, speaker, message):
        """Update chat history - OPTIMIZED VERSION with queuing"""
        # Clean up the speaker name and ensure it's added to history
        if speaker in ["AI", "Assistant"]:
            speaker = self.model_name
        elif speaker == "You":
            speaker = "You"
        else:
            # For any other speaker (like character names), use as is
            pass

        # Add to pending updates queue instead of directly to history
        self.pending_chat_updates.append((speaker, message))
        print(f"[DEBUG] Chat: Queued message: {speaker}: {message[:50]}...")

    def clear_input(self):
        self.input_box.delete(0, tk.END)

    def disable_input_controls(self):
        self.input_box.configure(state="disabled")
        self.send_btn.configure(state="disabled")
        self.voice_btn.configure(state="disabled")

    def enable_input_controls(self):
        self.input_box.configure(state="normal")
        self.send_btn.configure(state="normal")
        self.voice_btn.configure(state="normal")

    def set_voice_button_text(self, text):
        """Update the voice button text"""
        self.voice_btn.configure(text="🎤" if text == "Voice Input" else "⏹️")

    def get_avatar(self):
        """Get the avatar instance"""
        return self.avatar

    def _handle_send(self):
        """Handle send button click or Enter key press"""
        text = self.input_box.get()
        if text.strip():  # Only send if there's non-whitespace text
            self.clear_input()  # Clear input immediately after getting the text
            self.on_send_callback(text)

    def _handle_voice(self):
        """Handle voice button click"""
        if not self.is_processing:
            self.on_voice_toggle_callback()

    def set_model_switch_callback(self, callback):
        """Set callback for model switching"""
        self.on_model_switch = callback

    def _show_character_menu(self):
        """Show character selection dropdown menu"""
        if self.is_processing:
            return

        # Create popup menu
        menu = tk.Menu(
            self.root,
            tearoff=0,
            bg="#2d2d44",
            fg="white",
            activebackground="#3d3d5c",
            activeforeground="white",
            font=("Arial", 11),
        )

        # Get available characters from the asset/model directory
        character_dirs = []
        model_path = Path("asset/model")
        if model_path.exists():
            for item in model_path.iterdir():
                if (
                    item.is_dir()
                    and (item / "character_model" / "character_model.yaml").exists()
                ):
                    character_dirs.append(item.name)

        # If no characters found, use default list
        if not character_dirs:
            character_dirs = ["Amelia", "Eveland", "Gura"]

        # Add character options to menu
        for character in character_dirs:
            # Add checkmark for current character
            checkmark = "✓ " if character == self.model_name else "  "
            menu.add_command(
                label=f"{checkmark}{character}",
                command=lambda c=character: self._select_character(c),
                font=(
                    "Arial",
                    11,
                    "bold" if character == self.model_name else "normal",
                ),
            )

        # Show menu at button position
        x = self.character_btn.winfo_rootx()
        y = self.character_btn.winfo_rooty() + self.character_btn.winfo_height()
        menu.post(x, y)

    def _select_character(self, character_name):
        """Handle character selection from menu"""
        if character_name == self.model_name:
            return  # Already selected

        print(f"\n=== Switching to {character_name} ===")

        # Update model name and environment variable
        self.model_name = character_name
        os.environ["VOICE_TYPE"] = character_name

        # Only update character button color
        if character_name == "Amelia":
            self.character_btn.configure(bg="#ffd05c")
        elif character_name == "Eveland":
            self.character_btn.configure(bg="#318fc5")
        else:  # Gura
            self.character_btn.configure(bg="#4a90e2")

        # Notify user of character change
        self.chat_history.append(
            (
                "System",
                f"Switched to {character_name} character",
                datetime.now().strftime("%I:%M %p"),
            )
        )

        # Clean up old avatar
        if self.avatar:
            self.avatar.cleanup()
            self.avatar = None

        # Hide the frame temporarily
        self.wx_frame.Hide()

        # Create new avatar instance
        print(f"Reinitializing avatar for {character_name}...")
        self.avatar = AnimatedCharacter(self.wx_frame, WINDOW_WIDTH, WINDOW_HEIGHT)
        self.wx_frame.Show()

        # Force a refresh of both frames
        self.wx_frame.Refresh()
        self.avatar_embed.update()

        # Call the model switch callback if provided
        if hasattr(self, "on_model_switch"):
            self.on_model_switch(character_name)

    def _show_background_menu(self):
        """Show background selection dropdown menu"""
        if self.is_processing:
            return

        # Create popup menu
        menu = tk.Menu(
            self.root,
            tearoff=0,
            bg="#2d2d44",
            fg="white",
            activebackground="#3d3d5c",
            activeforeground="white",
            font=("Arial", 11),
        )

        # Get available backgrounds from the avatar
        if hasattr(self, "avatar") and self.avatar:
            background_list = self.avatar.get_background_list()
            current_bg_idx = self.avatar.get_selected_background_index()
        else:
            background_list = []
            current_bg_idx = -1

        # Add background options to menu
        for i, background in enumerate(background_list):
            # Add checkmark for current background
            checkmark = "✓ " if i == current_bg_idx else "  "
            menu.add_command(
                label=f"{checkmark}{background}",
                command=lambda idx=i: self._select_background(idx),
                font=("Arial", 11, "bold" if i == current_bg_idx else "normal"),
            )

        # Add separator
        menu.add_separator()

        # Add "Add New Background" option
        menu.add_command(
            label="  + Add New Background",
            command=self._add_new_background,
            font=("Arial", 11, "italic"),
        )

        # Show menu at button position
        x = self.background_btn.winfo_rootx()
        y = self.background_btn.winfo_rooty() + self.background_btn.winfo_height()
        menu.post(x, y)

    def _select_background(self, background_idx):
        """Handle background selection from menu"""
        if hasattr(self, "avatar") and self.avatar:
            self.avatar.select_background(background_idx)
            # Force a refresh of the avatar display
            if hasattr(self.avatar, "panel") and self.avatar.panel:
                self.avatar.panel.Refresh()

    def _add_new_background(self):
        """Add a new background image"""
        from tkinter import filedialog

        # Open file dialog to select image
        file_path = filedialog.askopenfilename(
            title="Select Background Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("All files", "*.*"),
            ],
        )

        if file_path and hasattr(self, "avatar") and self.avatar:
            try:
                self.avatar.add_background(file_path)
                # Force a refresh of the avatar display
                if hasattr(self.avatar, "panel") and self.avatar.panel:
                    self.avatar.panel.Refresh()
                print(f"Background added: {file_path}")
            except Exception as e:
                print(f"Error adding background: {e}")
                # You could show an error dialog here

    def cleanup(self):
        """Clean up resources before shutdown"""
        try:
            # Cancel all timers
            if self.ui_update_timer:
                self.root.after_cancel(self.ui_update_timer)
            if self.chat_update_timer:
                self.root.after_cancel(self.chat_update_timer)
            if self.subtitle_update_timer:
                self.root.after_cancel(self.subtitle_update_timer)
            if self.subtitle_timer:
                self.root.after_cancel(self.subtitle_timer)

            # Clean up subtitle window
            self.cleanup_subtitles()

            # Shutdown thread pool
            if hasattr(self, "thread_pool"):
                self.thread_pool.shutdown(wait=False)

            # Clean up avatar
            if self.avatar:
                self.avatar.cleanup()

            print("[DEBUG] GUI cleanup completed")
        except Exception as e:
            print(f"[ERROR] Failed to cleanup GUI: {e}")


def main():
    MODEL_NAME = os.getenv("VOICE_TYPE", "Amelia")

    # Verify model directory exists
    model_dir = Path(f"asset/model/{MODEL_NAME}")
    if not model_dir.exists():
        print(f"Error: Model directory not found: {model_dir}")
        print("Available models:")
        for model in Path("asset/model").glob("*"):
            if model.is_dir():
                print(f"- {model.name}")
        sys.exit(1)

    # Initialize all components
    init_handler = InitializationHandler(model_name=MODEL_NAME)
    components = init_handler.initialize_all()

    # Initialize tkinter root
    root = tk.Tk()
    root.title("Vbot")
    root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
    root.configure(bg="#2b2b3b")

    # Create Ollama Handler
    ollama_handler = init_handler.create_ollama_handler()

    # Create a wrapper for voice input processing
    def handle_voice_input(text, timings):
        # Skip showing the transcribed text as subtitle
        ollama_handler.handle_text_input_simple(text)

    # Initialize GUI
    gui = ModernChatGUI(
        root,
        ollama_handler.handle_text_input_simple,
        lambda: components["audio_processor"].toggle_listening(
            gui, handle_voice_input, ollama_handler.is_processing
        ),
    )

    # Set GUI in Ollama Handler and configure audio duration callback
    ollama_handler.gui = gui

    # Handle model switching
    def handle_model_switch(new_model):
        if gui.is_processing:
            return

        print(f"\n=== Switching to {new_model} model ===")

        # Update Ollama handler
        ollama_handler.set_model(new_model)

        # Create new initialization handler for the new model
        print(f"Initializing components for {new_model}...")
        init_handler = InitializationHandler(model_name=new_model)
        new_components = init_handler.initialize_for_character_switch()

        # Update only the components that need to change for different characters
        # Don't reinitialize audio_processor to avoid faster_whisper/onnxruntime issues
        components["inference_handler"] = new_components["inference_handler"]
        components["tts_model"] = new_components["tts_model"]
        components["emotion_handler"] = new_components["emotion_handler"]

        # Update Ollama handler with new components
        ollama_handler.tts_model = new_components["tts_model"]
        ollama_handler.inference_handler = new_components["inference_handler"]
        ollama_handler.emotion_handler = new_components["emotion_handler"]
        # Keep the existing audio_processor to avoid reinitialization issues

        print(f"Components reinitialized for {new_model}")

        # Update window title
        root.title(f"Vbot - {new_model}")

        # Update GUI's voice toggle callback with existing audio processor
        gui.on_voice_toggle_callback = lambda: components[
            "audio_processor"
        ].toggle_listening(gui, handle_voice_input, ollama_handler.is_processing)

        print("=== Model switch complete ===\n")

    # Set the model switch callback
    gui.set_model_switch_callback(handle_model_switch)

    # Register cleanup on window close
    root.protocol(
        "WM_DELETE_WINDOW",
        lambda: (
            gui.cleanup(),
            components["docker_handler"].cleanup(),
            root.destroy(),
        ),
    )

    # Start the application
    root.mainloop()


if __name__ == "__main__":
    main()
