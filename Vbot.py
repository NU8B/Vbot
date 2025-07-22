import os
import sys
import tkinter as tk
import warnings
from pathlib import Path
import wx
from datetime import datetime
import re

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from utils.gui import ChatGUI
from utils.initialization_utils import InitializationHandler
from utils.avatar import AnimatedCharacter

# Suppress all warnings
warnings.filterwarnings("ignore")

AVATAR_SIZE = 800  # Define avatar size as a constant

class ModernChatGUI(ChatGUI):
    def __init__(self, root, on_send_callback, on_voice_toggle_callback):
        # Initialize wx.App first (needed for avatar)
        self.wx_app = wx.App()
        
        # Get model name from environment
        self.model_name = os.getenv('VOICE_TYPE', 'Amelia')
        
        # Get background color from bg_color.txt or use model-specific defaults
        bg_color_path = Path(f"asset/model/{self.model_name}/bg_color.txt")
        try:
            with open(bg_color_path, 'r') as f:
                self.bg_color = f.read().strip()
        except:
            # Default background colors for each model
            self.bg_color = "#2b2b3b" if self.model_name == "Eveland" else "#ffd05c"
        
        # Initialize parent class but don't create its widgets
        self.root = root
        self.on_send_callback = on_send_callback
        self.on_voice_toggle_callback = on_voice_toggle_callback
        self.is_processing = False
        
        # Available models
        self.available_models = ["Amelia", "Eveland"]
        self.current_model_index = 0 if self.model_name == "Amelia" else 1
        
        # Main container - use character's background color
        self.main_container = tk.Frame(root, bg=self.bg_color)
        self.main_container.pack(expand=True, fill=tk.BOTH)
        
        # Avatar container - match character's background
        self.avatar_container = tk.Frame(self.main_container, bg=self.bg_color, width=AVATAR_SIZE, height=AVATAR_SIZE)
        self.avatar_container.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        self.avatar_container.pack_propagate(False)
        
        # Subtitle window for custom rendering and transparency
        self.subtitle_window = tk.Toplevel(self.root)
        self.subtitle_window.overrideredirect(True)  # Borderless
        self.subtitle_window.attributes('-alpha', 0.75)  # Semi-transparent
        self.subtitle_window.configure(bg='black')
        self.subtitle_window.withdraw()  # Initially hidden

        self.subtitle_label = tk.Label(
            self.subtitle_window,
            text="",
            font=('Arial', 18, 'bold'),
            fg='white',
            bg='black',
            justify='center'
        )
        self.subtitle_label.pack(expand=True, fill=tk.BOTH, padx=15, pady=10)

        # Bind subtitle positioning to main window changes
        self.root.bind("<Configure>", self._position_subtitle_window)
        
        # Create and initialize avatar
        print("Creating avatar frame...")
        self.wx_frame = wx.Frame(
            None,
            style=wx.BORDER_NONE | wx.FRAME_NO_TASKBAR,  # Remove window decorations and taskbar entry
            size=(AVATAR_SIZE, AVATAR_SIZE)
        )
        self.wx_frame.SetBackgroundColour('#2b2b3b')
        
        print("Initializing avatar...")
        self.avatar = AnimatedCharacter(self.wx_frame, AVATAR_SIZE, AVATAR_SIZE)
        self.wx_window_id = self.wx_frame.GetHandle()
        
        # Create a tkinter window to embed the wx.Frame
        self.avatar_embed = tk.Frame(self.avatar_container, bg='#2b2b3b')
        self.avatar_embed.pack(expand=True, fill=tk.BOTH)
        
        # Bottom bar container - darker purple theme
        self.bottom_bar = tk.Frame(self.main_container, bg='#1a1a2e', height=60)
        self.bottom_bar.pack(fill=tk.X, side=tk.BOTTOM, padx=20, pady=20)
        self.bottom_bar.pack_propagate(False)
        
        # Input box - darker theme
        self.input_box = tk.Entry(
            self.bottom_bar,
            bg='#5b5b6b',
            fg='white',
            insertbackground='white',
            font=('Arial', 12),
            relief=tk.FLAT
        )
        self.input_box.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=(20, 10), pady=10)
        
        # Button container - match bottom bar
        button_container = tk.Frame(self.bottom_bar, bg='#1a1a2e')
        button_container.pack(side=tk.RIGHT, padx=(0, 20))
        
        # Model switch button - keep model-specific colors
        self.switch_btn = tk.Button(
            button_container,
            text="🔄",
            font=('Arial', 14),
            bg='#ffd05c' if self.model_name == "Amelia" else '#318fc5',
            fg='black',
            relief=tk.FLAT,
            command=self._handle_model_switch
        )
        self.switch_btn.pack(side=tk.RIGHT, padx=5, pady=10)
        
        # History button - darker purple theme
        self.history_btn = tk.Button(
            button_container,
            text="💭",
            font=('Arial', 14),
            bg='#2d2d44',
            fg='white',
            relief=tk.FLAT,
            command=self.show_history
        )
        self.history_btn.pack(side=tk.RIGHT, padx=5, pady=10)
        
        # Voice button - darker purple theme
        self.voice_btn = tk.Button(
            button_container,
            text="🎤",
            font=('Arial', 14),
            bg='#2d2d44',
            fg='white',
            relief=tk.FLAT,
            command=self._handle_voice
        )
        self.voice_btn.pack(side=tk.RIGHT, padx=5, pady=10)
        
        # Send button - darker purple theme
        self.send_btn = tk.Button(
            button_container,
            text="➤",
            font=('Arial', 14),
            bg='#2d2d44',
            fg='white',
            relief=tk.FLAT,
            command=self._handle_send
        )
        self.send_btn.pack(side=tk.RIGHT, padx=5, pady=10)
        
        # Bind enter key to send
        self.input_box.bind("<Return>", lambda e: self._handle_send())
        
        # Chat history storage
        self.chat_history = []
        self.history_window = None
        
        # Style buttons on hover - darker purple theme
        for btn in [self.send_btn, self.voice_btn, self.history_btn]:
            btn.bind("<Enter>", lambda e, b=btn: b.configure(bg='#3d3d5c'))
            btn.bind("<Leave>", lambda e, b=btn: b.configure(bg='#2d2d44'))
        
        # Special hover effect for switch button based on current model
        def switch_hover_enter(e):
            self.switch_btn.configure(
                bg='#e6bb53' if self.model_name == "Amelia" else '#2b7eb3'
            )
        
        def switch_hover_leave(e):
            self.switch_btn.configure(
                bg='#ffd05c' if self.model_name == "Amelia" else '#318fc5'
            )
        
        self.switch_btn.bind("<Enter>", switch_hover_enter)
        self.switch_btn.bind("<Leave>", switch_hover_leave)
        
        # Embed wx.Frame into tkinter with a slight delay
        self.root.after(100, self._embed_wx_frame)
        
        # Start wx update timer
        self.root.after(10, self._update_wx)

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
            style = style & ~(win32con.WS_CAPTION | win32con.WS_THICKFRAME | win32con.WS_SYSMENU)
            win32gui.SetWindowLong(self.wx_window_id, win32con.GWL_STYLE, style)
            
            # Show the wx.Frame
            self.wx_frame.Show()
            
            # Position the wx.Frame
            win32gui.SetWindowPos(
                self.wx_window_id,
                win32con.HWND_TOP,
                0, 0,
                AVATAR_SIZE, AVATAR_SIZE,
                win32con.SWP_SHOWWINDOW | win32con.SWP_FRAMECHANGED  # Add SWP_FRAMECHANGED to apply style changes
            )
            
            print("Avatar frame embedded successfully")
            
        except Exception as e:
            print(f"Error embedding wx.Frame: {str(e)}")
            import traceback
            traceback.print_exc()

    def _update_wx(self):
        """Update wx.App"""
        try:
            self.wx_app.Yield()
            self.root.after(10, self._update_wx)
        except Exception as e:
            print(f"Error updating wx.App: {str(e)}")

    def show_history(self):
        if self.history_window is None or not self.history_window.winfo_exists():
            self.history_window = tk.Toplevel(self.root)
            self.history_window.title("Chat History")
            self.history_window.geometry("500x700")
            self.history_window.configure(bg='#1e1e2d')
            
            # Create a frame to hold the text widget and scrollbar
            history_frame = tk.Frame(self.history_window, bg='#1e1e2d')
            history_frame.pack(expand=True, fill=tk.BOTH, padx=15, pady=15)
            
            # Add scrollbar
            scrollbar = tk.Scrollbar(history_frame)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Create text widget for history with improved styles
            history_text = tk.Text(
                history_frame,
                bg='#1e1e2d',
                fg='white',
                font=('Arial', 12),
                wrap=tk.WORD,
                relief=tk.FLAT,
                padx=15,
                pady=15,
                spacing1=10,  # Add space before each line
                spacing3=10,  # Add space after each line
                yscrollcommand=scrollbar.set
            )
            history_text.pack(expand=True, fill=tk.BOTH)
            scrollbar.config(command=history_text.yview)
            
            # Configure tags for different message types
            history_text.tag_configure("user", foreground="#4CAF50", font=('Arial', 12, 'bold'))
            history_text.tag_configure("ai", foreground="#2196F3", font=('Arial', 12, 'bold'))
            history_text.tag_configure("timestamp", foreground="#9E9E9E", font=('Arial', 10, 'italic'))
            history_text.tag_configure("message", foreground="white", font=('Arial', 12))
            
            # Display chat history with improved formatting
            for message in self.chat_history:
                if isinstance(message, str):  # Handle old format messages
                    speaker, content = message.split(": ", 1)
                    history_text.insert(tk.END, f"{speaker}: ", "user" if speaker == "You" else "ai")
                    history_text.insert(tk.END, f"{content}\n\n", "message")
                else:  # Handle new format messages with timestamp
                    speaker, content, timestamp = message
                    history_text.insert(tk.END, f"{timestamp}\n", "timestamp")
                    history_text.insert(tk.END, f"{speaker}: ", "user" if speaker == "You" else "ai")
                    history_text.insert(tk.END, f"{content}\n\n", "message")
            
            history_text.configure(state='disabled')

    def update_chat(self, speaker, message):
        """Update chat history and show subtitle for AI responses"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%I:%M %p")  # 12-hour format with AM/PM
        
        # Clean up the speaker name
        if speaker in ["AI", "Assistant"]:
            speaker = self.model_name
            self.chat_history.append((speaker, message, timestamp))
            self.show_subtitle(message)
        elif speaker == "You":  # Only add user messages to chat history
            self.chat_history.append((speaker, message, timestamp))
        # Ignore "You said" messages completely

    def clear_input(self):
        self.input_box.delete(0, tk.END)

    def disable_input_controls(self):
        self.input_box.configure(state='disabled')
        self.send_btn.configure(state='disabled')
        self.voice_btn.configure(state='disabled')

    def enable_input_controls(self):
        self.input_box.configure(state='normal')
        self.send_btn.configure(state='normal')
        self.voice_btn.configure(state='normal')

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

    def _position_subtitle_window(self, event=None):
        """Keep the subtitle window positioned relative to the main window."""
        if self.subtitle_window.winfo_ismapped():
            main_win_width = self.root.winfo_width()
            main_win_height = self.root.winfo_height()
            x = self.root.winfo_x() + (main_win_width - int(main_win_width * 0.9)) // 2
            y = self.root.winfo_y() + main_win_height - 120 - 90  # height and y-offset

            width = int(main_win_width * 0.9)
            height = 120
            
            self.subtitle_window.geometry(f"{width}x{height}+{x}+{y}")
            self.subtitle_label.configure(wraplength=width - 30)

    def show_subtitle(self, text):
        """Show subtitle with specified text using a custom transparent window."""
        self.subtitle_label.configure(text=text)
        if not self.subtitle_window.winfo_ismapped():
            self.subtitle_window.deiconify()
        self._position_subtitle_window()
        self.root.update_idletasks()

    def show_subtitle_anime_style(self, text):
        """Show subtitle with anime-style fast updates and visual effects."""
        # Quick update without heavy operations for anime-style speed
        self.subtitle_label.configure(text=text)
        if not self.subtitle_window.winfo_ismapped():
            self.subtitle_window.deiconify()
            self._position_subtitle_window()
        # Use update() instead of update_idletasks() for immediate visual refresh
        self.subtitle_window.update()

    def hide_subtitle(self):
        """Hide the subtitle"""
        self.subtitle_window.withdraw()
        self.root.update_idletasks()

    def update_subtitle(self, text):
        """Update subtitle text and ensure it's visible"""
        self.show_subtitle(text)

    def update_subtitle_anime_style(self, text):
        """Update subtitle with anime-style timing - ultra-fast updates"""
        self.show_subtitle_anime_style(text)

    def _handle_model_switch(self):
        """Handle model switch button click"""
        if not self.is_processing:
            # Switch to next model
            self.current_model_index = (self.current_model_index + 1) % len(self.available_models)
            new_model = self.available_models[self.current_model_index]
            
            # Update model name and environment variable
            self.model_name = new_model
            os.environ['VOICE_TYPE'] = new_model
            
            # Get new background color
            bg_color_path = Path(f"asset/model/{new_model}/bg_color.txt")
            try:
                with open(bg_color_path, 'r') as f:
                    self.bg_color = f.read().strip()
            except:
                self.bg_color = "#2b2b3b" if new_model == "Eveland" else "#ffd05c"
            
            # Update UI colors
            self.main_container.configure(bg=self.bg_color)
            self.avatar_container.configure(bg=self.bg_color)
            self.avatar_embed.configure(bg=self.bg_color)
            
            # Update switch button color
            self.switch_btn.configure(
                bg='#ffd05c' if new_model == "Amelia" else '#318fc5'
            )
            
            # Notify user of model change
            self.chat_history.append(f"System: Switched to {new_model} model")
            
            # Clean up old avatar
            if self.avatar:
                self.avatar.cleanup()
                self.avatar = None
            
            # Update wx frame background color
            self.wx_frame.SetBackgroundColour(self.bg_color)
            
            # Hide the frame temporarily
            self.wx_frame.Hide()
            
            # Create new avatar instance
            print(f"Reinitializing avatar for {new_model}...")
            self.avatar = AnimatedCharacter(self.wx_frame, AVATAR_SIZE, AVATAR_SIZE)
            
            # Show frame and start animation
            self.wx_frame.Show()
            self.avatar.start_animation()
            
            # Force a refresh of both frames
            self.wx_frame.Refresh()
            self.avatar_embed.update()
            
            # Call the model switch callback if provided
            if hasattr(self, 'on_model_switch'):
                self.on_model_switch(new_model)

    def set_model_switch_callback(self, callback):
        """Set callback for model switching"""
        self.on_model_switch = callback

def main():
    MODEL_NAME = os.getenv('VOICE_TYPE', 'Amelia')
    
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
    root.geometry("1200x800")
    root.configure(bg='#2b2b3b')
    
    # Create Ollama Handler
    ollama_handler = init_handler.create_ollama_handler()
    
    # Create a wrapper for voice input processing
    def handle_voice_input(text, timings):
        # Skip showing the transcribed text as subtitle
        ollama_handler.handle_text_input(text)

    # Initialize GUI
    gui = ModernChatGUI(
        root,
        ollama_handler.handle_text_input,
        lambda: components["audio_processor"].toggle_listening(
            gui,
            handle_voice_input,
            ollama_handler.is_processing
        )
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
        new_components = init_handler.initialize_all()
        
        # Update all voice-related components
        components["inference_handler"] = new_components["inference_handler"]
        components["audio_processor"] = new_components["audio_processor"]
        components["tts_model"] = new_components["tts_model"]
        
        # Update Ollama handler with new components
        ollama_handler.tts_model = new_components["tts_model"]
        ollama_handler.audio_processor = new_components["audio_processor"]
        ollama_handler.inference_handler = new_components["inference_handler"]
        
        print(f"Components reinitialized for {new_model}")
        
        # Update window title
        root.title(f"Vbot - {new_model}")
        
        # Update GUI's voice toggle callback with new audio processor
        gui.on_voice_toggle_callback = lambda: components["audio_processor"].toggle_listening(
            gui,
            handle_voice_input,
            ollama_handler.is_processing
        )
        
        print("=== Model switch complete ===\n")

    # Set the model switch callback
    gui.set_model_switch_callback(handle_model_switch)
    
    # Register cleanup on window close
    root.protocol(
        "WM_DELETE_WINDOW",
        lambda: (components["docker_handler"].cleanup(), root.destroy()),
    )
    
    # Start the application
    root.mainloop()

if __name__ == "__main__":
    main() 