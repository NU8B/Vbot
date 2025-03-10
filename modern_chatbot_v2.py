import os
import sys
import tkinter as tk
import warnings
from pathlib import Path
import wx

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from utils.gui import ChatGUI
from utils.initialization_utils import InitializationHandler
from utils.avatar import AnimatedCharacter

# Suppress all warnings
warnings.filterwarnings("ignore")

class ModernChatGUI(ChatGUI):
    def __init__(self, root, on_send_callback, on_voice_toggle_callback):
        # Initialize wx.App first (needed for avatar)
        self.wx_app = wx.App()
        
        # Get model name from environment
        self.model_name = os.getenv('VOICE_TYPE', 'Amelia Watson')
        
        # Initialize parent class but don't create its widgets
        self.root = root
        self.on_send_callback = on_send_callback
        self.on_voice_toggle_callback = on_voice_toggle_callback
        self.is_processing = False
        
        # Main container
        self.main_container = tk.Frame(root, bg='#2b2b3b')
        self.main_container.pack(expand=True, fill=tk.BOTH)
        
        # Avatar container (centered, fixed size)
        self.avatar_container = tk.Frame(self.main_container, bg='#2b2b3b', width=512, height=512)
        self.avatar_container.pack(expand=True, padx=20, pady=(20, 0))  # Reduced bottom padding
        self.avatar_container.pack_propagate(False)  # Maintain fixed size
        
        # Subtitle label (transparent background)
        self.subtitle_frame = tk.Frame(self.main_container, bg='#000000')
        self.subtitle_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        self.subtitle_label = tk.Label(
            self.subtitle_frame,
            text="",
            font=('Arial', 16, 'bold'),
            fg='white',
            bg='#000000',
            wraplength=800  # Allow text to wrap
        )
        self.subtitle_label.pack(expand=True)
        
        # Initially hide the subtitle
        self.subtitle_frame.pack_forget()
        
        # Create and initialize avatar
        print("Creating avatar frame...")
        self.wx_frame = wx.Frame(None, title="", size=(512, 512))
        self.wx_frame.SetBackgroundColour('#2b2b3b')
        
        print("Initializing avatar...")
        self.avatar = AnimatedCharacter(self.wx_frame, 512, 512)
        self.wx_window_id = self.wx_frame.GetHandle()
        
        # Create a tkinter window to embed the wx.Frame
        self.avatar_embed = tk.Frame(self.avatar_container, bg='#2b2b3b')
        self.avatar_embed.pack(expand=True, fill=tk.BOTH)
        
        # Bottom bar container
        self.bottom_bar = tk.Frame(self.main_container, bg='#232334', height=60)
        self.bottom_bar.pack(fill=tk.X, side=tk.BOTTOM, padx=20, pady=20)
        self.bottom_bar.pack_propagate(False)
        
        # Input box
        self.input_box = tk.Entry(
            self.bottom_bar,
            bg='#1e1e2d',
            fg='white',
            insertbackground='white',
            font=('Arial', 12),
            relief=tk.FLAT
        )
        self.input_box.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=(20, 10), pady=10)
        
        # Button container
        button_container = tk.Frame(self.bottom_bar, bg='#232334')
        button_container.pack(side=tk.RIGHT, padx=(0, 20))
        
        # History button
        self.history_btn = tk.Button(
            button_container,
            text="💭",
            font=('Arial', 14),
            bg='#ffd05c',
            fg='black',
            relief=tk.FLAT,
            command=self.show_history
        )
        self.history_btn.pack(side=tk.RIGHT, padx=5, pady=10)
        
        # Voice button
        self.voice_btn = tk.Button(
            button_container,
            text="🎤",
            font=('Arial', 14),
            bg='#ffd05c',
            fg='black',
            relief=tk.FLAT,
            command=self._handle_voice
        )
        self.voice_btn.pack(side=tk.RIGHT, padx=5, pady=10)
        
        # Send button
        self.send_btn = tk.Button(
            button_container,
            text="➤",
            font=('Arial', 14),
            bg='#ffd05c',
            fg='black',
            relief=tk.FLAT,
            command=self._handle_send
        )
        self.send_btn.pack(side=tk.RIGHT, padx=5, pady=10)
        
        # Bind enter key to send
        self.input_box.bind("<Return>", lambda e: self._handle_send())
        
        # Chat history storage
        self.chat_history = []
        self.history_window = None
        
        # Style buttons on hover
        for btn in [self.send_btn, self.voice_btn, self.history_btn]:
            btn.bind("<Enter>", lambda e, b=btn: b.configure(bg='#e6bb53'))
            btn.bind("<Leave>", lambda e, b=btn: b.configure(bg='#ffd05c'))
        
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
            
            # Show the wx.Frame
            self.wx_frame.Show()
            
            # Position the wx.Frame
            win32gui.SetWindowPos(
                self.wx_window_id,
                win32con.HWND_TOP,
                0, 0,
                512, 512,
                win32con.SWP_SHOWWINDOW
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
            self.history_window.geometry("400x600")
            self.history_window.configure(bg='#2b2b3b')
            
            # Create text widget for history
            history_text = tk.Text(
                self.history_window,
                bg='#1e1e2d',
                fg='white',
                font=('Arial', 12),
                wrap=tk.WORD,
                relief=tk.FLAT
            )
            history_text.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
            
            # Display chat history
            for message in self.chat_history:
                history_text.insert(tk.END, message + "\n")
            history_text.configure(state='disabled')

    def update_chat(self, speaker, message):
        """Update chat history and show subtitle for AI responses"""
        # Clean up the speaker name
        if speaker in ["AI", "Assistant"]:
            speaker = self.model_name
            self.chat_history.append(f"{speaker}: {message}")
            self.show_subtitle(message)
        elif speaker == "You":  # Only add user messages to chat history
            self.chat_history.append(f"{speaker}: {message}")
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

    def show_subtitle(self, text):
        """Show subtitle with specified text"""
        # Make the subtitle background semi-transparent black
        self.subtitle_frame.configure(bg='#000000')
        self.subtitle_label.configure(
            text=text,
            bg='#000000',
            fg='white',
            font=('Arial', 16, 'bold')
        )
        self.subtitle_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        self.root.update_idletasks()

    def hide_subtitle(self):
        """Hide the subtitle"""
        self.subtitle_frame.pack_forget()
        self.root.update_idletasks()

    def update_subtitle(self, text):
        """Update subtitle text and ensure it's visible"""
        self.subtitle_label.configure(text=text)
        if not self.subtitle_frame.winfo_ismapped():
            self.subtitle_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        self.root.update_idletasks()

    def set_subtitle_duration(self, duration_ms):
        """Set a timer to hide the subtitle after the specified duration"""
        self.root.after(duration_ms, self.hide_subtitle)

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
    root.title("AI Assistant")
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
    
    # Override the inference handler's play_audio method to handle subtitle timing
    original_play_audio = components["inference_handler"].play_audio
    def play_audio_with_subtitle(speech, duration, avatar, audio_processor):
        # Convert duration from seconds to milliseconds and add 1 second (1000ms)
        subtitle_duration = int(duration * 1000) + 1000
        gui.set_subtitle_duration(subtitle_duration)
        # Call original play_audio method
        original_play_audio(speech, duration, avatar, audio_processor)

    # Replace the play_audio method with our new version
    components["inference_handler"].play_audio = play_audio_with_subtitle
    
    # Register cleanup on window close
    root.protocol(
        "WM_DELETE_WINDOW",
        lambda: (components["docker_handler"].cleanup(), root.destroy()),
    )
    
    # Start the application
    root.mainloop()

if __name__ == "__main__":
    main() 