import tkinter as tk
from tkinter import scrolledtext, ttk
from .avatar import AnimatedAvatar


class ChatGUI:
    def __init__(self, root, on_send_callback, on_voice_toggle_callback):
        self.root = root
        self.root.title("AI Chatbot")
        self.root.geometry("700x400")

        self.on_send = on_send_callback
        self.on_voice_toggle = on_voice_toggle_callback

        self.setup_gui()

    def setup_gui(self):
        # Avatar frame
        self.avatar_frame = ttk.Frame(self.root)
        self.avatar_frame.pack(side=tk.LEFT, padx=10, pady=10)
        self.avatar_canvas = tk.Canvas(self.avatar_frame, width=200, height=200)
        self.avatar_canvas.pack()
        self.avatar = AnimatedAvatar(self.avatar_canvas, 200, 200)

        # Chat frame
        self.chat_frame = ttk.Frame(self.root)
        self.chat_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=10, pady=10)
        self.text_area = scrolledtext.ScrolledText(
            self.chat_frame, wrap=tk.WORD, width=50, height=20
        )
        self.text_area.pack(expand=True, fill=tk.BOTH)

        # Input frame
        self.input_frame = ttk.Frame(self.chat_frame)
        self.input_frame.pack(pady=5)
        self.input_entry = ttk.Entry(self.input_frame, width=40)
        self.input_entry.pack(side=tk.LEFT, padx=5)

        # Bind Enter key to send
        self.input_entry.bind("<Return>", lambda e: self._on_send_click())

        # Buttons
        self.send_button = ttk.Button(
            self.input_frame, text="Send", command=self._on_send_click
        )
        self.send_button.pack(side=tk.LEFT)

        self.voice_button = ttk.Button(
            self.input_frame, text="Voice Input", command=self._on_voice_click
        )
        self.voice_button.pack(side=tk.LEFT, padx=5)

    def _on_send_click(self):
        text = self.input_entry.get().strip()
        if text:
            # Clear input box before processing
            self.input_entry.delete(0, tk.END)
            # Update GUI immediately
            self.root.update_idletasks()
            # Send the text
            self.on_send(text)

    def _on_voice_click(self):
        self.on_voice_toggle()

    def update_chat(self, speaker, text):
        """Add a message to the chat area"""
        self.text_area.insert(tk.END, f"{speaker}: {text}\n")
        self.text_area.see(tk.END)

    def disable_input_controls(self):
        """Disable all input controls"""
        self.input_entry.config(state="disabled")
        self.send_button.config(state="disabled")
        self.voice_button.config(state="disabled")

    def enable_input_controls(self):
        """Enable all input controls"""
        self.input_entry.config(state="normal")
        self.send_button.config(state="normal")
        self.voice_button.config(state="normal")

        # Set focus back to input entry
        self.input_entry.focus_set()

    def set_voice_button_text(self, text):
        """Update the voice button text"""
        self.voice_button.config(text=text)

    def get_avatar(self):
        """Get the avatar instance"""
        return self.avatar
