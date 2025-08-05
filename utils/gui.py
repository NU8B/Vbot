import tkinter as tk
from tkinter import scrolledtext, ttk
import wx
import threading
import torch
import wx.siplib
import win32gui
import win32con
from PIL import Image, ImageTk

from .avatar import AnimatedCharacter


class ChatGUI:
    def __init__(self, root, on_send_callback, on_voice_toggle_callback):
        self.root = root
        self.root.title("AI Chatbot")
        self.root.geometry("1200x800")  # Increased window size to fit larger avatar

        # Configure style
        style = ttk.Style()
        style.configure("Custom.TButton", padding=10, font=("Helvetica", 12))
        style.configure("Mic.TButton", padding=10, font=("Helvetica", 12))

        self.on_send = on_send_callback
        self.on_voice_toggle = on_voice_toggle_callback

        # Create main container
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(expand=True, fill=tk.BOTH)

        # Create avatar frame on the left
        self.avatar_frame = ttk.Frame(self.main_container, width=512, height=512)
        self.avatar_frame.pack(side=tk.LEFT, padx=10, pady=10)
        self.avatar_frame.pack_propagate(False)  # Prevent frame from shrinking

        # Create chat frame on the right
        self.chat_frame = ttk.Frame(self.main_container)
        self.chat_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=10, pady=10)

        # Setup wxPython
        self.wx_app = None
        self.avatar = None
        self.setup_wx()

        # Setup chat interface
        self.setup_chat()

    def setup_wx(self):
        """Setup wxPython"""
        # Create wx.App in the main thread
        if not wx.GetApp():
            self.wx_app = wx.App(False)

        # Create a parent frame
        self.wx_frame = wx.Frame(
            None,
            title="Avatar",
            size=(512, 512),
            style=wx.FRAME_NO_TASKBAR | wx.BORDER_NONE,
        )

        # Get the window handle of the avatar frame
        avatar_hwnd = self.avatar_frame.winfo_id()

        # Create the avatar
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.avatar = AnimatedCharacter(self.wx_frame, 512, 512, device)

        # Show the frame
        self.wx_frame.Show()

        # Embed the wx frame into the tkinter frame
        try:
            # Set the parent window
            win32gui.SetParent(self.wx_frame.GetHandle(), avatar_hwnd)

            # Remove window decorations
            style = win32gui.GetWindowLong(
                self.wx_frame.GetHandle(), win32con.GWL_STYLE
            )
            style = style & ~win32con.WS_CAPTION & ~win32con.WS_THICKFRAME
            win32gui.SetWindowLong(self.wx_frame.GetHandle(), win32con.GWL_STYLE, style)

            # Position the window
            win32gui.SetWindowPos(
                self.wx_frame.GetHandle(),
                None,
                0,
                0,
                512,
                512,
                win32con.SWP_NOMOVE | win32con.SWP_NOZORDER,
            )
        except Exception as e:
            print(f"Error embedding window: {str(e)}")

        # Set up a timer to process wx events
        def process_wx_events():
            self.wx_app.ProcessPendingEvents()
            self.root.after(10, process_wx_events)  # Schedule next update

        # Start processing wx events
        self.root.after(10, process_wx_events)

    def setup_chat(self):
        # Chat area with custom font and colors
        self.text_area = scrolledtext.ScrolledText(
            self.chat_frame,
            wrap=tk.WORD,
            width=50,
            height=20,
            font=("Helvetica", 12),
            bg="#FFFFFF",
        )
        self.text_area.pack(expand=True, fill=tk.BOTH)

        # Configure tags for different speakers
        self.text_area.tag_configure("user", foreground="#007AFF")  # Blue for user
        self.text_area.tag_configure("ai", foreground="#FF2D55")  # Pink for AI

        # Input frame
        self.input_frame = ttk.Frame(self.chat_frame)
        self.input_frame.pack(pady=10)

        # Input entry with larger font
        self.input_entry = ttk.Entry(self.input_frame, width=40, font=("Helvetica", 12))
        self.input_entry.pack(side=tk.LEFT, padx=5)

        # Bind Enter key to send
        self.input_entry.bind("<Return>", lambda e: self._on_send_click())

        # Send button with custom style
        self.send_button = ttk.Button(
            self.input_frame,
            text="Send",
            command=self._on_send_click,
            style="Custom.TButton",
        )
        self.send_button.pack(side=tk.LEFT, padx=5)

        # Voice button with mic icon (🎤)
        self.voice_button = ttk.Button(
            self.input_frame,
            text="🎤",
            command=self._on_voice_click,
            style="Mic.TButton",
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
        """Add a message to the chat area with color"""
        if speaker == "You" or speaker == "You said":
            self.text_area.insert(tk.END, f"{speaker}: ", "user")
            self.text_area.insert(tk.END, f"{text}\n")
        else:
            self.text_area.insert(tk.END, f"{speaker}: ", "ai")
            self.text_area.insert(tk.END, f"{text}\n")
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
        self.input_entry.focus_set()

    def set_voice_button_text(self, text):
        """Update the voice button text"""
        # Keep the mic icon, just change color when active/inactive
        self.voice_button.config(text="🎤")

    def get_avatar(self):
        """Get the avatar instance"""
        return self.avatar
