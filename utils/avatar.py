import tkinter as tk
from PIL import Image, ImageTk
import threading
import time
import os
from pathlib import Path


class AnimatedAvatar:
    def __init__(self, canvas, width, height):
        self.canvas = canvas
        self.width = width
        self.height = height

        # Load images directly
        self.mouth_closed = Image.open("./asset/pic-close.png")
        self.mouth_open = Image.open("./asset/pic-open.png")

        # Resize images to fit canvas
        self.mouth_closed = self.mouth_closed.resize((width, height))
        self.mouth_open = self.mouth_open.resize((width, height))

        # Convert to PhotoImage
        self.mouth_closed_photo = ImageTk.PhotoImage(self.mouth_closed)
        self.mouth_open_photo = ImageTk.PhotoImage(self.mouth_open)

        # Create image on canvas at top-left corner
        self.image_on_canvas = self.canvas.create_image(
            0, 0, anchor=tk.NW, image=self.mouth_closed_photo
        )

    def animate_mouth(self, duration):
        """Animate mouth opening and closing for speech"""
        frames = int(duration * 10)  # 10 frames per second
        for i in range(frames):
            if i % 2 == 0:
                self.canvas.itemconfig(
                    self.image_on_canvas, image=self.mouth_open_photo
                )
            else:
                self.canvas.itemconfig(
                    self.image_on_canvas, image=self.mouth_closed_photo
                )
            self.canvas.update()
            time.sleep(0.1)
        # Reset to closed mouth
        self.canvas.itemconfig(self.image_on_canvas, image=self.mouth_closed_photo)
