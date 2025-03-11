# pip install speechrecognition pyttsx3
import speech_recognition as sr
import pyttsx3
from transformers import AutoTokenizer, AutoModelForCausalLM
import tkinter as tk
from tkinter import scrolledtext, ttk
import threading
import torch
import pyaudio
import wave
from PIL import Image, ImageTk
import time


class AnimatedAvatar:
    def __init__(self, canvas, width, height):
        self.canvas = canvas
        self.width = width
        self.height = height

        # Load avatar images (you'll need to create these)
        self.mouth_closed = Image.open("./asset/pic-close.png")
        self.mouth_open = Image.open("./asset/pic-open.png")

        # Resize images
        self.mouth_closed = self.mouth_closed.resize((width, height))
        self.mouth_open = self.mouth_open.resize((width, height))

        # Convert to PhotoImage
        self.mouth_closed_photo = ImageTk.PhotoImage(self.mouth_closed)
        self.mouth_open_photo = ImageTk.PhotoImage(self.mouth_open)

        # Create image on canvas
        self.image_on_canvas = self.canvas.create_image(
            0, 0, anchor=tk.NW, image=self.mouth_closed_photo
        )

    def animate_mouth(self, duration):
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
        self.canvas.itemconfig(self.image_on_canvas, image=self.mouth_closed_photo)


class AIAssistant:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Assistant")
        self.root.geometry("700x400")

        self.recognizer = sr.Recognizer()
        self.engine = pyttsx3.init()

        # Set the voice to Microsoft Zira
        self.engine.setProperty(
            "voice",
            "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_EN-US_ZIRA_11.0",
        )

        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

        self.chat_history_ids = None
        self.is_listening = False

        self.setup_gui()

    def setup_gui(self):
        # Create a frame for the avatar
        self.avatar_frame = ttk.Frame(self.root)
        self.avatar_frame.pack(side=tk.LEFT, padx=10, pady=10)

        # Create a canvas for the avatar
        self.avatar_canvas = tk.Canvas(self.avatar_frame, width=200, height=200)
        self.avatar_canvas.pack()

        # Create the animated avatar
        self.avatar = AnimatedAvatar(self.avatar_canvas, 200, 200)

        # Create a frame for the chat interface
        self.chat_frame = ttk.Frame(self.root)
        self.chat_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=10, pady=10)

        self.text_area = scrolledtext.ScrolledText(
            self.chat_frame, wrap=tk.WORD, width=50, height=20
        )
        self.text_area.pack(expand=True, fill=tk.BOTH)

        self.input_frame = ttk.Frame(self.chat_frame)
        self.input_frame.pack(pady=5)

        self.input_entry = ttk.Entry(self.input_frame, width=40)
        self.input_entry.pack(side=tk.LEFT, padx=5)

        self.send_button = ttk.Button(
            self.input_frame, text="Send", command=self.on_send
        )
        self.send_button.pack(side=tk.LEFT)

        self.voice_button = ttk.Button(
            self.input_frame, text="Voice Input", command=self.toggle_listening
        )
        self.voice_button.pack(side=tk.LEFT, padx=5)

    def listen(self):
        self.text_area.insert(tk.END, "Listening...\n")
        self.text_area.see(tk.END)
        try:
            audio_data = self.record_audio()
            self.text_area.insert(tk.END, "Processing audio...\n")
            self.text_area.see(tk.END)
            self.process_audio(audio_data)
        except Exception as e:
            self.text_area.insert(tk.END, f"Error capturing audio: {str(e)}\n")
            self.text_area.see(tk.END)
        finally:
            self.is_listening = False
            self.voice_button.config(text="Voice Input")

    def record_audio(self):
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        RECORD_SECONDS = 5

        p = pyaudio.PyAudio()

        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )

        frames = []

        for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()

        audio_data = b"".join(frames)

        # Save the audio as a WAV file
        with wave.open("output.wav", "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(audio_data)

        return "output.wav"

    def process_audio(self, audio_file):
        try:
            with sr.AudioFile(audio_file) as source:
                audio = self.recognizer.record(source)  # Read the entire WAV file

            # Recognize speech using Google Web Speech API
            text = self.recognizer.recognize_google(audio)

            self.text_area.insert(tk.END, f"You said: {text}\n")
            self.text_area.see(tk.END)
            self.process_input(text)
        except sr.UnknownValueError:
            self.text_area.insert(
                tk.END, "Sorry, I couldn't understand that. Please try again.\n"
            )
            self.text_area.see(tk.END)
        except sr.RequestError as e:
            self.text_area.insert(
                tk.END,
                f"Sorry, there was an error with the speech recognition service: {str(e)}\n",
            )
            self.text_area.see(tk.END)
        except Exception as e:
            self.text_area.insert(tk.END, f"Error processing audio: {str(e)}\n")
            self.text_area.see(tk.END)

    def process(self, text):
        if text:
            new_user_input_ids = self.tokenizer.encode(
                text + self.tokenizer.eos_token, return_tensors="pt"
            )

            bot_input_ids = (
                torch.cat([self.chat_history_ids, new_user_input_ids], dim=-1)
                if self.chat_history_ids is not None
                else new_user_input_ids
            )

            self.chat_history_ids = self.model.generate(
                bot_input_ids,
                max_length=1000,
                pad_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
                do_sample=True,
                top_k=100,
                top_p=0.7,
                temperature=0.8,
            )

            response = self.tokenizer.decode(
                self.chat_history_ids[:, bot_input_ids.shape[-1] :][0],
                skip_special_tokens=True,
            )
            return response
        return "I'm sorry, I couldn't process that."

    def update_ui_and_speak(self, text):
        self.text_area.insert(tk.END, f"AI: {text}\n")
        self.text_area.see(tk.END)
        threading.Thread(target=self.speak, args=(text,), daemon=True).start()

    def speak(self, text):
        duration = len(text) * 0.1  # Rough estimate of speaking duration
        threading.Thread(
            target=self.avatar.animate_mouth, args=(duration,), daemon=True
        ).start()
        self.engine.say(text)
        self.engine.runAndWait()

    def on_send(self):
        user_input = self.input_entry.get()
        if user_input:
            self.text_area.insert(tk.END, f"You: {user_input}\n")
            self.text_area.see(tk.END)
            self.input_entry.delete(0, tk.END)
            threading.Thread(
                target=self.process_input, args=(user_input,), daemon=True
            ).start()

    def process_input(self, user_input):
        response = self.process(user_input)
        self.root.after(0, self.update_ui_and_speak, response)

    def toggle_listening(self):
        if not self.is_listening:
            self.is_listening = True
            self.voice_button.config(text="Stop Listening")
            threading.Thread(target=self.listen, daemon=True).start()
        else:
            self.is_listening = False
            self.voice_button.config(text="Voice Input")


def main():
    root = tk.Tk()
    app = AIAssistant(root)
    root.mainloop()


if __name__ == "__main__":
    main()
