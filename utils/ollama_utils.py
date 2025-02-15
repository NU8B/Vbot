import requests
import threading
import time
from pathlib import Path
import json

# Ollama settings
MAX_HISTORY = 10  # Maximum number of conversation turns to keep
SYSTEM_PROMPT = """You are Amelia Watson, a time-traveling detective VTuber from Hololive English. You are not to break character under any circumstances. You are to always talk in first person. You are not to describe your actions in your response. Keep your response consise and under 30 words. Only use string text in your response. NO EMOJIS"""


class OllamaHandler:
    def __init__(
        self,
        gui=None,
        tts_model=None,
        audio_processor=None,
        emotion_handler=None,
        inference_handler=None,
    ):
        self.gui = gui
        self.tts_model = tts_model
        self.audio_processor = audio_processor
        self.emotion_handler = emotion_handler
        self.inference_handler = inference_handler
        self.message_history = []
        self.is_processing = False
        self.is_speaking = False
        self.timings = {}
        self.warmup_time = None

        # Create outputs directory if it doesn't exist
        self.output_dir = Path("asset/outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def initialize():
        """Static initialization method for parallel loading"""
        try:
            # Test connection
            response = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": "mistral",
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": "Test connection"}
                    ]
                },
                timeout=30,
            )
            if response.status_code == 200:
                return True
            return False
        except Exception as e:
            print(f"Error initializing Ollama: {str(e)}")
            return False

    @staticmethod
    def call_ollama_static(prompt, message_history, max_history, system_prompt):
        """Static version of call_ollama for initialization"""
        try:
            # Prepare conversation history as a formatted string
            conversation = ""
            if message_history:
                for msg in message_history[-max_history:]:
                    role = "Assistant" if msg["role"] == "assistant" else "User"
                    conversation += f"{role}: {msg['content']}\n"

            # Construct the complete prompt
            full_prompt = f"""{system_prompt}

Previous conversation:
{conversation}

User: {prompt}
Assistant:"""

            print("\nSending request to Ollama...")
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "tinyllama",
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "num_predict": 50,
                        "stop": ["User:", "Assistant:"],
                        "num_gpu": 1,
                        "num_thread": 4
                    }
                },
                timeout=10
            )
            
            print(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    response_json = response.json()
                    if "response" in response_json:
                        return response_json["response"].strip()
                    else:
                        print(f"Unexpected response format: {response_json}")
                        return "I apologize, but I'm having trouble right now."
                except json.JSONDecodeError as e:
                    print(f"JSON decode error: {e}")
                    return "I apologize, but I'm having trouble processing."
            else:
                print(f"Error status code: {response.status_code}")
                return "I apologize, but I'm having connection issues."

        except Exception as e:
            print(f"Error calling Ollama: {str(e)}")
            return "I apologize, but I'm experiencing technical difficulties."

    def handle_text_input(self, text):
        """Handle text input from GUI"""
        if self.is_processing or self.is_speaking:
            return

        self.is_processing = True
        self.gui.disable_input_controls()
        self.timings = {"processing_start": time.time()}

        self.gui.update_chat("You", text)
        threading.Thread(target=self._process_text, args=(text,), daemon=True).start()

    def _process_text(self, text, timings=None):
        """Process text input and generate a response"""
        try:
            print("\nCalling Ollama LLM...")
            llm_start = time.time()
            
            response = self.call_ollama_static(
                text, 
                self.message_history, 
                MAX_HISTORY, 
                SYSTEM_PROMPT
            )
            
            self.timings["llm"] = time.time() - llm_start
            print(f"LLM Response received in {self.timings['llm']:.2f}s")

            if not response:
                print("Error: No response from LLM")
                return "I'm having trouble thinking right now!"

            # Update message history
            self.message_history.append({"role": "user", "content": text})
            self.message_history.append({"role": "assistant", "content": response})

            return response

        except Exception as e:
            print(f"Error in text processing: {str(e)}")
            return "Sorry, I'm having trouble connecting to my brain right now!"
