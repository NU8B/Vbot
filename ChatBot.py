import os
import sys
import tkinter as tk
import warnings

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from utils.gui import ChatGUI
from utils.initialization_utils import InitializationHandler

# Suppress all warnings
warnings.filterwarnings("ignore")

# Model selection - can be changed to "Eveland" or "Amelia"
MODEL_NAME = "Amelia"  # Default model

# Initialize all components with selected model
init_handler = InitializationHandler(model_name=MODEL_NAME)
components = init_handler.initialize_all()

# Initialize tkinter root
root = tk.Tk()

# Create Ollama Handler with all components
ollama_handler = init_handler.create_ollama_handler()

# Initialize GUI with callbacks
gui = ChatGUI(
    root,
    ollama_handler.handle_text_input,
    lambda: components["audio_processor"].toggle_listening(
        gui,
        ollama_handler._process_text,
        ollama_handler.is_processing,
    ),
)

# Set the GUI in Ollama Handler after creation
ollama_handler.gui = gui

# Register cleanup on window close
root.protocol(
    "WM_DELETE_WINDOW",
    lambda: (components["docker_handler"].cleanup(), root.destroy()),
)

# Start the application
root.mainloop()
