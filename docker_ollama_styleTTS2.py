import os
import tkinter as tk
import warnings
from pathlib import Path
import nltk
import nltk.data
import time

from util.audio_utils import AudioProcessor
from util.gui import ChatGUI
from util.inference_styleTTS2 import StyleTTS2Inference
from util.docker_utils import DockerHandler
from util.ollama_utils import OllamaHandler
from util.timing_utils import ParallelInitializer

"""Initialize NLTK
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
nltk.download("punkt", quiet=True)
nltk_data_dir = nltk.data.path[0]  # Get the first NLTK data directory

# Verify the file exists
punkt_file = Path(nltk_data_dir) / "tokenizers" / "punkt" / "english.pickle"
if punkt_file.exists():
    print(f"Found punkt file at: {punkt_file}")
else:
    print(f"Punkt file not found at expected location: {punkt_file}")

try:
    tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
    print("Punkt is properly installed")
except LookupError as e:
    print(f"Punkt is not installed correctly: {str(e)}")
    print(f"Current NLTK data paths: {nltk.data.path}")
"""
# Suppress all warnings
warnings.filterwarnings("ignore")
os.makedirs("./cache", exist_ok=True)

init_start = time.time()

# Initialize Docker client first as it's needed for Ollama
docker_handler = DockerHandler()

# Group 1: Ollama warmup and StyleTTS2 init
group1_start = time.time()
parallel_init_1 = ParallelInitializer()
parallel_init_1.add_task("ollama", OllamaHandler.initialize)
parallel_init_1.add_task("tts", StyleTTS2Inference)
results_1 = parallel_init_1.run()
group1_time = time.time() - group1_start

# Get results from first group
warmup_time = parallel_init_1.get_result("ollama")
tts_model = parallel_init_1.get_result("tts")

# Group 2: Whisper and reference style (parallel)
group2_start = time.time()
parallel_init_2 = ParallelInitializer()
parallel_init_2.add_task("whisper", AudioProcessor)
parallel_init_2.add_task("ref_style", lambda: tts_model.compute_style("asset/ref.wav"))
results_2 = parallel_init_2.run()
group2_time = time.time() - group2_start

# Get results from second group
audio_processor = parallel_init_2.get_result("whisper")
ref_style = parallel_init_2.get_result("ref_style")

# Initialize tkinter root
root = tk.Tk()

# Create Ollama Handler first (without GUI)
ollama_handler = OllamaHandler(None, tts_model, audio_processor, ref_style)
ollama_handler.warmup_time = warmup_time  # Store the warmup time

# Initialize GUI with callbacks
gui = ChatGUI(
    root,
    ollama_handler.handle_text_input,
    lambda: audio_processor.toggle_listening(
        gui,
        ollama_handler._process_text,
        ollama_handler.is_processing,
    ),
)

# Set the GUI in Ollama Handler after creation
ollama_handler.gui = gui

# Calculate total initialization time
total_init_time = time.time() - init_start
print(f"\nTotal initialization time: {total_init_time:.2f}s")
print(f"├─ Docker setup: {docker_handler.setup_time:.2f}s")
print(f"├─ Group 1: {group1_time:.2f}s")
print(f"│  ├─ Ollama warm-up")
print(f"│  └─ StyleTTS2")
print(f"└─ Group 2: {group2_time:.2f}s")
print(f"   ├─ Whisper")
print(f"   └─ Reference style")

# Register cleanup on window close
root.protocol(
    "WM_DELETE_WINDOW",
    lambda: (docker_handler.cleanup(), root.destroy()),
)

# Start the application
root.mainloop()
