import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import soundfile as sf
import warnings
from util.inference_styleTTS2 import StyleTTS2Inference
import time

# Suppress warnings
warnings.filterwarnings("ignore")


text = "Hello, this is a test of the Style-TTS 2 system. (Should not read this.) Pass 1. [Should not read this.] Pass 2. {Should not read this.} Pass 3. *Should not read this.* Pass 4. 5984. This is just a longer test so I can see if there's any change! I hope so because I'v been trying to make it faster all this time!"
tts = StyleTTS2Inference()

style = tts.compute_style("asset/ref.wav")

start = time.time()
print("\nGenerating speech...")
audio = tts.inference(
    text=text,
    ref_s=style,
    alpha=0.3,  # Style mixing parameter
    beta=0.7,  # Style mixing parameter
    diffusion_steps=5,  # Number of diffusion steps
    embedding_scale=1.0,  # Embedding scale for diffusion
)
print(f"Time taken: {time.time() - start:.2f}s")
output_path = "asset/output.wav"
print("Saved to:", output_path)
sf.write(output_path, audio, 24000)
