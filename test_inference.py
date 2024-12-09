import soundfile as sf
import warnings
from util.inference_styleTTS2 import StyleTTS2Inference
import time

# Suppress warnings
warnings.filterwarnings("ignore")


text = "Hello, this is a test of the Style-TTS 2 system. It's currently 1945 2005 and 2015."
tts = StyleTTS2Inference()

style = tts.compute_style("asset/ref.wav")

start = time.time()
print("\nGenerating speech...")
audio = tts.inference(
    text=text,
    ref_s=style,
    alpha=0.3,  # Style mixing parameter
    beta=0.7,  # Style mixing parameter
    diffusion_steps=20,  # Number of diffusion steps
    embedding_scale=1.0,  # Embedding scale for diffusion
)
print(f"Time taken: {time.time() - start:.2f}s")
output_path = "output.wav"
sf.write(output_path, audio, 24000)
