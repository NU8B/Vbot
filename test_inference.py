import soundfile as sf
import warnings
import time
from util.inference_styleTTS2 import StyleTTS2Inference

# Suppress warnings
warnings.filterwarnings("ignore")


def test_tts():
    # Initialize the TTS model
    print("Initializing TTS model...")
    init_start = time.time()
    tts = StyleTTS2Inference()
    init_time = time.time() - init_start
    print(f"Model initialization took {init_time:.2f}s")

    # might have buzz sound if text not formatted correctly
    text = "Hello, this is a test of the Style-TTS 2 system. It's currently 1945 2005 and 2015."

    # Compute style from reference
    print("\nComputing style from reference audio...")
    style_start = time.time()
    style = tts.compute_style("asset/ref.wav")
    style_time = time.time() - style_start
    print(f"Style computation took {style_time:.2f}s")

    # Generate speech
    print("\nGenerating speech...")
    inference_start = time.time()
    audio = tts.inference(
        text=text,
        ref_s=style,
        alpha=0.3,  # Style mixing parameter
        beta=0.7,  # Style mixing parameter
        diffusion_steps=20,  # Number of diffusion steps
        embedding_scale=1.0,  # Embedding scale for diffusion
    )
    inference_time = time.time() - inference_start
    print(f"Speech generation took {inference_time:.2f}s")

    # Save the generated audio
    print("\nSaving audio...")
    save_start = time.time()
    output_path = "output.wav"
    sf.write(output_path, audio, 24000)
    save_time = time.time() - save_start
    print(f"Audio saving took {save_time:.2f}s")

    # Print total time
    total_time = init_time + style_time + inference_time + save_time
    print(f"\nTotal processing time: {total_time:.2f}s")
    print(f"├─ Initialization: {init_time:.2f}s")
    print(f"├─ Style computation: {style_time:.2f}s")
    print(f"├─ Speech generation: {inference_time:.2f}s")
    print(f"└─ Audio saving: {save_time:.2f}s")


if __name__ == "__main__":
    test_tts()
