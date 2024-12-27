import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import soundfile as sf
import warnings
from utils.inference_styleTTS2 import StyleTTS2Inference
from utils.emotion_utils import (
    EmotionHandler,
    EMOTION_CONFIG,
    DIFFUSION_STEPS,
    ALPHA,
    BETA,
    EMBEDDING_SCALE,
)
import time
from pathlib import Path

# Suppress warnings
warnings.filterwarnings("ignore")

# Create outputs directory if it doesn't exist
output_dir = Path("asset/outputs")
output_dir.mkdir(parents=True, exist_ok=True)

# Initialize emotion handler and TTS
print("Initializing models...")
emotion_handler = EmotionHandler()
tts = StyleTTS2Inference()

# Load neutral style as default
print("\nLoading neutral style...")
default_style = tts.compute_style("asset/ref_sound/neutral.wav")

# Warm up inference with a short text
print("\nWarming up inference...")
warmup_text = "This is a warm-up inference."
warmup_start = time.time()
_ = tts.inference(
    text=warmup_text,
    ref_s=default_style,
    alpha=ALPHA,
    beta=BETA,
    diffusion_steps=DIFFUSION_STEPS,
    embedding_scale=EMBEDDING_SCALE,
)
print(f"Warm-up took {time.time() - warmup_start:.2f}s")

# Test prompts for different emotions
test_prompts = {
    # Happy emotions
    "joy": "I'm so incredibly happy and excited! This is the best day ever! I can't stop smiling!",
    "admiration": "You've done such an amazing job! Your work is absolutely brilliant and inspiring!",
    "amusement": "That's hilarious! I can't stop laughing at that joke, it's so funny!",
    # Sad emotions
    "sadness": "I feel so heartbroken and lost. Everything seems so dark and hopeless right now.",
    "disappointment": "I really thought things would work out differently. This isn't what I expected at all.",
    "grief": "I miss them so much. The pain of losing someone so dear is unbearable.",
    # Angry emotions
    "anger": "This is absolutely unacceptable! I'm furious about how this was handled!",
    "annoyance": "This keeps happening over and over. It's really getting on my nerves.",
    "disapproval": "I strongly disagree with this decision. This is not the right way to do things.",
    # Surprised emotions
    "surprise": "Wow! I absolutely did not see that coming! This is completely unexpected!",
    "realization": "Oh! Now I finally understand what's been happening all along!",
    # Neutral emotions
    "neutral": "The weather today is partly cloudy with a temperature of 20 degrees celsius.",
    "caring": "Let me help you with that. I want to make sure you're comfortable.",
    "curiosity": "I wonder how this works. Could you explain the process to me?",
}

# Test each emotion
print("\nTesting emotion detection and speech generation for different emotions:")
print("-" * 80)

for emotion_name, text in test_prompts.items():
    print(f"\nTesting prompt for {emotion_name}:")
    print(f"Text: {text}")

    # Detect emotion
    detected_emotion = emotion_handler.classify_emotion(text)
    print(f"Detected emotion: {detected_emotion}")

    # Generate speech using parameters from EMOTION_CONFIG
    start = time.time()
    print("Generating speech...")

    audio = tts.inference(
        text=text,
        ref_s=tts.compute_style(
            f"asset/ref_sound/{EMOTION_CONFIG[detected_emotion]['file']}"
        ),
        alpha=EMOTION_CONFIG[detected_emotion]["alpha"],
        beta=EMOTION_CONFIG[detected_emotion]["beta"],
        diffusion_steps=DIFFUSION_STEPS,
        embedding_scale=EMOTION_CONFIG[detected_emotion]["embedding_scale"],
    )

    duration = time.time() - start
    print(f"Time taken: {duration:.2f}s")

    # Save audio in the outputs directory
    output_path = output_dir / f"{emotion_name}.wav"
    sf.write(str(output_path), audio, 24000)
    print(f"Saved to: {output_path}")
    print("-" * 80)

# New tests with the same text for different emotions
additional_text = "This is a test sentence to evaluate different emotions."
additional_emotions = ["joy", "sadness", "anger", "surprise", "caring"]

print("\nTesting additional emotions with the same text:")
print("-" * 80)

for emotion_name in additional_emotions:
    print(f"\nTesting prompt for {emotion_name}:")
    print(f"Text: {additional_text}")

    # Use predefined emotion without detection
    print(f"Using predefined emotion: {emotion_name}")

    # Generate speech using parameters from EMOTION_CONFIG
    start = time.time()
    print("Generating speech...")

    audio = tts.inference(
        text=additional_text,
        ref_s=tts.compute_style(
            f"asset/ref_sound/{EMOTION_CONFIG[emotion_name]['file']}"
        ),
        alpha=EMOTION_CONFIG[emotion_name]["alpha"],
        beta=EMOTION_CONFIG[emotion_name]["beta"],
        diffusion_steps=DIFFUSION_STEPS,
        embedding_scale=EMOTION_CONFIG[emotion_name]["embedding_scale"],
    )

    duration = time.time() - start
    print(f"Time taken: {duration:.2f}s")

    # Save audio in the outputs directory
    output_path = output_dir / f"{emotion_name}_additional.wav"
    sf.write(str(output_path), audio, 24000)
    print(f"Saved to: {output_path}")
    print("-" * 80)
