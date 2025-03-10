import os
import sys
import shutil

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

# List of models to test
MODELS_TO_TEST = ["Amelia", "Eveland"]

# Base output directory
BASE_OUTPUT_DIR = Path("asset/outputs")


def clear_model_style_cache(model_name):
    """Clear style cache for a specific model"""
    cache_dir = Path("cache/style") / model_name
    if cache_dir.exists():
        print(f"Clearing style cache for {model_name}...")
        shutil.rmtree(cache_dir)
        print(f"Style cache cleared for {model_name}.")
    else:
        print(f"No style cache found for {model_name}.")


def ensure_output_directory(path):
    """Ensure output directory exists and is writable"""
    path.mkdir(parents=True, exist_ok=True)
    # Test if directory is writable
    test_file = path / ".test"
    try:
        test_file.touch()
        test_file.unlink()
    except Exception as e:
        raise RuntimeError(f"Cannot write to output directory {path}: {str(e)}")


# Test prompts optimized for each emotion
EMOTION_SPECIFIC_PROMPTS = {
    "joy": "I just got promoted at work! This is the happiest day of my life! I won't let anybody ruin it!",
    "sadness": "I had to say goodbye to my best friend today, I'm so sad.",
    "anger": "They completely destroyed the project I spent months working on!",
    "surprise": "I can't believe I just won the lottery! This is unbelievable!",
    "neutral": "The meeting is scheduled for three o'clock in the conference room.",
}

# Core emotions matching our reference sound files
CORE_EMOTIONS = ["joy", "sadness", "anger", "surprise", "neutral"]

# Generic text to test across all models with core emotions
GENERIC_TEST_TEXT = "This is a test sentence to compare different models and emotions."


def test_model(model_name):
    print(f"\nTesting model: {model_name}")
    print("=" * 80)

    # Create model-specific output directory
    output_dir = BASE_OUTPUT_DIR / model_name
    ensure_output_directory(output_dir)
    print(f"Output directory: {output_dir}")

    # Clear model-specific cache before testing
    clear_model_style_cache(model_name)

    # Initialize models with model name
    print("Initializing models...")
    emotion_handler = EmotionHandler(model_name=model_name)
    tts = StyleTTS2Inference(model_name=model_name)

    # Load neutral style as default
    print("\nLoading neutral style...")
    neutral_path = f"asset/ref_sound/{EMOTION_CONFIG['neutral']['file'][model_name]}"
    default_style = tts.compute_style(neutral_path)

    # Warm up inference
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

    # Test 1: Emotion-specific prompts
    print("\nTesting emotion-specific prompts:")
    print("-" * 80)

    for emotion, text in EMOTION_SPECIFIC_PROMPTS.items():
        print(f"\nTesting {emotion} with optimized prompt:")
        print(f"Text: {text}")

        # Generate speech using parameters from EMOTION_CONFIG
        start = time.time()
        print("Generating speech...")

        ref_path = f"asset/ref_sound/{EMOTION_CONFIG[emotion]['file'][model_name]}"
        # Compute style for each inference to avoid cross-model contamination
        ref_style = tts.compute_style(ref_path)
        audio = tts.inference(
            text=text,
            ref_s=ref_style,
            alpha=EMOTION_CONFIG[emotion]["alpha"],
            beta=EMOTION_CONFIG[emotion]["beta"],
            diffusion_steps=DIFFUSION_STEPS,
            embedding_scale=EMOTION_CONFIG[emotion]["embedding_scale"],
            speed=EMOTION_CONFIG[emotion]["speed"],
        )

        duration = time.time() - start
        print(f"Time taken: {duration:.2f}s")

        # Save audio
        output_path = output_dir / f"specific_{emotion}.wav"
        sf.write(str(output_path), audio, 24000)
        if output_path.exists():
            print(f"Successfully saved to: {output_path}")
        else:
            print(f"Failed to save to: {output_path}")

    # Test 2: Same text with core emotions only
    print("\nTesting generic text with core emotions:")
    print("-" * 80)

    for emotion in CORE_EMOTIONS:
        print(f"\nTesting {emotion} with generic text:")
        print(f"Text: {GENERIC_TEST_TEXT}")

        start = time.time()
        print("Generating speech...")

        ref_path = f"asset/ref_sound/{EMOTION_CONFIG[emotion]['file'][model_name]}"
        # Compute style for each inference to avoid cross-model contamination
        ref_style = tts.compute_style(ref_path)
        audio = tts.inference(
            text=GENERIC_TEST_TEXT,
            ref_s=ref_style,
            alpha=EMOTION_CONFIG[emotion]["alpha"],
            beta=EMOTION_CONFIG[emotion]["beta"],
            diffusion_steps=DIFFUSION_STEPS,
            embedding_scale=EMOTION_CONFIG[emotion]["embedding_scale"],
            speed=EMOTION_CONFIG[emotion]["speed"],
        )

        duration = time.time() - start
        print(f"Time taken: {duration:.2f}s")

        # Save audio
        output_path = output_dir / f"generic_{emotion}.wav"
        sf.write(str(output_path), audio, 24000)
        if output_path.exists():
            print(f"Successfully saved to: {output_path}")
        else:
            print(f"Failed to save to: {output_path}")

    # Clear the model's cache after testing to free memory
    clear_model_style_cache(model_name)


def main():
    print("Starting multiple model inference test")
    print("=" * 80)

    # Ensure base output directory exists
    ensure_output_directory(BASE_OUTPUT_DIR)
    print(f"Base output directory: {BASE_OUTPUT_DIR}")

    for model_name in MODELS_TO_TEST:
        try:
            test_model(model_name)
        except Exception as e:
            print(f"Error testing model {model_name}: {str(e)}")
            continue

    print("\nTesting completed!")
    print(f"All outputs have been saved to: {BASE_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
