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

# List of models to test (repo IDs)
MODELS_TO_TEST = [
    "nonoJDWAOIDAWKDA/Amelia10_ft_StyleTTS2",
    # "nonoJDWAOIDAWKDA/new_ft_StyleTTS2",
]

# Base output directory
BASE_OUTPUT_DIR = Path("asset/outputs")


def clear_style_cache():
    """Clear all style cache directories"""
    cache_dir = Path("cache/style")
    if cache_dir.exists():
        print("Clearing style cache...")
        shutil.rmtree(cache_dir)
        print("Style cache cleared.")
    else:
        print("No style cache found.")


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


def test_model(repo_id):
    print(f"\nTesting model: {repo_id}")
    print("=" * 80)

    # Create model-specific output directory
    model_name = repo_id.split("/")[-1]
    output_dir = BASE_OUTPUT_DIR / model_name
    ensure_output_directory(output_dir)
    print(f"Output directory: {output_dir}")

    # Initialize models
    print("Initializing models...")
    emotion_handler = EmotionHandler()
    tts = StyleTTS2Inference(repo_id=repo_id)

    # Load neutral style as default
    print("\nLoading neutral style...")
    default_style = tts.compute_style("asset/ref_sound/neutral.wav")

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

        audio = tts.inference(
            text=text,
            ref_s=tts.compute_style(
                f"asset/ref_sound/{EMOTION_CONFIG[emotion]['file']}"
            ),
            alpha=EMOTION_CONFIG[emotion]["alpha"],
            beta=EMOTION_CONFIG[emotion]["beta"],
            diffusion_steps=DIFFUSION_STEPS,
            embedding_scale=EMOTION_CONFIG[emotion]["embedding_scale"],
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

        audio = tts.inference(
            text=GENERIC_TEST_TEXT,
            ref_s=tts.compute_style(
                f"asset/ref_sound/{EMOTION_CONFIG[emotion]['file']}"
            ),
            alpha=EMOTION_CONFIG[emotion]["alpha"],
            beta=EMOTION_CONFIG[emotion]["beta"],
            diffusion_steps=DIFFUSION_STEPS,
            embedding_scale=EMOTION_CONFIG[emotion]["embedding_scale"],
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

    # Test 3: Custom text with neutral voice
    print("\nTesting custom text with custom reference voice:")
    print("-" * 80)

    custom_text = "I really like our new original song tho. Every single day I wake up and it's stuck in my head."
    print(f"Text: {custom_text}")

    start = time.time()
    print("Generating speech...")

    audio = tts.inference(
        text=custom_text,
        ref_s=tts.compute_style("asset/ref_sound/custom.wav"),
        alpha=0.3,
        beta=0.7,
        diffusion_steps=DIFFUSION_STEPS,
        embedding_scale=1,
    )

    duration = time.time() - start
    print(f"Time taken: {duration:.2f}s")

    # Save audio
    output_path = output_dir / "custom_reference.wav"
    sf.write(str(output_path), audio, 24000)
    if output_path.exists():
        print(f"Successfully saved to: {output_path}")
    else:
        print(f"Failed to save to: {output_path}")


def main():
    print("Starting multiple model inference test")
    print("=" * 80)

    # Clear all style cache before starting tests
    clear_style_cache()

    # Ensure base output directory exists
    ensure_output_directory(BASE_OUTPUT_DIR)
    print(f"Base output directory: {BASE_OUTPUT_DIR}")

    for repo_id in MODELS_TO_TEST:
        try:
            test_model(repo_id)
        except Exception as e:
            print(f"Error testing model {repo_id}: {str(e)}")
            continue

    print("\nTesting completed!")
    print(f"All outputs have been saved to: {BASE_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
