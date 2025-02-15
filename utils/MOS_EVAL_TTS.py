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
    "nonoJDWAOIDAWKDA/new_ft_StyleTTS2",
]

# Base output directory
BASE_OUTPUT_DIR = Path("mos_evaluation/static/audio")


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


# Test prompts optimized for each emotion (4 per emotion)
EMOTION_SPECIFIC_PROMPTS = {
    "joy": [
        "I just got promoted at work! This is the happiest day of my life!",
        "We just won the championship! All our hard work paid off!",
        "My best friend surprised me with tickets to my favorite band!",
        "I'm getting married to the love of my life next month!",
    ],
    "sadness": [
        "I had to say goodbye to my best friend today, I'm going to miss him.",
        "I miss the past so much, I can't stop crying.",
        "I feel like things will never get better.",
        "I can't believe I lost my job, I'm so sad.",
    ],
    "anger": [
        "They completely destroyed the project I spent months working on!",
        "I can't believe they lied to my face about everything!",
        "This is absolutely unacceptable! I demand to speak to the manager!",
        "How dare they spread rumors about me behind my back!",
    ],
    "surprise": [
        "I can't believe I just won the lottery! This is unbelievable!",
        "Wait, what? You're telling me we're going to Paris tomorrow?",
        "No way! They actually accepted my proposal on the first try!",
        "I never expected to find you here of all places!",
    ],
    "neutral": [
        "The meeting is scheduled for three o'clock in the conference room.",
        "Please remember to submit your reports by Friday.",
        "The weather forecast predicts rain for the weekend.",
        "The train will arrive at platform six in ten minutes.",
    ],
}

# Generic test prompts
GENERIC_TEST_PROMPTS = [
    "This is a test sentence to compare different models and emotions.",
    "The quick brown fox jumps over the lazy dog.",
]

# Dynamic test prompts (for emotion classification)
DYNAMIC_TEST_PROMPTS = [
    "I can't wait to tell you about my amazing vacation!",
    "Why won't this computer just work properly for once?",
    "I think we should consider all options carefully before deciding.",
    "Did you hear what happened at the party last night?",
    "Sometimes I sit and think about the meaning of life.",
    "Can you believe how beautiful the sunset is today?",
    "I've been working on this project for weeks now.",
    "Let's meet up for coffee and catch up soon.",
    "The traffic today is absolutely terrible.",
    "I just finished reading the most incredible book.",
    "I'm so excited to see you tomorrow!",
    "I'm so lost and alone in this big city.",
    "I don't know what to do with my life.",
    "Did you know about that?" "I don't care what they said about me at all.",
]

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

    for emotion, prompts in EMOTION_SPECIFIC_PROMPTS.items():
        for idx, text in enumerate(prompts, 1):
            print(f"\nTesting {emotion} with optimized prompt {idx}:")
            print(f"Text: {text}")

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
            output_path = output_dir / f"specific_{emotion}_{idx}.wav"
            sf.write(str(output_path), audio, 24000)
            if output_path.exists():
                print(f"Successfully saved to: {output_path}")
            else:
                print(f"Failed to save to: {output_path}")

    # Test 2: Generic text with core emotions
    print("\nTesting generic text with core emotions:")
    print("-" * 80)

    for emotion in CORE_EMOTIONS:
        for idx, text in enumerate(GENERIC_TEST_PROMPTS, 1):
            print(f"\nTesting {emotion} with generic text {idx}:")
            print(f"Text: {text}")

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

            output_path = output_dir / f"generic_{emotion}_{idx}.wav"
            sf.write(str(output_path), audio, 24000)
            if output_path.exists():
                print(f"Successfully saved to: {output_path}")
            else:
                print(f"Failed to save to: {output_path}")

    # Test 3: Dynamic emotion classification
    print("\nTesting dynamic emotion classification:")
    print("-" * 80)

    emotion_handler = EmotionHandler()

    for idx, text in enumerate(DYNAMIC_TEST_PROMPTS, 1):
        print(f"\nTesting dynamic prompt {idx}:")
        print(f"Text: {text}")

        # Classify emotion
        detected_emotion = emotion_handler.classify_emotion(text)
        print(f"Detected emotion: {detected_emotion}")

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

        output_path = output_dir / f"dynamic_{idx}_{detected_emotion}.wav"
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
