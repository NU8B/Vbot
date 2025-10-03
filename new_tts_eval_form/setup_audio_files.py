"""
Script to copy audio files from TEST_multiple_model_inference.py outputs
to the new_tts_eval_form static directory.

Run this from the Vbot project root:
    python new_tts_eval_form/setup_audio_files.py
"""

import shutil
from pathlib import Path

# Define paths
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "asset" / "outputs"
STATIC_AUDIO_DIR = Path(__file__).parent / "static" / "audio"

# Models to copy
MODELS = ["Amelia", "Eveland", "Gura", "Amelia_new"]

# Emotions
EMOTIONS = ["joy", "sadness", "anger", "surprise", "neutral"]


def main():
    print("Setting up audio files for TTS Evaluation Form")
    print("=" * 60)

    # Create static/audio directory structure
    for model in MODELS:
        model_dir = STATIC_AUDIO_DIR / model
        model_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {model_dir}")

    # Copy audio files
    total_copied = 0
    # 4 prompts per emotion (specific) + 1 generic per emotion = 5 files per emotion
    # 5 emotions × 5 files = 25 files per model
    # 4 models × 25 files = 100 files total
    total_expected = len(MODELS) * len(EMOTIONS) * 5

    for model in MODELS:
        source_dir = OUTPUTS_DIR / model
        dest_dir = STATIC_AUDIO_DIR / model

        if not source_dir.exists():
            print(f"⚠️  Warning: Source directory not found: {source_dir}")
            continue

        print(f"\nCopying files for {model}...")
        model_copied = 0

        for emotion in EMOTIONS:
            # Copy specific emotion files (4 prompts per emotion)
            for i in range(1, 5):
                source_file = source_dir / f"specific_{emotion}_{i}.wav"
                dest_file = dest_dir / f"specific_{emotion}_{i}.wav"

                if source_file.exists():
                    shutil.copy2(source_file, dest_file)
                    model_copied += 1
                    total_copied += 1
                    print(f"  ✓ Copied: specific_{emotion}_{i}.wav")
                else:
                    print(f"  ✗ Missing: specific_{emotion}_{i}.wav")

            # Copy generic emotion file (1 per emotion)
            source_file = source_dir / f"generic_{emotion}.wav"
            dest_file = dest_dir / f"generic_{emotion}.wav"

            if source_file.exists():
                shutil.copy2(source_file, dest_file)
                model_copied += 1
                total_copied += 1
                print(f"  ✓ Copied: generic_{emotion}.wav")
            else:
                print(f"  ✗ Missing: generic_{emotion}.wav")

        print(f"  Total for {model}: {model_copied}/25 files")

    print("\n" + "=" * 60)
    print(f"Setup complete!")
    print(f"Total files copied: {total_copied}/{total_expected}")

    if total_copied == total_expected:
        print("✓ All audio files successfully copied!")
        print("\nYou can now run the evaluation form:")
        print("  cd new_tts_eval_form")
        print("  python app.py")
    else:
        print(f"⚠️  Warning: Only {total_copied} of {total_expected} files were copied.")
        print(
            "Make sure you've run utils/TEST_multiple_model_inference.py to generate all audio files."
        )


if __name__ == "__main__":
    main()
