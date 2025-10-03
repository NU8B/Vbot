"""
Quick diagnostic script to check if audio files are set up correctly
"""

from pathlib import Path

# Define paths
STATIC_AUDIO_DIR = Path(__file__).parent / "static" / "audio"
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "asset" / "outputs"

MODELS = ["Amelia", "Eveland", "Gura", "Amelia_new"]
EMOTIONS = ["joy", "sadness", "anger", "surprise", "neutral"]


def check_source_files():
    """Check if source files exist in asset/outputs"""
    print("=" * 60)
    print("CHECKING SOURCE FILES (asset/outputs/)")
    print("=" * 60)

    if not OUTPUTS_DIR.exists():
        print(f"❌ Outputs directory not found: {OUTPUTS_DIR}")
        return False

    print(f"✓ Outputs directory exists: {OUTPUTS_DIR}\n")

    total_found = 0
    for model in MODELS:
        model_dir = OUTPUTS_DIR / model
        if not model_dir.exists():
            print(f"❌ {model}: Directory not found")
            continue

        files = list(model_dir.glob("*.wav"))
        print(f"✓ {model}: {len(files)} files found")
        total_found += len(files)

    print(f"\nTotal source files: {total_found}/100")
    return total_found > 0


def check_static_files():
    """Check if files are copied to static/audio"""
    print("\n" + "=" * 60)
    print("CHECKING STATIC FILES (new_tts_eval_form/static/audio/)")
    print("=" * 60)

    if not STATIC_AUDIO_DIR.exists():
        print(f"❌ Static audio directory not found: {STATIC_AUDIO_DIR}")
        print("   Run: python new_tts_eval_form/setup_audio_files.py")
        return False

    print(f"✓ Static audio directory exists: {STATIC_AUDIO_DIR}\n")

    total_found = 0
    for model in MODELS:
        model_dir = STATIC_AUDIO_DIR / model
        if not model_dir.exists():
            print(f"❌ {model}: Directory not found")
            continue

        files = list(model_dir.glob("*.wav"))
        print(f"✓ {model}: {len(files)} files")

        # Show breakdown
        specific_files = list(model_dir.glob("specific_*.wav"))
        generic_files = list(model_dir.glob("generic_*.wav"))
        print(f"   - Specific: {len(specific_files)}, Generic: {len(generic_files)}")

        total_found += len(files)

    print(f"\nTotal static files: {total_found}/100")
    return total_found == 100


def check_expected_files():
    """List expected files for one model as example"""
    print("\n" + "=" * 60)
    print("EXPECTED FILES (example for one model)")
    print("=" * 60)

    print("\nEach model should have these 25 files:")
    for emotion in EMOTIONS:
        print(f"\n{emotion.capitalize()}:")
        for i in range(1, 5):
            print(f"  - specific_{emotion}_{i}.wav")
        print(f"  - generic_{emotion}.wav")


def main():
    print("\n" + "🔍 TTS EVALUATION FORM - SETUP DIAGNOSTIC" + "\n")

    has_source = check_source_files()
    has_static = check_static_files()
    check_expected_files()

    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)

    if not has_source:
        print("\n❌ Source files not found!")
        print("   → Run: python utils/TEST_multiple_model_inference.py")
        print("   This will generate all 100 audio files in asset/outputs/")
    elif not has_static:
        print("\n❌ Static files not copied!")
        print("   → Run: python new_tts_eval_form/setup_audio_files.py")
        print("   This will copy files from asset/outputs/ to static/audio/")
    else:
        print("\n✅ All files are set up correctly!")
        print("   → Run: cd new_tts_eval_form && python app.py")
        print("   → Open: http://localhost:5000")

    print()


if __name__ == "__main__":
    main()
