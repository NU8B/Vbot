# import nltk
# nltk.download()
import os
import sys
import shutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import soundfile as sf
import warnings
from utils.inference_styleTTS2 import StyleTTS2Inference
from utils.emotion_utils import (
    EmotionHandler,
    DIFFUSION_STEPS,
    get_model_params,
    create_emotion_config,
)
import time
from pathlib import Path

# Suppress warnings
warnings.filterwarnings("ignore")

# List of models to test
MODELS_TO_TEST = ["Amelia"]

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


# Test prompts optimized for each emotion - 4 prompts per emotion per character
EMOTION_SPECIFIC_PROMPTS = {
    "Gura": {
        "joy": [
            "I just got promoted at work! This is the happiest day of my life! I won't let anybody ruin it!",
            "I love spending time with my family! These moments bring me so much joy and warmth!",
            "Today has been absolutely wonderful! Everything is going perfectly and I feel so happy!",
            "This delicious meal makes me so happy! I could eat this every single day!",
        ],
        "sadness": [
            "I had to say goodbye to my best friend today, I'm so sad.",
            "I lost something very precious to me. I don't know if I'll ever feel the same again.",
            "The news broke my heart. I just want to cry and be alone for a while.",
            "I feel so lonely and empty inside. Nothing seems to matter anymore.",
        ],
        "anger": [
            "They completely destroyed the project I spent months working on!",
            "This is unacceptable! I've had enough of being treated this way!",
            "How dare they betray my trust like that! I'm absolutely furious!",
            "I'm sick and tired of these constant lies and excuses!",
        ],
        "surprise": [
            "I can't believe I just won the lottery! This is unbelievable!",
            "Wait, what? You're telling me this actually happened? I'm completely shocked!",
            "No way! I never expected this in a million years! This is incredible!",
            "What?! Are you serious right now? This is absolutely mind-blowing!",
        ],
        "neutral": [
            "The meeting is scheduled for three o'clock in the conference room.",
            "According to the report, the results are consistent with our previous findings.",
            "Please ensure that all documents are submitted by the end of the week.",
            "The system will undergo maintenance between two and four in the afternoon.",
        ],
    },
    "Wilson": {
        "joy": [
            "The research is progressing beautifully! I'm thrilled with how smoothly everything is going!",
            "This is exactly what I've been hoping for! My hard work finally paid off!",
            "I feel so grateful and blessed! Life is treating me wonderfully right now!",
            "What a beautiful day! The sunshine and fresh air make me feel alive!",
        ],
        "sadness": [
            "My heart feels heavy with sorrow. I don't know how to move forward.",
            "The disappointment is crushing me. All my efforts were for nothing.",
            "I can't stop thinking about what went wrong. The regret is unbearable.",
            "They're gone now, and I'll never get another chance. I'm devastated.",
        ],
        "anger": [
            "This is ridiculous! I refuse to tolerate this incompetence any longer!",
            "You broke your promise again! I'm done giving you second chances!",
            "The injustice of this situation makes my blood boil! This isn't fair!",
            "I warned you this would happen! Now look at the mess we're in!",
        ],
        "surprise": [
            "Oh my goodness! I didn't see that coming at all! What a twist!",
            "That's impossible! How could this even happen? I'm in complete disbelief!",
            "You're kidding me, right? This is too extraordinary to be true!",
            "Wow! That caught me completely off guard! I never imagined this!",
        ],
        "neutral": [
            "The data shows a steady increase over the past three quarters.",
            "We need to finalize the budget allocation before next Tuesday.",
            "The instructions are clearly outlined in the user manual on page twelve.",
            "Please submit your feedback using the online form provided.",
        ],
    },
    "Amelia": {
        "joy": [
            "The investigation is going great! I'm having so much fun solving this mystery!",
            "I love this adventure! Every moment brings me pure joy and excitement!",
            "This is wonderful! I'm genuinely happy with how things are turning out!",
            "Being with my friends makes me so cheerful! I cherish these happy times!",
        ],
        "sadness": [
            "I miss the old days so much. Everything feels different now and it hurts.",
            "My confidence is shattered. I thought I could do better than this.",
            "I let everyone down. The weight of this failure is crushing my spirit.",
            "Sometimes I feel so alone in this. Nobody understands what I'm going through.",
        ],
        "anger": [
            "Are you serious right now?! I can't believe you'd do something so thoughtless!",
            "That's it! I've been patient long enough but this is too much!",
            "You're really testing my limits here! This behavior is completely unacceptable!",
            "I'm fed up with this nonsense! Someone needs to take responsibility!",
        ],
        "surprise": [
            "Wait, hold on! Did that really just happen?! I'm completely stunned!",
            "What in the world?! This is beyond anything I could have imagined!",
            "Are you telling me the truth?! I'm in total shock right now!",
            "Holy smokes! I never would have predicted this in a hundred years!",
        ],
        "neutral": [
            "The evidence suggests we should proceed with the standard protocol.",
            "I'll need to review the case files before making any conclusions.",
            "The schedule indicates our next appointment is at four o'clock.",
            "Please make sure to document everything according to procedure.",
        ],
    },
    "Eveland": {
        "joy": [
            "This novel is absolutely delightful! Reading it brings me such happiness!",
            "I'm truly content with how everything has unfolded. Life feels fulfilling!",
            "The performance was magnificent! I'm so pleased I attended this evening!",
            "What a lovely conversation! I feel genuinely joyful sharing these moments!",
        ],
        "sadness": [
            "The melancholy weighs on me heavily. I cannot shake this somber feeling.",
            "I'm drowning in disappointment. My expectations have crumbled to dust.",
            "The separation pains me deeply. I fear this emptiness will never fade.",
            "Everything feels meaningless now. The sadness has consumed my enthusiasm.",
        ],
        "anger": [
            "This is absolutely preposterous! I will not stand for such disrespect!",
            "Your negligence is infuriating! Have you no sense of responsibility?!",
            "I am outraged by this behavior! This crosses every line imaginable!",
            "How utterly disgraceful! This incompetence is beyond frustrating!",
        ],
        "surprise": [
            "Good heavens! I never anticipated such an extraordinary turn of events!",
            "What?! This revelation is absolutely staggering! I'm flabbergasted!",
            "Impossible! How could this possibly be?! I'm utterly dumbfounded!",
            "My word! This is the most astonishing thing I've ever witnessed!",
        ],
        "neutral": [
            "The manuscript requires careful editing before publication.",
            "I shall attend the seminar on Thursday at half past two.",
            "The library closes at eight in the evening on weekdays.",
            "Please reference chapter seven for additional information.",
        ],
    },
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

    # Get model-specific parameters and config
    params = get_model_params(model_name)
    emotion_config = create_emotion_config(model_name)

    # Load neutral style as default
    print("\nLoading neutral style...")
    neutral_path = f"asset/ref_sound/{emotion_config['neutral']['file'][model_name]}"
    default_style = tts.compute_style(neutral_path)

    # Warm up inference
    print("\nWarming up inference...")
    warmup_text = "This is a warm-up inference."
    warmup_start = time.time()
    _ = tts.inference(
        text=warmup_text,
        ref_s=default_style,
        alpha=params["ALPHA"],
        beta=params["BETA"],
        diffusion_steps=DIFFUSION_STEPS,
        embedding_scale=params["EMBEDDING_SCALE"],
    )
    print(f"Warm-up took {time.time() - warmup_start:.2f}s")

    # Test 1: Emotion-specific prompts
    print("\nTesting emotion-specific prompts:")
    print("-" * 80)

    for emotion, prompts in EMOTION_SPECIFIC_PROMPTS[model_name].items():
        for idx, text in enumerate(prompts, 1):
            print(f"\nTesting {emotion} with prompt {idx}/4:")
            print(f"Text: {text}")

            # Generate speech using parameters from emotion_config
            start = time.time()
            print("Generating speech...")

            ref_path = f"asset/ref_sound/{emotion_config[emotion]['file'][model_name]}"
            # Compute style for each inference to avoid cross-model contamination
            ref_style = tts.compute_style(ref_path)
            audio = tts.inference(
                text=text,
                ref_s=ref_style,
                alpha=emotion_config[emotion]["alpha"],
                beta=emotion_config[emotion]["beta"],
                diffusion_steps=DIFFUSION_STEPS,
                embedding_scale=emotion_config[emotion]["embedding_scale"],
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

    # Test 2: Same text with core emotions only
    print("\nTesting generic text with core emotions:")
    print("-" * 80)

    for emotion in CORE_EMOTIONS:
        print(f"\nTesting {emotion} with generic text:")
        print(f"Text: {GENERIC_TEST_TEXT}")

        start = time.time()
        print("Generating speech...")

        ref_path = f"asset/ref_sound/{emotion_config[emotion]['file'][model_name]}"
        # Compute style for each inference to avoid cross-model contamination
        ref_style = tts.compute_style(ref_path)
        audio = tts.inference(
            text=GENERIC_TEST_TEXT,
            ref_s=ref_style,
            alpha=emotion_config[emotion]["alpha"],
            beta=emotion_config[emotion]["beta"],
            diffusion_steps=DIFFUSION_STEPS,
            embedding_scale=emotion_config[emotion]["embedding_scale"],
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
