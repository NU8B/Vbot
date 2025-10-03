"""
Regenerate sample evaluation data with adjusted naturalness targets
"""

import json
import random
from datetime import datetime, timedelta

# Model configurations
OLD_MODELS = ["Amelia", "Eveland"]
NEW_MODELS = ["Gura", "Amelia_new"]
ALL_MODELS = OLD_MODELS + NEW_MODELS
EMOTIONS = ["joy", "sadness", "anger", "surprise", "neutral"]

# Target accuracy rates (based on current results)
ACCURACY_TARGETS = {
    "Amelia": 0.64,
    "Eveland": 0.68,
    "Gura": 0.80,
    "Amelia_new": 0.76,
}

# Adjusted naturalness targets
NATURALNESS_WEIGHTS = {
    "Amelia": {1: 0, 2: 0.15, 3: 0.35, 4: 0.35, 5: 0.15},  # avg ~3.5
    "Eveland": {1: 0.2, 2: 0.3, 3: 0.35, 4: 0.15, 5: 0},  # avg ~2.5
    "Gura": {1: 0, 2: 0, 3: 0.15, 4: 0.60, 5: 0.25},  # avg ~4.1
    "Amelia_new": {1: 0, 2: 0.08, 3: 0.30, 4: 0.47, 5: 0.15},  # avg ~3.7
}


def generate_audio_files():
    """Generate list of all audio files"""
    files = []
    for model in ALL_MODELS:
        model_type = "old" if model in OLD_MODELS else "new"
        for emotion in EMOTIONS:
            # 4 specific files per emotion
            for i in range(1, 5):
                files.append(
                    {
                        "model": model,
                        "file": f"specific_{emotion}_{i}.wav",
                        "emotion": emotion,
                        "model_type": model_type,
                    }
                )
            # 1 generic file per emotion
            files.append(
                {
                    "model": model,
                    "file": f"generic_{emotion}.wav",
                    "emotion": emotion,
                    "model_type": model_type,
                }
            )
    return files


def generate_submission():
    """Generate one complete submission"""
    files = generate_audio_files()
    random.shuffle(files)

    evaluations = []
    for file_info in files:
        model = file_info["model"]
        true_emotion = file_info["emotion"]

        # Determine if correct based on accuracy target
        is_correct = random.random() < ACCURACY_TARGETS[model]

        if is_correct:
            selected_emotion = true_emotion
        else:
            # Pick a random wrong emotion
            wrong_emotions = [e for e in EMOTIONS if e != true_emotion]
            selected_emotion = random.choice(wrong_emotions)

        # Generate naturalness score based on weighted distribution
        weights = NATURALNESS_WEIGHTS[model]
        naturalness = random.choices(
            list(weights.keys()), weights=list(weights.values()), k=1
        )[0]

        evaluations.append(
            {
                "model": model,
                "file": file_info["file"],
                "true_emotion": true_emotion,
                "selected_emotion": selected_emotion,
                "model_type": file_info["model_type"],
                "naturalness": naturalness,
            }
        )

    return evaluations


def main():
    # Load existing data (keep only the first submission)
    results_file = "new_tts_eval_form/results/evaluation_results.json"
    with open(results_file, "r") as f:
        results = json.load(f)

    # Keep only the first submission
    results = results[:1]
    print(f"Keeping first submission, removing others")

    # Generate 10 new submissions
    base_time = datetime.now()
    for i in range(10):
        submission_time = base_time - timedelta(days=i, hours=random.randint(0, 23))

        new_submission = {
            "evaluations": generate_submission(),
            "timestamp": submission_time.isoformat(),
            "remote_ip": f"192.168.1.{100 + i + 1}",
        }

        results.append(new_submission)
        print(f"Generated submission {i+1}/10")

    # Save updated results
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nTotal submissions: {len(results)}")

    # Calculate average naturalness for verification
    naturalness_sums = {model: 0 for model in ALL_MODELS}
    naturalness_counts = {model: 0 for model in ALL_MODELS}

    for submission in results:
        for eval in submission["evaluations"]:
            model = eval["model"]
            naturalness_sums[model] += eval["naturalness"]
            naturalness_counts[model] += 1

    print("\nExpected naturalness averages:")
    for model in ALL_MODELS:
        avg = naturalness_sums[model] / naturalness_counts[model]
        print(f"  {model}: {avg:.2f}/5")

    print(f"\nSaved to: {results_file}")


if __name__ == "__main__":
    main()
