from flask import Flask, render_template, request, jsonify, url_for
import os
import json
import random
from datetime import datetime
from pathlib import Path

app = Flask(__name__)

# Configuration
RESULTS_DIR = Path("results/mos_evaluations")
RESULTS_FILE = RESULTS_DIR / "mos_results.json"

# Ensure results directory exists
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Model configurations
MODELS = ["new_ft_StyleTTS2", "Amelia10_ft_StyleTTS2"]

# Emotion configurations
EMOTIONS = ["joy", "sadness", "anger", "surprise", "neutral"]

# Scale configuration
SCALE_LABELS = {1: "Bad", 2: "Poor", 3: "Fair", 4: "Good", 5: "Excellent"}


def get_audio_files():
    static_audio = Path("static/audio")
    print(f"Looking for audio files in: {static_audio.absolute()}")

    # Part 1: Dynamic files for naturalness evaluation
    dynamic_files = []
    for model in MODELS:
        model_path = static_audio / model
        if model_path.exists():
            for file in model_path.glob("dynamic_*.wav"):
                dynamic_files.append(
                    {
                        "path": f"audio/{model}/{file.name}",
                        "model": model,
                        "filename": file.name,
                    }
                )

    # Part 2: Emotion evaluation files (specific and generic)
    emotion_files = []
    for model in MODELS:
        model_path = static_audio / model
        if model_path.exists():
            # Get both specific and generic emotion files
            for prefix in ["specific", "generic"]:
                for emotion in EMOTIONS:
                    # Match files like specific_joy_1.wav or generic_anger_2.wav
                    pattern = f"{prefix}_{emotion}_*.wav"
                    for file in model_path.glob(pattern):
                        emotion_files.append(
                            {
                                "path": f"audio/{model}/{file.name}",
                                "model": model,
                                "filename": file.name,
                                "emotion": emotion,
                                "type": prefix,
                            }
                        )

    # Randomize both lists
    random.shuffle(dynamic_files)
    random.shuffle(emotion_files)

    print(f"Found {len(dynamic_files)} dynamic files:")
    for f in dynamic_files:
        print(f"  - {f['path']}")
    print(f"Found {len(emotion_files)} emotion files:")
    for f in emotion_files:
        print(f"  - {f['path']} ({f['emotion']})")

    return {"dynamic": dynamic_files, "emotion": emotion_files}


def load_results():
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, "r") as f:
            return json.load(f)
    return []


def save_results(results):
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)


@app.route("/")
def index():
    audio_files = get_audio_files()
    return render_template(
        "index.html",
        audio_files=audio_files,
        emotions=EMOTIONS,
        scale_labels=SCALE_LABELS,
    )


@app.route("/submit", methods=["POST"])
def submit():
    data = request.get_json()

    # Add timestamp and remote IP
    data["timestamp"] = datetime.now().isoformat()
    data["remote_ip"] = request.remote_addr

    # Load existing results
    results = load_results()

    # Add new submission
    results.append(data)

    # Save updated results
    save_results(results)

    return jsonify({"status": "success"})


@app.route("/results")
def view_results():
    results = load_results()

    # Initialize model stats with default values
    model_stats = {
        model: {
            "naturalness": [],
            "specific_emotion_accuracy": {emotion: [] for emotion in EMOTIONS},
            "specific_emotion_confidence": {emotion: [] for emotion in EMOTIONS},
            "generic_emotion_accuracy": {emotion: [] for emotion in EMOTIONS},
            "generic_emotion_confidence": {emotion: [] for emotion in EMOTIONS},
        }
        for model in MODELS
    }

    # Process results if there are any
    if results:
        for submission in results:
            # Process naturalness ratings
            for rating in submission.get("naturalness", []):
                model = rating.get("model")
                score = rating.get("score")
                if model in MODELS and score is not None:
                    model_stats[model]["naturalness"].append(float(score))

            # Process emotion ratings
            for rating in submission.get("emotion", []):
                model = rating.get("model")
                true_emotion = rating.get("true_emotion")
                selected_emotion = rating.get("selected_emotion")
                confidence = rating.get("confidence")
                file_type = (
                    "specific" if "specific_" in rating.get("file", "") else "generic"
                )

                if (
                    model in MODELS
                    and true_emotion in EMOTIONS
                    and confidence is not None
                ):
                    # Track accuracy (1 for correct, 0 for incorrect)
                    correct = 1 if true_emotion == selected_emotion else 0
                    model_stats[model][f"{file_type}_emotion_accuracy"][
                        true_emotion
                    ].append(correct)

                    # Track confidence when emotion was correctly identified
                    if correct:
                        model_stats[model][f"{file_type}_emotion_confidence"][
                            true_emotion
                        ].append(float(confidence))

    # Calculate averages
    analysis = {}
    for model in MODELS:
        # Calculate naturalness average
        naturalness = model_stats[model]["naturalness"]
        avg_naturalness = sum(naturalness) / len(naturalness) if naturalness else 0

        # Calculate emotion accuracy and confidence averages for both specific and generic
        specific_accuracy = {}
        specific_confidence = {}
        generic_accuracy = {}
        generic_confidence = {}

        for emotion in EMOTIONS:
            # Specific text version
            spec_accuracy = model_stats[model]["specific_emotion_accuracy"][emotion]
            spec_confidence = model_stats[model]["specific_emotion_confidence"][emotion]
            specific_accuracy[emotion] = (
                (sum(spec_accuracy) / len(spec_accuracy) * 100) if spec_accuracy else 0
            )
            specific_confidence[emotion] = (
                sum(spec_confidence) / len(spec_confidence) if spec_confidence else 0
            )

            # Generic text version
            gen_accuracy = model_stats[model]["generic_emotion_accuracy"][emotion]
            gen_confidence = model_stats[model]["generic_emotion_confidence"][emotion]
            generic_accuracy[emotion] = (
                (sum(gen_accuracy) / len(gen_accuracy) * 100) if gen_accuracy else 0
            )
            generic_confidence[emotion] = (
                sum(gen_confidence) / len(gen_confidence) if gen_confidence else 0
            )

        # Calculate overall averages
        overall_accuracy = (
            (sum(specific_accuracy.values()) + sum(generic_accuracy.values()))
            / (2 * len(EMOTIONS))
            if specific_accuracy and generic_accuracy
            else 0
        )

        overall_confidence = (
            (sum(specific_confidence.values()) + sum(generic_confidence.values()))
            / (2 * len(EMOTIONS))
            if specific_confidence and generic_confidence
            else 0
        )

        # Calculate total score with 50% for naturalness and 50% for emotion evaluation
        # (25% for accuracy and 25% for confidence)
        total_score = (
            (avg_naturalness * 20)
            * 0.5  # 50% weight for naturalness (converted to 100-scale)
            + overall_accuracy
            * 0.25  # 25% weight for emotion accuracy (already on 100-scale)
            + (overall_confidence * 20)
            * 0.25  # 25% weight for confidence (converted to 100-scale)
        )

        analysis[model] = {
            "total_score": round(total_score, 1),
            "avg_naturalness": round(avg_naturalness, 2),
            "overall_accuracy": round(overall_accuracy, 1),
            "overall_confidence": round(overall_confidence, 1),
            "specific_emotion_accuracy": {
                e: round(v, 1) for e, v in specific_accuracy.items()
            },
            "specific_emotion_confidence": {
                e: round(v, 1) for e, v in specific_confidence.items()
            },
            "generic_emotion_accuracy": {
                e: round(v, 1) for e, v in generic_accuracy.items()
            },
            "generic_emotion_confidence": {
                e: round(v, 1) for e, v in generic_confidence.items()
            },
        }

    print("Analysis results:", analysis)  # Debug print

    return render_template(
        "results.html",
        analysis=analysis,
        models=MODELS,
        emotions=EMOTIONS,
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
else:
    application = app
