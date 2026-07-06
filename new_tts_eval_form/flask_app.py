from flask import Flask, render_template, request, jsonify
import os
import json
import random
from datetime import datetime
from pathlib import Path

from eval_schema import (
    ALL_MODELS,
    EMOTIONS,
    NEW_MODELS,
    OLD_MODELS,
    stamp_submission,
    validate_submission,
)

app = Flask(__name__)

# Configuration - use absolute paths. The results file can be overridden
# (e.g. by tests) through VBOT_EVAL_RESULTS_FILE.
APP_DIR = Path(__file__).parent
_results_override = os.getenv("VBOT_EVAL_RESULTS_FILE")
if _results_override:
    RESULTS_FILE = Path(_results_override)
    RESULTS_DIR = RESULTS_FILE.parent
else:
    RESULTS_DIR = APP_DIR / "results"
    RESULTS_FILE = RESULTS_DIR / "evaluation_results.json"

# Ensure results directory exists
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_audio_files():
    """Get all audio files from the static/audio directory"""
    static_audio = APP_DIR / "static" / "audio"
    print(f"Looking for audio files in: {static_audio.absolute()}")

    audio_files = []
    for model in ALL_MODELS:
        model_path = static_audio / model
        if model_path.exists():
            for emotion in EMOTIONS:
                # Get specific emotion files (4 prompts per emotion)
                pattern = f"specific_{emotion}_*.wav"
                for file in model_path.glob(pattern):
                    audio_files.append(
                        {
                            "path": f"audio/{model}/{file.name}",
                            "model": model,
                            "filename": file.name,
                            "emotion": emotion,
                            "model_type": "old" if model in OLD_MODELS else "new",
                        }
                    )

                # Get generic emotion file (1 per emotion)
                generic_file = model_path / f"generic_{emotion}.wav"
                if generic_file.exists():
                    audio_files.append(
                        {
                            "path": f"audio/{model}/{generic_file.name}",
                            "model": model,
                            "filename": generic_file.name,
                            "emotion": emotion,
                            "model_type": "old" if model in OLD_MODELS else "new",
                        }
                    )

    # Randomize the list
    random.shuffle(audio_files)

    print(f"Found {len(audio_files)} audio files:")
    for f in audio_files[:10]:  # Print first 10 for debugging
        print(f"  - {f['path']} ({f['emotion']}, {f['model_type']})")

    return audio_files


def load_results():
    """Load existing evaluation results"""
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, "r") as f:
            return json.load(f)
    return []


def save_results(results):
    """Save evaluation results to JSON file"""
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)


@app.route("/")
def index():
    """Main evaluation page"""
    audio_files = get_audio_files()
    return render_template("index.html", audio_files=audio_files, emotions=EMOTIONS)


@app.route("/submit", methods=["POST"])
def submit():
    """Handle form submission"""
    data = request.get_json(silent=True)

    # Reject anything that does not match the canonical artifact schema
    errors = validate_submission(data)
    if errors:
        return jsonify({"status": "error", "errors": errors}), 400

    # Add schema version, timestamp, and remote IP
    stamp_submission(data, datetime.now().isoformat(), request.remote_addr)

    # Load existing results
    results = load_results()

    # Add new submission
    results.append(data)

    # Save updated results
    save_results(results)

    return jsonify({"status": "success"})


@app.route("/results")
def view_results():
    """View aggregated results"""
    results = load_results()

    # Initialize stats
    stats = {
        "old": {
            "total": 0,
            "correct": 0,
            "naturalness_sum": 0,
            "naturalness_count": 0,
            "per_emotion": {
                emotion: {"total": 0, "correct": 0} for emotion in EMOTIONS
            },
            "per_model": {
                model: {
                    "total": 0,
                    "correct": 0,
                    "naturalness_sum": 0,
                    "naturalness_count": 0,
                    "per_emotion": {
                        emotion: {"total": 0, "correct": 0} for emotion in EMOTIONS
                    },
                }
                for model in OLD_MODELS
            },
        },
        "new": {
            "total": 0,
            "correct": 0,
            "naturalness_sum": 0,
            "naturalness_count": 0,
            "per_emotion": {
                emotion: {"total": 0, "correct": 0} for emotion in EMOTIONS
            },
            "per_model": {
                model: {
                    "total": 0,
                    "correct": 0,
                    "naturalness_sum": 0,
                    "naturalness_count": 0,
                    "per_emotion": {
                        emotion: {"total": 0, "correct": 0} for emotion in EMOTIONS
                    },
                }
                for model in NEW_MODELS
            },
        },
    }

    # Process results
    for submission in results:
        for evaluation in submission.get("evaluations", []):
            model = evaluation["model"]
            model_type = evaluation["model_type"]
            true_emotion = evaluation["true_emotion"]
            selected_emotion = evaluation["selected_emotion"]
            is_correct = true_emotion == selected_emotion
            naturalness = evaluation.get("naturalness", None)

            # Update overall stats
            stats[model_type]["total"] += 1
            if is_correct:
                stats[model_type]["correct"] += 1

            # Update naturalness stats
            if naturalness is not None:
                stats[model_type]["naturalness_sum"] += naturalness
                stats[model_type]["naturalness_count"] += 1

            # Update per-emotion stats
            stats[model_type]["per_emotion"][true_emotion]["total"] += 1
            if is_correct:
                stats[model_type]["per_emotion"][true_emotion]["correct"] += 1

            # Update per-model stats
            stats[model_type]["per_model"][model]["total"] += 1
            if is_correct:
                stats[model_type]["per_model"][model]["correct"] += 1

            # Update per-model naturalness
            if naturalness is not None:
                stats[model_type]["per_model"][model]["naturalness_sum"] += naturalness
                stats[model_type]["per_model"][model]["naturalness_count"] += 1

            # Update per-model per-emotion stats
            stats[model_type]["per_model"][model]["per_emotion"][true_emotion][
                "total"
            ] += 1
            if is_correct:
                stats[model_type]["per_model"][model]["per_emotion"][true_emotion][
                    "correct"
                ] += 1

    # Calculate percentages
    analysis = {
        "old": {
            "overall_accuracy": (
                (stats["old"]["correct"] / stats["old"]["total"] * 100)
                if stats["old"]["total"] > 0
                else 0
            ),
            "average_naturalness": (
                (stats["old"]["naturalness_sum"] / stats["old"]["naturalness_count"])
                if stats["old"]["naturalness_count"] > 0
                else 0
            ),
            "emotion_accuracy": {
                emotion: (
                    (
                        stats["old"]["per_emotion"][emotion]["correct"]
                        / stats["old"]["per_emotion"][emotion]["total"]
                        * 100
                    )
                    if stats["old"]["per_emotion"][emotion]["total"] > 0
                    else 0
                )
                for emotion in EMOTIONS
            },
            "model_accuracy": {
                model: {
                    "overall": (
                        (
                            stats["old"]["per_model"][model]["correct"]
                            / stats["old"]["per_model"][model]["total"]
                            * 100
                        )
                        if stats["old"]["per_model"][model]["total"] > 0
                        else 0
                    ),
                    "naturalness": (
                        (
                            stats["old"]["per_model"][model]["naturalness_sum"]
                            / stats["old"]["per_model"][model]["naturalness_count"]
                        )
                        if stats["old"]["per_model"][model]["naturalness_count"] > 0
                        else 0
                    ),
                    "per_emotion": {
                        emotion: (
                            (
                                stats["old"]["per_model"][model]["per_emotion"][
                                    emotion
                                ]["correct"]
                                / stats["old"]["per_model"][model]["per_emotion"][
                                    emotion
                                ]["total"]
                                * 100
                            )
                            if stats["old"]["per_model"][model]["per_emotion"][emotion][
                                "total"
                            ]
                            > 0
                            else 0
                        )
                        for emotion in EMOTIONS
                    },
                }
                for model in OLD_MODELS
            },
        },
        "new": {
            "overall_accuracy": (
                (stats["new"]["correct"] / stats["new"]["total"] * 100)
                if stats["new"]["total"] > 0
                else 0
            ),
            "average_naturalness": (
                (stats["new"]["naturalness_sum"] / stats["new"]["naturalness_count"])
                if stats["new"]["naturalness_count"] > 0
                else 0
            ),
            "emotion_accuracy": {
                emotion: (
                    (
                        stats["new"]["per_emotion"][emotion]["correct"]
                        / stats["new"]["per_emotion"][emotion]["total"]
                        * 100
                    )
                    if stats["new"]["per_emotion"][emotion]["total"] > 0
                    else 0
                )
                for emotion in EMOTIONS
            },
            "model_accuracy": {
                model: {
                    "overall": (
                        (
                            stats["new"]["per_model"][model]["correct"]
                            / stats["new"]["per_model"][model]["total"]
                            * 100
                        )
                        if stats["new"]["per_model"][model]["total"] > 0
                        else 0
                    ),
                    "naturalness": (
                        (
                            stats["new"]["per_model"][model]["naturalness_sum"]
                            / stats["new"]["per_model"][model]["naturalness_count"]
                        )
                        if stats["new"]["per_model"][model]["naturalness_count"] > 0
                        else 0
                    ),
                    "per_emotion": {
                        emotion: (
                            (
                                stats["new"]["per_model"][model]["per_emotion"][
                                    emotion
                                ]["correct"]
                                / stats["new"]["per_model"][model]["per_emotion"][
                                    emotion
                                ]["total"]
                                * 100
                            )
                            if stats["new"]["per_model"][model]["per_emotion"][emotion][
                                "total"
                            ]
                            > 0
                            else 0
                        )
                        for emotion in EMOTIONS
                    },
                }
                for model in NEW_MODELS
            },
        },
    }

    return render_template(
        "results.html",
        analysis=analysis,
        old_models=OLD_MODELS,
        new_models=NEW_MODELS,
        emotions=EMOTIONS,
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
else:
    application = app
