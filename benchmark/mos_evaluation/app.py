from flask import Flask, render_template, request, jsonify, url_for
import os
import json
from datetime import datetime
from pathlib import Path

app = Flask(__name__)

# Configuration
RESULTS_DIR = Path("results/mos_evaluations")
RESULTS_FILE = RESULTS_DIR / "mos_results.json"

# Ensure results directory exists
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Model configurations with their custom reference files
MODELS = {
    "new_ft_StyleTTS2": {
        "id": "nonoJDWAOIDAWKDA/new_ft_StyleTTS2",
        "reference": "new_custom_reference.wav",
    },
    "Amelia10_ft_StyleTTS2": {
        "id": "nonoJDWAOIDAWKDA/Amelia10_ft_StyleTTS2",
        "reference": "Amelia10_custom_reference.wav",
    },
}

# Emotion configurations
EMOTIONS = ["joy", "sadness", "anger", "surprise", "neutral"]


# Audio file mappings
def get_audio_urls():
    static_audio = Path("static/audio")

    # Fixed original audio URL
    original_url = url_for("static", filename="audio/original.wav")

    # Generate URLs for model outputs
    generated = {}
    for model_name, model_info in MODELS.items():
        generated[model_name] = {
            # Add custom reference file
            "reference": url_for("static", filename=f'audio/{model_info["reference"]}')
        }
        for emotion in EMOTIONS:
            # Both specific and generic versions
            for prefix in ["specific", "generic"]:
                filename = f"{model_name}_{prefix}_{emotion}.wav"
                if (static_audio / filename).exists():
                    generated[model_name][f"{prefix}_{emotion}"] = url_for(
                        "static", filename=f"audio/{filename}"
                    )

    return {"original": original_url, "generated": generated}


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
    audio_urls = get_audio_urls()
    return render_template(
        "index.html",
        models=MODELS,
        emotions=EMOTIONS,
        audio_urls=audio_urls,
        has_generic=True,  # Flag to indicate we have generic versions
    )


@app.route("/submit", methods=["POST"])
def submit():
    data = request.get_json()

    # Add timestamp and remote IP (for tracking unique submissions)
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
    return render_template("results.html", results=results)


if __name__ == "__main__":
    # For local development
    app.run(debug=True, host="0.0.0.0", port=5000)
else:
    # For production/PythonAnywhere
    application = app
