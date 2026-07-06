"""
Canonical schema for Vbot TTS evaluation artifacts.

This module is the single source of truth for what a human TTS evaluation
submission looks like on disk. The Flask form validates against it before
persisting, and downstream tooling (promotion gates, analysis notebooks)
can rely on every stored record matching SCHEMA_VERSION.

Stored artifact shape (one JSON list of submissions):

    [
      {
        "schema_version": 1,
        "timestamp": "2026-07-06T12:34:56.789012",   # server-stamped, ISO 8601
        "remote_ip": "127.0.0.1",                    # server-stamped
        "evaluations": [
          {
            "model": "Amelia",              # one of ALL_MODELS
            "model_type": "old",            # one of MODEL_TYPES
            "file": "specific_joy_1.wav",   # source audio filename (optional)
            "true_emotion": "joy",          # one of EMOTIONS
            "selected_emotion": "joy",      # one of EMOTIONS
            "naturalness": 4                # optional int in [1, 5]
          },
          ...
        ]
      },
      ...
    ]
"""

SCHEMA_VERSION = 1

# Model roster under evaluation. "old" = current production voices,
# "new" = candidate voices being compared against them.
OLD_MODELS = ["Amelia", "Eveland"]
NEW_MODELS = ["Gura", "Amelia_new"]
ALL_MODELS = OLD_MODELS + NEW_MODELS

MODEL_TYPES = ("old", "new")

EMOTIONS = ["joy", "sadness", "anger", "surprise", "neutral"]

NATURALNESS_MIN = 1
NATURALNESS_MAX = 5


def validate_submission(data):
    """Validate a raw /submit payload against the schema.

    Returns a list of human-readable error strings; an empty list means the
    payload is valid. Server-stamped fields (timestamp, remote_ip,
    schema_version) are not expected in the incoming payload.
    """
    errors = []

    if not isinstance(data, dict):
        return ["submission must be a JSON object"]

    evaluations = data.get("evaluations")
    if not isinstance(evaluations, list) or not evaluations:
        return ["'evaluations' must be a non-empty list"]

    for index, evaluation in enumerate(evaluations):
        prefix = f"evaluations[{index}]"

        if not isinstance(evaluation, dict):
            errors.append(f"{prefix}: must be an object")
            continue

        model = evaluation.get("model")
        if model not in ALL_MODELS:
            errors.append(f"{prefix}: unknown model {model!r} (expected one of {ALL_MODELS})")

        model_type = evaluation.get("model_type")
        if model_type not in MODEL_TYPES:
            errors.append(f"{prefix}: invalid model_type {model_type!r} (expected one of {list(MODEL_TYPES)})")
        elif model in ALL_MODELS:
            expected_type = "old" if model in OLD_MODELS else "new"
            if model_type != expected_type:
                errors.append(f"{prefix}: model {model!r} must have model_type {expected_type!r}, got {model_type!r}")

        for field in ("true_emotion", "selected_emotion"):
            value = evaluation.get(field)
            if value not in EMOTIONS:
                errors.append(f"{prefix}: invalid {field} {value!r} (expected one of {EMOTIONS})")

        naturalness = evaluation.get("naturalness")
        if naturalness is not None:
            if not isinstance(naturalness, int) or isinstance(naturalness, bool):
                errors.append(f"{prefix}: naturalness must be an integer, got {naturalness!r}")
            elif not NATURALNESS_MIN <= naturalness <= NATURALNESS_MAX:
                errors.append(
                    f"{prefix}: naturalness {naturalness} out of range "
                    f"[{NATURALNESS_MIN}, {NATURALNESS_MAX}]"
                )

        file_name = evaluation.get("file")
        if file_name is not None and not isinstance(file_name, str):
            errors.append(f"{prefix}: file must be a string, got {file_name!r}")

    return errors


def stamp_submission(data, timestamp, remote_ip):
    """Attach server-side metadata to a validated submission."""
    data["schema_version"] = SCHEMA_VERSION
    data["timestamp"] = timestamp
    data["remote_ip"] = remote_ip
    return data
