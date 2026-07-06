"""
GoEmotions label -> voice-style bucket mapping.

Extracted from utils/emotion_utils.py so that stdlib-only tooling
(scripts/monitoring_report.py, tests) can use the mapping without pulling
in transformers/torch. utils/emotion_utils.py re-exports both names, so
runtime code and the eval scripts keep importing from there — this module
is the single source of truth either way.
"""

# Centralized emotion definitions with audio file mappings.
# Format: emotion_name: (audio_type, alpha_param, beta_param, embedding_param)
# Covers the 28 GoEmotions labels emitted by the runtime classifier,
# mapped onto the 5 voice-style buckets used for TTS delivery.
EMOTION_DEFINITIONS = {
    # Neutral emotions - use neutral audio and default parameters
    "neutral": ("neutral", "ALPHA", "BETA", "EMBEDDING_SCALE"),
    "confusion": ("neutral", "ALPHA", "BETA", "EMBEDDING_SCALE"),
    "caring": ("neutral", "ALPHA", "BETA", "EMBEDDING_SCALE"),
    "curiosity": ("neutral", "ALPHA", "BETA", "EMBEDDING_SCALE"),
    "desire": ("neutral", "ALPHA", "BETA", "EMBEDDING_SCALE"),
    "relief": ("neutral", "ALPHA", "BETA", "EMBEDDING_SCALE"),
    # Happy emotions - use happy audio and happy parameters
    "admiration": ("happy", "HAPPY_ALPHA", "HAPPY_BETA", "HAPPY_EMBEDDING_SCALE"),
    "amusement": ("happy", "HAPPY_ALPHA", "HAPPY_BETA", "HAPPY_EMBEDDING_SCALE"),
    "approval": ("happy", "HAPPY_ALPHA", "HAPPY_BETA", "HAPPY_EMBEDDING_SCALE"),
    "excitement": ("happy", "HAPPY_ALPHA", "HAPPY_BETA", "HAPPY_EMBEDDING_SCALE"),
    "gratitude": ("happy", "HAPPY_ALPHA", "HAPPY_BETA", "HAPPY_EMBEDDING_SCALE"),
    "joy": ("happy", "HAPPY_ALPHA", "HAPPY_BETA", "HAPPY_EMBEDDING_SCALE"),
    "love": ("happy", "HAPPY_ALPHA", "HAPPY_BETA", "HAPPY_EMBEDDING_SCALE"),
    "optimism": ("happy", "HAPPY_ALPHA", "HAPPY_BETA", "HAPPY_EMBEDDING_SCALE"),
    "pride": ("happy", "HAPPY_ALPHA", "HAPPY_BETA", "HAPPY_EMBEDDING_SCALE"),
    # Sad emotions - use sad audio and sad parameters
    "disappointment": ("sad", "SAD_ALPHA", "SAD_BETA", "SAD_EMBEDDING_SCALE"),
    "embarrassment": ("sad", "SAD_ALPHA", "SAD_BETA", "SAD_EMBEDDING_SCALE"),
    "fear": ("sad", "SAD_ALPHA", "SAD_BETA", "SAD_EMBEDDING_SCALE"),
    "grief": ("sad", "SAD_ALPHA", "SAD_BETA", "SAD_EMBEDDING_SCALE"),
    "nervousness": ("sad", "SAD_ALPHA", "SAD_BETA", "SAD_EMBEDDING_SCALE"),
    "remorse": ("sad", "SAD_ALPHA", "SAD_BETA", "SAD_EMBEDDING_SCALE"),
    "sadness": ("sad", "SAD_ALPHA", "SAD_BETA", "SAD_EMBEDDING_SCALE"),
    # Angry emotions - use angry audio and angry parameters
    "disapproval": ("angry", "ANGRY_ALPHA", "ANGRY_BETA", "ANGRY_EMBEDDING_SCALE"),
    "disgust": ("angry", "ANGRY_ALPHA", "ANGRY_BETA", "ANGRY_EMBEDDING_SCALE"),
    "anger": ("angry", "ANGRY_ALPHA", "ANGRY_BETA", "ANGRY_EMBEDDING_SCALE"),
    "annoyance": ("angry", "ANGRY_ALPHA", "ANGRY_BETA", "ANGRY_EMBEDDING_SCALE"),
    # Surprised emotions - use surprised audio and surprise parameters
    "realization": (
        "surprised",
        "SURPRISE_ALPHA",
        "SURPRISE_BETA",
        "SURPRISE_EMBEDDING_SCALE",
    ),
    "surprise": (
        "surprised",
        "SURPRISE_ALPHA",
        "SURPRISE_BETA",
        "SURPRISE_EMBEDDING_SCALE",
    ),
}


def get_emotion_bucket(label):
    """Map a classifier label to its runtime voice-style bucket.

    Unknown labels fall back to neutral, matching the runtime's defensive
    behavior elsewhere in the pipeline.
    """
    return EMOTION_DEFINITIONS.get(label, ("neutral",))[0]
