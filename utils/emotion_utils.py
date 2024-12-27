from transformers import pipeline
import torch

# Emotion model settings
EMOTION_MODEL_NAME = "SamLowe/roberta-base-go_emotions"

DIFFUSION_STEPS = 5  # Adjust this to trade off quality vs speed


ALPHA = 0.3
BETA = 0.7
EMBEDDING_SCALE = 1.0

ANGRY_ALPHA = 0.3
ANGRY_BETA = 0.7
ANGRY_EMBEDDING_SCALE = 1.0

HAPPY_ALPHA = 0.3
HAPPY_BETA = 0.7
HAPPY_EMBEDDING_SCALE = 1.0

SAD_ALPHA = 0.3
SAD_BETA = 0.7
SAD_EMBEDDING_SCALE = 1.0

SURPRISE_ALPHA = 0.3
SURPRISE_BETA = 0.7
SURPRISE_EMBEDDING_SCALE = 1.0

# Emotion to voice style mapping with inference parameters
EMOTION_CONFIG = {
    # Format: "emotion": {"file": "style_file.wav", "alpha": float, "beta": float, "embedding_scale": float}
    # Neutral emotions - using neutral.wav
    "neutral": {
        "file": "neutral.wav",
        "alpha": ALPHA,
        "beta": BETA,
        "embedding_scale": EMBEDDING_SCALE,
    },
    "confusion": {
        "file": "neutral.wav",
        "alpha": ALPHA,
        "beta": BETA,
        "embedding_scale": EMBEDDING_SCALE,
    },
    "caring": {
        "file": "neutral.wav",
        "alpha": ALPHA,
        "beta": BETA,
        "embedding_scale": EMBEDDING_SCALE,
    },
    "curiosity": {
        "file": "neutral.wav",
        "alpha": ALPHA,
        "beta": BETA,
        "embedding_scale": EMBEDDING_SCALE,
    },
    "desire": {
        "file": "neutral.wav",
        "alpha": ALPHA,
        "beta": BETA,
        "embedding_scale": EMBEDDING_SCALE,
    },
    "relief": {
        "file": "neutral.wav",
        "alpha": ALPHA,
        "beta": BETA,
        "embedding_scale": EMBEDDING_SCALE,
    },
    # Happy emotions - using happy.wav
    "admiration": {
        "file": "happy.wav",
        "alpha": HAPPY_ALPHA,
        "beta": HAPPY_BETA,
        "embedding_scale": HAPPY_EMBEDDING_SCALE,
    },
    "amusement": {
        "file": "happy.wav",
        "alpha": HAPPY_ALPHA,
        "beta": HAPPY_BETA,
        "embedding_scale": HAPPY_EMBEDDING_SCALE,
    },
    "approval": {
        "file": "happy.wav",
        "alpha": HAPPY_ALPHA,
        "beta": HAPPY_BETA,
        "embedding_scale": HAPPY_EMBEDDING_SCALE,
    },
    "excitement": {
        "file": "happy.wav",
        "alpha": HAPPY_ALPHA,
        "beta": HAPPY_BETA,
        "embedding_scale": HAPPY_EMBEDDING_SCALE,
    },
    "gratitude": {
        "file": "happy.wav",
        "alpha": HAPPY_ALPHA,
        "beta": HAPPY_BETA,
        "embedding_scale": HAPPY_EMBEDDING_SCALE,
    },
    "joy": {
        "file": "happy.wav",
        "alpha": HAPPY_ALPHA,
        "beta": HAPPY_BETA,
        "embedding_scale": HAPPY_EMBEDDING_SCALE,
    },
    "love": {
        "file": "happy.wav",
        "alpha": HAPPY_ALPHA,
        "beta": HAPPY_BETA,
        "embedding_scale": HAPPY_EMBEDDING_SCALE,
    },
    "optimism": {
        "file": "happy.wav",
        "alpha": HAPPY_ALPHA,
        "beta": HAPPY_BETA,
        "embedding_scale": HAPPY_EMBEDDING_SCALE,
    },
    "pride": {
        "file": "happy.wav",
        "alpha": HAPPY_ALPHA,
        "beta": HAPPY_BETA,
        "embedding_scale": HAPPY_EMBEDDING_SCALE,
    },
    # Sad emotions - using sad.wav
    "disappointment": {
        "file": "sad.wav",
        "alpha": SAD_ALPHA,
        "beta": SAD_BETA,
        "embedding_scale": SAD_EMBEDDING_SCALE,
    },
    "embarrassment": {
        "file": "sad.wav",
        "alpha": SAD_ALPHA,
        "beta": SAD_BETA,
        "embedding_scale": SAD_EMBEDDING_SCALE,
    },
    "fear": {
        "file": "sad.wav",
        "alpha": SAD_ALPHA,
        "beta": SAD_BETA,
        "embedding_scale": SAD_EMBEDDING_SCALE,
    },
    "grief": {
        "file": "sad.wav",
        "alpha": SAD_ALPHA,
        "beta": SAD_BETA,
        "embedding_scale": SAD_EMBEDDING_SCALE,
    },
    "nervousness": {
        "file": "sad.wav",
        "alpha": SAD_ALPHA,
        "beta": SAD_BETA,
        "embedding_scale": SAD_EMBEDDING_SCALE,
    },
    "remorse": {
        "file": "sad.wav",
        "alpha": SAD_ALPHA,
        "beta": SAD_BETA,
        "embedding_scale": SAD_EMBEDDING_SCALE,
    },
    "sadness": {
        "file": "sad.wav",
        "alpha": SAD_ALPHA,
        "beta": SAD_BETA,
        "embedding_scale": SAD_EMBEDDING_SCALE,
    },
    # Angry emotions - using angry.wav
    "disapproval": {
        "file": "angry.wav",
        "alpha": ANGRY_ALPHA,
        "beta": ANGRY_BETA,
        "embedding_scale": ANGRY_EMBEDDING_SCALE,
    },
    "disgust": {
        "file": "angry.wav",
        "alpha": ANGRY_ALPHA,
        "beta": ANGRY_BETA,
        "embedding_scale": ANGRY_EMBEDDING_SCALE,
    },
    "anger": {
        "file": "angry.wav",
        "alpha": ANGRY_ALPHA,
        "beta": ANGRY_BETA,
        "embedding_scale": ANGRY_EMBEDDING_SCALE,
    },
    "annoyance": {
        "file": "angry.wav",
        "alpha": ANGRY_ALPHA,
        "beta": ANGRY_BETA,
        "embedding_scale": ANGRY_EMBEDDING_SCALE,
    },
    # Surprised emotions - using surprised.wav
    "realization": {
        "file": "surprised.wav",
        "alpha": SURPRISE_ALPHA,
        "beta": SURPRISE_BETA,
        "embedding_scale": SURPRISE_EMBEDDING_SCALE,
    },
    "surprise": {
        "file": "surprised.wav",
        "alpha": SURPRISE_ALPHA,
        "beta": SURPRISE_BETA,
        "embedding_scale": SURPRISE_EMBEDDING_SCALE,
    },
}

# For backward compatibility
EMOTION_MAPPING = {
    emotion: config["file"] for emotion, config in EMOTION_CONFIG.items()
}


class EmotionHandler:
    def __init__(self):
        # Initialize emotion classifier with RoBERTa
        self.emotion_classifier = pipeline(
            "text-classification",
            model=EMOTION_MODEL_NAME,
            top_k=1,
            truncation=True,
            device=-1,
            framework="pt",  # Force PyTorch backend
            model_kwargs={
                "low_cpu_mem_usage": True,
                "torch_dtype": torch.float32,
            },
        )

        # Warm up classifier silently
        self.classify_emotion("Test message for warmup.")

    def classify_emotion(self, text):
        """Classify the emotion of the given text using RoBERTa."""
        result = self.emotion_classifier(text)
        emotion = result[0][0]["label"]
        confidence = result[0][0]["score"]

        # Use neutral for low confidence predictions
        if confidence < 0.25:
            emotion = "neutral"

        return emotion

    def get_style_for_emotion(self, emotion):
        """Get the corresponding voice style file for an emotion."""
        return EMOTION_MAPPING.get(emotion, "neutral.wav")
