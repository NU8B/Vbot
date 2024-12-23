from transformers import pipeline
import torch

# Emotion model settings
EMOTION_MODEL_NAME = "SamLowe/roberta-base-go_emotions"

# Global inference settings
DIFFUSION_STEPS = 5  # Adjust this to trade off quality vs speed

# Emotion to voice style mapping with inference parameters
EMOTION_CONFIG = {
    # Format: "emotion": {"file": "style_file.wav", "alpha": float, "beta": float, "embedding_scale": float}
    # Neutral emotions - minimal style modification
    "neutral": {
        "file": "neutral.wav",
        "alpha": 0.3,
        "beta": 0.7,
        "embedding_scale": 1.0,
    },
    "confusion": {
        "file": "neutral.wav",
        "alpha": 0.3,
        "beta": 0.7,
        "embedding_scale": 1.0,
    },
    "caring": {
        "file": "neutral.wav",
        "alpha": 0.3,
        "beta": 0.7,
        "embedding_scale": 1.0,
    },
    "curiosity": {
        "file": "neutral.wav",
        "alpha": 0.3,
        "beta": 0.7,
        "embedding_scale": 1.0,
    },
    "desire": {
        "file": "neutral.wav",
        "alpha": 0.3,
        "beta": 0.7,
        "embedding_scale": 1.0,
    },
    "relief": {
        "file": "neutral.wav",
        "alpha": 0.3,
        "beta": 0.7,
        "embedding_scale": 1.0,
    },
    # Happy emotions - more expressive, higher embedding scale
    "admiration": {
        "file": "happy.wav",
        "alpha": 0.3,
        "beta": 0.7,
        "embedding_scale": 1.5,
    },
    "amusement": {
        "file": "happy.wav",
        "alpha": 0.3,
        "beta": 0.7,
        "embedding_scale": 1.5,
    },
    "approval": {
        "file": "happy.wav",
        "alpha": 0.3,
        "beta": 0.7,
        "embedding_scale": 1.5,
    },
    "excitement": {
        "file": "happy.wav",
        "alpha": 0.5,
        "beta": 0.9,
        "embedding_scale": 2.0,
    },
    "gratitude": {
        "file": "happy.wav",
        "alpha": 0.3,
        "beta": 0.7,
        "embedding_scale": 1.5,
    },
    "joy": {"file": "happy.wav", "alpha": 0.5, "beta": 0.9, "embedding_scale": 2.0},
    "love": {"file": "happy.wav", "alpha": 0.5, "beta": 0.9, "embedding_scale": 2.0},
    "optimism": {
        "file": "happy.wav",
        "alpha": 0.3,
        "beta": 0.7,
        "embedding_scale": 1.5,
    },
    "pride": {"file": "happy.wav", "alpha": 0.5, "beta": 0.9, "embedding_scale": 2.0},
    # Sad emotions - lower beta for more reference emotion
    "disappointment": {
        "file": "sad.wav",
        "alpha": 0.3,
        "beta": 0.5,
        "embedding_scale": 1.5,
    },
    "embarrassment": {
        "file": "sad.wav",
        "alpha": 0.3,
        "beta": 0.5,
        "embedding_scale": 1.5,
    },
    "fear": {"file": "sad.wav", "alpha": 0.5, "beta": 0.7, "embedding_scale": 2.0},
    "grief": {"file": "sad.wav", "alpha": 0.5, "beta": 0.7, "embedding_scale": 2.0},
    "nervousness": {
        "file": "sad.wav",
        "alpha": 0.3,
        "beta": 0.5,
        "embedding_scale": 1.5,
    },
    "remorse": {"file": "sad.wav", "alpha": 0.5, "beta": 0.7, "embedding_scale": 2.0},
    "sadness": {"file": "sad.wav", "alpha": 0.5, "beta": 0.7, "embedding_scale": 2.0},
    # Angry emotions - high beta and embedding scale for strong emotion
    "disapproval": {
        "file": "angry.wav",
        "alpha": 0.5,
        "beta": 0.9,
        "embedding_scale": 2.0,
    },
    "disgust": {"file": "angry.wav", "alpha": 0.5, "beta": 0.9, "embedding_scale": 2.0},
    "anger": {"file": "angry.wav", "alpha": 0.5, "beta": 0.95, "embedding_scale": 2.0},
    "annoyance": {
        "file": "angry.wav",
        "alpha": 0.5,
        "beta": 0.9,
        "embedding_scale": 2.0,
    },
    # Surprised emotions - high embedding scale for expressiveness
    "realization": {
        "file": "surprised.wav",
        "alpha": 0.5,
        "beta": 0.9,
        "embedding_scale": 2.0,
    },
    "surprise": {
        "file": "surprised.wav",
        "alpha": 0.5,
        "beta": 0.9,
        "embedding_scale": 2.0,
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
