from transformers import pipeline
import torch

# Emotion model settings
EMOTION_MODEL_NAME = "SamLowe/roberta-base-go_emotions"

DIFFUSION_STEPS = 10  # Adjust this to trade off quality vs speed


ALPHA = 0.3
BETA = 0.7
EMBEDDING_SCALE = 1.0

ANGRY_ALPHA = 0.4
ANGRY_BETA = 0.7
ANGRY_EMBEDDING_SCALE = 1.5

HAPPY_ALPHA = 0.6
HAPPY_BETA = 0.7
HAPPY_EMBEDDING_SCALE = 1.5

SAD_ALPHA = 0.7
SAD_BETA = 0.9
SAD_EMBEDDING_SCALE = 2.0

SURPRISE_ALPHA = 0.4
SURPRISE_BETA = 0.7
SURPRISE_EMBEDDING_SCALE = 1.5

# Speed settings for different emotion types
NEUTRAL_SPEED = 1.05  # Normal speed
ANGRY_SPEED = 1.3  # 25% faster
HAPPY_SPEED = 1.15  # 15% faster
SAD_SPEED = 0.85  # 15% slower
SURPRISE_SPEED = 1.4  # 20% faster

# Emotion to voice style mapping with inference parameters
EMOTION_CONFIG = {
    # Format: "emotion": {"file": "style_file.wav", "alpha": float, "beta": float, "embedding_scale": float, "speed": float}
    # Neutral emotions - using neutral.wav
    "neutral": {
        "file": "neutral.wav",
        "alpha": ALPHA,
        "beta": BETA,
        "embedding_scale": EMBEDDING_SCALE,
        "speed": NEUTRAL_SPEED,
    },
    "confusion": {
        "file": "neutral.wav",
        "alpha": ALPHA,
        "beta": BETA,
        "embedding_scale": EMBEDDING_SCALE,
        "speed": NEUTRAL_SPEED,
    },
    "caring": {
        "file": "neutral.wav",
        "alpha": ALPHA,
        "beta": BETA,
        "embedding_scale": EMBEDDING_SCALE,
        "speed": NEUTRAL_SPEED,
    },
    "curiosity": {
        "file": "neutral.wav",
        "alpha": ALPHA,
        "beta": BETA,
        "embedding_scale": EMBEDDING_SCALE,
        "speed": NEUTRAL_SPEED,
    },
    "desire": {
        "file": "neutral.wav",
        "alpha": ALPHA,
        "beta": BETA,
        "embedding_scale": EMBEDDING_SCALE,
        "speed": NEUTRAL_SPEED,
    },
    "relief": {
        "file": "neutral.wav",
        "alpha": ALPHA,
        "beta": BETA,
        "embedding_scale": EMBEDDING_SCALE,
        "speed": NEUTRAL_SPEED,
    },
    # Happy emotions - using happy.wav
    "admiration": {
        "file": "happy.wav",
        "alpha": HAPPY_ALPHA,
        "beta": HAPPY_BETA,
        "embedding_scale": HAPPY_EMBEDDING_SCALE,
        "speed": HAPPY_SPEED,
    },
    "amusement": {
        "file": "happy.wav",
        "alpha": HAPPY_ALPHA,
        "beta": HAPPY_BETA,
        "embedding_scale": HAPPY_EMBEDDING_SCALE,
        "speed": HAPPY_SPEED,
    },
    "approval": {
        "file": "happy.wav",
        "alpha": HAPPY_ALPHA,
        "beta": HAPPY_BETA,
        "embedding_scale": HAPPY_EMBEDDING_SCALE,
        "speed": HAPPY_SPEED,
    },
    "excitement": {
        "file": "happy.wav",
        "alpha": HAPPY_ALPHA,
        "beta": HAPPY_BETA,
        "embedding_scale": HAPPY_EMBEDDING_SCALE,
        "speed": HAPPY_SPEED,
    },
    "gratitude": {
        "file": "happy.wav",
        "alpha": HAPPY_ALPHA,
        "beta": HAPPY_BETA,
        "embedding_scale": HAPPY_EMBEDDING_SCALE,
        "speed": HAPPY_SPEED,
    },
    "joy": {
        "file": "happy.wav",
        "alpha": HAPPY_ALPHA,
        "beta": HAPPY_BETA,
        "embedding_scale": HAPPY_EMBEDDING_SCALE,
        "speed": HAPPY_SPEED,
    },
    "love": {
        "file": "happy.wav",
        "alpha": HAPPY_ALPHA,
        "beta": HAPPY_BETA,
        "embedding_scale": HAPPY_EMBEDDING_SCALE,
        "speed": HAPPY_SPEED,
    },
    "optimism": {
        "file": "happy.wav",
        "alpha": HAPPY_ALPHA,
        "beta": HAPPY_BETA,
        "embedding_scale": HAPPY_EMBEDDING_SCALE,
        "speed": HAPPY_SPEED,
    },
    "pride": {
        "file": "happy.wav",
        "alpha": HAPPY_ALPHA,
        "beta": HAPPY_BETA,
        "embedding_scale": HAPPY_EMBEDDING_SCALE,
        "speed": HAPPY_SPEED,
    },
    # Sad emotions - using sad.wav
    "disappointment": {
        "file": "sad.wav",
        "alpha": SAD_ALPHA,
        "beta": SAD_BETA,
        "embedding_scale": SAD_EMBEDDING_SCALE,
        "speed": SAD_SPEED,
    },
    "embarrassment": {
        "file": "sad.wav",
        "alpha": SAD_ALPHA,
        "beta": SAD_BETA,
        "embedding_scale": SAD_EMBEDDING_SCALE,
        "speed": SAD_SPEED,
    },
    "fear": {
        "file": "sad.wav",
        "alpha": SAD_ALPHA,
        "beta": SAD_BETA,
        "embedding_scale": SAD_EMBEDDING_SCALE,
        "speed": SAD_SPEED,
    },
    "grief": {
        "file": "sad.wav",
        "alpha": SAD_ALPHA,
        "beta": SAD_BETA,
        "embedding_scale": SAD_EMBEDDING_SCALE,
        "speed": SAD_SPEED,
    },
    "nervousness": {
        "file": "sad.wav",
        "alpha": SAD_ALPHA,
        "beta": SAD_BETA,
        "embedding_scale": SAD_EMBEDDING_SCALE,
        "speed": SAD_SPEED,
    },
    "remorse": {
        "file": "sad.wav",
        "alpha": SAD_ALPHA,
        "beta": SAD_BETA,
        "embedding_scale": SAD_EMBEDDING_SCALE,
        "speed": SAD_SPEED,
    },
    "sadness": {
        "file": "sad.wav",
        "alpha": SAD_ALPHA,
        "beta": SAD_BETA,
        "embedding_scale": SAD_EMBEDDING_SCALE,
        "speed": SAD_SPEED,
    },
    # Angry emotions - using angry.wav
    "disapproval": {
        "file": "angry.wav",
        "alpha": ANGRY_ALPHA,
        "beta": ANGRY_BETA,
        "embedding_scale": ANGRY_EMBEDDING_SCALE,
        "speed": ANGRY_SPEED,
    },
    "disgust": {
        "file": "angry.wav",
        "alpha": ANGRY_ALPHA,
        "beta": ANGRY_BETA,
        "embedding_scale": ANGRY_EMBEDDING_SCALE,
        "speed": ANGRY_SPEED,
    },
    "anger": {
        "file": "angry.wav",
        "alpha": ANGRY_ALPHA,
        "beta": ANGRY_BETA,
        "embedding_scale": ANGRY_EMBEDDING_SCALE,
        "speed": ANGRY_SPEED,
    },
    "annoyance": {
        "file": "angry.wav",
        "alpha": ANGRY_ALPHA,
        "beta": ANGRY_BETA,
        "embedding_scale": ANGRY_EMBEDDING_SCALE,
        "speed": ANGRY_SPEED,
    },
    # Surprised emotions - using surprised.wav
    "realization": {
        "file": "surprised.wav",
        "alpha": SURPRISE_ALPHA,
        "beta": SURPRISE_BETA,
        "embedding_scale": SURPRISE_EMBEDDING_SCALE,
        "speed": SURPRISE_SPEED,
    },
    "surprise": {
        "file": "surprised.wav",
        "alpha": SURPRISE_ALPHA,
        "beta": SURPRISE_BETA,
        "embedding_scale": SURPRISE_EMBEDDING_SCALE,
        "speed": SURPRISE_SPEED,
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
        if confidence < 0.3:
            emotion = "neutral"

        return emotion

    def get_style_for_emotion(self, emotion):
        """Get the corresponding voice style file for an emotion."""
        return EMOTION_MAPPING.get(emotion, "neutral.wav")
