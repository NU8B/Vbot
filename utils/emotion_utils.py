from transformers import pipeline
import torch

# Emotion model settings
EMOTION_MODEL_NAME = "SamLowe/roberta-base-go_emotions"

DIFFUSION_STEPS = 20  # Adjust this to trade off quality vs speed

# Model-specific parameters
MODEL_PARAMS = {
    "Amelia": {
        "ALPHA": 0.3,
        "BETA": 0.7,
        "EMBEDDING_SCALE": 0.4,
        "ANGRY_ALPHA": 0.3,
        "ANGRY_BETA": 0.7,
        "ANGRY_EMBEDDING_SCALE": 0.5,
        "HAPPY_ALPHA": 0.3,
        "HAPPY_BETA": 0.7,
        "HAPPY_EMBEDDING_SCALE": 0.5,
        "SAD_ALPHA": 0.3,
        "SAD_BETA": 0.7,
        "SAD_EMBEDDING_SCALE": 0.5,
        "SURPRISE_ALPHA": 0.3,
        "SURPRISE_BETA": 0.7,
        "SURPRISE_EMBEDDING_SCALE": 0.5,
    },
    "Eveland": {
        "ALPHA": 0.3,
        "BETA": 0.7,
        "EMBEDDING_SCALE": 0.5,
        "ANGRY_ALPHA": 0.3,
        "ANGRY_BETA": 0.7,
        "ANGRY_EMBEDDING_SCALE": 0.5,
        "HAPPY_ALPHA": 0.3,
        "HAPPY_BETA": 0.7,
        "HAPPY_EMBEDDING_SCALE": 0.5,
        "SAD_ALPHA": 0.3,
        "SAD_BETA": 0.7,
        "SAD_EMBEDDING_SCALE": 0.5,
        "SURPRISE_ALPHA": 0.3,
        "SURPRISE_BETA": 0.7,
        "SURPRISE_EMBEDDING_SCALE": 0.5,
    },
    "Gura": {
        "ALPHA": 0.3,
        "BETA": 0.7,
        "EMBEDDING_SCALE": 0.4,
        "ANGRY_ALPHA": 0.3,
        "ANGRY_BETA": 0.7,
        "ANGRY_EMBEDDING_SCALE": 0.5,
        "HAPPY_ALPHA": 0.3,
        "HAPPY_BETA": 0.7,
        "HAPPY_EMBEDDING_SCALE": 0.5,
        "SAD_ALPHA": 0.3,
        "SAD_BETA": 0.7,
        "SAD_EMBEDDING_SCALE": 0.5,
        "SURPRISE_ALPHA": 0.3,
        "SURPRISE_BETA": 0.7,
        "SURPRISE_EMBEDDING_SCALE": 0.5,
    },
}


# Helper function to get model-specific parameters
def get_model_params(model_name="Amelia"):
    return MODEL_PARAMS[model_name]


# Emotion to voice style mapping with inference parameters
def create_emotion_config(model_name="Amelia"):
    params = get_model_params(model_name)

    return {
        # Format: "emotion": {"file": {"model_name": "file.wav"}, "alpha": float, "beta": float, "embedding_scale": float, "speed": float}
        # Neutral emotions
        "neutral": {
            "file": {
                "Amelia": "Amelia/neutral.wav",
                "Eveland": "Eveland/neutral.wav",
                "Gura": "Gura/neutral.wav",
            },
            "alpha": params["ALPHA"],
            "beta": params["BETA"],
            "embedding_scale": params["EMBEDDING_SCALE"],
        },
        "confusion": {
            "file": {
                "Amelia": "Amelia/neutral.wav",
                "Eveland": "Eveland/neutral.wav",
                "Gura": "Gura/neutral.wav",
            },
            "alpha": params["ALPHA"],
            "beta": params["BETA"],
            "embedding_scale": params["EMBEDDING_SCALE"],
        },
        "caring": {
            "file": {
                "Amelia": "Amelia/neutral.wav",
                "Eveland": "Eveland/neutral.wav",
                "Gura": "Gura/neutral.wav",
            },
            "alpha": params["ALPHA"],
            "beta": params["BETA"],
            "embedding_scale": params["EMBEDDING_SCALE"],
        },
        "curiosity": {
            "file": {
                "Amelia": "Amelia/neutral.wav",
                "Eveland": "Eveland/neutral.wav",
                "Gura": "Gura/neutral.wav",
            },
            "alpha": params["ALPHA"],
            "beta": params["BETA"],
            "embedding_scale": params["EMBEDDING_SCALE"],
        },
        "desire": {
            "file": {
                "Amelia": "Amelia/neutral.wav",
                "Eveland": "Eveland/neutral.wav",
                "Gura": "Gura/neutral.wav",
            },
            "alpha": params["ALPHA"],
            "beta": params["BETA"],
            "embedding_scale": params["EMBEDDING_SCALE"],
        },
        "relief": {
            "file": {
                "Amelia": "Amelia/neutral.wav",
                "Eveland": "Eveland/neutral.wav",
                "Gura": "Gura/neutral.wav",
            },
            "alpha": params["ALPHA"],
            "beta": params["BETA"],
            "embedding_scale": params["EMBEDDING_SCALE"],
        },
        # Happy emotions
        "admiration": {
            "file": {
                "Amelia": "Amelia/happy.wav",
                "Eveland": "Eveland/happy.wav",
                "Gura": "Gura/happy.wav",
            },
            "alpha": params["HAPPY_ALPHA"],
            "beta": params["HAPPY_BETA"],
            "embedding_scale": params["HAPPY_EMBEDDING_SCALE"],
        },
        "amusement": {
            "file": {
                "Amelia": "Amelia/happy.wav",
                "Eveland": "Eveland/happy.wav",
                "Gura": "Gura/happy.wav",
            },
            "alpha": params["HAPPY_ALPHA"],
            "beta": params["HAPPY_BETA"],
            "embedding_scale": params["HAPPY_EMBEDDING_SCALE"],
        },
        "approval": {
            "file": {
                "Amelia": "Amelia/happy.wav",
                "Eveland": "Eveland/happy.wav",
                "Gura": "Gura/happy.wav",
            },
            "alpha": params["HAPPY_ALPHA"],
            "beta": params["HAPPY_BETA"],
            "embedding_scale": params["HAPPY_EMBEDDING_SCALE"],
        },
        "excitement": {
            "file": {
                "Amelia": "Amelia/happy.wav",
                "Eveland": "Eveland/happy.wav",
                "Gura": "Gura/happy.wav",
            },
            "alpha": params["HAPPY_ALPHA"],
            "beta": params["HAPPY_BETA"],
            "embedding_scale": params["HAPPY_EMBEDDING_SCALE"],
        },
        "gratitude": {
            "file": {
                "Amelia": "Amelia/happy.wav",
                "Eveland": "Eveland/happy.wav",
                "Gura": "Gura/happy.wav",
            },
            "alpha": params["HAPPY_ALPHA"],
            "beta": params["HAPPY_BETA"],
            "embedding_scale": params["HAPPY_EMBEDDING_SCALE"],
        },
        "joy": {
            "file": {
                "Amelia": "Amelia/happy.wav",
                "Eveland": "Eveland/happy.wav",
                "Gura": "Gura/happy.wav",
            },
            "alpha": params["HAPPY_ALPHA"],
            "beta": params["HAPPY_BETA"],
            "embedding_scale": params["HAPPY_EMBEDDING_SCALE"],
        },
        "love": {
            "file": {
                "Amelia": "Amelia/happy.wav",
                "Eveland": "Eveland/happy.wav",
                "Gura": "Gura/happy.wav",
            },
            "alpha": params["HAPPY_ALPHA"],
            "beta": params["HAPPY_BETA"],
            "embedding_scale": params["HAPPY_EMBEDDING_SCALE"],
        },
        "optimism": {
            "file": {
                "Amelia": "Amelia/happy.wav",
                "Eveland": "Eveland/happy.wav",
                "Gura": "Gura/happy.wav",
            },
            "alpha": params["HAPPY_ALPHA"],
            "beta": params["HAPPY_BETA"],
            "embedding_scale": params["HAPPY_EMBEDDING_SCALE"],
        },
        "pride": {
            "file": {
                "Amelia": "Amelia/happy.wav",
                "Eveland": "Eveland/happy.wav",
                "Gura": "Gura/happy.wav",
            },
            "alpha": params["HAPPY_ALPHA"],
            "beta": params["HAPPY_BETA"],
            "embedding_scale": params["HAPPY_EMBEDDING_SCALE"],
        },
        # Sad emotions
        "disappointment": {
            "file": {
                "Amelia": "Amelia/sad.wav",
                "Eveland": "Eveland/sad.wav",
                "Gura": "Gura/sad.wav",
            },
            "alpha": params["SAD_ALPHA"],
            "beta": params["SAD_BETA"],
            "embedding_scale": params["SAD_EMBEDDING_SCALE"],
        },
        "embarrassment": {
            "file": {
                "Amelia": "Amelia/sad.wav",
                "Eveland": "Eveland/sad.wav",
                "Gura": "Gura/sad.wav",
            },
            "alpha": params["SAD_ALPHA"],
            "beta": params["SAD_BETA"],
            "embedding_scale": params["SAD_EMBEDDING_SCALE"],
        },
        "fear": {
            "file": {
                "Amelia": "Amelia/sad.wav",
                "Eveland": "Eveland/sad.wav",
                "Gura": "Gura/sad.wav",
            },
            "alpha": params["SAD_ALPHA"],
            "beta": params["SAD_BETA"],
            "embedding_scale": params["SAD_EMBEDDING_SCALE"],
        },
        "grief": {
            "file": {
                "Amelia": "Amelia/sad.wav",
                "Eveland": "Eveland/sad.wav",
                "Gura": "Gura/sad.wav",
            },
            "alpha": params["SAD_ALPHA"],
            "beta": params["SAD_BETA"],
            "embedding_scale": params["SAD_EMBEDDING_SCALE"],
        },
        "nervousness": {
            "file": {
                "Amelia": "Amelia/sad.wav",
                "Eveland": "Eveland/sad.wav",
                "Gura": "Gura/sad.wav",
            },
            "alpha": params["SAD_ALPHA"],
            "beta": params["SAD_BETA"],
            "embedding_scale": params["SAD_EMBEDDING_SCALE"],
        },
        "remorse": {
            "file": {
                "Amelia": "Amelia/sad.wav",
                "Eveland": "Eveland/sad.wav",
                "Gura": "Gura/sad.wav",
            },
            "alpha": params["SAD_ALPHA"],
            "beta": params["SAD_BETA"],
            "embedding_scale": params["SAD_EMBEDDING_SCALE"],
        },
        "sadness": {
            "file": {
                "Amelia": "Amelia/sad.wav",
                "Eveland": "Eveland/sad.wav",
                "Gura": "Gura/sad.wav",
            },
            "alpha": params["SAD_ALPHA"],
            "beta": params["SAD_BETA"],
            "embedding_scale": params["SAD_EMBEDDING_SCALE"],
        },
        # Angry emotions
        "disapproval": {
            "file": {
                "Amelia": "Amelia/angry.wav",
                "Eveland": "Eveland/angry.wav",
                "Gura": "Gura/angry.wav",
            },
            "alpha": params["ANGRY_ALPHA"],
            "beta": params["ANGRY_BETA"],
            "embedding_scale": params["ANGRY_EMBEDDING_SCALE"],
        },
        "disgust": {
            "file": {
                "Amelia": "Amelia/angry.wav",
                "Eveland": "Eveland/angry.wav",
                "Gura": "Gura/angry.wav",
            },
            "alpha": params["ANGRY_ALPHA"],
            "beta": params["ANGRY_BETA"],
            "embedding_scale": params["ANGRY_EMBEDDING_SCALE"],
        },
        "anger": {
            "file": {
                "Amelia": "Amelia/angry.wav",
                "Eveland": "Eveland/angry.wav",
                "Gura": "Gura/angry.wav",
            },
            "alpha": params["ANGRY_ALPHA"],
            "beta": params["ANGRY_BETA"],
            "embedding_scale": params["ANGRY_EMBEDDING_SCALE"],
        },
        "annoyance": {
            "file": {
                "Amelia": "Amelia/angry.wav",
                "Eveland": "Eveland/angry.wav",
                "Gura": "Gura/angry.wav",
            },
            "alpha": params["ANGRY_ALPHA"],
            "beta": params["ANGRY_BETA"],
            "embedding_scale": params["ANGRY_EMBEDDING_SCALE"],
        },
        # Surprised emotions
        "realization": {
            "file": {
                "Amelia": "Amelia/surprised.wav",
                "Eveland": "Eveland/surprised.wav",
                "Gura": "Gura/surprised.wav",
            },
            "alpha": params["SURPRISE_ALPHA"],
            "beta": params["SURPRISE_BETA"],
            "embedding_scale": params["SURPRISE_EMBEDDING_SCALE"],
        },
        "surprise": {
            "file": {
                "Amelia": "Amelia/surprised.wav",
                "Eveland": "Eveland/surprised.wav",
                "Gura": "Gura/surprised.wav",
            },
            "alpha": params["SURPRISE_ALPHA"],
            "beta": params["SURPRISE_BETA"],
            "embedding_scale": params["SURPRISE_EMBEDDING_SCALE"],
        },
    }


# Initialize EMOTION_CONFIG with default model (Amelia)
EMOTION_CONFIG = create_emotion_config("Amelia")


# For backward compatibility - now requires model name
def get_emotion_file(emotion, model_name="Amelia"):
    return EMOTION_CONFIG[emotion]["file"][model_name]


# Update EmotionHandler to support multiple models
class EmotionHandler:
    def __init__(self, model_name="Amelia"):
        self.model_name = model_name
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

        # Create model-specific emotion config
        self.emotion_config = create_emotion_config(model_name)

        # Warm up classifier silently
        self.classify_emotion("Test message for warmup.")

    def classify_emotion(self, text):
        """Classify the emotion of the given text using RoBERTa."""
        result = self.emotion_classifier(text)
        emotion = result[0][0]["label"]
        confidence = result[0][0]["score"]

        # Store confidence for later use
        self._last_confidence = confidence

        # Use neutral for low confidence predictions
        if confidence < 0.3:
            emotion = "neutral"

        return emotion

    def get_style_for_emotion(self, emotion):
        """Get the corresponding voice style file for an emotion."""
        return self.emotion_config[emotion]["file"][self.model_name]

    def get_last_confidence(self):
        """Return the confidence score of the last emotion classification"""
        return getattr(self, "_last_confidence", 0.3)

    def get_base_emotion(self, emotion):
        """Get the base emotion category"""
        emotion_categories = {
            "happy": [
                "admiration",
                "amusement",
                "approval",
                "excitement",
                "gratitude",
                "joy",
                "love",
                "optimism",
                "pride",
            ],
            "sad": [
                "disappointment",
                "embarrassment",
                "fear",
                "grief",
                "nervousness",
                "remorse",
                "sadness",
            ],
            "angry": ["disapproval", "disgust", "anger", "annoyance"],
            "surprise": ["realization", "surprise"],
            "neutral": [
                "neutral",
                "confusion",
                "caring",
                "curiosity",
                "desire",
                "relief",
            ],
        }

        for base, variants in emotion_categories.items():
            if emotion in variants:
                return base
        return "neutral"


# Emotion mapping dictionary - maps emotions to style file names (without .wav extension)
EMOTION_MAPPING = {
    # Core emotions
    "joy": "happy",
    "happy": "happy",
    "sadness": "sad",
    "sad": "sad",
    "anger": "angry",
    "angry": "angry",
    "surprise": "surprised",
    "surprised": "surprised",
    "neutral": "neutral",
    # Additional emotion mappings
    "fear": "surprised",
    "disgust": "angry",
    "confused": "surprised",
    "excited": "happy",
    "frustrated": "angry",
    "worried": "sad",
    "annoyed": "angry",
    "pleased": "happy",
    "disappointed": "sad",
}
