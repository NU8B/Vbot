from transformers import pipeline
import torch

# Emotion model settings
EMOTION_MODEL_NAME = "SamLowe/roberta-base-go_emotions"

DIFFUSION_STEPS = 3  # Further reduced for much faster TTS generation (was 5)

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
    "Shiori": {
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
    "Wilson": {
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

    # Define available character models and their fallbacks
    AVAILABLE_CHARACTERS = ["Amelia", "Eveland", "Gura", "Shiori", "Wilson"]

    # Centralized emotion definitions with audio file mappings
    # Format: emotion_name: (audio_type, alpha_param, beta_param, embedding_param)
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

    # Character audio file mapping with fallbacks for missing characters
    CHARACTER_AUDIO_MAPPING = {
        "Amelia": {
            "neutral": "Amelia/neutral.wav",
            "happy": "Amelia/happy.wav",
            "sad": "Amelia/sad.wav",
            "angry": "Amelia/angry.wav",
            "surprised": "Amelia/surprised.wav",
        },
        "Eveland": {
            "neutral": "Eveland/neutral.wav",
            "happy": "Eveland/happy.wav",
            "sad": "Eveland/sad.wav",
            "angry": "Eveland/angry.wav",
            "surprised": "Eveland/surprised.wav",
        },
        "Gura": {
            "neutral": "Gura/neutral.wav",
            "happy": "Gura/happy.wav",
            "sad": "Gura/sad.wav",
            "angry": "Gura/angry.wav",
            "surprised": "Gura/surprised.wav",
        },
        "Shiori": {
            # PLACEHOLDER: Using Amelia's audio until Shiori's audio files are available
            "neutral": "Amelia/neutral.wav",
            "happy": "Amelia/happy.wav",
            "sad": "Amelia/sad.wav",
            "angry": "Amelia/angry.wav",
            "surprised": "Amelia/surprised.wav",
        },
        "Wilson": {
            "neutral": "Wilson/neutral.wav",
            "happy": "Wilson/happy.wav",
            "sad": "Wilson/sad.wav",
            "angry": "Wilson/angry.wav",
            "surprised": "Wilson/surprised.wav",
        },
    }

    # Build the emotion config dynamically
    emotion_config = {}

    for emotion_name, (
        audio_type,
        alpha_param,
        beta_param,
        embedding_param,
    ) in EMOTION_DEFINITIONS.items():
        # Build file mapping for all characters
        file_mapping = {}
        for character in AVAILABLE_CHARACTERS:
            if character in CHARACTER_AUDIO_MAPPING:
                file_mapping[character] = CHARACTER_AUDIO_MAPPING[character][audio_type]
            else:
                # Fallback to Amelia if character not found
                file_mapping[character] = CHARACTER_AUDIO_MAPPING["Amelia"][audio_type]

        # Add emotion configuration
        emotion_config[emotion_name] = {
            "file": file_mapping,
            "alpha": params[alpha_param],
            "beta": params[beta_param],
            "embedding_scale": params[embedding_param],
            "speed": 1.0,
        }

    return emotion_config


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


def add_new_character(character_name, audio_files=None, use_fallback=True):
    """
    Utility function to easily add a new character to the system.

    Args:
        character_name (str): Name of the new character
        audio_files (dict): Optional dictionary with audio file mappings
                           Format: {"neutral": "path/to/neutral.wav", "happy": "path/to/happy.wav", ...}
        use_fallback (bool): If True, uses Amelia's audio files as fallback for missing files

    Example:
        # Add new character with all audio files
        add_new_character("Ina", {
            "neutral": "Ina/neutral.wav",
            "happy": "Ina/happy.wav",
            "sad": "Ina/sad.wav",
            "angry": "Ina/angry.wav",
            "surprised": "Ina/surprised.wav"
        })

        # Add new character using Amelia's audio as placeholder
        add_new_character("Ina", use_fallback=True)
    """
    # This function is for documentation purposes - the actual character addition
    # should be done by modifying the CHARACTER_AUDIO_MAPPING and AVAILABLE_CHARACTERS
    # in the create_emotion_config function above.

    print(
        f"""
To add character '{character_name}':

1. Add '{character_name}' to AVAILABLE_CHARACTERS list in create_emotion_config()
2. Add entry to CHARACTER_AUDIO_MAPPING:
   "{character_name}": {{
       "neutral": "{character_name}/neutral.wav" if audio_files else "Amelia/neutral.wav",
       "happy": "{character_name}/happy.wav" if audio_files else "Amelia/happy.wav", 
       "sad": "{character_name}/sad.wav" if audio_files else "Amelia/sad.wav",
       "angry": "{character_name}/angry.wav" if audio_files else "Amelia/angry.wav",
       "surprised": "{character_name}/surprised.wav" if audio_files else "Amelia/surprised.wav"
   }}
3. Add character parameters to MODEL_PARAMS at the top of the file
4. That's it! All emotions will be automatically configured.
"""
    )

    return True
