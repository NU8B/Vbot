"""
Tests for shared emotion classifier ownership.
==============================================
Pins the consolidation from 2026-07-06: every EmotionHandler must reuse one
process-wide classifier pipeline (the duplicate-instance pattern cost
~3.4 GB RAM / ~12s at startup). The pipeline factory is stubbed so CI never
downloads the RoBERTa model.
"""

import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils import emotion_utils


class FakePipeline:
    """Stands in for the transformers pipeline; records calls."""

    instances = 0

    def __init__(self):
        FakePipeline.instances += 1
        self.calls = []

    def __call__(self, text):
        self.calls.append(text)
        return [[{"label": "joy", "score": 0.9}]]


@pytest.fixture
def stubbed_classifier(monkeypatch):
    """Reset the singleton and replace the pipeline factory with a stub."""
    FakePipeline.instances = 0
    monkeypatch.setattr(emotion_utils, "_shared_classifier", None)
    monkeypatch.setattr(emotion_utils, "pipeline", lambda *args, **kwargs: FakePipeline())
    yield
    # monkeypatch restores attributes; ensure no stub leaks to other tests.
    emotion_utils._shared_classifier = None


class TestSharedClassifier:
    def test_handlers_share_one_classifier(self, stubbed_classifier):
        first = emotion_utils.EmotionHandler(model_name="Amelia")
        second = emotion_utils.EmotionHandler(model_name="Gura")
        third = emotion_utils.EmotionHandler(model_name="Wilson")

        assert first.emotion_classifier is second.emotion_classifier
        assert second.emotion_classifier is third.emotion_classifier
        assert FakePipeline.instances == 1

    def test_warmup_happens_once_total(self, stubbed_classifier):
        emotion_utils.EmotionHandler(model_name="Amelia")
        emotion_utils.EmotionHandler(model_name="Gura")

        classifier = emotion_utils.get_shared_emotion_classifier()
        warmups = [call for call in classifier.calls if "warmup" in call.lower()]
        assert len(warmups) == 1

    def test_handlers_keep_character_specific_config(self, stubbed_classifier):
        amelia = emotion_utils.EmotionHandler(model_name="Amelia")
        gura = emotion_utils.EmotionHandler(model_name="Gura")

        assert amelia.emotion_config["joy"]["file"]["Amelia"] == "Amelia/happy.wav"
        assert gura.emotion_config["joy"]["file"]["Gura"] == "Gura/happy.wav"

    def test_classification_flow_unchanged(self, stubbed_classifier):
        handler = emotion_utils.EmotionHandler(model_name="Amelia")
        emotion = handler.classify_emotion("What a wonderful day!")
        assert emotion == "joy"
        assert handler.get_last_confidence() == pytest.approx(0.9)
