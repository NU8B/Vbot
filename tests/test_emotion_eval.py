"""
Tests for the emotion pipeline evaluation harness.
==================================================
Covers scripts/emotion_eval.py metric math, threshold behavior, sweep, and
gate logic with synthetic predictions (no model download), plus contracts
on the frozen datasets and the runtime label->bucket mapping.
"""

import json
import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from emotion_eval import (
    BUCKETS,
    apply_runtime_threshold,
    check_gate,
    evaluate_buckets,
    load_dataset,
    sweep_thresholds,
)
from utils.emotion_utils import EMOTION_DEFINITIONS, get_emotion_bucket

# The 28 GoEmotions taxonomy labels emitted by the runtime classifier.
GOEMOTIONS_LABELS = {
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
}


def make_items(gold_buckets):
    return [{"text": f"t{i}", "gold_bucket": bucket} for i, bucket in enumerate(gold_buckets)]


class TestMappingContract:
    def test_mapping_covers_exactly_goemotions_taxonomy(self):
        assert set(EMOTION_DEFINITIONS.keys()) == GOEMOTIONS_LABELS, (
            "EMOTION_DEFINITIONS must cover the classifier's 28 labels "
            "exactly; a gap sends real labels to the neutral fallback."
        )

    def test_all_labels_map_to_known_buckets(self):
        for label in GOEMOTIONS_LABELS:
            assert get_emotion_bucket(label) in BUCKETS

    def test_unknown_label_falls_back_to_neutral(self):
        assert get_emotion_bucket("this-is-not-a-label") == "neutral"


class TestThresholdBehavior:
    def test_confident_prediction_kept(self):
        assert apply_runtime_threshold("joy", 0.9, 0.3) == "joy"

    def test_low_confidence_becomes_neutral(self):
        assert apply_runtime_threshold("anger", 0.1, 0.3) == "neutral"

    def test_zero_threshold_keeps_everything(self):
        assert apply_runtime_threshold("anger", 0.01, 0.0) == "anger"


class TestEvaluateBuckets:
    def test_perfect_predictions(self):
        items = make_items(["happy", "sad", "angry", "surprised", "neutral"])
        predictions = [
            ("joy", 0.9),
            ("sadness", 0.9),
            ("anger", 0.9),
            ("surprise", 0.9),
            ("neutral", 0.9),
        ]
        report = evaluate_buckets(items, predictions, threshold=0.3)
        assert report["accuracy"] == 1.0
        assert report["macro_f1"] == 1.0
        assert report["per_bucket"]["happy"]["f1"] == 1.0

    def test_confusion_matrix_counts(self):
        items = make_items(["happy", "happy", "sad"])
        predictions = [("joy", 0.9), ("sadness", 0.9), ("sadness", 0.9)]
        report = evaluate_buckets(items, predictions, threshold=0.3)
        assert report["confusion"]["happy"]["happy"] == 1
        assert report["confusion"]["happy"]["sad"] == 1
        assert report["confusion"]["sad"]["sad"] == 1
        assert report["accuracy"] == pytest.approx(2 / 3)

    def test_threshold_reroutes_low_confidence_to_neutral(self):
        items = make_items(["happy", "neutral"])
        predictions = [("joy", 0.2), ("curiosity", 0.2)]
        report = evaluate_buckets(items, predictions, threshold=0.3)
        # Low-confidence joy becomes neutral: wrong for the happy item,
        # right for the neutral one.
        assert report["confusion"]["happy"]["neutral"] == 1
        assert report["confusion"]["neutral"]["neutral"] == 1

    def test_fine_label_maps_through_runtime_buckets(self):
        # "gratitude" is a happy-bucket label at runtime.
        items = make_items(["happy"])
        report = evaluate_buckets(items, [("gratitude", 0.9)], threshold=0.3)
        assert report["accuracy"] == 1.0


class TestSweep:
    def test_sweep_recommends_best_macro_f1(self):
        # Predictions are correct but low-confidence (0.4): thresholds above
        # 0.4 destroy accuracy, so the recommendation must be <= 0.4.
        items = make_items(["happy", "sad", "angry"])
        predictions = [("joy", 0.4), ("sadness", 0.4), ("anger", 0.4)]
        sweep = sweep_thresholds(items, predictions, [0.0, 0.3, 0.6])
        assert sweep["recommended_threshold"] in (0.0, 0.3)
        by_threshold = {row["threshold"]: row for row in sweep["rows"]}
        assert by_threshold[0.6]["accuracy"] == 0.0
        assert by_threshold[0.3]["accuracy"] == 1.0


class TestGate:
    def make_report(self, macro_f1=0.8, happy_f1=0.8):
        per_bucket = {bucket: {"precision": 0.8, "recall": 0.8, "f1": 0.8, "support": 10} for bucket in BUCKETS}
        per_bucket["happy"]["f1"] = happy_f1
        return {"macro_f1": macro_f1, "per_bucket": per_bucket}

    def test_no_regression_passes(self):
        baseline = {"slice": self.make_report()}
        current = {"slice": self.make_report(macro_f1=0.85)}
        assert check_gate(current, baseline) == []

    def test_macro_f1_regression_fails(self):
        baseline = {"slice": self.make_report(macro_f1=0.80)}
        current = {"slice": self.make_report(macro_f1=0.70)}
        failures = check_gate(current, baseline)
        assert any("macro_f1" in failure for failure in failures)

    def test_per_bucket_regression_fails(self):
        baseline = {"slice": self.make_report(happy_f1=0.9)}
        current = {"slice": self.make_report(happy_f1=0.7)}
        failures = check_gate(current, baseline)
        assert any("happy" in failure for failure in failures)

    def test_tolerance_allows_small_drop(self):
        baseline = {"slice": self.make_report(macro_f1=0.805)}
        current = {"slice": self.make_report(macro_f1=0.800)}
        assert check_gate(current, baseline, tolerance=0.01) == []

    def test_missing_dataset_fails(self):
        baseline = {"slice": self.make_report()}
        assert check_gate({}, baseline)


class TestFrozenDatasets:
    @pytest.mark.parametrize("filename", ["goemotions_test_slice.json", "vtuber_domain_slice.json"])
    def test_dataset_contract(self, filename):
        path = os.path.join(PROJECT_ROOT, "evaluation", "emotion", filename)
        assert os.path.isfile(path), f"frozen dataset missing: {filename}"

        name, items = load_dataset(path)
        assert name
        assert len(items) >= 50
        for item in items:
            assert item["text"].strip()
            assert item["gold_bucket"] in BUCKETS

    def test_domain_slice_covers_all_buckets(self):
        path = os.path.join(PROJECT_ROOT, "evaluation", "emotion", "vtuber_domain_slice.json")
        _, items = load_dataset(path)
        covered = {item["gold_bucket"] for item in items}
        assert covered == set(BUCKETS)

    def test_goemotions_slice_gold_labels_map_consistently(self):
        path = os.path.join(PROJECT_ROOT, "evaluation", "emotion", "goemotions_test_slice.json")
        _, items = load_dataset(path)
        for item in items:
            assert get_emotion_bucket(item["gold_label"]) == item["gold_bucket"]
