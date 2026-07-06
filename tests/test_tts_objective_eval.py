"""
Tests for the TTS objective-metrics command.
============================================
Covers scripts/tts_objective_eval.py pure logic (text normalization, WER,
aggregation, battery contract) and the promotion gate's artifact
unwrapping. No models, audio, or network.
"""

import json
import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "new_tts_eval_form"))

from tts_objective_eval import (
    BATTERY_VERSION,
    SENTENCE_BATTERY,
    aggregate_metrics,
    normalize_text,
    word_error_rate,
)


class TestNormalizeText:
    def test_lowercase_and_punctuation_stripped(self):
        assert normalize_text("Hello, World! It's fine.") == ["hello", "world", "it's", "fine"]

    def test_empty(self):
        assert normalize_text("") == []


class TestWordErrorRate:
    def test_perfect_transcript(self):
        assert word_error_rate("the cat sat", "The cat sat.") == 0.0

    def test_one_substitution(self):
        assert word_error_rate("the cat sat", "the bat sat") == pytest.approx(1 / 3)

    def test_deletion(self):
        assert word_error_rate("the cat sat down", "the cat sat") == pytest.approx(1 / 4)

    def test_insertion(self):
        assert word_error_rate("the cat sat", "the big cat sat") == pytest.approx(1 / 3)

    def test_everything_wrong(self):
        assert word_error_rate("alpha beta", "gamma delta") == 1.0

    def test_empty_reference_with_hypothesis(self):
        assert word_error_rate("", "something") == 1.0

    def test_empty_both(self):
        assert word_error_rate("", "") == 0.0

    def test_wer_can_exceed_one_on_long_hypothesis(self):
        # WER is unbounded above; a rambling transcript must not be hidden.
        assert word_error_rate("hi", "one two three four") > 1.0


class TestAggregation:
    def test_means(self):
        per_sentence = [
            {"speaker_similarity": 0.8, "wer": 0.1},
            {"speaker_similarity": 0.6, "wer": 0.3},
        ]
        metrics = aggregate_metrics(per_sentence)
        assert metrics["speaker_similarity"] == pytest.approx(0.7)
        assert metrics["wer"] == pytest.approx(0.2)

    def test_empty(self):
        assert aggregate_metrics([]) == {}


class TestBatteryContract:
    def test_battery_is_versioned_and_stable_size(self):
        assert BATTERY_VERSION == 1
        assert len(SENTENCE_BATTERY) == 10

    def test_battery_sentences_are_speakable(self):
        for sentence in SENTENCE_BATTERY:
            assert sentence.strip()
            assert sentence[-1] in ".!?"
            assert "*" not in sentence


class TestGateUnwrapping:
    def test_gate_reads_full_artifact(self, tmp_path):
        from promotion_gate import _load_json

        artifact = {
            "schema_version": 1,
            "kind": "tts_objective_eval",
            "gate_metrics": {"speaker_similarity": 0.82, "wer": 0.07},
        }
        path = tmp_path / "artifact.json"
        path.write_text(json.dumps(artifact))
        assert _load_json(str(path)) == {"speaker_similarity": 0.82, "wer": 0.07}

    def test_gate_still_reads_flat_dict(self, tmp_path):
        from promotion_gate import _load_json

        path = tmp_path / "flat.json"
        path.write_text(json.dumps({"speaker_similarity": 0.9}))
        assert _load_json(str(path)) == {"speaker_similarity": 0.9}

    def test_end_to_end_gate_regression_via_artifacts(self, tmp_path):
        # Candidate similarity drops below baseline -> gate must fail.
        from promotion_gate import evaluate_promotion

        submissions = [
            {
                "evaluations": [
                    {
                        "model": "Gura",
                        "model_type": "new",
                        "true_emotion": "joy",
                        "selected_emotion": "joy",
                        "naturalness": 4,
                    }
                ]
            }
        ]
        report = evaluate_promotion(
            submissions,
            objective_candidate={"speaker_similarity": 0.70, "wer": 0.05},
            objective_baseline={"speaker_similarity": 0.85, "wer": 0.05},
        )
        assert report["passed"] is False
