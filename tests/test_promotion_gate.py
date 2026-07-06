"""
Tests for the TTS model-promotion gate.
=======================================
Covers new_tts_eval_form/promotion_gate.py: human-eval aggregation,
threshold checks, objective no-regression checks, and CLI exit codes.
"""

import json
import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "new_tts_eval_form"))

from promotion_gate import aggregate_human_eval, evaluate_promotion, main


def make_submissions(correct=8, incorrect=2, naturalness=4, model_type="new"):
    """Build a stored artifact with a given accuracy and naturalness."""
    evaluations = []
    for _ in range(correct):
        evaluations.append(
            {
                "model": "Gura",
                "model_type": model_type,
                "true_emotion": "joy",
                "selected_emotion": "joy",
                "naturalness": naturalness,
            }
        )
    for _ in range(incorrect):
        evaluations.append(
            {
                "model": "Gura",
                "model_type": model_type,
                "true_emotion": "joy",
                "selected_emotion": "anger",
                "naturalness": naturalness,
            }
        )
    return [{"schema_version": 1, "evaluations": evaluations}]


class TestAggregation:
    def test_accuracy_and_naturalness(self):
        stats = aggregate_human_eval(make_submissions(correct=3, incorrect=1, naturalness=4))
        assert stats["total"] == 4
        assert stats["accuracy"] == 75.0
        assert stats["naturalness_avg"] == 4.0

    def test_other_model_type_ignored(self):
        submissions = make_submissions(model_type="old")
        stats = aggregate_human_eval(submissions, model_type="new")
        assert stats["total"] == 0
        assert stats["accuracy"] is None

    def test_missing_naturalness_not_counted(self):
        submissions = make_submissions(correct=2, incorrect=0, naturalness=None)
        for evaluation in submissions[0]["evaluations"]:
            del evaluation["naturalness"]
        stats = aggregate_human_eval(submissions)
        assert stats["naturalness_count"] == 0
        assert stats["naturalness_avg"] is None


class TestPromotionChecks:
    def test_passing_candidate_promotes(self):
        report = evaluate_promotion(make_submissions(correct=8, incorrect=2))
        assert report["passed"] is True

    def test_low_accuracy_blocks_promotion(self):
        report = evaluate_promotion(make_submissions(correct=2, incorrect=8))
        assert report["passed"] is False
        failed = [c for c in report["checks"] if not c["passed"]]
        assert any("emotion accuracy" in c["name"] for c in failed)

    def test_low_naturalness_blocks_promotion(self):
        report = evaluate_promotion(make_submissions(naturalness=2))
        assert report["passed"] is False
        failed = [c for c in report["checks"] if not c["passed"]]
        assert any("naturalness" in c["name"] for c in failed)

    def test_naturalness_skipped_when_not_collected(self):
        submissions = make_submissions()
        for evaluation in submissions[0]["evaluations"]:
            del evaluation["naturalness"]
        report = evaluate_promotion(submissions)
        naturalness = next(c for c in report["checks"] if c["name"] == "naturalness")
        assert naturalness["passed"] is True
        assert "skipped" in naturalness["detail"]

    def test_no_candidate_data_blocks_promotion(self):
        report = evaluate_promotion(make_submissions(model_type="old"))
        assert report["passed"] is False

    def test_threshold_override(self):
        report = evaluate_promotion(
            make_submissions(correct=6, incorrect=4),
            thresholds={"min_emotion_accuracy": 90.0},
        )
        assert report["passed"] is False


class TestObjectiveRegression:
    def test_speaker_similarity_regression_blocks(self):
        report = evaluate_promotion(
            make_submissions(),
            objective_candidate={"speaker_similarity": 0.70},
            objective_baseline={"speaker_similarity": 0.80},
        )
        assert report["passed"] is False

    def test_wer_increase_is_a_regression(self):
        # WER is lower-is-better; an increase must fail intelligibility.
        report = evaluate_promotion(
            make_submissions(),
            objective_candidate={"wer": 0.20},
            objective_baseline={"wer": 0.10},
        )
        assert report["passed"] is False

    def test_equal_or_better_metrics_pass(self):
        report = evaluate_promotion(
            make_submissions(),
            objective_candidate={"speaker_similarity": 0.85, "stoi": 0.92, "wer": 0.08},
            objective_baseline={"speaker_similarity": 0.85, "stoi": 0.90, "wer": 0.10},
        )
        assert report["passed"] is True

    def test_tolerance_allows_small_drop(self):
        report = evaluate_promotion(
            make_submissions(),
            thresholds={"regression_tolerance": 0.05},
            objective_candidate={"speaker_similarity": 0.78},
            objective_baseline={"speaker_similarity": 0.80},
        )
        assert report["passed"] is True

    def test_disjoint_metrics_skip_cleanly(self):
        report = evaluate_promotion(
            make_submissions(),
            objective_candidate={"pesq": 3.1},
            objective_baseline={"speaker_similarity": 0.8},
        )
        # Nothing comparable: both regression groups skip, human checks decide.
        assert report["passed"] is True


class TestCLI:
    def test_cli_exit_codes(self, tmp_path, capsys):
        passing = tmp_path / "pass.json"
        passing.write_text(json.dumps(make_submissions(correct=9, incorrect=1)))
        failing = tmp_path / "fail.json"
        failing.write_text(json.dumps(make_submissions(correct=1, incorrect=9)))

        assert main([str(passing)]) == 0
        assert "PROMOTE" in capsys.readouterr().out

        assert main([str(failing)]) == 1
        assert "DO NOT PROMOTE" in capsys.readouterr().out
