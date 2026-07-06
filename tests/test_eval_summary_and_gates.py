"""
Tests for the TTS self-gate and the eval scorecard renderer.
============================================================
Covers scripts/tts_objective_eval.py check_regression/load_gate_metrics and
scripts/eval_summary.py section rendering. Pure logic, CI-safe.
"""

import json
import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from eval_summary import summarize_emotion, summarize_persona, summarize_tts
from tts_objective_eval import check_regression, load_gate_metrics


class TestTTSSelfGate:
    def test_no_regression_passes(self):
        current = {"speaker_similarity": 0.83, "wer": 0.05}
        baseline = {"speaker_similarity": 0.82, "wer": 0.06}
        assert check_regression(current, baseline) == []

    def test_similarity_drop_fails(self):
        current = {"speaker_similarity": 0.70, "wer": 0.05}
        baseline = {"speaker_similarity": 0.82, "wer": 0.05}
        failures = check_regression(current, baseline)
        assert any("speaker_similarity" in failure for failure in failures)

    def test_wer_rise_fails(self):
        current = {"speaker_similarity": 0.82, "wer": 0.20}
        baseline = {"speaker_similarity": 0.82, "wer": 0.05}
        failures = check_regression(current, baseline)
        assert any("wer" in failure for failure in failures)

    def test_tolerance_absorbs_noise(self):
        current = {"speaker_similarity": 0.805, "wer": 0.065}
        baseline = {"speaker_similarity": 0.82, "wer": 0.058}
        assert check_regression(current, baseline, tolerance=0.02) == []

    def test_load_gate_metrics_unwraps_artifact(self, tmp_path):
        artifact = {"kind": "tts_objective_eval", "gate_metrics": {"wer": 0.05, "speaker_similarity": 0.8}}
        path = tmp_path / "a.json"
        path.write_text(json.dumps(artifact))
        assert load_gate_metrics(str(path)) == {"wer": 0.05, "speaker_similarity": 0.8}

    def test_committed_baseline_is_loadable(self):
        baseline_path = os.path.join(PROJECT_ROOT, "evaluation", "baselines", "tts_objective_Amelia_baseline.json")
        metrics = load_gate_metrics(baseline_path)
        assert 0 < metrics["speaker_similarity"] <= 1
        assert metrics["wer"] >= 0


class TestScorecardSections:
    def test_emotion_section_with_delta(self):
        report = {
            "runtime_threshold_report": {"accuracy": 0.78, "macro_f1": 0.79},
        }
        current = {
            "model": "m",
            "runtime_threshold": 0.15,
            "datasets": {"domain": report},
        }
        baseline = {"datasets": {"domain": {"runtime_threshold_report": {"accuracy": 0.75, "macro_f1": 0.76}}}}
        section = summarize_emotion(current, baseline)
        assert "| domain | 0.780 | 0.790 | +0.030 |" in section

    def test_tts_section_without_baseline(self):
        current = {
            "character": "Amelia",
            "repo_id": "x",
            "battery_version": 1,
            "gate_metrics": {"speaker_similarity": 0.82, "wer": 0.06},
        }
        section = summarize_tts(current, None)
        assert "0.820" in section
        assert "n/a" in section

    def test_missing_artifacts_degrade_gracefully(self):
        assert "_No emotion eval artifact found._" in summarize_emotion(None, None)
        assert "_No persona judgment artifact found._" in summarize_persona(None, None)
        assert "_No TTS objective artifact found._" in summarize_tts(None, None)

    def test_persona_section_break_rate(self):
        current = {
            "judge_model": "mistral:latest",
            "judge_prompt_version": 2,
            "generator_model": "stheno",
            "characters": {
                "Wilson": {
                    "aggregate": {
                        "judged": 10,
                        "unparsable": 0,
                        "avg_persona_voice": 3.8,
                        "avg_engagement": 4.8,
                        "avg_kayfabe": 4.2,
                        "kayfabe_break_rate": 0.2,
                    }
                }
            },
        }
        section = summarize_persona(current, None)
        assert "| Wilson | 3.80 | 4.80 | 4.20 | 20% | n/a |" in section
