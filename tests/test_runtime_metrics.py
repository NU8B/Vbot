"""
Tests for the runtime metrics logger and report.
================================================
Covers utils/runtime_metrics.py (JSONL logging, failure safety) and
scripts/metrics_report.py (percentiles, summarize). CI-safe.
"""

import json
import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from metrics_report import percentile, summarize
from utils.runtime_metrics import log_interaction, read_records


class TestLogInteraction:
    def test_appends_jsonl_records(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VBOT_METRICS_DIR", str(tmp_path))

        assert log_interaction(character="Amelia", path="simple", llm_latency_s=0.5)
        assert log_interaction(character="Gura", path="streaming", chunks=3)

        files = list(tmp_path.glob("metrics_*.jsonl"))
        assert len(files) == 1
        records = read_records(str(files[0]))
        assert len(records) == 2
        assert records[0]["character"] == "Amelia"
        assert records[0]["timestamp"]
        assert records[1]["chunks"] == 3

    def test_unwritable_directory_returns_false_without_raising(self, tmp_path, monkeypatch):
        blocker = tmp_path / "blocker"
        blocker.write_text("a file where the metrics dir should be")
        monkeypatch.setenv("VBOT_METRICS_DIR", str(blocker))

        assert log_interaction(character="Amelia") is False

    def test_read_records_skips_corrupt_lines(self, tmp_path):
        path = tmp_path / "metrics.jsonl"
        path.write_text('{"a": 1}\nnot json at all\n{"b": 2}\n')
        records = read_records(str(path))
        assert records == [{"a": 1}, {"b": 2}]


class TestPercentile:
    def test_median_and_p95(self):
        values = list(range(1, 101))
        assert percentile(values, 50) == 50
        assert percentile(values, 95) == 95

    def test_single_value(self):
        assert percentile([7.0], 50) == 7.0
        assert percentile([7.0], 95) == 7.0

    def test_empty(self):
        assert percentile([], 50) is None


class TestSummarize:
    def make_records(self):
        return [
            {"path": "simple", "character": "Amelia", "emotion": "joy", "llm_latency_s": 0.5, "tts_latency_s": 1.5},
            {"path": "simple", "character": "Amelia", "emotion": "neutral", "llm_latency_s": 0.7, "tts_latency_s": 2.0},
            {
                "path": "streaming",
                "character": "Gura",
                "emotion": "joy",
                "llm_latency_s": 0.6,
                "time_to_first_audio_s": 1.8,
            },
        ]

    def test_counts_and_latency_stats(self):
        report = summarize(self.make_records())
        assert report["interactions"] == 3
        assert report["by_path"] == {"simple": 2, "streaming": 1}
        assert report["by_emotion"]["joy"] == 2
        assert report["latency"]["llm_latency_s"]["n"] == 3
        assert report["latency"]["tts_latency_s"]["max"] == 2.0
        assert report["latency"]["time_to_first_audio_s"]["p50"] == 1.8

    def test_none_latencies_excluded(self):
        report = summarize([{"path": "simple", "llm_latency_s": None}])
        assert "llm_latency_s" not in report["latency"]

    def test_empty_records(self):
        report = summarize([])
        assert report["interactions"] == 0
        assert report["latency"] == {}


class TestRuntimeWiring:
    def test_ollama_utils_logs_both_paths(self):
        path = os.path.join(PROJECT_ROOT, "utils", "ollama_utils.py")
        with open(path, "r", encoding="utf-8") as file:
            source = file.read()
        assert source.count("log_interaction(") >= 2, "both the simple and streaming TTS paths must log runtime metrics"
