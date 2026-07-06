"""
Tests for the runtime monitoring gate.
======================================
Covers scripts/monitoring_report.py (SLO evaluation, window splitting,
emotion-bucket drift via total variation distance, report assembly, exit
codes) and the committed SLO config contract. CI-safe: only stdlib-backed
modules are imported.
"""

import json
import os
import sys
from datetime import datetime, timedelta

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from monitoring_report import (
    BUCKETS,
    DEFAULT_SLO_CONFIG,
    build_report,
    bucket_distribution,
    check_drift,
    check_slos,
    load_slo_config,
    main,
    split_window,
    total_variation_distance,
)

NOW = datetime(2026, 7, 6, 12, 0, 0)


def slo_config():
    return load_slo_config(DEFAULT_SLO_CONFIG)


def simple_record(days_ago=0, emotion="joy", llm=1.0, tts=2.0, duration=10.0):
    return {
        "timestamp": (NOW - timedelta(days=days_ago)).isoformat(),
        "character": "Amelia",
        "path": "simple",
        "llm_latency_s": llm,
        "emotion": emotion,
        "tts_latency_s": tts,
        "audio_duration_s": duration,
        "response_words": 30,
    }


def streaming_record(days_ago=0, emotion="joy", ttfa=1.4, chunks=3, played=3, errors=0):
    return {
        "timestamp": (NOW - timedelta(days=days_ago)).isoformat(),
        "character": "Gura",
        "path": "streaming",
        "llm_latency_s": 1.0,
        "emotion": emotion,
        "chunks": chunks,
        "chunks_played": played,
        "time_to_first_audio_s": ttfa,
        "tts_latency_s": 4.0,
        "pipeline_errors": errors,
        "response_words": 40,
    }


class TestSloConfigContract:
    def test_committed_config_is_valid(self):
        config = slo_config()
        assert config["schema_version"] == 1
        assert config["kind"] == "runtime_slos"
        for name in (
            "llm_latency_s_p95",
            "time_to_first_audio_s_p95",
            "tts_rtf_p95",
            "pipeline_error_rate",
            "chunk_completion_rate",
        ):
            assert name in config["slos"]
            assert config["slos"][name]["min_samples"] >= 1
        assert 0 < config["drift"]["emotion_bucket_tvd_max"] <= 1
        assert config["drift"]["min_samples_per_window"] >= 1

    def test_wrong_kind_rejected(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text('{"kind": "something_else"}')
        with pytest.raises(ValueError):
            load_slo_config(str(bad))


class TestSplitWindow:
    def test_splits_by_timestamp(self):
        records = [simple_record(days_ago=0), simple_record(days_ago=3), simple_record(days_ago=30)]
        current, reference = split_window(records, window_days=7, now=NOW)
        assert len(current) == 2
        assert len(reference) == 1

    def test_unparsable_timestamps_stay_in_current_window(self):
        records = [{"timestamp": "not-a-date", "llm_latency_s": 1.0}, {"llm_latency_s": 2.0}]
        current, reference = split_window(records, window_days=7, now=NOW)
        assert len(current) == 2
        assert reference == []


class TestCheckSlos:
    def test_all_pass_on_healthy_telemetry(self):
        records = [simple_record() for _ in range(5)] + [streaming_record() for _ in range(5)]
        results = check_slos(records, slo_config())
        assert all(r["status"] == "pass" for r in results.values())

    def test_llm_latency_violation(self):
        records = [simple_record(llm=9.0) for _ in range(5)] + [streaming_record() for _ in range(5)]
        results = check_slos(records, slo_config())
        assert results["llm_latency_s_p95"]["status"] == "fail"

    def test_rtf_uses_simple_path_only(self):
        # Streaming records have no audio_duration_s -> only 3 RTF samples,
        # below min_samples, so no verdict even though values are terrible.
        records = [simple_record(tts=9.0, duration=1.0) for _ in range(3)] + [streaming_record() for _ in range(5)]
        results = check_slos(records, slo_config())
        assert results["tts_rtf_p95"]["status"] == "insufficient_data"

    def test_pipeline_error_and_completion_rates(self):
        records = [streaming_record(chunks=4, played=2, errors=2) for _ in range(5)]
        results = check_slos(records, slo_config())
        assert results["pipeline_error_rate"]["status"] == "fail"
        assert results["pipeline_error_rate"]["value"] == 0.5
        assert results["chunk_completion_rate"]["status"] == "fail"
        assert results["chunk_completion_rate"]["value"] == 0.5

    def test_insufficient_data_never_fails(self):
        results = check_slos([simple_record(llm=99.0)], slo_config())
        assert all(r["status"] == "insufficient_data" for r in results.values())


class TestDrift:
    def test_bucket_distribution_maps_labels(self):
        records = [simple_record(emotion="joy"), simple_record(emotion="gratitude"), simple_record(emotion="anger")]
        dist, n = bucket_distribution(records)
        assert n == 3
        assert dist["happy"] == pytest.approx(2 / 3)
        assert dist["angry"] == pytest.approx(1 / 3)

    def test_tvd_bounds(self):
        same = {b: 0.2 for b in BUCKETS}
        assert total_variation_distance(same, same) == 0.0
        disjoint_a = {"happy": 1.0}
        disjoint_b = {"sad": 1.0}
        assert total_variation_distance(disjoint_a, disjoint_b) == pytest.approx(1.0)

    def test_insufficient_data_below_min_samples(self):
        current = [simple_record() for _ in range(5)]
        reference = [simple_record(days_ago=30) for _ in range(5)]
        result = check_drift(current, reference, slo_config()["drift"])
        assert result["status"] == "insufficient_data"
        assert result["tvd"] is None

    def test_detects_distribution_shift(self):
        drift_cfg = {"emotion_bucket_tvd_max": 0.25, "min_samples_per_window": 30}
        current = [simple_record(emotion="sadness") for _ in range(30)]
        reference = [simple_record(days_ago=30, emotion="joy") for _ in range(30)]
        result = check_drift(current, reference, drift_cfg)
        assert result["status"] == "fail"
        assert result["tvd"] == pytest.approx(1.0)

    def test_stable_distribution_passes(self):
        drift_cfg = {"emotion_bucket_tvd_max": 0.25, "min_samples_per_window": 30}
        current = [simple_record(emotion="joy") for _ in range(30)]
        reference = [simple_record(days_ago=30, emotion="amusement") for _ in range(30)]
        result = check_drift(current, reference, drift_cfg)
        assert result["status"] == "pass"
        assert result["tvd"] == pytest.approx(0.0)


class TestBuildReport:
    def test_violations_collected(self):
        records = [simple_record(llm=9.0) for _ in range(5)] + [streaming_record() for _ in range(5)]
        report = build_report(records, window_days=7, slo_config=slo_config(), now=NOW)
        assert report["kind"] == "runtime_monitoring"
        assert report["schema_version"] == 1
        assert "llm_latency_s_p95" in report["violations"]

    def test_healthy_report_has_no_violations(self):
        records = [simple_record() for _ in range(5)] + [streaming_record() for _ in range(5)]
        report = build_report(records, window_days=7, slo_config=slo_config(), now=NOW)
        assert report["violations"] == []
        assert report["records_in_window"] == 10


class TestMain:
    def write_jsonl(self, directory, records):
        path = os.path.join(str(directory), "metrics_2026-07-06.jsonl")
        with open(path, "w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record) + "\n")

    def test_exit_zero_and_artifact_on_healthy_data(self, tmp_path, capsys):
        self.write_jsonl(tmp_path, [simple_record() for _ in range(5)] + [streaming_record() for _ in range(5)])
        artifact = tmp_path / "report.json"
        code = main(["--metrics-dir", str(tmp_path), "--json", str(artifact), "--window-days", "36500"])
        assert code == 0
        report = json.loads(artifact.read_text())
        assert report["kind"] == "runtime_monitoring"
        assert "all checks passed" in capsys.readouterr().out

    def test_exit_one_on_violation(self, tmp_path, capsys):
        self.write_jsonl(tmp_path, [simple_record(llm=9.0) for _ in range(5)])
        code = main(["--metrics-dir", str(tmp_path), "--window-days", "36500"])
        assert code == 1
        assert "VIOLATIONS" in capsys.readouterr().out

    def test_exit_zero_on_empty_directory(self, tmp_path, capsys):
        code = main(["--metrics-dir", str(tmp_path)])
        assert code == 0
        assert "nothing to monitor" in capsys.readouterr().out
