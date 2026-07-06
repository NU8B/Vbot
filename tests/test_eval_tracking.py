"""
Tests for the optional MLflow tracking layer.
=============================================
Covers scripts/eval_tracking.py: silent no-op without mlflow, correct
logging calls with a stubbed mlflow module, and failure isolation. mlflow
itself is never imported in CI.
"""

import os
import sys
import types
from contextlib import contextmanager

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

import eval_tracking


class FakeMlflow(types.ModuleType):
    def __init__(self, fail_on_metrics=False):
        super().__init__("mlflow")
        self.fail_on_metrics = fail_on_metrics
        self.calls = []

    def set_tracking_uri(self, uri):
        self.calls.append(("uri", uri))

    def set_experiment(self, name):
        self.calls.append(("experiment", name))

    @contextmanager
    def start_run(self, run_name=None):
        self.calls.append(("run", run_name))
        yield

    def set_tags(self, tags):
        self.calls.append(("tags", tags))

    def log_params(self, params):
        self.calls.append(("params", params))

    def log_metrics(self, metrics):
        if self.fail_on_metrics:
            raise RuntimeError("store unavailable")
        self.calls.append(("metrics", metrics))

    def log_artifact(self, path):
        self.calls.append(("artifact", path))


class TestNoMlflow:
    def test_silent_noop_when_mlflow_missing(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "mlflow", None)  # import -> ImportError
        assert eval_tracking.log_eval_run("exp", "run", metrics={"a": 1}) is False


class TestWithStubbedMlflow:
    def test_logs_params_metrics_and_artifact(self, monkeypatch, tmp_path):
        fake = FakeMlflow()
        monkeypatch.setitem(sys.modules, "mlflow", fake)
        artifact = tmp_path / "artifact.json"
        artifact.write_text("{}")

        logged = eval_tracking.log_eval_run(
            "emotion-eval",
            "run-1",
            params={"model": "m", "threshold": 0.15},
            metrics={"macro_f1": 0.64, "skipme": None, "flag": True},
            artifact=str(artifact),
        )

        assert logged is True
        recorded = dict(
            (name, value) for name, value in fake.calls if name in ("experiment", "params", "metrics", "artifact")
        )
        assert recorded["experiment"] == "emotion-eval"
        assert recorded["params"] == {"model": "m", "threshold": "0.15"}
        # None and bool values must be filtered out of metrics.
        assert recorded["metrics"] == {"macro_f1": 0.64}
        assert recorded["artifact"] == str(artifact)

    def test_missing_artifact_file_not_logged(self, monkeypatch):
        fake = FakeMlflow()
        monkeypatch.setitem(sys.modules, "mlflow", fake)
        eval_tracking.log_eval_run("exp", "run", artifact="does/not/exist.json")
        assert not any(name == "artifact" for name, _ in fake.calls)

    def test_tracking_failure_never_raises(self, monkeypatch):
        fake = FakeMlflow(fail_on_metrics=True)
        monkeypatch.setitem(sys.modules, "mlflow", fake)
        assert eval_tracking.log_eval_run("exp", "run", metrics={"a": 1.0}) is False
