"""
Flask tests for the TTS evaluation form.
========================================
Covers the /, /submit, and /results routes of new_tts_eval_form/flask_app.py,
including schema validation of submitted artifacts. Results are written to a
temp file via VBOT_EVAL_RESULTS_FILE so real evaluation data is never touched.
"""

import importlib
import json
import os
import sys

import pytest

pytest.importorskip("flask", reason="flask is required for eval-form tests")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FORM_DIR = os.path.join(PROJECT_ROOT, "new_tts_eval_form")


@pytest.fixture
def eval_app(tmp_path, monkeypatch):
    """Import flask_app fresh with an isolated results file."""
    results_file = tmp_path / "results" / "evaluation_results.json"
    monkeypatch.setenv("VBOT_EVAL_RESULTS_FILE", str(results_file))
    monkeypatch.syspath_prepend(FORM_DIR)

    import flask_app as module

    # Reload so module-level RESULTS_FILE picks up the env override even if
    # another test already imported the module.
    module = importlib.reload(module)
    module.app.config["TESTING"] = True
    return module, results_file


def make_submission(**overrides):
    evaluation = {
        "model": "Amelia",
        "model_type": "old",
        "file": "specific_joy_1.wav",
        "true_emotion": "joy",
        "selected_emotion": "joy",
        "naturalness": 4,
    }
    evaluation.update(overrides)
    return {"evaluations": [evaluation]}


class TestIndexRoute:
    def test_index_returns_evaluation_form(self, eval_app):
        module, _ = eval_app
        response = module.app.test_client().get("/")
        assert response.status_code == 200
        assert b"emotion" in response.data.lower()


class TestSubmitRoute:
    def test_valid_submission_persisted_with_schema_stamp(self, eval_app):
        module, results_file = eval_app
        response = module.app.test_client().post("/submit", json=make_submission())
        assert response.status_code == 200
        assert response.get_json()["status"] == "success"

        stored = json.loads(results_file.read_text())
        assert len(stored) == 1
        record = stored[0]
        assert record["schema_version"] == 1
        assert record["timestamp"]
        assert "remote_ip" in record
        assert record["evaluations"][0]["selected_emotion"] == "joy"

    def test_submissions_append(self, eval_app):
        module, results_file = eval_app
        client = module.app.test_client()
        client.post("/submit", json=make_submission())
        client.post("/submit", json=make_submission(selected_emotion="anger"))

        stored = json.loads(results_file.read_text())
        assert len(stored) == 2

    def test_non_json_body_rejected(self, eval_app):
        module, results_file = eval_app
        response = module.app.test_client().post("/submit", data="not json")
        assert response.status_code == 400
        assert not results_file.exists()

    def test_missing_evaluations_rejected(self, eval_app):
        module, _ = eval_app
        response = module.app.test_client().post("/submit", json={"evaluations": []})
        assert response.status_code == 400
        assert response.get_json()["status"] == "error"

    def test_unknown_emotion_rejected(self, eval_app):
        module, _ = eval_app
        payload = make_submission(selected_emotion="confusion")
        response = module.app.test_client().post("/submit", json=payload)
        assert response.status_code == 400
        assert any("selected_emotion" in error for error in response.get_json()["errors"])

    def test_unknown_model_rejected(self, eval_app):
        module, _ = eval_app
        payload = make_submission(model="NotARealModel")
        response = module.app.test_client().post("/submit", json=payload)
        assert response.status_code == 400

    def test_model_type_mismatch_rejected(self, eval_app):
        module, _ = eval_app
        # Amelia is an old model; claiming it is new would corrupt /results stats.
        payload = make_submission(model_type="new")
        response = module.app.test_client().post("/submit", json=payload)
        assert response.status_code == 400

    def test_naturalness_out_of_range_rejected(self, eval_app):
        module, _ = eval_app
        payload = make_submission(naturalness=9)
        response = module.app.test_client().post("/submit", json=payload)
        assert response.status_code == 400


class TestResultsRoute:
    def test_results_renders_with_no_data(self, eval_app):
        module, _ = eval_app
        response = module.app.test_client().get("/results")
        assert response.status_code == 200

    def test_results_renders_after_submissions(self, eval_app):
        module, _ = eval_app
        client = module.app.test_client()
        client.post("/submit", json=make_submission())
        client.post(
            "/submit",
            json=make_submission(model="Gura", model_type="new", selected_emotion="sadness"),
        )

        response = client.get("/results")
        assert response.status_code == 200
