"""
Cleanup tests for the model preloader.
======================================
Verifies ModelPreloader.cleanup_unused_models() and cleanup_all() release
models correctly. utils.preloader imports the full desktop initialization
stack at module level, so a stub utils.initialization_utils is injected
before import — CI never loads audio drivers or TTS models here.
"""

import importlib
import os
import sys
import types

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


class RecordingInitHandler:
    """Stands in for InitializationHandler; records cleanup calls."""

    def __init__(self, fail_cleanup=False):
        self.cleaned_up = False
        self.fail_cleanup = fail_cleanup

    def cleanup(self):
        if self.fail_cleanup:
            raise RuntimeError("simulated cleanup failure")
        self.cleaned_up = True


@pytest.fixture
def preloader_module(monkeypatch):
    """Import utils.preloader with initialization_utils stubbed out."""
    stub = types.ModuleType("utils.initialization_utils")
    stub.InitializationHandler = RecordingInitHandler
    monkeypatch.setitem(sys.modules, "utils.initialization_utils", stub)

    sys.modules.pop("utils.preloader", None)
    module = importlib.import_module("utils.preloader")
    yield module
    sys.modules.pop("utils.preloader", None)


def make_loaded_preloader(preloader_module, model_names, **handler_kwargs):
    preloader = preloader_module.ModelPreloader()
    for name in model_names:
        preloader.loaded_models[name] = {
            "init_handler": RecordingInitHandler(**handler_kwargs),
            "components": {},
            "ollama_handler": object(),
            "model_name": name,
        }
    return preloader


class TestCleanupUnusedModels:
    def test_keeps_only_requested_model(self, preloader_module):
        preloader = make_loaded_preloader(preloader_module, ["Amelia", "Gura", "Wilson"])
        handlers = {name: data["init_handler"] for name, data in preloader.loaded_models.items()}

        preloader.cleanup_unused_models(keep_model="Amelia")

        assert set(preloader.loaded_models.keys()) == {"Amelia"}
        assert handlers["Amelia"].cleaned_up is False
        assert handlers["Gura"].cleaned_up is True
        assert handlers["Wilson"].cleaned_up is True

    def test_model_removed_even_when_cleanup_fails(self, preloader_module):
        preloader = make_loaded_preloader(preloader_module, ["Amelia", "Gura"], fail_cleanup=True)

        preloader.cleanup_unused_models(keep_model="Amelia")

        # A failing handler must not leak the model reference.
        assert set(preloader.loaded_models.keys()) == {"Amelia"}

    def test_missing_init_handler_is_tolerated(self, preloader_module):
        preloader = preloader_module.ModelPreloader()
        preloader.loaded_models["Gura"] = {"components": {}, "model_name": "Gura"}

        preloader.cleanup_unused_models(keep_model="Amelia")

        assert preloader.loaded_models == {}


class TestCleanupAll:
    def test_all_models_cleaned_and_cleared(self, preloader_module):
        preloader = make_loaded_preloader(preloader_module, ["Amelia", "Eveland", "Shiori"])
        handlers = list(data["init_handler"] for data in preloader.loaded_models.values())

        preloader.cleanup_all()

        assert preloader.loaded_models == {}
        assert all(handler.cleaned_up for handler in handlers)

    def test_cleanup_all_survives_handler_failure(self, preloader_module):
        preloader = make_loaded_preloader(preloader_module, ["Amelia"], fail_cleanup=True)

        preloader.cleanup_all()

        assert preloader.loaded_models == {}
