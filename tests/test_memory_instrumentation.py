"""
Tests for runtime memory instrumentation.
=========================================
Covers MemoryManager.log_memory and track_cuda_peak from
utils/performance_boost.py. CPU-safe: CUDA paths no-op without a GPU.
"""

import ast
import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.performance_boost import MemoryManager, track_cuda_peak


class TestLogMemory:
    def test_returns_info_and_prints_label(self, capsys):
        info = MemoryManager.log_memory("unit-test-label")
        captured = capsys.readouterr().out
        assert "unit-test-label" in captured
        assert "RAM" in captured
        assert "system_ram" in info
        assert 0 <= info["system_ram"] <= 100


class TestTrackCudaPeak:
    def test_yields_and_survives_without_cuda(self):
        executed = []
        with track_cuda_peak("unit-test-component"):
            executed.append(True)
        assert executed == [True]

    def test_exceptions_propagate(self):
        with pytest.raises(ValueError):
            with track_cuda_peak("failing-component"):
                raise ValueError("component load failed")


class TestInstrumentationWiring:
    """The desktop modules must keep their instrumentation call sites."""

    def _source(self, *parts):
        with open(os.path.join(PROJECT_ROOT, *parts), "r", encoding="utf-8") as file:
            return file.read()

    def test_switch_path_logs_memory(self):
        source = self._source("utils", "seamless_interface.py")
        assert source.count("memory_manager.log_memory") >= 3, (
            "character-switch memory logging was removed from " "seamless_interface._handle_model_switch"
        )

    def test_startup_tracks_tts_peak_and_phases(self):
        source = self._source("utils", "initialization_utils.py")
        assert "track_cuda_peak" in source
        assert source.count("memory_manager.log_memory") >= 3

    def test_touched_desktop_modules_still_parse(self):
        for module in ("seamless_interface.py", "initialization_utils.py"):
            ast.parse(self._source("utils", module))
