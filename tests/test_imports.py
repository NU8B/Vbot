"""
Vbot Import & Configuration Tests
===================================
Lightweight tests that verify all core modules can be imported
and critical configurations are present. These tests do NOT require
a GPU, Docker, or any ML model downloads.

These are designed to run in GitHub Actions CI on every push.
"""

import ast
import importlib
import os
import sys

import pytest

# Add project root to path (same pattern used throughout the Vbot codebase)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def _parse_project_file(*path_parts):
    """Parse a project file without importing desktop/GPU-only dependencies."""
    with open(os.path.join(PROJECT_ROOT, *path_parts), "r", encoding="utf-8") as file:
        return ast.parse(file.read())


def _module_defines(path_parts, name, node_type):
    tree = _parse_project_file(*path_parts)
    return any(isinstance(node, node_type) and node.name == name for node in tree.body)


# ============================================================
# Test: Core module imports
# ============================================================


class TestCoreImports:
    """Verify that all core utility modules can be imported without crashing."""

    def test_import_emotion_utils(self):
        """emotion_utils.py should import cleanly (RoBERTa config, emotion mappings)."""
        mod = importlib.import_module("utils.emotion_utils")
        assert hasattr(mod, "EMOTION_MODEL_NAME")
        assert hasattr(mod, "MODEL_PARAMS")

    def test_import_docker_utils(self):
        """docker_utils.py should import cleanly (DockerHandler class)."""
        mod = importlib.import_module("utils.docker_utils")
        assert hasattr(mod, "DockerHandler")

    def test_import_performance_boost(self):
        """performance_boost.py should import cleanly (MemoryManager, LazyLoader)."""
        mod = importlib.import_module("utils.performance_boost")
        assert hasattr(mod, "MemoryManager")
        assert hasattr(mod, "LazyLoader")

    def test_import_tts_utils(self):
        """TTS_utils.py should import cleanly."""
        mod = importlib.import_module("utils.TTS_utils")
        assert hasattr(mod, "InferenceHandler")

    def test_import_user_preferences(self):
        """user_preferences.py should import cleanly."""
        mod = importlib.import_module("utils.user_preferences")
        assert hasattr(mod, "should_show_welcome_screen")
        assert hasattr(mod, "get_last_selected_avatar")


class TestDesktopSourceContracts:
    """Check desktop/audio modules without importing Windows or audio drivers in CI."""

    def test_audio_utils_defines_audio_processor(self):
        assert _module_defines(("utils", "audio_utils.py"), "AudioProcessor", ast.ClassDef)

    def test_gui_defines_chat_gui(self):
        assert _module_defines(("utils", "gui.py"), "ChatGUI", ast.ClassDef)

    def test_ollama_utils_defines_handler(self):
        assert _module_defines(("utils", "ollama_utils.py"), "OllamaHandler", ast.ClassDef)

    def test_ollama_prompts_include_core_characters(self):
        # Prompts moved to the versioned registry (stdlib-only, safe to
        # import in CI); ollama_utils re-imports MODEL_PROMPTS from there,
        # which tests/test_prompt_contract.py verifies at source level.
        from utils.character_prompts import MODEL_PROMPTS

        expected_characters = {"Amelia", "Eveland", "Gura", "Shiori", "Wilson"}
        assert set(MODEL_PROMPTS.keys()) == expected_characters


# ============================================================
# Test: Critical configurations
# ============================================================


class TestCriticalConfigs:
    """Verify that critical constants and configurations exist and are valid."""

    def test_all_five_characters_in_emotion_params(self):
        """MODEL_PARAMS in emotion_utils must contain all 5 character configs."""
        from utils.emotion_utils import MODEL_PARAMS

        expected_characters = {"Amelia", "Eveland", "Gura", "Shiori", "Wilson"}
        assert set(MODEL_PARAMS.keys()) == expected_characters, (
            f"MODEL_PARAMS is missing characters. " f"Expected: {expected_characters}, Got: {set(MODEL_PARAMS.keys())}"
        )

    def test_emotion_params_have_required_keys(self):
        """Each character's emotion params must have ALPHA, BETA, EMBEDDING_SCALE."""
        from utils.emotion_utils import MODEL_PARAMS

        required_keys = {"ALPHA", "BETA", "EMBEDDING_SCALE"}
        for character, params in MODEL_PARAMS.items():
            for key in required_keys:
                assert key in params, f"Character '{character}' is missing required key '{key}' " f"in MODEL_PARAMS"

    def test_emotion_model_name_is_set(self):
        """EMOTION_MODEL_NAME should be a non-empty string."""
        from utils.emotion_utils import EMOTION_MODEL_NAME

        assert isinstance(EMOTION_MODEL_NAME, str)
        assert len(EMOTION_MODEL_NAME) > 0

    def test_diffusion_steps_is_positive(self):
        """DIFFUSION_STEPS should be a positive integer."""
        from utils.emotion_utils import DIFFUSION_STEPS

        assert isinstance(DIFFUSION_STEPS, int)
        assert DIFFUSION_STEPS > 0


# ============================================================
# Test: Audio segmenter quality thresholds
# ============================================================


class TestQualityThresholds:
    """Verify that audio quality gate thresholds are configured correctly."""

    def test_speech_thresholds_exist(self):
        """SPEECH_THRESHOLDS dict must exist with required keys."""
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "Data_prep", "audio_preprocessor"))
        from audio_segmenter_v2 import SPEECH_THRESHOLDS

        required_keys = {"stoi", "pesq", "zcr", "spec_cent", "spectral_flatness", "spectral_rolloff", "rms_energy"}
        for key in required_keys:
            assert key in SPEECH_THRESHOLDS, f"Missing threshold key: {key}"

    def test_stoi_threshold_in_range(self):
        """STOI threshold should be between 0 and 1."""
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "Data_prep", "audio_preprocessor"))
        from audio_segmenter_v2 import SPEECH_THRESHOLDS

        stoi_val = SPEECH_THRESHOLDS["stoi"]
        assert 0 < stoi_val <= 1.0, f"STOI threshold {stoi_val} is out of range (0, 1]"

    def test_pesq_threshold_in_range(self):
        """PESQ threshold should be between 1.0 and 5.0 (MOS scale)."""
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "Data_prep", "audio_preprocessor"))
        from audio_segmenter_v2 import SPEECH_THRESHOLDS

        pesq_val = SPEECH_THRESHOLDS["pesq"]
        assert 1.0 <= pesq_val <= 5.0, f"PESQ threshold {pesq_val} is out of MOS range [1, 5]"

    def test_upper_bounds_exist(self):
        """SPEECH_UPPER_BOUNDS dict must exist with required keys."""
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "Data_prep", "audio_preprocessor"))
        from audio_segmenter_v2 import SPEECH_UPPER_BOUNDS

        required_keys = {"zcr", "spec_cent", "spectral_flatness", "spectral_rolloff", "rms_energy", "mfcc_var"}
        for key in required_keys:
            assert key in SPEECH_UPPER_BOUNDS, f"Missing upper bound key: {key}"

    def test_duration_settings_are_valid(self):
        """Audio segmenter duration settings should be logically consistent."""
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "Data_prep", "audio_preprocessor"))
        from audio_segmenter_v2 import DISCARD_THRESHOLD, MAX_DURATION, MIN_DURATION

        assert (
            DISCARD_THRESHOLD < MIN_DURATION
        ), f"DISCARD_THRESHOLD ({DISCARD_THRESHOLD}) must be < MIN_DURATION ({MIN_DURATION})"
        assert MIN_DURATION < MAX_DURATION, f"MIN_DURATION ({MIN_DURATION}) must be < MAX_DURATION ({MAX_DURATION})"


# ============================================================
# Test: Project structure
# ============================================================


class TestProjectStructure:
    """Verify critical project files and directories exist."""

    def test_requirements_txt_exists(self):
        assert os.path.isfile(os.path.join(PROJECT_ROOT, "requirements.txt"))

    def test_vbot_spec_exists(self):
        assert os.path.isfile(os.path.join(PROJECT_ROOT, "vbot.spec"))

    def test_readme_exists(self):
        assert os.path.isfile(os.path.join(PROJECT_ROOT, "README.md"))

    def test_styletts2_directory_exists(self):
        assert os.path.isdir(os.path.join(PROJECT_ROOT, "StyleTTS2"))

    def test_data_prep_directory_exists(self):
        assert os.path.isdir(os.path.join(PROJECT_ROOT, "Data_prep"))

    def test_utils_directory_exists(self):
        assert os.path.isdir(os.path.join(PROJECT_ROOT, "utils"))

    def test_mcp_server_exists(self):
        assert os.path.isfile(os.path.join(PROJECT_ROOT, "vbot_mcp_server.py"))
