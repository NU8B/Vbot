"""
Vbot Data Pipeline Unit Tests
==============================
Tests for the data pipeline logic — text normalization, number conversion,
and quality threshold validation. These tests do NOT require a GPU, Docker,
or any ML model downloads.

These are designed to run in GitHub Actions CI on every push.
"""

import os
import sys

import pytest

# Add project root and Data_prep to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "Data_prep"))


# ============================================================
# Test: Number-to-words conversion
# ============================================================


class TestNumToWords:
    """Test the num_to_words function used in TTS data preparation."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from data_StyleTTS2 import num_to_words

        self.num_to_words = num_to_words

    def test_zero(self):
        assert self.num_to_words(0) == "zero"

    def test_single_digits(self):
        assert self.num_to_words(1) == "one"
        assert self.num_to_words(5) == "five"
        assert self.num_to_words(9) == "nine"

    def test_teens(self):
        assert self.num_to_words(11) == "eleven"
        assert self.num_to_words(13) == "thirteen"
        assert self.num_to_words(19) == "nineteen"

    def test_tens(self):
        assert self.num_to_words(20) == "twenty"
        assert self.num_to_words(42) == "forty two"
        assert self.num_to_words(99) == "ninety nine"

    def test_hundreds(self):
        assert self.num_to_words(100) == "one hundred"
        assert self.num_to_words(256) == "two hundred fifty six"
        assert self.num_to_words(999) == "nine hundred ninety nine"

    def test_thousands(self):
        assert self.num_to_words(1000) == "one thousand"
        assert self.num_to_words(2024) == "two thousand twenty four"
        assert self.num_to_words(9999) == "nine thousand nine hundred ninety nine"


# ============================================================
# Test: Text normalization
# ============================================================


class TestTextNormalization:
    """Test normalize_text_for_tts function used in data preparation."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from data_StyleTTS2 import normalize_text_for_tts

        self.normalize = normalize_text_for_tts

    def test_whitespace_normalization(self):
        """Multiple spaces should be collapsed to single spaces."""
        result = self.normalize("hello    world")
        assert "  " not in result
        assert "hello" in result and "world" in result

    def test_ellipsis_normalization(self):
        """Multiple dots should be reduced to a single dot."""
        result = self.normalize("hello... world")
        assert "..." not in result

    def test_excessive_punctuation(self):
        """Multiple exclamation/question marks should be reduced."""
        result = self.normalize("wow!!!")
        assert "!!!" not in result
        result2 = self.normalize("really???")
        assert "???" not in result2

    def test_contraction_expansion(self):
        """Common contractions should be expanded for phonemization."""
        result = self.normalize("I can't believe it")
        assert "cannot" in result

        result2 = self.normalize("I won't go")
        assert "will not" in result2

    def test_parentheses_removal(self):
        """Content in parentheses should be removed."""
        result = self.normalize("hello (this is noise) world")
        assert "noise" not in result
        assert "hello" in result and "world" in result

    def test_special_characters_removed(self):
        """Special characters like &, @, #, etc. should be replaced with spaces."""
        result = self.normalize("price is $100 & tax")
        assert "$" not in result
        assert "&" not in result

    def test_em_dash_normalization(self):
        """Em/en dashes should be converted to hyphens."""
        result = self.normalize("hello\u2014world")  # em-dash
        assert "\u2014" not in result

    def test_empty_string(self):
        """Empty strings should not crash."""
        result = self.normalize("")
        assert isinstance(result, str)

    def test_already_clean_text(self):
        """Clean text should pass through mostly unchanged."""
        text = "Hello, this is a test."
        result = self.normalize(text)
        assert "Hello" in result
        assert "test" in result


# ============================================================
# Test: Ordinal conversion
# ============================================================


class TestOrdinalToWords:
    """Test the ordinal_to_words function."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from data_StyleTTS2 import ordinal_to_words

        self.ordinal = ordinal_to_words

    def test_first(self):
        assert self.ordinal(1) == "first"

    def test_second(self):
        assert self.ordinal(2) == "second"

    def test_third(self):
        assert self.ordinal(3) == "third"


# ============================================================
# Test: Quality threshold validation logic
# ============================================================


class TestQualityGateLogic:
    """Test that quality gate thresholds are internally consistent."""

    @pytest.fixture(autouse=True)
    def setup(self):
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "Data_prep", "audio_preprocessor"))
        from audio_segmenter_v2 import SPEECH_THRESHOLDS, SPEECH_UPPER_BOUNDS

        self.thresholds = SPEECH_THRESHOLDS
        self.upper_bounds = SPEECH_UPPER_BOUNDS

    def test_lower_bounds_less_than_upper_bounds_zcr(self):
        """ZCR lower threshold must be less than upper bound."""
        assert (
            self.thresholds["zcr"] < self.upper_bounds["zcr"]
        ), f"ZCR lower ({self.thresholds['zcr']}) >= upper ({self.upper_bounds['zcr']})"

    def test_lower_bounds_less_than_upper_bounds_spec_cent(self):
        """Spectral centroid lower threshold must be less than upper bound."""
        assert (
            self.thresholds["spec_cent"] < self.upper_bounds["spec_cent"]
        ), f"spec_cent lower ({self.thresholds['spec_cent']}) >= upper ({self.upper_bounds['spec_cent']})"

    def test_lower_bounds_less_than_upper_bounds_spectral_flatness(self):
        """Spectral flatness lower threshold must be less than upper bound."""
        assert self.thresholds["spectral_flatness"] < self.upper_bounds["spectral_flatness"], (
            f"spectral_flatness lower ({self.thresholds['spectral_flatness']}) >= "
            f"upper ({self.upper_bounds['spectral_flatness']})"
        )

    def test_lower_bounds_less_than_upper_bounds_rms_energy(self):
        """RMS energy lower threshold must be less than upper bound."""
        assert self.thresholds["rms_energy"] < self.upper_bounds["rms_energy"], (
            f"rms_energy lower ({self.thresholds['rms_energy']}) >= " f"upper ({self.upper_bounds['rms_energy']})"
        )

    def test_all_thresholds_are_numeric(self):
        """All threshold values should be numeric (int or float) or None."""
        for key, value in self.thresholds.items():
            assert value is None or isinstance(
                value, (int, float)
            ), f"Threshold '{key}' has non-numeric value: {value} ({type(value)})"

    def test_all_upper_bounds_are_numeric(self):
        """All upper bound values should be numeric."""
        for key, value in self.upper_bounds.items():
            assert isinstance(
                value, (int, float)
            ), f"Upper bound '{key}' has non-numeric value: {value} ({type(value)})"
