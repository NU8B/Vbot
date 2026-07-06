"""
Response-cleanup tests for the TTS-safe text filter.
====================================================
Covers utils/response_filter.py, the pure extraction of
OllamaHandler._filter_action_text(). These tests run in CI with no GPU,
Docker, or desktop dependencies.
"""

import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.response_filter import FALLBACK_RESPONSE, filter_action_text


class TestActionTextRemoval:
    """Roleplay markup must never reach the speech pipeline."""

    def test_clean_text_passes_through(self):
        assert filter_action_text("I am on the case.") == "I am on the case."

    def test_asterisk_action_removed(self):
        assert filter_action_text("*chuckles* Time travel is easy.") == "Time travel is easy."

    def test_multiple_actions_removed(self):
        result = filter_action_text("*gasps* No way! *whips out pocket watch* Follow me!")
        assert result == "No way! Follow me!"

    def test_multiword_action_removed(self):
        assert filter_action_text("*adjusts monocle slowly* Elementary.") == "Elementary."

    def test_stray_asterisks_removed(self):
        assert filter_action_text("I am **very** sure.") == "I am very sure."

    def test_parenthesized_text_removed(self):
        assert filter_action_text("Sure thing (in gremlin mode) detective.") == "Sure thing detective."

    def test_bracketed_text_removed(self):
        assert filter_action_text("Hello [waves at camera] friend.") == "Hello friend."

    def test_whitespace_collapsed(self):
        assert filter_action_text("Too   many    spaces.") == "Too many spaces."

    def test_emoji_removed(self):
        # Found by the runtime LLM benchmark: models emit emoji under bait
        # prompts, and emoji are unspeakable for StyleTTS2.
        assert filter_action_text("So fin-tastic! \U0001f988 Love it! \U0001f389") == "So fin-tastic! Love it!"

    def test_emoji_only_response_returns_fallback(self):
        assert filter_action_text("\U0001f988\U0001f389\U0001f60a") == FALLBACK_RESPONSE


class TestEmptyFallback:
    """Filtering must never return unspeakable output."""

    def test_empty_input_returns_fallback(self):
        assert filter_action_text("") == FALLBACK_RESPONSE

    def test_whitespace_only_returns_fallback(self):
        assert filter_action_text("   ") == FALLBACK_RESPONSE

    def test_action_only_response_returns_fallback(self):
        assert filter_action_text("*laughs softly*") == FALLBACK_RESPONSE

    def test_markup_only_response_returns_fallback(self):
        assert filter_action_text("(sighs) [dramatic pause] ***") == FALLBACK_RESPONSE

    def test_trimmed_to_empty_returns_fallback(self):
        # A one-letter response gets punctuated to "K.", then trimmed as an
        # incomplete word; the filter must fall back instead of returning "".
        assert filter_action_text("K") == FALLBACK_RESPONSE


class TestSentenceCompletion:
    """Cut-off generations should end as natural speech."""

    def test_incomplete_trailing_sentence_dropped(self):
        result = filter_action_text("The mystery is solved. But wait there is")
        assert result == "The mystery is solved."

    def test_single_incomplete_sentence_gets_period(self):
        assert filter_action_text("I will check the timeline") == "I will check the timeline."

    def test_complete_punctuation_preserved(self):
        assert filter_action_text("Really?") == "Really?"
        assert filter_action_text("Amazing!") == "Amazing!"

    def test_exclamation_normalized_when_trailing_fragment_dropped(self):
        # Known quirk inherited from the original implementation: rebuilding
        # from sentence fragments joins with periods, so "!" becomes ".".
        result = filter_action_text("No way! And then she")
        assert result == "No way."

    def test_single_char_final_fragment_trimmed(self):
        # After punctuation is appended, "t." is two chars and treated as a
        # generation cut-off.
        assert filter_action_text("Let me think about t") == "Let me think about."

    def test_short_final_words_kept(self):
        # With their appended punctuation these are three chars, above the
        # trim threshold.
        assert filter_action_text("Time to go") == "Time to go."
        assert filter_action_text("That is it") == "That is it."


class TestRealWorldSamples:
    """Shapes of output the runtime has actually had to clean up."""

    def test_typical_roleplay_response(self):
        raw = "*Gremlin Mode activated* Heh, you think you can beat ME? (cracks knuckles) Bring it on!"
        assert filter_action_text(raw) == "Heh, you think you can beat ME? Bring it on!"

    def test_mixed_markup_response(self):
        raw = "Ara ara~ [smiles mysteriously] The archives hold many secrets... *turns page*"
        assert filter_action_text(raw) == "Ara ara~ The archives hold many secrets..."


class TestHandlerDelegation:
    """OllamaHandler._filter_action_text must stay wired to the pure filter."""

    def test_handler_source_delegates_to_response_filter(self):
        import ast

        path = os.path.join(PROJECT_ROOT, "utils", "ollama_utils.py")
        with open(path, "r", encoding="utf-8") as file:
            tree = ast.parse(file.read())

        imports_filter = any(
            isinstance(node, ast.ImportFrom)
            and node.module == "utils.response_filter"
            and any(alias.name == "filter_action_text" for alias in node.names)
            for node in ast.walk(tree)
        )
        assert imports_filter, "ollama_utils.py no longer imports utils.response_filter"
