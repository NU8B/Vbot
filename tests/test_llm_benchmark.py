"""
Tests for the runtime LLM benchmark scoring and artifacts.
==========================================================
Covers scripts/llm_benchmark.py metric functions, aggregation, and artifact
schema. No network or Ollama required — the live benchmark run is manual.
"""

import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from llm_benchmark import (
    CHARACTER_PROFILES,
    PROMPT_BATTERY,
    aggregate_results,
    build_artifact,
    load_model_prompts,
    score_response,
)

AMELIA = CHARACTER_PROFILES["Amelia"]


class TestScoreResponse:
    def test_clean_in_character_response(self):
        metrics = score_response("Elementary! I traced the clue through the timeline myself.", AMELIA)
        assert metrics["tts_safe"] is True
        assert metrics["brevity_ok"] is True
        assert metrics["first_person"] is True
        assert metrics["character_break"] is False
        assert metrics["persona_hit"] is True

    def test_action_text_flags_tts_violation(self):
        metrics = score_response("*whips out pocket watch* Time to investigate!", AMELIA)
        assert metrics["tts_safe"] is False
        assert metrics["tts_violations"]["asterisks"] == 2
        assert metrics["filter_changed_response"] is True

    def test_parentheses_and_brackets_counted(self):
        metrics = score_response("Sure (in gremlin mode) let's go [laughs].", AMELIA)
        assert metrics["tts_violations"]["parentheses"] == 1
        assert metrics["tts_violations"]["brackets"] == 1
        assert metrics["tts_safe"] is False

    def test_emoji_flags_violation(self):
        metrics = score_response("Time to solve mysteries! \U0001f50d\U0001f575", AMELIA)
        assert metrics["tts_violations"]["emoji"] == 2
        assert metrics["tts_safe"] is False

    def test_verbose_response_fails_brevity(self):
        long_response = "word " * 45
        metrics = score_response(long_response.strip() + ".", AMELIA)
        assert metrics["word_count"] > 30
        assert metrics["brevity_ok"] is False

    def test_character_break_detected(self):
        metrics = score_response("As an AI language model, I cannot travel through time.", AMELIA)
        assert metrics["character_break"] is True

    def test_self_described_ai_is_a_break(self):
        # Caught live in benchmark run 1: Wilson introduced himself as
        # "a highly advanced AI" without using any "as an AI" phrasing.
        metrics = score_response(
            "I am Wilson, a highly advanced AI designed to provide support.",
            CHARACTER_PROFILES["Wilson"],
        )
        assert metrics["character_break"] is True

    def test_an_air_of_is_not_a_break(self):
        metrics = score_response("I speak with an air of ancient wisdom.", CHARACTER_PROFILES["Shiori"])
        assert metrics["character_break"] is False

    def test_persona_miss_for_generic_response(self):
        metrics = score_response("That sounds nice. Have a good day.", AMELIA)
        assert metrics["persona_hit"] is False


class TestAggregation:
    def make_result(self, **metric_overrides):
        metrics = {
            "tts_safe": True,
            "tts_violations": {},
            "filter_changed_response": False,
            "word_count": 20,
            "brevity_ok": True,
            "first_person": True,
            "character_break": False,
            "persona_hit": True,
            "topic_hit": True,
        }
        metrics.update(metric_overrides)
        return {"metrics": metrics, "latency_s": 2.0, "tokens_per_s": 40.0}

    def test_rates_computed(self):
        results = [
            self.make_result(),
            self.make_result(tts_safe=False, brevity_ok=False),
        ]
        agg = aggregate_results(results)
        assert agg["responses"] == 2
        assert agg["tts_safety_rate"] == 0.5
        assert agg["brevity_rate"] == 0.5
        assert agg["persona_adherence_rate"] == 1.0
        assert agg["avg_latency_s"] == 2.0
        assert agg["avg_tokens_per_s"] == 40.0

    def test_empty_results(self):
        assert aggregate_results([]) == {}


class TestArtifactSchema:
    def test_artifact_carries_required_fields(self):
        artifact = build_artifact("http://localhost:11500", {"Amelia": {"results": [], "aggregate": {}}})
        assert artifact["schema_version"] == 1
        assert artifact["kind"] == "llm_benchmark"
        assert artifact["timestamp"]
        assert artifact["model"] == "stheno"
        assert "Amelia" in artifact["characters"]
        assert artifact["battery"] == PROMPT_BATTERY


class TestBenchmarkContracts:
    def test_profiles_cover_all_production_characters(self):
        prompts = load_model_prompts()
        assert set(CHARACTER_PROFILES.keys()) == set(prompts.keys()), (
            "Every character with a production prompt needs a benchmark " "profile, and vice versa."
        )

    def test_battery_includes_bait_prompts(self):
        ids = {p["id"] for p in PROMPT_BATTERY}
        assert {"roleplay_bait", "emoji_bait", "verbosity_bait", "identity_probe"} <= ids

    def test_battery_prompts_unique(self):
        ids = [p["id"] for p in PROMPT_BATTERY]
        assert len(ids) == len(set(ids))
