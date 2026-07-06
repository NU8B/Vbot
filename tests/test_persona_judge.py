"""
Tests for the LLM-as-judge persona scorer.
==========================================
Covers scripts/persona_judge.py prompt building, reply parsing, and
aggregation with canned judge replies. No network or Ollama required.
"""

import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from persona_judge import (
    DIMENSIONS,
    JUDGE_PROMPT_VERSION,
    aggregate_judgments,
    build_judge_prompt,
    parse_judge_reply,
)


class TestJudgePrompt:
    def test_prompt_carries_spec_message_and_response(self):
        prompt = build_judge_prompt(
            "You are Amelia Watson, a detective.",
            "Hi! How are you?",
            "Elementary! The timeline is stable today.",
        )
        assert "You are Amelia Watson" in prompt
        assert "Hi! How are you?" in prompt
        assert "Elementary!" in prompt
        for dimension in DIMENSIONS:
            assert dimension in prompt

    def test_prompt_version_is_pinned(self):
        # Bump JUDGE_PROMPT_VERSION whenever JUDGE_PROMPT_TEMPLATE changes;
        # judged artifacts are only comparable within a version.
        assert JUDGE_PROMPT_VERSION == 2

    def test_prompt_carries_kayfabe_hard_rule(self):
        # v2 exists because the v1 judge scored an explicit AI break at 5.
        prompt = build_judge_prompt("spec", "hi", "hello")
        assert "HARD RULE" in prompt
        assert "MUST be 1" in prompt


class TestParseJudgeReply:
    def test_clean_json(self):
        scores = parse_judge_reply('{"persona_voice": 4, "engagement": 5, "kayfabe": 3, "justification": "solid"}')
        assert scores == {
            "persona_voice": 4,
            "engagement": 5,
            "kayfabe": 3,
            "justification": "solid",
        }

    def test_fenced_json_with_prose(self):
        reply = 'Sure! Here is my evaluation:\n```json\n{"persona_voice": 2, "engagement": 3, "kayfabe": 5, "justification": "ok"}\n```\nHope that helps.'
        scores = parse_judge_reply(reply)
        assert scores["persona_voice"] == 2
        assert scores["kayfabe"] == 5

    def test_out_of_range_scores_clamped(self):
        scores = parse_judge_reply('{"persona_voice": 9, "engagement": 0, "kayfabe": 5, "justification": ""}')
        assert scores["persona_voice"] == 5
        assert scores["engagement"] == 1

    def test_float_scores_rounded(self):
        scores = parse_judge_reply('{"persona_voice": 3.6, "engagement": 4.2, "kayfabe": 4.5, "justification": ""}')
        assert scores["persona_voice"] == 4
        assert scores["engagement"] == 4

    def test_missing_dimension_rejected(self):
        assert parse_judge_reply('{"persona_voice": 4, "engagement": 5}') is None

    def test_non_numeric_score_rejected(self):
        reply = '{"persona_voice": "great", "engagement": 5, "kayfabe": 5}'
        assert parse_judge_reply(reply) is None

    def test_no_json_rejected(self):
        assert parse_judge_reply("I would rate this response quite highly.") is None

    def test_broken_json_rejected(self):
        assert parse_judge_reply('{"persona_voice": 4,,,}') is None


class TestAggregation:
    def make_judgment(self, voice=4, engagement=4, kayfabe=5):
        return {
            "prompt_id": "x",
            "scores": {
                "persona_voice": voice,
                "engagement": engagement,
                "kayfabe": kayfabe,
                "justification": "",
            },
        }

    def test_means_and_break_rate(self):
        judgments = [
            self.make_judgment(voice=5, kayfabe=5),
            self.make_judgment(voice=3, kayfabe=1),
        ]
        agg = aggregate_judgments(judgments)
        assert agg["judged"] == 2
        assert agg["avg_persona_voice"] == 4.0
        assert agg["kayfabe_break_rate"] == 0.5

    def test_unparsable_judgments_counted_not_averaged(self):
        judgments = [self.make_judgment(voice=5), {"prompt_id": "y", "scores": None}]
        agg = aggregate_judgments(judgments)
        assert agg["judged"] == 1
        assert agg["unparsable"] == 1
        assert agg["avg_persona_voice"] == 5.0

    def test_all_unparsable(self):
        agg = aggregate_judgments([{"prompt_id": "y", "scores": None}])
        assert agg == {"judged": 0, "unparsable": 1}
