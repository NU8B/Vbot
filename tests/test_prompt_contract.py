"""
Prompt-contract tests for TTS-safe character prompts.
=====================================================
Every character prompt in MODEL_PROMPTS feeds a speech pipeline, not a text
chat. These tests enforce the contract each prompt must carry so that LLM
output stays speakable: no emojis, no parentheses, no action text, no
asterisks, concise first-person replies.

MODEL_PROMPTS is read via AST (not imported) so CI never loads the desktop
LLM/TTS stack. This mirrors the pattern in tests/test_imports.py.
"""

import ast
import os
import re
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

EXPECTED_CHARACTERS = {"Amelia", "Eveland", "Gura", "Shiori", "Wilson"}

# Instructions every prompt must state, as (label, case-insensitive pattern).
REQUIRED_CLAUSES = [
    ("emoji ban", r"NO EMOJIS"),
    ("parenthesis ban", r"NO PARENTHESIS"),
    ("action-text ban", r"NO ACTION TEXT"),
    ("asterisk ban", r"NEVER use asterisks"),
    ("conciseness cap", r"concise and under 30 words"),
    ("first-person voice", r"first person"),
    ("character lock", r"not to break character"),
    ("plain-text output", r"Only use string text"),
]

# Codepoint ranges that cover emoji and pictographs a TTS prompt must not
# contain as literal characters.
EMOJI_PATTERN = re.compile(
    "["
    "\U0001f300-\U0001faff"  # symbols, pictographs, emoticons, extended-A
    "\U00002600-\U000027bf"  # misc symbols and dingbats
    "\U0001f1e6-\U0001f1ff"  # regional indicators
    "\U0000fe0f"  # variation selector-16
    "]"
)


def _load_model_prompts():
    path = os.path.join(PROJECT_ROOT, "utils", "ollama_utils.py")
    with open(path, "r", encoding="utf-8") as file:
        tree = ast.parse(file.read())

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "MODEL_PROMPTS":
                    return ast.literal_eval(node.value)
    raise AssertionError("MODEL_PROMPTS not found in utils/ollama_utils.py")


MODEL_PROMPTS = _load_model_prompts()


class TestPromptRoster:
    """The prompt set must cover exactly the shipped characters."""

    def test_all_core_characters_have_prompts(self):
        assert set(MODEL_PROMPTS.keys()) == EXPECTED_CHARACTERS

    def test_prompt_roster_matches_emotion_configs(self):
        from utils.emotion_utils import MODEL_PARAMS

        assert set(MODEL_PROMPTS.keys()) == set(MODEL_PARAMS.keys()), (
            "Characters with an LLM prompt must match characters with "
            "emotion params; a mismatch breaks character switching."
        )


@pytest.mark.parametrize("character", sorted(EXPECTED_CHARACTERS))
class TestTTSSafeConstraints:
    """Each prompt must carry every TTS-safety clause."""

    @pytest.mark.parametrize("label, pattern", REQUIRED_CLAUSES)
    def test_required_clause_present(self, character, label, pattern):
        prompt = MODEL_PROMPTS[character]
        assert re.search(pattern, prompt, re.IGNORECASE), (
            f"{character} prompt is missing its {label} clause " f"(expected to match /{pattern}/i)"
        )

    def test_prompt_is_substantial(self, character):
        prompt = MODEL_PROMPTS[character]
        assert isinstance(prompt, str)
        assert len(prompt.strip()) > 200, (
            f"{character} prompt looks truncated; persona plus TTS " "constraints should not fit in 200 characters"
        )

    def test_prompt_contains_no_literal_emoji(self, character):
        prompt = MODEL_PROMPTS[character]
        found = EMOJI_PATTERN.findall(prompt)
        assert not found, (
            f"{character} prompt contains literal emoji {found}; prompts "
            "must model the no-emoji behavior they demand"
        )

    def test_prompt_defines_persona_traits(self, character):
        prompt = MODEL_PROMPTS[character]
        assert "Key traits" in prompt, (
            f"{character} prompt has no 'Key traits' section; persona " "guidance and TTS constraints are both required"
        )
