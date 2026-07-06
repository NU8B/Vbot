"""
Response cleanup for TTS-safe speech text.

Extracted from OllamaHandler._filter_action_text so the logic is a pure,
import-light function that can be unit tested in CI without pulling in
torch, tkinter, or the desktop audio stack.

The prompt layer asks the LLM not to emit roleplay markup, but the runtime
still defends itself here when the model does.
"""

import re

# Returned when filtering strips a response down to nothing.
FALLBACK_RESPONSE = "Hello! How can I help you today?"

# Short trailing words that are legitimate sentence endings and must not be
# trimmed by the incomplete-word heuristic.
_VALID_SHORT_WORDS = frozenset(
    {
        "a", "an", "at", "in", "on", "to", "of", "is", "it", "he", "she",
        "we", "me", "my", "up", "go", "no", "so", "do", "if", "or", "as",
        "by", "be", "am", "hi", "oh", "ah", "ha", "ho", "la", "ma", "pa",
        "ta", "ya",
    }
)

_ACTION_TEXT = re.compile(r"\*[^*]*\*")
_STRAY_ASTERISKS = re.compile(r"\*+")
_PARENTHESES = re.compile(r"\([^)]*\)")
_BRACKETS = re.compile(r"\[[^\]]*\]")
_MULTI_WHITESPACE = re.compile(r"\s+")
_SENTENCE_SPLIT = re.compile(r"[.!?]+")
# Emoji and pictographs are unspeakable; the 2026-07-06 LLM benchmark showed
# models emit them under emoji-bait prompts despite the prompt ban.
_EMOJI = re.compile(
    "["
    "\U0001f300-\U0001faff"  # symbols, pictographs, emoticons, extended-A
    "\U00002600-\U000027bf"  # misc symbols and dingbats
    "\U0001f1e6-\U0001f1ff"  # regional indicators
    "\U0000fe0f"  # variation selector-16
    "]"
)


def filter_action_text(text):
    """Remove roleplay markup and ensure the text ends as natural speech.

    Strips asterisk-wrapped actions, stray asterisks, parenthesized and
    bracketed stage directions, and emoji, collapses whitespace, then trims
    incomplete trailing sentences/words and guarantees final punctuation.
    Returns FALLBACK_RESPONSE if nothing speakable remains.
    """
    filtered_text = _ACTION_TEXT.sub("", text)
    filtered_text = _STRAY_ASTERISKS.sub("", filtered_text)
    filtered_text = _PARENTHESES.sub("", filtered_text)
    filtered_text = _BRACKETS.sub("", filtered_text)
    filtered_text = _EMOJI.sub("", filtered_text)
    filtered_text = _MULTI_WHITESPACE.sub(" ", filtered_text).strip()

    if not filtered_text or filtered_text.isspace():
        return FALLBACK_RESPONSE

    # If the text ends mid-sentence, keep only complete sentences when there
    # are any; otherwise close the single sentence with a period.
    if filtered_text[-1] not in ".!?":
        sentences = _SENTENCE_SPLIT.split(filtered_text)
        if len(sentences) > 1:
            complete_sentences = sentences[:-1]
            filtered_text = ". ".join(complete_sentences) + "."
        else:
            filtered_text = filtered_text.rstrip() + "."

    # A very short trailing word is likely a generation cut-off unless it is
    # a known valid short word.
    words = filtered_text.split()
    if words:
        last_word = words[-1]
        if len(last_word) <= 2 and last_word.lower() not in _VALID_SHORT_WORDS:
            words = words[:-1]
            filtered_text = " ".join(words)
            if filtered_text and filtered_text[-1] not in ".!?":
                filtered_text = filtered_text.rstrip() + "."

    # The word trim above can empty the text entirely (e.g. a one-letter
    # response); never hand an empty string to the TTS pipeline.
    if not filtered_text:
        return FALLBACK_RESPONSE

    return filtered_text
