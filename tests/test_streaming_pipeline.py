"""
Tests for the sentence-streaming TTS pipeline core.
===================================================
Covers utils/streaming_pipeline.py: chunking policy and producer/consumer
orchestration with fake synth/play callables. CI-safe, no audio or models.
"""

import os
import sys
import time

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.streaming_pipeline import StreamingTTSPipeline, chunk_sentences


class TestChunkSentences:
    def test_empty_text(self):
        assert chunk_sentences("") == []
        assert chunk_sentences("   ") == []

    def test_single_sentence_single_chunk(self):
        assert chunk_sentences("The case is closed now.") == ["The case is closed now."]

    def test_first_chunk_is_single_sentence(self):
        # Time to first sound is dominated by the first chunk's synthesis,
        # so the first chunk stays one sentence even when two would fit.
        text = "I found the clue. It was under the rug."
        assert chunk_sentences(text) == ["I found the clue.", "It was under the rug."]

    def test_later_chunks_batch_two_sentences(self):
        text = "First fact here. Second fact here. Third fact appears now."
        assert chunk_sentences(text) == [
            "First fact here.",
            "Second fact here. Third fact appears now.",
        ]

    def test_word_cap_splits_early(self):
        # 28 + 5 words would exceed the 30-word cap, so the tail must split off.
        long_sentence = "word " * 28
        text = f"{long_sentence.strip()}. Short tail sentence follows here."
        chunks = chunk_sentences(text, max_words=30)
        assert len(chunks) == 2
        assert chunks[1] == "Short tail sentence follows here."

    def test_word_cap_boundary_stays_together(self):
        # Exactly at the cap (25 + 5 = 30 words) fits one chunk when the
        # first-chunk sentence cap is lifted.
        long_sentence = "word " * 25
        text = f"{long_sentence.strip()}. Short tail sentence follows here."
        assert len(chunk_sentences(text, max_words=30, first_max_sentences=2)) == 1

    def test_oversized_sentence_kept_whole(self):
        long_sentence = ("word " * 50).strip() + "."
        assert chunk_sentences(long_sentence) == [long_sentence]

    def test_tiny_trailing_fragment_merged(self):
        # "Okay." must not cost its own ~0.85s synthesis call; it merges back
        # into the previous chunk.
        text = "Here is the first sentence. Here is the second one. Okay."
        assert chunk_sentences(text) == [
            "Here is the first sentence.",
            "Here is the second one. Okay.",
        ]

    def test_text_without_punctuation(self):
        assert chunk_sentences("hello there friend") == ["hello there friend"]

    def test_question_and_exclamation_boundaries(self):
        text = "Really? That is amazing! Tell me everything about it. And then some more."
        chunks = chunk_sentences(text)
        # A tiny FIRST chunk is desirable: it reaches the speaker fastest.
        assert chunks == [
            "Really?",
            "That is amazing! Tell me everything about it.",
            "And then some more.",
        ]


class RecordingPlayer:
    def __init__(self, play_seconds=0.0, fail_on=None):
        self.play_seconds = play_seconds
        self.fail_on = fail_on or set()
        self.played = []

    def __call__(self, chunk, audio, duration, index, total):
        if index in self.fail_on:
            raise RuntimeError(f"playback device lost at chunk {index}")
        self.played.append((index, chunk, audio, duration, total))
        if self.play_seconds:
            time.sleep(self.play_seconds)


class TestStreamingPipeline:
    def test_all_chunks_played_in_order(self):
        player = RecordingPlayer()
        pipeline = StreamingTTSPipeline(synthesize=lambda chunk: (f"audio:{chunk}", 1.5), play=player)
        report = pipeline.run(["one.", "two.", "three."])

        assert report["played"] == 3
        assert [p[0] for p in player.played] == [0, 1, 2]
        assert player.played[0][2] == "audio:one."
        assert report["errors"] == []
        assert report["time_to_first_audio"] is not None

    def test_empty_chunk_list(self):
        report = StreamingTTSPipeline(lambda c: ("a", 1.0), RecordingPlayer()).run([])
        assert report["played"] == 0
        assert report["total_time"] == 0.0

    def test_synthesis_overlaps_playback(self):
        # 4 chunks, synth 80ms each, playback 120ms each. Serial would take
        # 4*80 + 4*120 = 800ms; overlapped should approach 80 + 4*120 = 560ms.
        def slow_synth(chunk):
            time.sleep(0.08)
            return "audio", 0.12

        player = RecordingPlayer(play_seconds=0.12)
        report = StreamingTTSPipeline(slow_synth, player).run(["a.", "b.", "c.", "d."])

        assert report["played"] == 4
        assert report["total_time"] < 0.75, (
            f"pipeline took {report['total_time']:.2f}s; synthesis does not " "appear to overlap playback"
        )
        # First audio must not wait for all synthesis to finish.
        assert report["time_to_first_audio"] < 0.25

    def test_failed_synthesis_skips_chunk_and_continues(self):
        def flaky_synth(chunk):
            if "bad" in chunk:
                raise ValueError("cuda hiccup")
            return "audio", 0.5

        player = RecordingPlayer()
        report = StreamingTTSPipeline(flaky_synth, player).run(["good one.", "bad one.", "good two."])

        assert report["played"] == 2
        assert len(report["errors"]) == 1
        assert "synthesize[1]" in report["errors"][0]

    def test_none_synthesis_counts_as_skip(self):
        pipeline = StreamingTTSPipeline(synthesize=lambda chunk: None, play=RecordingPlayer())
        report = pipeline.run(["one.", "two."])
        assert report["skipped"] == 2
        assert report["played"] == 0

    def test_playback_failure_does_not_stop_stream(self):
        player = RecordingPlayer(fail_on={0})
        pipeline = StreamingTTSPipeline(lambda c: ("audio", 0.5), player)
        report = pipeline.run(["one.", "two."])

        assert report["played"] == 1
        assert len(report["errors"]) == 1
        assert "play[0]" in report["errors"][0]
