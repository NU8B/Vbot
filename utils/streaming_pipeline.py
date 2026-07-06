"""
Sentence-streaming TTS pipeline core.

Pure, import-light orchestration for chunked speech synthesis: split a
response into speakable chunks, synthesize ahead in a producer thread, and
play chunks in order while the next ones are being synthesized.

Why chunks of 1-2 sentences: StyleTTS2 synthesis has a fixed ~0.85s per-call
overhead plus ~0.02s/word (measured 2026-07-06 on the RTX 5080, see
.private/for-review-latency-and-brevity.md). Chunking too finely re-pays
the overhead per chunk; not chunking at all serializes the whole response
ahead of playback. With RTF 0.14-0.40, one chunk of playback buys enough
time to synthesize several more.

This module knows nothing about Ollama, tkinter, StyleTTS2, or audio
devices — synthesis and playback are injected callables — so it is fully
unit-testable in CI.
"""

import queue
import re
import threading
import time

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

# A queue slot holds synthesized audio waiting for playback. Two slots keeps
# the producer ~2 chunks ahead at most, bounding memory and keeping emotion
# pacing close to the spoken text.
_MAX_BUFFERED_CHUNKS = 2

_END_OF_STREAM = object()


def chunk_sentences(text, max_sentences=2, max_words=30, min_words=3, first_max_sentences=1):
    """Split text into speakable chunks of at most `max_sentences` sentences
    and roughly `max_words` words.

    The FIRST chunk is capped at `first_max_sentences` (default 1) — time to
    first sound is dominated by the first chunk's synthesis, so it should be
    as small as a natural sentence allows. Later chunks batch up to
    `max_sentences` to amortize the fixed per-call synthesis overhead.

    Tiny trailing fragments (under `min_words`) are merged into the previous
    chunk so no synthesis call is wasted on "Okay." style stubs. A sentence
    longer than `max_words` stays its own chunk — splitting mid-sentence
    sounds worse than one long call.
    """
    text = text.strip()
    if not text:
        return []

    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    if not sentences:
        return [text]

    chunks = []
    current = []
    current_words = 0

    for sentence in sentences:
        words = len(sentence.split())
        sentence_cap = first_max_sentences if not chunks else max_sentences
        if current and (len(current) >= sentence_cap or current_words + words > max_words):
            chunks.append(" ".join(current))
            current = []
            current_words = 0
        current.append(sentence)
        current_words += words

    if current:
        chunks.append(" ".join(current))

    # Merge a tiny trailing chunk into its predecessor.
    if len(chunks) >= 2 and len(chunks[-1].split()) < min_words:
        chunks[-2] = chunks[-2] + " " + chunks[-1]
        chunks.pop()

    return chunks


class StreamingTTSPipeline:
    """Producer/consumer pipeline: synthesize chunks ahead, play in order.

    synthesize(chunk_text) -> (audio, duration_seconds) or None to skip.
    play(chunk_text, audio, duration, index, total) -> blocks until the
        chunk has finished playing.

    run() blocks until every chunk is played (call it from a background
    thread when a GUI is involved) and returns a metrics report.
    """

    def __init__(self, synthesize, play, max_buffered=_MAX_BUFFERED_CHUNKS):
        self.synthesize = synthesize
        self.play = play
        self.max_buffered = max_buffered

    def run(self, chunks):
        report = {
            "chunks": len(chunks),
            "played": 0,
            "skipped": 0,
            "errors": [],
            "synth_times": [],
            "time_to_first_audio": None,
            "total_time": None,
        }
        if not chunks:
            report["total_time"] = 0.0
            return report

        start = time.time()
        buffer = queue.Queue(maxsize=self.max_buffered)

        def producer():
            for index, chunk in enumerate(chunks):
                try:
                    synth_start = time.time()
                    result = self.synthesize(chunk)
                    report["synth_times"].append(time.time() - synth_start)
                except Exception as exc:  # noqa: BLE001 - keep speaking on failure
                    report["errors"].append(f"synthesize[{index}]: {exc}")
                    result = None
                if result is None:
                    report["skipped"] += 1
                    continue
                audio, duration = result
                buffer.put((index, chunk, audio, duration))
            buffer.put(_END_OF_STREAM)

        producer_thread = threading.Thread(
            target=producer, daemon=True, name="TTSChunkProducer"
        )
        producer_thread.start()

        total = len(chunks)
        while True:
            item = buffer.get()
            if item is _END_OF_STREAM:
                break
            index, chunk, audio, duration = item
            if report["time_to_first_audio"] is None:
                report["time_to_first_audio"] = time.time() - start
            try:
                self.play(chunk, audio, duration, index, total)
                report["played"] += 1
            except Exception as exc:  # noqa: BLE001 - finish remaining chunks
                report["errors"].append(f"play[{index}]: {exc}")

        producer_thread.join(timeout=5)
        report["total_time"] = time.time() - start
        return report
