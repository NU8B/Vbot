"""
TTS objective-metrics evaluation — the promotion gate's producer.

Synthesizes a frozen sentence battery with a character's StyleTTS2 voice
and computes machine-scored quality metrics, writing an artifact whose
`gate_metrics` block feeds new_tts_eval_form/promotion_gate.py
(--objective-candidate / --objective-baseline).

Metrics:
  speaker_similarity  Resemblyzer voice-embedding cosine similarity between
                      each generated utterance and the character's
                      reference recordings (asset/ref_sound/<char>/).
                      Higher = the voice still sounds like the character.
  wer                 Word error rate: faster-whisper transcribes the
                      generated audio, diffed against the input text.
                      Lower = more intelligible speech.

Deliberately EXCLUDED: PESQ and STOI. Both are intrusive metrics that
require a time-aligned reference of the SAME utterance; the archived
benchmark compared generated audio against reference clips with different
content, which yields meaningless scores. They remain valid in the
data-prep pipeline (clean-vs-degraded same-signal) but not here. The
promotion gate compares only the metrics both artifacts share, so this
subset plugs in cleanly.

Usage (GPU for synthesis; whisper + resemblyzer run on CPU):
    python scripts/tts_objective_eval.py --character Amelia
    python scripts/tts_objective_eval.py --character Amelia \
        --repo-id someuser/Amelia_new_ft_StyleTTS2 --label candidate

Compare a candidate against a baseline artifact:
    python new_tts_eval_form/promotion_gate.py results.json \
        --objective-baseline <baseline artifact> \
        --objective-candidate <candidate artifact>
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

SCHEMA_VERSION = 1

CHARACTERS = ("Amelia", "Eveland", "Gura", "Shiori", "Wilson")

# Frozen, character-neutral, phonetically varied battery. Changing it makes
# artifacts incomparable — bump BATTERY_VERSION if you must.
BATTERY_VERSION = 1
SENTENCE_BATTERY = [
    "The quick brown fox jumps over the lazy dog near the riverbank.",
    "Yesterday I visited the museum and saw paintings from another century.",
    "Please remember to water the plants before you leave this evening.",
    "Seven silver spoons sat shining on the small kitchen shelf.",
    "The weather forecast predicts light rain followed by clear skies.",
    "Everyone agreed the concert was the best performance of the year.",
    "Bring me the blue notebook that is lying on the wooden desk.",
    "The train departs at nine fifteen from the central station platform.",
    "Curiosity often leads to the most unexpected and wonderful discoveries.",
    "Thank you for listening so patiently to this entire announcement.",
]

DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "asset", "outputs", "tts_objective_eval")
_PUNCTUATION = re.compile(r"[^\w\s']")


def normalize_text(text):
    """Lowercase and strip punctuation so WER measures words, not commas."""
    return _PUNCTUATION.sub(" ", text.lower()).split()


def word_error_rate(reference, hypothesis):
    """Levenshtein distance over words divided by reference length."""
    ref = normalize_text(reference) if isinstance(reference, str) else reference
    hyp = normalize_text(hypothesis) if isinstance(hypothesis, str) else hypothesis
    if not ref:
        return 0.0 if not hyp else 1.0

    previous = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, start=1):
        current = [i] + [0] * len(hyp)
        for j, hyp_word in enumerate(hyp, start=1):
            substitution = previous[j - 1] + (ref_word != hyp_word)
            current[j] = min(previous[j] + 1, current[j - 1] + 1, substitution)
        previous = current
    return previous[-1] / len(ref)


def aggregate_metrics(per_sentence):
    """Mean speaker similarity and WER over the battery."""
    if not per_sentence:
        return {}
    return {
        "speaker_similarity": sum(s["speaker_similarity"] for s in per_sentence)
        / len(per_sentence),
        "wer": sum(s["wer"] for s in per_sentence) / len(per_sentence),
    }


DEFAULT_GATE_TOLERANCE = 0.02


def check_regression(current, baseline, tolerance=DEFAULT_GATE_TOLERANCE):
    """Compare gate metrics against a baseline. Returns failure strings.

    speaker_similarity is higher-is-better, wer is lower-is-better; a drop
    (or rise) beyond `tolerance` counts as a regression.
    """
    failures = []
    if current["speaker_similarity"] < baseline["speaker_similarity"] - tolerance:
        failures.append(
            "speaker_similarity regressed "
            f"{baseline['speaker_similarity']:.3f} -> {current['speaker_similarity']:.3f} "
            f"(tolerance {tolerance})"
        )
    if current["wer"] > baseline["wer"] + tolerance:
        failures.append(
            f"wer regressed {baseline['wer']:.3f} -> {current['wer']:.3f} "
            f"(tolerance {tolerance})"
        )
    return failures


def load_gate_metrics(path):
    """Read gate metrics from a flat dict or a full artifact."""
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    return payload.get("gate_metrics", payload)


def load_reference_embedding(encoder, character):
    """Mean voice embedding over the character's reference recordings."""
    import numpy as np
    from resemblyzer import preprocess_wav

    ref_dir = os.path.join(PROJECT_ROOT, "asset", "ref_sound", character)
    wavs = [f for f in os.listdir(ref_dir) if f.endswith(".wav")]
    if not wavs:
        raise FileNotFoundError(f"no reference wavs in {ref_dir}")
    embeddings = [
        encoder.embed_utterance(preprocess_wav(os.path.join(ref_dir, name)))
        for name in wavs
    ]
    reference = np.mean(embeddings, axis=0)
    return reference / np.linalg.norm(reference)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Vbot TTS objective metrics (gate producer)")
    parser.add_argument("--character", required=True, choices=CHARACTERS)
    parser.add_argument(
        "--repo-id",
        help="Override HF repo for the TTS weights (evaluate a candidate model)",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Artifact label, e.g. 'baseline' or 'candidate' (default: repo default/production)",
    )
    parser.add_argument("--whisper-model", default="base")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--baseline",
        help="Baseline artifact (e.g. evaluation/baselines/...) to gate against",
    )
    parser.add_argument("--tolerance", type=float, default=DEFAULT_GATE_TOLERANCE)
    args = parser.parse_args(argv)

    import numpy as np
    import soundfile as sf
    from faster_whisper import WhisperModel
    from resemblyzer import VoiceEncoder, preprocess_wav

    from utils.emotion_utils import DIFFUSION_STEPS, create_emotion_config
    from utils.inference_styleTTS2 import StyleTTS2Inference

    print(f"loading TTS ({args.character}, repo={args.repo_id or 'production default'})...")
    tts = StyleTTS2Inference(model_name=args.character, repo_id=args.repo_id)
    emotion_params = create_emotion_config(args.character)["neutral"]
    style = tts.compute_style(
        os.path.join(
            PROJECT_ROOT, "asset", "ref_sound", emotion_params["file"][args.character]
        )
    )

    print("loading voice encoder + whisper...")
    encoder = VoiceEncoder("cpu")
    reference_embedding = load_reference_embedding(encoder, args.character)
    whisper = WhisperModel(args.whisper_model, device="cpu", compute_type="int8")

    audio_dir = os.path.join(args.output_dir, "audio", args.label or args.character)
    os.makedirs(audio_dir, exist_ok=True)

    per_sentence = []
    for index, sentence in enumerate(SENTENCE_BATTERY):
        start = time.time()
        wav = tts.inference(
            text=sentence,
            ref_s=style,
            alpha=emotion_params["alpha"],
            beta=emotion_params["beta"],
            diffusion_steps=DIFFUSION_STEPS,
            embedding_scale=emotion_params["embedding_scale"],
        )
        wav = np.asarray(wav)
        synth_time = time.time() - start

        wav_path = os.path.join(audio_dir, f"battery_{index:02d}.wav")
        sf.write(wav_path, wav, 24000)

        embedding = encoder.embed_utterance(preprocess_wav(wav_path))
        similarity = float(np.dot(embedding / np.linalg.norm(embedding), reference_embedding))

        segments, _ = whisper.transcribe(wav_path, language="en")
        transcript = " ".join(segment.text.strip() for segment in segments)
        wer = word_error_rate(sentence, transcript)

        per_sentence.append(
            {
                "index": index,
                "text": sentence,
                "transcript": transcript,
                "speaker_similarity": similarity,
                "wer": wer,
                "audio_seconds": len(wav) / 24000,
                "synth_seconds": synth_time,
            }
        )
        print(
            f"  [{index:02d}] sim={similarity:.3f} wer={wer:.2f} "
            f"({synth_time:.1f}s synth) {transcript[:50]!r}"
        )

    gate_metrics = aggregate_metrics(per_sentence)

    artifact = {
        "schema_version": SCHEMA_VERSION,
        "kind": "tts_objective_eval",
        "timestamp": datetime.now().isoformat(),
        "character": args.character,
        "repo_id": args.repo_id or getattr(tts, "repo_id", None),
        "label": args.label,
        "battery_version": BATTERY_VERSION,
        "whisper_model": args.whisper_model,
        "per_sentence": per_sentence,
        "gate_metrics": gate_metrics,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    label = args.label or args.character
    out_path = os.path.join(args.output_dir, f"tts_objective_{label}_{stamp}.json")
    with open(out_path, "w", encoding="utf-8") as file:
        json.dump(artifact, file, indent=2, ensure_ascii=False)

    print(f"\nspeaker_similarity={gate_metrics['speaker_similarity']:.3f} "
          f"wer={gate_metrics['wer']:.3f}")
    print(f"artifact: {out_path}")

    from eval_tracking import log_eval_run

    log_eval_run(
        experiment="tts-objective",
        run_name=f"{args.character}_{label}_{stamp}",
        params={
            "character": args.character,
            "repo_id": artifact["repo_id"],
            "battery_version": BATTERY_VERSION,
            "whisper_model": args.whisper_model,
        },
        metrics=gate_metrics,
        artifact=out_path,
    )

    if args.baseline:
        failures = check_regression(
            gate_metrics, load_gate_metrics(args.baseline), args.tolerance
        )
        if failures:
            print("\nGATE: FAIL")
            for failure in failures:
                print(f"  [FAIL] {failure}")
            return 1
        print("\nGATE: PASS (no regression vs baseline)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
