"""
Render a markdown scorecard from the latest eval artifacts vs baselines.

Used by the model-eval workflow to populate the GitHub Actions job summary
($GITHUB_STEP_SUMMARY), and handy locally:

    python scripts/eval_summary.py            # print markdown to stdout
    python scripts/eval_summary.py --out path # also write to a file

Reads the newest artifact of each kind from asset/outputs/ and the
committed baselines from evaluation/baselines/. Sections degrade
gracefully when an artifact kind has not been produced yet.
"""

import argparse
import glob
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE_DIR = os.path.join(PROJECT_ROOT, "evaluation", "baselines")
OUTPUTS = os.path.join(PROJECT_ROOT, "asset", "outputs")


def _load(path):
    if not path or not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _latest(pattern):
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None


def _delta(current, baseline, decimals=3):
    if baseline is None:
        return "n/a"
    diff = current - baseline
    return f"{diff:+.{decimals}f}"


def summarize_emotion(current, baseline):
    if not current:
        return "## Emotion pipeline\n\n_No emotion eval artifact found._\n"
    lines = [
        "## Emotion pipeline",
        "",
        f"Model: `{current['model']}` | runtime threshold: {current['runtime_threshold']}",
        "",
        "| dataset | accuracy | macro-F1 | Δ macro-F1 vs baseline |",
        "| --- | --- | --- | --- |",
    ]
    baseline_sets = (baseline or {}).get("datasets", {})
    for name, report in current["datasets"].items():
        run = report["runtime_threshold_report"]
        base = baseline_sets.get(name, {}).get("runtime_threshold_report")
        lines.append(
            f"| {name} | {run['accuracy']:.3f} | {run['macro_f1']:.3f} | "
            f"{_delta(run['macro_f1'], base['macro_f1'] if base else None)} |"
        )
    return "\n".join(lines) + "\n"


def summarize_tts(current, baseline):
    if not current:
        return "## TTS objective metrics\n\n_No TTS objective artifact found._\n"
    metrics = current["gate_metrics"]
    base = (baseline or {}).get("gate_metrics")
    lines = [
        "## TTS objective metrics",
        "",
        f"Character: `{current['character']}` | repo: `{current.get('repo_id')}` | battery v{current['battery_version']}",
        "",
        "| metric | value | Δ vs baseline |",
        "| --- | --- | --- |",
        f"| speaker_similarity | {metrics['speaker_similarity']:.3f} | "
        f"{_delta(metrics['speaker_similarity'], base['speaker_similarity'] if base else None)} |",
        f"| wer | {metrics['wer']:.3f} | "
        f"{_delta(metrics['wer'], base['wer'] if base else None)} |",
    ]
    return "\n".join(lines) + "\n"


def summarize_persona(current, baseline):
    if not current:
        return "## LLM persona fidelity (report-only)\n\n_No persona judgment artifact found._\n"
    lines = [
        "## LLM persona fidelity (report-only)",
        "",
        f"Judge: `{current['judge_model']}` (prompt v{current['judge_prompt_version']}) "
        f"| generator: `{current['generator_model']}`",
        "",
        "| character | voice | engagement | kayfabe | break rate | Δ voice vs ref |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    reference = (baseline or {}).get("characters", {})
    for character, report in current["characters"].items():
        agg = report["aggregate"]
        if not agg.get("judged"):
            lines.append(f"| {character} | — | — | — | — | unparsable |")
            continue
        ref_agg = reference.get(character, {}).get("aggregate", {})
        ref_voice = ref_agg.get("avg_persona_voice")
        lines.append(
            f"| {character} | {agg['avg_persona_voice']:.2f} | {agg['avg_engagement']:.2f} | "
            f"{agg['avg_kayfabe']:.2f} | {agg['kayfabe_break_rate']:.0%} | "
            f"{_delta(agg['avg_persona_voice'], ref_voice, 2)} |"
        )
    return "\n".join(lines) + "\n"


def summarize_llm_contract(current):
    if not current:
        return "## LLM contract benchmark\n\n_No llm_benchmark artifact found._\n"
    lines = [
        "## LLM contract benchmark",
        "",
        f"Model: `{current['model']}` | battery: {len(current['battery'])} prompts",
        "",
        "| character | TTS-safe | brevity | persona hits | breaks (heuristic) | avg latency |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for character, report in current["characters"].items():
        agg = report["aggregate"]
        lines.append(
            f"| {character} | {agg['tts_safety_rate']:.0%} | {agg['brevity_rate']:.0%} | "
            f"{agg['persona_adherence_rate']:.0%} | {agg['character_break_rate']:.0%} | "
            f"{agg['avg_latency_s']:.1f}s |"
        )
    return "\n".join(lines) + "\n"


def _tts_section_with_matching_baseline():
    """Compare the newest TTS artifact against ITS character's baseline."""
    current = _load(_latest(os.path.join(OUTPUTS, "tts_objective_eval", "tts_objective_*.json")))
    baseline = None
    if current:
        baseline = _load(
            os.path.join(BASELINE_DIR, f"tts_objective_{current['character']}_baseline.json")
        )
    return summarize_tts(current, baseline)


def build_summary():
    sections = [
        "# Vbot model evaluation scorecard",
        "",
        summarize_emotion(
            _load(_latest(os.path.join(OUTPUTS, "emotion_eval", "emotion_eval_*.json"))),
            _load(os.path.join(BASELINE_DIR, "emotion_eval_baseline.json")),
        ),
        summarize_llm_contract(
            _load(_latest(os.path.join(OUTPUTS, "llm_benchmark", "llm_benchmark_*.json")))
        ),
        summarize_persona(
            _load(_latest(os.path.join(OUTPUTS, "llm_benchmark", "persona_judged_*.json"))),
            _load(os.path.join(BASELINE_DIR, "persona_judged_reference.json")),
        ),
        _tts_section_with_matching_baseline(),
    ]
    return "\n".join(sections)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Render eval scorecard markdown")
    parser.add_argument("--out", help="Also write the markdown to this file")
    args = parser.parse_args(argv)

    summary = build_summary()
    print(summary)
    if args.out:
        with open(args.out, "a", encoding="utf-8") as file:
            file.write(summary + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
