"""
Compare Vbot LLM benchmark and persona-judge artifacts.

This is the reporting layer for prompt-version experiments: run an older
benchmark/judge pair, run the candidate prompt version, then render a small
Markdown delta report that shows whether the change improved the targeted
behavior without damaging persona quality.

Usage:
    python scripts/compare_llm_artifacts.py old_benchmark.json new_benchmark.json
    python scripts/compare_llm_artifacts.py old.json new.json --old-judge old_judged.json --new-judge new_judged.json
    python scripts/compare_llm_artifacts.py old.json new.json --out report.md
"""

import argparse
import json
import os

BENCHMARK_METRICS = (
    ("tts_safety_rate", "TTS-safe", "pct"),
    ("brevity_rate", "brevity", "pct"),
    ("persona_adherence_rate", "persona hits", "pct"),
    ("character_break_rate", "breaks", "pct"),
    ("avg_word_count", "avg words", "num"),
    ("avg_latency_s", "avg latency", "seconds"),
)

JUDGE_METRICS = (
    ("avg_persona_voice", "voice", "score"),
    ("avg_engagement", "engagement", "score"),
    ("avg_kayfabe", "kayfabe", "score"),
    ("kayfabe_break_rate", "judge breaks", "pct"),
)


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _versions_text(artifact):
    versions = artifact.get("prompt_versions")
    if not versions:
        return "not recorded"
    return ", ".join(f"{name}=v{version}" for name, version in sorted(versions.items()))


def _fmt(value, kind):
    if value is None:
        return "-"
    if kind == "pct":
        return f"{value:.0%}"
    if kind == "seconds":
        return f"{value:.2f}s"
    if kind == "score":
        return f"{value:.2f}"
    return f"{value:.1f}"


def _fmt_delta(delta, kind):
    if delta is None:
        return "n/a"
    if kind == "pct":
        return f"{delta:+.0%}"
    if kind == "seconds":
        return f"{delta:+.2f}s"
    if kind == "score":
        return f"{delta:+.2f}"
    return f"{delta:+.1f}"


def _aggregate(artifact, character):
    return artifact.get("characters", {}).get(character, {}).get("aggregate", {})


def _characters(*artifacts):
    names = set()
    for artifact in artifacts:
        if artifact:
            names.update(artifact.get("characters", {}).keys())
    return sorted(names)


def _metric_delta(old, new, key):
    old_value = old.get(key)
    new_value = new.get(key)
    if not isinstance(old_value, (int, float)) or isinstance(old_value, bool):
        return None
    if not isinstance(new_value, (int, float)) or isinstance(new_value, bool):
        return None
    return new_value - old_value


def benchmark_table(old_artifact, new_artifact):
    lines = [
        "## LLM benchmark delta",
        "",
        "| character | metric | old | new | delta |",
        "| --- | --- | --- | --- | --- |",
    ]
    for character in _characters(old_artifact, new_artifact):
        old = _aggregate(old_artifact, character)
        new = _aggregate(new_artifact, character)
        for key, label, kind in BENCHMARK_METRICS:
            lines.append(
                f"| {character} | {label} | {_fmt(old.get(key), kind)} | "
                f"{_fmt(new.get(key), kind)} | {_fmt_delta(_metric_delta(old, new, key), kind)} |"
            )
    return "\n".join(lines)


def judge_table(old_artifact, new_artifact):
    if not old_artifact or not new_artifact:
        return "## Persona judge delta\n\n_No complete old/new judge pair supplied._"

    lines = [
        "## Persona judge delta",
        "",
        "| character | metric | old | new | delta |",
        "| --- | --- | --- | --- | --- |",
    ]
    for character in _characters(old_artifact, new_artifact):
        old = _aggregate(old_artifact, character)
        new = _aggregate(new_artifact, character)
        for key, label, kind in JUDGE_METRICS:
            lines.append(
                f"| {character} | {label} | {_fmt(old.get(key), kind)} | "
                f"{_fmt(new.get(key), kind)} | {_fmt_delta(_metric_delta(old, new, key), kind)} |"
            )
    return "\n".join(lines)


def build_report(old_benchmark, new_benchmark, old_judge=None, new_judge=None):
    lines = [
        "# Vbot LLM artifact comparison",
        "",
        "## Inputs",
        "",
        f"- old benchmark: `{os.path.basename(old_benchmark['_path'])}`",
        f"- new benchmark: `{os.path.basename(new_benchmark['_path'])}`",
        f"- old prompt versions: {_versions_text(old_benchmark)}",
        f"- new prompt versions: {_versions_text(new_benchmark)}",
    ]
    if old_judge or new_judge:
        lines.extend(
            [
                f"- old judge: `{os.path.basename(old_judge['_path'])}`" if old_judge else "- old judge: n/a",
                f"- new judge: `{os.path.basename(new_judge['_path'])}`" if new_judge else "- new judge: n/a",
            ]
        )
    lines.extend(["", benchmark_table(old_benchmark, new_benchmark), "", judge_table(old_judge, new_judge), ""])
    return "\n".join(lines)


def load_artifact_with_path(path):
    artifact = load_json(path)
    artifact["_path"] = path
    return artifact


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare Vbot LLM benchmark/persona artifacts")
    parser.add_argument("old_benchmark")
    parser.add_argument("new_benchmark")
    parser.add_argument("--old-judge")
    parser.add_argument("--new-judge")
    parser.add_argument("--out", help="Write Markdown report to this path")
    args = parser.parse_args(argv)

    report = build_report(
        load_artifact_with_path(args.old_benchmark),
        load_artifact_with_path(args.new_benchmark),
        load_artifact_with_path(args.old_judge) if args.old_judge else None,
        load_artifact_with_path(args.new_judge) if args.new_judge else None,
    )
    print(report)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as file:
            file.write(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
