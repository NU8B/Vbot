"""
Summarize Vbot runtime interaction metrics.

Reads the JSONL files written by utils/runtime_metrics.py and reports
latency percentiles per pipeline stage, emotion distribution, and
path usage — "what did the system actually do" without a debugger.

Usage:
    python scripts/metrics_report.py                # newest daily file
    python scripts/metrics_report.py path/to.jsonl  # specific file
"""

import argparse
import glob
import os
import sys
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.runtime_metrics import metrics_dir, read_records

LATENCY_FIELDS = (
    "llm_latency_s",
    "tts_latency_s",
    "audio_duration_s",
    "time_to_first_audio_s",
)


def percentile(values, q):
    """Nearest-rank percentile; q in [0, 100]."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, round(q / 100 * len(ordered)))
    return ordered[rank - 1]


def summarize(records):
    """Aggregate interaction records into a report dict."""
    report = {
        "interactions": len(records),
        "by_path": dict(Counter(r.get("path", "unknown") for r in records)),
        "by_character": dict(Counter(r.get("character", "unknown") for r in records)),
        "by_emotion": dict(Counter(r.get("emotion", "unknown") for r in records)),
        "latency": {},
    }
    for field in LATENCY_FIELDS:
        values = [
            r[field] for r in records
            if isinstance(r.get(field), (int, float)) and not isinstance(r.get(field), bool)
        ]
        if values:
            report["latency"][field] = {
                "n": len(values),
                "p50": percentile(values, 50),
                "p95": percentile(values, 95),
                "max": max(values),
            }
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Summarize runtime metrics JSONL")
    parser.add_argument("file", nargs="?", help="metrics JSONL (default: newest daily file)")
    args = parser.parse_args(argv)

    path = args.file
    if not path:
        candidates = sorted(glob.glob(os.path.join(metrics_dir(), "metrics_*.jsonl")))
        if not candidates:
            print(f"No metrics files in {metrics_dir()} — run the app first.")
            return 1
        path = candidates[-1]

    records = read_records(path)
    report = summarize(records)

    print(f"metrics file: {path}")
    print(f"interactions: {report['interactions']}")
    print(f"paths: {report['by_path']}")
    print(f"characters: {report['by_character']}")
    print(f"emotions: {report['by_emotion']}")
    print(f"\n{'stage':<24} {'n':>4} {'p50':>8} {'p95':>8} {'max':>8}")
    print("-" * 56)
    for field, stats in report["latency"].items():
        print(
            f"{field:<24} {stats['n']:>4} {stats['p50']:>7.2f}s {stats['p95']:>7.2f}s "
            f"{stats['max']:>7.2f}s"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
