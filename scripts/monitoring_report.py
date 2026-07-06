"""
Runtime monitoring gate: SLO checks + drift detection over production telemetry.

Closes the observability loop that utils/runtime_metrics.py opens: the app
logs one JSONL record per interaction, scripts/metrics_report.py summarizes
a single file, and this script turns the accumulated telemetry into a
pass/fail verdict suitable for a scheduled CI job.

What it checks, against the committed budgets in evaluation/monitoring_slos.json:
  - SLOs: p95 LLM latency, p95 time-to-first-audio (streaming), p95 TTS
    real-time factor (simple path), streaming pipeline error rate, and
    chunk completion rate. Each budget has a min_samples guard so a quiet
    day cannot produce a fake verdict.
  - Drift: total variation distance between the emotion-bucket distribution
    of the current window (default: last 7 days) and all older telemetry.
    Only issued when both windows meet min_samples_per_window; with fewer
    records the verdict is "insufficient_data", never a guess.

Exit code 1 on any SLO violation or drift alert, 0 otherwise (including
insufficient data - a monitor that fails when there is nothing to measure
just trains people to ignore it).

Usage:
    python scripts/monitoring_report.py                       # all telemetry
    python scripts/monitoring_report.py --window-days 7       # explicit window
    python scripts/monitoring_report.py --json out.json       # artifact for CI/MLflow
"""

import argparse
import glob
import json
import os
import sys
from collections import Counter
from datetime import datetime, timedelta

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from eval_tracking import log_eval_run
from metrics_report import percentile
from utils.emotion_buckets import get_emotion_bucket
from utils.runtime_metrics import metrics_dir, read_records

SCHEMA_VERSION = 1
DEFAULT_SLO_CONFIG = os.path.join(PROJECT_ROOT, "evaluation", "monitoring_slos.json")
BUCKETS = ("angry", "happy", "neutral", "sad", "surprised")


def load_slo_config(path):
    with open(path, "r", encoding="utf-8") as file:
        config = json.load(file)
    if config.get("kind") != "runtime_slos":
        raise ValueError(f"{path} is not a runtime_slos config")
    return config


def load_all_records(directory):
    """Read every daily metrics file, oldest first."""
    records = []
    for path in sorted(glob.glob(os.path.join(directory, "metrics_*.jsonl"))):
        records.extend(read_records(path))
    return records


def parse_timestamp(record):
    try:
        return datetime.fromisoformat(record["timestamp"])
    except (KeyError, TypeError, ValueError):
        return None


def split_window(records, window_days, now=None):
    """Split records into (current_window, reference) by timestamp.

    Records without a parsable timestamp count toward the current window so
    they are never silently dropped from the SLO checks.
    """
    now = now or datetime.now()
    cutoff = now - timedelta(days=window_days)
    current, reference = [], []
    for record in records:
        stamp = parse_timestamp(record)
        if stamp is not None and stamp < cutoff:
            reference.append(record)
        else:
            current.append(record)
    return current, reference


def _numeric(record, field):
    value = record.get(field)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    return None


def _slo_result(value, n, budget, min_samples):
    """Build one SLO entry; budget is {'max': x} or {'min': x}."""
    result = {"value": value, "n": n, "budget": budget}
    if n < min_samples or value is None:
        result["status"] = "insufficient_data"
    elif "max" in budget and value > budget["max"]:
        result["status"] = "fail"
    elif "min" in budget and value < budget["min"]:
        result["status"] = "fail"
    else:
        result["status"] = "pass"
    return result


def check_slos(records, slo_config):
    """Evaluate every configured SLO over the given records."""
    slos = slo_config["slos"]
    results = {}

    llm = [v for r in records if (v := _numeric(r, "llm_latency_s")) is not None]
    spec = slos["llm_latency_s_p95"]
    results["llm_latency_s_p95"] = _slo_result(percentile(llm, 95), len(llm), {"max": spec["max"]}, spec["min_samples"])

    ttfa = [v for r in records if (v := _numeric(r, "time_to_first_audio_s")) is not None]
    spec = slos["time_to_first_audio_s_p95"]
    results["time_to_first_audio_s_p95"] = _slo_result(
        percentile(ttfa, 95), len(ttfa), {"max": spec["max"]}, spec["min_samples"]
    )

    rtf = []
    for record in records:
        synth = _numeric(record, "tts_latency_s")
        duration = _numeric(record, "audio_duration_s")
        if synth is not None and duration is not None and duration > 0:
            rtf.append(synth / duration)
    spec = slos["tts_rtf_p95"]
    results["tts_rtf_p95"] = _slo_result(percentile(rtf, 95), len(rtf), {"max": spec["max"]}, spec["min_samples"])

    streaming = [r for r in records if _numeric(r, "chunks") is not None]
    total_chunks = sum(r["chunks"] for r in streaming)
    errors = sum(_numeric(r, "pipeline_errors") or 0 for r in streaming)
    played = sum(_numeric(r, "chunks_played") or 0 for r in streaming)

    spec = slos["pipeline_error_rate"]
    error_rate = errors / total_chunks if total_chunks else None
    results["pipeline_error_rate"] = _slo_result(error_rate, len(streaming), {"max": spec["max"]}, spec["min_samples"])

    spec = slos["chunk_completion_rate"]
    completion = played / total_chunks if total_chunks else None
    results["chunk_completion_rate"] = _slo_result(
        completion, len(streaming), {"min": spec["min"]}, spec["min_samples"]
    )

    return results


def bucket_distribution(records):
    """Emotion-bucket shares for records that carry an emotion label."""
    counts = Counter(get_emotion_bucket(r["emotion"]) for r in records if isinstance(r.get("emotion"), str))
    total = sum(counts.values())
    if not total:
        return {}, 0
    return {bucket: counts.get(bucket, 0) / total for bucket in BUCKETS}, total


def total_variation_distance(p, q):
    """0.5 * sum |p_b - q_b| over the shared bucket space; range [0, 1]."""
    return 0.5 * sum(abs(p.get(b, 0.0) - q.get(b, 0.0)) for b in BUCKETS)


def check_drift(current, reference, drift_config):
    """Compare current-window emotion buckets against older telemetry."""
    current_dist, current_n = bucket_distribution(current)
    reference_dist, reference_n = bucket_distribution(reference)
    min_samples = drift_config["min_samples_per_window"]

    result = {
        "current": {"n": current_n, "distribution": current_dist},
        "reference": {"n": reference_n, "distribution": reference_dist},
        "tvd_max": drift_config["emotion_bucket_tvd_max"],
    }
    if current_n < min_samples or reference_n < min_samples:
        result["status"] = "insufficient_data"
        result["tvd"] = None
    else:
        tvd = total_variation_distance(current_dist, reference_dist)
        result["tvd"] = tvd
        result["status"] = "fail" if tvd > drift_config["emotion_bucket_tvd_max"] else "pass"
    return result


def build_report(records, window_days, slo_config, now=None):
    """Assemble the full monitoring report dict (the JSON artifact)."""
    current, reference = split_window(records, window_days, now=now)
    slo_results = check_slos(current, slo_config)
    drift_result = check_drift(current, reference, slo_config["drift"])

    violations = [name for name, r in slo_results.items() if r["status"] == "fail"]
    if drift_result["status"] == "fail":
        violations.append("emotion_bucket_drift")

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "runtime_monitoring",
        "timestamp": (now or datetime.now()).isoformat(),
        "window_days": window_days,
        "records_total": len(records),
        "records_in_window": len(current),
        "by_path": dict(Counter(r.get("path", "unknown") for r in current)),
        "by_character": dict(Counter(r.get("character", "unknown") for r in current)),
        "slos": slo_results,
        "drift": drift_result,
        "violations": violations,
    }


def print_report(report):
    print(f"runtime monitoring report ({report['timestamp']})")
    print(
        f"window: last {report['window_days']} days -> {report['records_in_window']} of "
        f"{report['records_total']} interactions"
    )
    print(f"paths: {report['by_path']}  characters: {report['by_character']}")

    print(f"\n{'SLO':<28} {'value':>10} {'budget':>10} {'n':>5}  status")
    print("-" * 68)
    for name, result in report["slos"].items():
        budget = result["budget"]
        bound = f"<={budget['max']}" if "max" in budget else f">={budget['min']}"
        value = f"{result['value']:.3f}" if result["value"] is not None else "-"
        print(f"{name:<28} {value:>10} {bound:>10} {result['n']:>5}  {result['status']}")

    drift = report["drift"]
    print(f"\nemotion-bucket drift: {drift['status']}", end="")
    if drift["tvd"] is not None:
        print(f" (TVD {drift['tvd']:.3f}, max {drift['tvd_max']})")
    else:
        print(
            f" (need {report['window_days']}-day window and reference each with enough samples; "
            f"have {drift['current']['n']} / {drift['reference']['n']})"
        )
    for side in ("current", "reference"):
        dist = drift[side]["distribution"]
        if dist:
            shares = ", ".join(f"{b}={dist[b]:.2f}" for b in BUCKETS)
            print(f"  {side} (n={drift[side]['n']}): {shares}")

    if report["violations"]:
        print(f"\nVIOLATIONS: {', '.join(report['violations'])}")
    else:
        print("\nall checks passed (or lacked data for a verdict)")


def flatten_metrics(report):
    """Headline numbers for MLflow indexing."""
    metrics = {"records_in_window": report["records_in_window"], "violations": len(report["violations"])}
    for name, result in report["slos"].items():
        if result["value"] is not None:
            metrics[name] = result["value"]
    if report["drift"]["tvd"] is not None:
        metrics["emotion_bucket_tvd"] = report["drift"]["tvd"]
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description="Runtime SLO + drift monitoring gate")
    parser.add_argument("--metrics-dir", default=None, help="metrics directory (default: app telemetry dir)")
    parser.add_argument("--window-days", type=int, default=7, help="current-window size in days (default 7)")
    parser.add_argument("--slo-config", default=DEFAULT_SLO_CONFIG, help="committed SLO budgets JSON")
    parser.add_argument("--json", dest="json_out", default=None, help="write the report artifact here")
    args = parser.parse_args(argv)

    slo_config = load_slo_config(args.slo_config)
    directory = args.metrics_dir or metrics_dir()
    records = load_all_records(directory)
    if not records:
        print(f"No telemetry in {directory} — nothing to monitor yet.")
        return 0

    report = build_report(records, args.window_days, slo_config)
    print_report(report)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=2)
        print(f"\nartifact written: {args.json_out}")

    log_eval_run(
        experiment="runtime-monitoring",
        run_name=f"monitor_{report['timestamp'][:19]}",
        params={"window_days": args.window_days, "slo_config": os.path.basename(args.slo_config)},
        metrics=flatten_metrics(report),
        artifact=args.json_out,
        tags={"kind": "runtime_monitoring"},
    )

    return 1 if report["violations"] else 0


if __name__ == "__main__":
    sys.exit(main())
