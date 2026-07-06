"""
Emotion pipeline evaluation and threshold calibration for Vbot.

Measures the RUNTIME emotion behavior, not just the raw classifier: a text
is classified (top-1 label + confidence), low-confidence predictions fall
back to neutral (the runtime threshold), and the label is mapped to one of
the 5 voice-style buckets via utils.emotion_utils.EMOTION_DEFINITIONS —
exactly what EmotionHandler.classify_emotion + the style lookup do.

Outputs per dataset: accuracy, macro-F1, per-bucket precision/recall/F1,
a bucket confusion matrix, and a threshold sweep so the hand-picked 0.3
runtime threshold can be replaced with a calibrated value.

Also acts as the classifier-swap gate: run with --model <hf-model> to
evaluate a candidate classifier and --baseline <artifact.json> to enforce
no-regression before promoting it.

Usage:
    python scripts/emotion_eval.py                       # frozen datasets, runtime model
    python scripts/emotion_eval.py --model some/other-classifier
    python scripts/emotion_eval.py --baseline asset/outputs/emotion_eval/<prev>.json
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.emotion_utils import (
    EMOTION_CONFIDENCE_THRESHOLD as RUNTIME_THRESHOLD,
    EMOTION_MODEL_NAME,
    get_emotion_bucket,
)

SCHEMA_VERSION = 1
BUCKETS = ("angry", "happy", "neutral", "sad", "surprised")
DEFAULT_SWEEP = [round(t * 0.05, 2) for t in range(0, 19)]  # 0.00 .. 0.90

DEFAULT_DATASETS = [
    os.path.join(PROJECT_ROOT, "evaluation", "emotion", "goemotions_test_slice.json"),
    os.path.join(PROJECT_ROOT, "evaluation", "emotion", "vtuber_domain_slice.json"),
]
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "asset", "outputs", "emotion_eval")

DEFAULT_GATE_TOLERANCE = 0.01  # allowed absolute macro/per-bucket F1 drop


def load_dataset(path):
    with open(path, "r", encoding="utf-8") as file:
        artifact = json.load(file)
    return artifact["name"], artifact["items"]


def predict_labels(texts, model_name, device=-1):
    """Top-1 label + confidence for each text, using the runtime's pipeline
    settings (CPU, truncation)."""
    from transformers import pipeline

    classifier = pipeline(
        "text-classification",
        model=model_name,
        top_k=1,
        truncation=True,
        device=device,
        framework="pt",
    )
    results = classifier(list(texts), batch_size=16)
    return [(r[0]["label"], float(r[0]["score"])) for r in results]


def apply_runtime_threshold(label, score, threshold):
    """Low-confidence predictions become neutral, as in classify_emotion."""
    return "neutral" if score < threshold else label


def evaluate_buckets(items, predictions, threshold):
    """Score bucket-level performance at a given confidence threshold."""
    confusion = {gold: {pred: 0 for pred in BUCKETS} for gold in BUCKETS}
    correct = 0

    for item, (label, score) in zip(items, predictions):
        gold = item["gold_bucket"]
        effective = apply_runtime_threshold(label, score, threshold)
        predicted = get_emotion_bucket(effective)
        confusion[gold][predicted] += 1
        if predicted == gold:
            correct += 1

    per_bucket = {}
    f1_values = []
    for bucket in BUCKETS:
        true_positive = confusion[bucket][bucket]
        support = sum(confusion[bucket].values())
        predicted_count = sum(confusion[gold][bucket] for gold in BUCKETS)
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_bucket[bucket] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        if support:
            f1_values.append(f1)

    total = len(items)
    neutral_predictions = sum(confusion[gold]["neutral"] for gold in BUCKETS)
    return {
        "threshold": threshold,
        "total": total,
        "accuracy": correct / total if total else 0.0,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "neutral_prediction_rate": neutral_predictions / total if total else 0.0,
        "per_bucket": per_bucket,
        "confusion": confusion,
    }


def sweep_thresholds(items, predictions, thresholds):
    """Evaluate every candidate threshold; recommend the macro-F1 argmax."""
    rows = [
        {
            "threshold": report["threshold"],
            "accuracy": report["accuracy"],
            "macro_f1": report["macro_f1"],
            "neutral_prediction_rate": report["neutral_prediction_rate"],
        }
        for report in (
            evaluate_buckets(items, predictions, threshold) for threshold in thresholds
        )
    ]
    best = max(rows, key=lambda row: row["macro_f1"])
    return {"rows": rows, "recommended_threshold": best["threshold"]}


def check_gate(current, baseline, tolerance=DEFAULT_GATE_TOLERANCE):
    """Compare per-dataset runtime metrics; return list of failure strings."""
    failures = []
    for name, report in baseline.items():
        if name not in current:
            failures.append(f"dataset '{name}' missing from current run")
            continue
        cur = current[name]
        if cur["macro_f1"] < report["macro_f1"] - tolerance:
            failures.append(
                f"{name}: macro_f1 regressed {report['macro_f1']:.3f} -> {cur['macro_f1']:.3f}"
            )
        for bucket, stats in report["per_bucket"].items():
            cur_f1 = cur["per_bucket"][bucket]["f1"]
            if cur_f1 < stats["f1"] - tolerance:
                failures.append(
                    f"{name}/{bucket}: f1 regressed {stats['f1']:.3f} -> {cur_f1:.3f}"
                )
    return failures


def main(argv=None):
    parser = argparse.ArgumentParser(description="Vbot emotion pipeline evaluation")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--model", default=EMOTION_MODEL_NAME)
    parser.add_argument("--baseline", help="previous artifact JSON to gate against")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    dataset_reports = {}
    runtime_reports = {}

    for path in args.datasets:
        name, items = load_dataset(path)
        print(f"\n[{name}] {len(items)} items — classifying with {args.model}...")
        start = time.time()
        predictions = predict_labels([item["text"] for item in items], args.model)
        elapsed = time.time() - start
        print(f"[{name}] classified in {elapsed:.1f}s ({elapsed / len(items) * 1000:.0f}ms/item)")

        runtime_report = evaluate_buckets(items, predictions, RUNTIME_THRESHOLD)
        sweep = sweep_thresholds(items, predictions, DEFAULT_SWEEP)

        dataset_reports[name] = {
            "dataset_path": os.path.relpath(path, PROJECT_ROOT),
            "items": len(items),
            "runtime_threshold_report": runtime_report,
            "threshold_sweep": sweep,
        }
        runtime_reports[name] = runtime_report

        print(f"[{name}] @runtime threshold {RUNTIME_THRESHOLD}:")
        print(
            f"  accuracy={runtime_report['accuracy']:.3f} "
            f"macro_f1={runtime_report['macro_f1']:.3f} "
            f"neutral_rate={runtime_report['neutral_prediction_rate']:.3f}"
        )
        for bucket in BUCKETS:
            stats = runtime_report["per_bucket"][bucket]
            print(
                f"  {bucket:<10} P={stats['precision']:.2f} R={stats['recall']:.2f} "
                f"F1={stats['f1']:.2f} (n={stats['support']})"
            )
        print(f"  recommended threshold (macro-F1): {sweep['recommended_threshold']}")

    artifact = {
        "schema_version": SCHEMA_VERSION,
        "kind": "emotion_eval",
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "runtime_threshold": RUNTIME_THRESHOLD,
        "datasets": dataset_reports,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = os.path.join(args.output_dir, f"emotion_eval_{stamp}.json")
    with open(out_path, "w", encoding="utf-8") as file:
        json.dump(artifact, file, indent=2, ensure_ascii=False)
    print(f"\nartifact: {out_path}")

    from eval_tracking import log_eval_run

    log_eval_run(
        experiment="emotion-eval",
        run_name=f"{os.path.basename(args.model)}_{stamp}",
        params={"model": args.model, "runtime_threshold": RUNTIME_THRESHOLD},
        metrics={
            f"{name}.{key}": report["runtime_threshold_report"][key]
            for name, report in dataset_reports.items()
            for key in ("accuracy", "macro_f1", "neutral_prediction_rate")
        },
        artifact=out_path,
    )

    if args.baseline:
        with open(args.baseline, "r", encoding="utf-8") as file:
            baseline_artifact = json.load(file)
        baseline_reports = {
            name: report["runtime_threshold_report"]
            for name, report in baseline_artifact["datasets"].items()
        }
        failures = check_gate(runtime_reports, baseline_reports)
        if failures:
            print("\nGATE: FAIL")
            for failure in failures:
                print(f"  [FAIL] {failure}")
            return 1
        print("\nGATE: PASS (no regression vs baseline)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
