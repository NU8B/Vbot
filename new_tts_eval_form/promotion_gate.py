"""
Model-promotion gate for candidate TTS voices.

Decides whether the candidate ("new") models in a human evaluation artifact
are good enough to replace the current ("old") production voices. The gate
enforces the promotion policy from the evaluation roadmap:

  1. minimum human emotion-recognition accuracy for candidates,
  2. minimum average naturalness, if naturalness ratings were collected,
  3. no regression in speaker similarity vs the baseline,
  4. no regression in intelligibility metrics vs the baseline.

Checks 1-2 read the human eval artifact produced by flask_app.py
(see eval_schema.py). Checks 3-4 read optional objective-metric JSON files
produced by the manual GPU evaluation flow; they are skipped when those
artifacts are not provided.

Usage:
    python promotion_gate.py results/evaluation_results.json \
        [--objective-candidate cand.json] [--objective-baseline base.json] \
        [--thresholds thresholds.json]

Exit code 0 means promotion is allowed, 1 means at least one check failed.
"""

import argparse
import json

# Tunable policy defaults. Override any subset via a --thresholds JSON file.
DEFAULT_THRESHOLDS = {
    # Five-emotion task; chance level is 20%.
    "min_emotion_accuracy": 50.0,
    # 1-5 Likert scale; only enforced when naturalness ratings exist.
    "min_naturalness": 3.0,
    # Allowed absolute drop vs baseline before a metric counts as regressed.
    "regression_tolerance": 0.0,
}

# Objective metrics the no-regression checks understand, and the direction
# that counts as "better".
SPEAKER_SIMILARITY_METRICS = {"speaker_similarity": "higher"}
INTELLIGIBILITY_METRICS = {"stoi": "higher", "pesq": "higher", "wer": "lower"}


def aggregate_human_eval(submissions, model_type="new"):
    """Aggregate emotion accuracy and naturalness for one model_type.

    Accepts the stored artifact shape: a list of submissions, each holding an
    "evaluations" list (see eval_schema.py).
    """
    total = 0
    correct = 0
    naturalness_sum = 0
    naturalness_count = 0

    for submission in submissions:
        for evaluation in submission.get("evaluations", []):
            if evaluation.get("model_type") != model_type:
                continue
            total += 1
            if evaluation.get("selected_emotion") == evaluation.get("true_emotion"):
                correct += 1
            naturalness = evaluation.get("naturalness")
            if isinstance(naturalness, int) and not isinstance(naturalness, bool):
                naturalness_sum += naturalness
                naturalness_count += 1

    return {
        "total": total,
        "accuracy": (correct / total * 100) if total else None,
        "naturalness_avg": (naturalness_sum / naturalness_count) if naturalness_count else None,
        "naturalness_count": naturalness_count,
    }


def _check(name, passed, detail):
    return {"name": name, "passed": passed, "detail": detail}


def _regression_checks(label, metric_directions, candidate, baseline, tolerance):
    """Compare candidate vs baseline for every metric both artifacts share."""
    checks = []
    shared = [m for m in metric_directions if m in candidate and m in baseline]
    if not shared:
        checks.append(
            _check(label, True, "skipped: no shared metrics between candidate and baseline artifacts")
        )
        return checks

    for metric in shared:
        cand, base = candidate[metric], baseline[metric]
        if metric_directions[metric] == "higher":
            regressed = cand < base - tolerance
        else:
            regressed = cand > base + tolerance
        checks.append(
            _check(
                f"{label}: {metric}",
                not regressed,
                f"candidate={cand:.4f} baseline={base:.4f} tolerance={tolerance}",
            )
        )
    return checks


def evaluate_promotion(
    submissions,
    thresholds=None,
    objective_candidate=None,
    objective_baseline=None,
):
    """Run all promotion checks. Returns {"passed": bool, "checks": [...], "stats": {...}}."""
    policy = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        policy.update(thresholds)

    stats = aggregate_human_eval(submissions, model_type="new")
    checks = []

    # 1. Minimum human emotion accuracy.
    if stats["accuracy"] is None:
        checks.append(
            _check("emotion accuracy", False, "no candidate ('new' model_type) evaluations found")
        )
    else:
        checks.append(
            _check(
                "emotion accuracy",
                stats["accuracy"] >= policy["min_emotion_accuracy"],
                f"{stats['accuracy']:.1f}% over {stats['total']} ratings "
                f"(minimum {policy['min_emotion_accuracy']}%)",
            )
        )

    # 2. Minimum naturalness, only if collected.
    if stats["naturalness_count"]:
        checks.append(
            _check(
                "naturalness",
                stats["naturalness_avg"] >= policy["min_naturalness"],
                f"average {stats['naturalness_avg']:.2f} over {stats['naturalness_count']} ratings "
                f"(minimum {policy['min_naturalness']})",
            )
        )
    else:
        checks.append(_check("naturalness", True, "skipped: no naturalness ratings collected"))

    # 3-4. Objective no-regression checks, only when both artifacts exist.
    if objective_candidate is not None and objective_baseline is not None:
        tolerance = policy["regression_tolerance"]
        checks.extend(
            _regression_checks(
                "speaker similarity", SPEAKER_SIMILARITY_METRICS,
                objective_candidate, objective_baseline, tolerance,
            )
        )
        checks.extend(
            _regression_checks(
                "intelligibility", INTELLIGIBILITY_METRICS,
                objective_candidate, objective_baseline, tolerance,
            )
        )
    else:
        checks.append(
            _check(
                "objective regression",
                True,
                "skipped: provide --objective-candidate and --objective-baseline to enable",
            )
        )

    return {"passed": all(check["passed"] for check in checks), "checks": checks, "stats": stats}


def _load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    # Accept both a flat metrics dict and a full tts_objective_eval artifact
    # (scripts/tts_objective_eval.py), whose metrics live under gate_metrics.
    if isinstance(payload, dict) and "gate_metrics" in payload:
        return payload["gate_metrics"]
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description="Vbot TTS model-promotion gate")
    parser.add_argument("results", help="Human evaluation artifact (evaluation_results.json)")
    parser.add_argument("--objective-candidate", help="Candidate objective-metrics JSON")
    parser.add_argument("--objective-baseline", help="Baseline objective-metrics JSON")
    parser.add_argument("--thresholds", help="JSON file overriding DEFAULT_THRESHOLDS")
    args = parser.parse_args(argv)

    report = evaluate_promotion(
        _load_json(args.results),
        thresholds=_load_json(args.thresholds) if args.thresholds else None,
        objective_candidate=_load_json(args.objective_candidate) if args.objective_candidate else None,
        objective_baseline=_load_json(args.objective_baseline) if args.objective_baseline else None,
    )

    print("Vbot model-promotion gate")
    print("=" * 40)
    for check in report["checks"]:
        marker = "PASS" if check["passed"] else "FAIL"
        print(f"[{marker}] {check['name']}: {check['detail']}")
    print("=" * 40)
    print("RESULT:", "PROMOTE" if report["passed"] else "DO NOT PROMOTE")

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
