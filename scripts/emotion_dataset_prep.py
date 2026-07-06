"""
Freeze the GoEmotions evaluation slice for the Vbot emotion pipeline.

Downloads the GoEmotions test split (Google Research, Apache-2.0), keeps
clean single-label examples, maps gold labels to Vbot's 5 runtime voice
buckets via utils.emotion_utils.EMOTION_DEFINITIONS, stratifies per bucket,
and freezes the slice into evaluation/emotion/goemotions_test_slice.json.

Freezing matters: the eval must be reproducible run-to-run and across
classifier swaps, so the dataset is committed to the repo rather than
re-downloaded at eval time.

Usage:
    python scripts/emotion_dataset_prep.py [--per-bucket 150] [--seed 7]
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime

import requests

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.emotion_utils import get_emotion_bucket

BASE_URL = "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data"
OUTPUT_PATH = os.path.join(
    PROJECT_ROOT, "evaluation", "emotion", "goemotions_test_slice.json"
)

SCHEMA_VERSION = 1

# GoEmotions masks entities with bracket tokens; those artifacts don't occur
# in runtime chat text, so masked rows are dropped rather than rewritten.
MASK_TOKENS = ("[NAME]", "[RELIGION]", "[TRANSGENDER]")


def fetch(path):
    response = requests.get(f"{BASE_URL}/{path}", timeout=60)
    response.raise_for_status()
    return response.text


def main(argv=None):
    parser = argparse.ArgumentParser(description="Freeze the GoEmotions eval slice")
    parser.add_argument("--per-bucket", type=int, default=150)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min-chars", type=int, default=10)
    parser.add_argument("--max-chars", type=int, default=200)
    args = parser.parse_args(argv)

    label_names = fetch("emotions.txt").splitlines()
    print(f"labels: {len(label_names)}")

    rows = fetch("test.tsv").splitlines()
    print(f"test split rows: {len(rows)}")

    candidates = {}
    for row in rows:
        parts = row.split("\t")
        if len(parts) < 2:
            continue
        text, label_ids = parts[0].strip(), parts[1].split(",")
        if len(label_ids) != 1:
            continue  # single-label examples only: unambiguous ground truth
        if not (args.min_chars <= len(text) <= args.max_chars):
            continue
        if any(token in text for token in MASK_TOKENS):
            continue
        label = label_names[int(label_ids[0])]
        bucket = get_emotion_bucket(label)
        candidates.setdefault(bucket, []).append(
            {"text": text, "gold_label": label, "gold_bucket": bucket}
        )

    rng = random.Random(args.seed)
    items = []
    for bucket in sorted(candidates):
        pool = candidates[bucket]
        rng.shuffle(pool)
        take = pool[: args.per_bucket]
        items.extend(take)
        print(f"  {bucket:<10} pool={len(pool):>5} frozen={len(take)}")

    artifact = {
        "schema_version": SCHEMA_VERSION,
        "kind": "emotion_eval_dataset",
        "name": "goemotions_test_slice",
        "source": f"{BASE_URL}/test.tsv (GoEmotions, Apache-2.0)",
        "created": datetime.now().isoformat(),
        "sampling": {
            "seed": args.seed,
            "per_bucket": args.per_bucket,
            "single_label_only": True,
            "min_chars": args.min_chars,
            "max_chars": args.max_chars,
            "masked_rows_dropped": list(MASK_TOKENS),
        },
        "items": items,
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(artifact, file, indent=2, ensure_ascii=False)
    print(f"frozen {len(items)} items -> {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
