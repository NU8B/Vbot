"""
Per-interaction runtime metrics for Vbot.

Appends one JSON line per conversation turn (LLM latency, emotion, TTS
timing, playback duration) to a daily JSONL file, so production behavior
can be inspected after the fact with scripts/metrics_report.py.

Rules:
  - Import-light (stdlib only) so it can never slow app startup.
  - Failure-safe: logging problems print a warning and return False; they
    must never break a conversation turn.
  - Location: asset/outputs/runtime_metrics/ (gitignored), overridable via
    VBOT_METRICS_DIR (used by tests).
"""

import json
import os
import threading
from datetime import datetime

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_DIR = os.path.join(_PROJECT_ROOT, "asset", "outputs", "runtime_metrics")

_write_lock = threading.Lock()


def metrics_dir():
    return os.getenv("VBOT_METRICS_DIR", _DEFAULT_DIR)


def log_interaction(**fields):
    """Append one interaction record. Returns True when written."""
    try:
        now = datetime.now()
        record = {"timestamp": now.isoformat()}
        record.update(fields)

        directory = metrics_dir()
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"metrics_{now:%Y-%m-%d}.jsonl")

        line = json.dumps(record, ensure_ascii=False, default=str)
        with _write_lock:
            with open(path, "a", encoding="utf-8") as file:
                file.write(line + "\n")
        return True
    except Exception as exc:  # noqa: BLE001 - metrics must never break the app
        print(f"[WARN] runtime metrics skipped: {exc}")
        return False


def read_records(path):
    """Read a JSONL metrics file, skipping unparsable lines."""
    records = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records
