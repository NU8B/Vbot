"""
Runtime-compatible LLM benchmark for Vbot characters.

Benchmarks the SAME language stack the app ships: the Dockerized Ollama
runtime (model alias `stheno`) driven by the production character prompts
from utils/ollama_utils.py. This replaces the archived HF-loading benchmark
(asset/old/benchmark/benchmark_LLM.py), which duplicated the 8B model
outside Docker and only scored a hard-coded Amelia profile.

Metrics per response:
  - TTS safety: violations of the prompt contract (asterisks, parentheses,
    brackets, emoji) that would pollute the speech pipeline, plus how much
    utils.response_filter would have to strip.
  - Brevity: word count vs the "under 30 words" instruction.
  - First person: does the character speak as itself.
  - Character break: "as an AI"-style break phrases.
  - Persona/topic adherence: per-character keyword profiles (extended from
    the archived Amelia-only profile to all five characters).
  - Latency and tokens/sec from Ollama's own counters.

The archived benchmark's BERT sentiment metric was dropped deliberately:
its implementation ignored the model prediction (returned the character
prior regardless of text), and star-rating sentiment cannot measure persona
emotion anyway.

Usage (requires Docker Desktop + the ollama container, like the app):
    python scripts/llm_benchmark.py                     # all characters
    python scripts/llm_benchmark.py --characters Amelia Gura
    python scripts/llm_benchmark.py --output-dir path/to/dir

Artifacts land in asset/outputs/llm_benchmark/ as schema-versioned JSON
with prompt, response, character, timestamp, and metric breakdown.
"""

import argparse
import ast
import json
import os
import re
import sys
import time
from datetime import datetime

import requests

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.response_filter import filter_action_text

SCHEMA_VERSION = 1

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11500")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))
RUNTIME_MODEL = "stheno"
# Mirrors MAX_LENGTH in utils/ollama_utils.py so the benchmark generates
# under the same budget as the app.
NUM_PREDICT = 150

DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "asset", "outputs", "llm_benchmark")

# Shared battery: ordinary interactions plus bait prompts that tempt the
# model into breaking the TTS contract.
PROMPT_BATTERY = [
    {"id": "greeting", "text": "Hi! How are you today?"},
    {"id": "self_intro", "text": "Tell me about yourself."},
    {"id": "favorite_activity", "text": "What do you love doing the most?"},
    {"id": "emotional_support", "text": "I had a really rough day at work today."},
    {"id": "roleplay_bait", "text": "*hugs you tightly* I missed you so much!"},
    {"id": "emoji_bait", "text": "Reply with lots of emojis! What's your favorite emoji?"},
    {
        "id": "verbosity_bait",
        "text": "Explain your entire life story in full detail, don't leave anything out.",
    },
    {"id": "identity_probe", "text": "Are you a real person or an AI?"},
    {"id": "game_invite", "text": "Want to play a game together later?"},
    {"id": "lore_question", "text": "What's the most interesting thing you've ever done?"},
]

# Persona/topic keyword profiles for every active character, extended from
# the archived Amelia-only profile. A response "adheres" when it contains at
# least one keyword; rates are aggregated over the whole battery.
CHARACTER_PROFILES = {
    "Amelia": {
        "persona_keywords": [
            "detective", "time", "travel", "watson", "case", "mystery",
            "investigate", "clue", "evidence", "pocket watch", "timeline",
        ],
        "topic_keywords": ["mystery", "case", "game", "gaming", "time travel", "adventure"],
    },
    "Eveland": {
        "persona_keywords": [
            "novel", "book", "write", "writing", "story", "stories",
            "author", "ink", "chapter", "sweden", "swedish", "read",
        ],
        "topic_keywords": ["book", "story", "horror", "romance", "writing", "novel"],
    },
    "Gura": {
        "persona_keywords": [
            "shark", "ocean", "sea", "water", "swim", "apex", "predator",
            "fish", "atlantis", "chomp", "fin",
        ],
        "topic_keywords": ["ocean", "sing", "singing", "rhythm", "game", "shark"],
    },
    "Shiori": {
        "persona_keywords": [
            "archive", "story", "stories", "book", "knowledge", "tale",
            "secret", "lore", "page", "mysterious", "novella",
        ],
        "topic_keywords": ["story", "book", "knowledge", "mystery", "archive", "tale"],
    },
    "Wilson": {
        "persona_keywords": [
            "help", "support", "here for you", "listen", "care", "guide",
            "steady", "trust", "together", "reliable",
        ],
        "topic_keywords": ["help", "support", "advice", "listen", "care"],
    },
}

# Word-boundary patterns so "an ai" does not false-positive on "an air of".
BREAK_PATTERNS = [
    re.compile(pattern)
    for pattern in (
        r"\bas an ai\b",
        r"\ban ai\b",
        r"\blanguage model\b",
        r"\bartificial intelligence\b",
        r"\badvanced ai\b",
        r"\bchatbot\b",
        r"\bmy programming\b",
        r"\bcannot roleplay\b",
    )
]

EMOJI_PATTERN = re.compile(
    "["
    "\U0001f300-\U0001faff"
    "\U00002600-\U000027bf"
    "\U0001f1e6-\U0001f1ff"
    "\U0000fe0f"
    "]"
)


def load_model_prompts():
    """Read MODEL_PROMPTS from utils/ollama_utils.py without importing the
    desktop stack (same AST pattern as the CI tests)."""
    path = os.path.join(PROJECT_ROOT, "utils", "ollama_utils.py")
    with open(path, "r", encoding="utf-8") as file:
        tree = ast.parse(file.read())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "MODEL_PROMPTS":
                    return ast.literal_eval(node.value)
    raise RuntimeError("MODEL_PROMPTS not found in utils/ollama_utils.py")


def score_response(text, profile):
    """Compute the full metric breakdown for one generated response."""
    lower = text.lower()
    words = text.split()

    violations = {
        "asterisks": text.count("*"),
        "parentheses": len(re.findall(r"\([^)]*\)", text)),
        "brackets": len(re.findall(r"\[[^\]]*\]", text)),
        "emoji": len(EMOJI_PATTERN.findall(text)),
    }
    filtered = filter_action_text(text)

    return {
        "tts_safe": not any(violations.values()),
        "tts_violations": violations,
        "filter_changed_response": filtered != text.strip(),
        "word_count": len(words),
        "brevity_ok": len(words) <= 30,
        "first_person": bool(re.search(r"\b(I|I'm|I've|my|me)\b", text)),
        "character_break": any(pattern.search(lower) for pattern in BREAK_PATTERNS),
        "persona_hit": any(k in lower for k in profile["persona_keywords"]),
        "topic_hit": any(k in lower for k in profile["topic_keywords"]),
    }


def aggregate_results(results):
    """Aggregate per-response metrics into per-character rates."""
    total = len(results)
    if total == 0:
        return {}

    def rate(key):
        return sum(1 for r in results if r["metrics"][key]) / total

    return {
        "responses": total,
        "tts_safety_rate": rate("tts_safe"),
        "filter_intervention_rate": rate("filter_changed_response"),
        "brevity_rate": rate("brevity_ok"),
        "avg_word_count": sum(r["metrics"]["word_count"] for r in results) / total,
        "first_person_rate": rate("first_person"),
        "character_break_rate": rate("character_break"),
        "persona_adherence_rate": rate("persona_hit"),
        "topic_adherence_rate": rate("topic_hit"),
        "avg_latency_s": sum(r["latency_s"] for r in results) / total,
        "avg_tokens_per_s": sum(r["tokens_per_s"] for r in results) / total,
    }


def check_ollama(host):
    """Return True when the Ollama server responds and lists the model."""
    try:
        response = requests.get(f"{host}/api/tags", timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"Ollama not reachable at {host}: {exc}")
        return False
    models = [m.get("name", "") for m in response.json().get("models", [])]
    if not any(name.split(":")[0] == RUNTIME_MODEL for name in models):
        print(f"Model '{RUNTIME_MODEL}' not found in Ollama. Available: {models}")
        return False
    return True


def call_character(host, system_prompt, user_prompt):
    """One non-streaming chat call shaped like the app's runtime request."""
    start = time.time()
    response = requests.post(
        f"{host}/api/chat",
        json={
            "model": RUNTIME_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"num_predict": NUM_PREDICT},
        },
        timeout=OLLAMA_TIMEOUT,
    )
    latency = time.time() - start
    response.raise_for_status()
    payload = response.json()

    eval_count = payload.get("eval_count", 0)
    eval_duration_s = payload.get("eval_duration", 0) / 1e9
    return {
        "text": payload.get("message", {}).get("content", "").strip(),
        "latency_s": latency,
        "prompt_eval_count": payload.get("prompt_eval_count", 0),
        "eval_count": eval_count,
        "tokens_per_s": (eval_count / eval_duration_s) if eval_duration_s > 0 else 0.0,
    }


def benchmark_character(host, character, system_prompt, battery):
    profile = CHARACTER_PROFILES[character]
    results = []
    for prompt in battery:
        print(f"  [{character}] {prompt['id']}...", end=" ", flush=True)
        reply = call_character(host, system_prompt, prompt["text"])
        metrics = score_response(reply["text"], profile)
        print(
            f"{reply['latency_s']:.1f}s, {metrics['word_count']}w,"
            f" {'safe' if metrics['tts_safe'] else 'VIOLATION'}"
        )
        results.append(
            {
                "prompt_id": prompt["id"],
                "user_prompt": prompt["text"],
                "response": reply["text"],
                "filtered_response": filter_action_text(reply["text"]),
                "latency_s": reply["latency_s"],
                "prompt_eval_count": reply["prompt_eval_count"],
                "eval_count": reply["eval_count"],
                "tokens_per_s": reply["tokens_per_s"],
                "metrics": metrics,
            }
        )
    return {"results": results, "aggregate": aggregate_results(results)}


def build_artifact(host, characters):
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "llm_benchmark",
        "timestamp": datetime.now().isoformat(),
        "ollama_host": host,
        "model": RUNTIME_MODEL,
        "num_predict": NUM_PREDICT,
        "battery": PROMPT_BATTERY,
        "characters": characters,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Vbot runtime LLM benchmark (Ollama)")
    parser.add_argument(
        "--characters",
        nargs="+",
        default=sorted(CHARACTER_PROFILES.keys()),
        help="Characters to benchmark (default: all)",
    )
    parser.add_argument("--host", default=OLLAMA_HOST, help="Ollama host URL")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    prompts = load_model_prompts()
    unknown = [c for c in args.characters if c not in prompts or c not in CHARACTER_PROFILES]
    if unknown:
        print(f"Unknown characters: {unknown}. Known: {sorted(CHARACTER_PROFILES)}")
        return 2

    if not check_ollama(args.host):
        print("Start Docker Desktop and the ollama container first (the app does this on launch).")
        return 1

    character_reports = {}
    for character in args.characters:
        print(f"\nBenchmarking {character} ({len(PROMPT_BATTERY)} prompts)...")
        character_reports[character] = benchmark_character(
            args.host, character, prompts[character], PROMPT_BATTERY
        )

    artifact = build_artifact(args.host, character_reports)

    os.makedirs(args.output_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = os.path.join(args.output_dir, f"llm_benchmark_{stamp}.json")
    with open(out_path, "w", encoding="utf-8") as file:
        json.dump(artifact, file, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 72}")
    header = (
        f"{'character':<10} {'tts_safe':>8} {'brevity':>8} {'persona':>8}"
        f" {'breaks':>7} {'avg_words':>9} {'avg_lat':>8} {'tok/s':>7}"
    )
    print(header)
    print("-" * 72)
    for character, report in character_reports.items():
        agg = report["aggregate"]
        print(
            f"{character:<10} {agg['tts_safety_rate']:>7.0%} {agg['brevity_rate']:>7.0%}"
            f" {agg['persona_adherence_rate']:>7.0%} {agg['character_break_rate']:>6.0%}"
            f" {agg['avg_word_count']:>9.1f} {agg['avg_latency_s']:>7.1f}s"
            f" {agg['avg_tokens_per_s']:>7.1f}"
        )
    print(f"{'=' * 72}\nArtifact: {out_path}")

    from eval_tracking import log_eval_run

    log_eval_run(
        experiment="llm-benchmark",
        run_name=f"{RUNTIME_MODEL}_{stamp}",
        params={"model": RUNTIME_MODEL, "num_predict": NUM_PREDICT, "battery": len(PROMPT_BATTERY)},
        metrics={
            f"{character}.{key}": report["aggregate"][key]
            for character, report in character_reports.items()
            for key in ("tts_safety_rate", "brevity_rate", "persona_adherence_rate",
                        "character_break_rate", "avg_latency_s", "avg_tokens_per_s")
        },
        artifact=out_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
