"""
LLM-as-judge persona fidelity scoring for Vbot characters.

Consumes an artifact produced by scripts/llm_benchmark.py and scores every
response with an INDEPENDENT judge model served by the same Ollama runtime
(default: mistral:latest — a different model family than the stheno
generator, avoiding self-preference bias). Generation and judgment are
deliberately separate scripts so old artifacts can be re-judged when the
rubric evolves, and different judge models can be compared on identical
responses.

The judge receives the character's actual production system prompt as the
persona specification — the rubric can never drift from what the runtime
demands — and scores three dimensions from 1-5:

  persona_voice  does it sound like this specific character?
  engagement     does it respond naturally to the user's message?
  kayfabe        does it stay fully in character (no AI/assistant leaks)?

Usage (requires the ollama container, like the benchmark):
    python scripts/persona_judge.py                       # newest artifact
    python scripts/persona_judge.py path/to/llm_benchmark_X.json
    python scripts/persona_judge.py --judge-model mistral:latest
"""

import argparse
import glob
import json
import os
import re
import sys
import time
from datetime import datetime

import requests

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from llm_benchmark import DEFAULT_OUTPUT_DIR, OLLAMA_HOST, OLLAMA_TIMEOUT, load_model_prompts

SCHEMA_VERSION = 1
# v2: added hard kayfabe rule + scoring anchors after the v1 calibration
# check showed mistral scoring an explicit "As an AI language model" break
# at kayfabe 5. Judged artifacts are only comparable within a version.
JUDGE_PROMPT_VERSION = 2
DEFAULT_JUDGE_MODEL = "mistral:latest"

DIMENSIONS = ("persona_voice", "engagement", "kayfabe")

JUDGE_PROMPT_TEMPLATE = """You are a strict evaluator of roleplay fidelity for a virtual companion app. Most responses you see are decent; your job is to find the flaws. Do not give 5s by default.

CHARACTER SPECIFICATION (the persona the assistant must embody):
---
{system_prompt}
---

USER MESSAGE:
{user_prompt}

CHARACTER'S RESPONSE:
{response}

Score the response on three dimensions, each an integer from 1 (bad) to 5 (excellent):
- "persona_voice": Does it sound like THIS specific character? 5 = unmistakably this character's vocabulary, themes, and tone. 3 = pleasant but generic, could be any friendly character. 1 = wrong character or no personality at all (e.g. corporate boilerplate).
- "engagement": Does it respond naturally and appropriately to the user's message? 1 = ignores or contradicts what the user said.
- "kayfabe": Does it stay fully inside the fiction?

HARD RULE for kayfabe — check this FIRST, before anything else:
If the response mentions being an AI, artificial intelligence, a language model, a chatbot, a program, an assistant, or "simulating"/"roleplaying" a persona, then kayfabe MUST be 1, no matter how good the rest of the response is. Example of a kayfabe=1 response: "As an AI language model, I can simulate this persona." Only score kayfabe 5 when there is zero hint that the character is artificial.

Reply with ONLY a JSON object, no other text:
{{"persona_voice": <1-5>, "engagement": <1-5>, "kayfabe": <1-5>, "justification": "<one short sentence>"}}"""


def build_judge_prompt(system_prompt, user_prompt, response):
    return JUDGE_PROMPT_TEMPLATE.format(
        system_prompt=system_prompt.strip(),
        user_prompt=user_prompt,
        response=response,
    )


def parse_judge_reply(text):
    """Extract the scores JSON from a judge reply.

    Tolerates fencing and surrounding prose. Returns a dict with integer
    scores clamped to [1, 5] plus the justification, or None when the reply
    is unusable.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    scores = {}
    for dimension in DIMENSIONS:
        value = payload.get(dimension)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        scores[dimension] = max(1, min(5, int(round(value))))
    scores["justification"] = str(payload.get("justification", "")).strip()
    return scores


def aggregate_judgments(judgments):
    """Mean per dimension plus the judged break rate (kayfabe <= 2)."""
    scored = [j for j in judgments if j.get("scores")]
    if not scored:
        return {"judged": 0, "unparsable": len(judgments)}

    aggregate = {"judged": len(scored), "unparsable": len(judgments) - len(scored)}
    for dimension in DIMENSIONS:
        aggregate[f"avg_{dimension}"] = sum(j["scores"][dimension] for j in scored) / len(scored)
    aggregate["kayfabe_break_rate"] = sum(1 for j in scored if j["scores"]["kayfabe"] <= 2) / len(scored)
    return aggregate


def call_judge(host, judge_model, prompt):
    response = requests.post(
        f"{host}/api/chat",
        json={
            "model": judge_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            # Deterministic-leaning judging; enough tokens for JSON + sentence.
            "options": {"temperature": 0, "num_predict": 200},
        },
        timeout=OLLAMA_TIMEOUT,
    )
    response.raise_for_status()
    return response.json().get("message", {}).get("content", "")


def judge_artifact(artifact, host, judge_model, prompts):
    """Score every response in a benchmark artifact. Returns judgments dict."""
    characters = {}
    for character, report in artifact["characters"].items():
        system_prompt = prompts[character]
        judgments = []
        for result in report["results"]:
            prompt = build_judge_prompt(system_prompt, result["user_prompt"], result["response"])
            start = time.time()
            reply = call_judge(host, judge_model, prompt)
            scores = parse_judge_reply(reply)
            if scores is None:
                # One retry: small judge models occasionally wrap or garble
                # the JSON on the first attempt.
                reply = call_judge(host, judge_model, prompt)
                scores = parse_judge_reply(reply)
            judgments.append(
                {
                    "prompt_id": result["prompt_id"],
                    "scores": scores,
                    "judge_latency_s": time.time() - start,
                }
            )
            status = (
                f"pv={scores['persona_voice']} en={scores['engagement']} " f"kf={scores['kayfabe']}"
                if scores
                else "UNPARSABLE"
            )
            print(f"  [{character}] {result['prompt_id']}: {status}")
        characters[character] = {
            "judgments": judgments,
            "aggregate": aggregate_judgments(judgments),
        }
    return characters


def find_latest_artifact():
    candidates = sorted(glob.glob(os.path.join(DEFAULT_OUTPUT_DIR, "llm_benchmark_*.json")))
    # Skip already-judged outputs of this script.
    candidates = [c for c in candidates if "judged" not in os.path.basename(c)]
    return candidates[-1] if candidates else None


def main(argv=None):
    parser = argparse.ArgumentParser(description="Judge persona fidelity of a benchmark artifact")
    parser.add_argument("artifact", nargs="?", help="llm_benchmark artifact JSON (default: newest)")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--host", default=OLLAMA_HOST)
    args = parser.parse_args(argv)

    artifact_path = args.artifact or find_latest_artifact()
    if not artifact_path:
        print("No llm_benchmark artifact found; run scripts/llm_benchmark.py first.")
        return 2
    with open(artifact_path, "r", encoding="utf-8") as file:
        artifact = json.load(file)
    print(f"judging: {artifact_path}")
    print(f"judge model: {args.judge_model} (generator was {artifact['model']})")

    prompts = load_model_prompts()
    characters = judge_artifact(artifact, args.host, args.judge_model, prompts)

    judged = {
        "schema_version": SCHEMA_VERSION,
        "kind": "persona_judgment",
        "timestamp": datetime.now().isoformat(),
        "judge_model": args.judge_model,
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "source_artifact": os.path.basename(artifact_path),
        "generator_model": artifact["model"],
        # Prompt versions the benchmark ran against (absent in artifacts
        # produced before the registry existed).
        "prompt_versions": artifact.get("prompt_versions"),
        "characters": characters,
    }

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = os.path.join(os.path.dirname(artifact_path), f"persona_judged_{stamp}.json")
    with open(out_path, "w", encoding="utf-8") as file:
        json.dump(judged, file, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 64}")
    print(f"{'character':<10} {'voice':>6} {'engage':>7} {'kayfabe':>8} {'breaks':>7} {'judged':>7}")
    print("-" * 64)
    for character, report in characters.items():
        agg = report["aggregate"]
        if not agg.get("judged"):
            print(f"{character:<10} (no parsable judgments)")
            continue
        print(
            f"{character:<10} {agg['avg_persona_voice']:>6.2f} {agg['avg_engagement']:>7.2f}"
            f" {agg['avg_kayfabe']:>8.2f} {agg['kayfabe_break_rate']:>6.0%}"
            f" {agg['judged']:>4}/{agg['judged'] + agg['unparsable']}"
        )
    print(f"{'=' * 64}\nartifact: {out_path}")

    from eval_tracking import log_eval_run

    log_eval_run(
        experiment="persona-judge",
        run_name=f"{args.judge_model.split(':')[0]}_v{JUDGE_PROMPT_VERSION}_{stamp}",
        params={
            "judge_model": args.judge_model,
            "judge_prompt_version": JUDGE_PROMPT_VERSION,
            "generator_model": artifact["model"],
            "source_artifact": os.path.basename(artifact_path),
            **{f"prompt_version_{name}": version for name, version in (artifact.get("prompt_versions") or {}).items()},
        },
        metrics={
            f"{character}.{key}": report["aggregate"][key]
            for character, report in characters.items()
            if report["aggregate"].get("judged")
            for key in ("avg_persona_voice", "avg_engagement", "avg_kayfabe", "kayfabe_break_rate")
        },
        artifact=out_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
