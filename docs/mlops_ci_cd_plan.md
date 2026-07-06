# Vbot MLOps and CI/CD Implementation

Last reviewed: 2026-07-06

This document summarizes the CI/CD and model-evaluation infrastructure currently implemented for Vbot.

The design separates two concerns:

- lightweight source checks that can run on ordinary GitHub-hosted CI.
- heavyweight model/runtime checks that run manually on a prepared Windows GPU environment.

That split keeps normal code review fast while still giving model changes a measurable promotion path.

## Runtime Scope

Vbot's runtime combines:

- desktop GUI and character switching.
- Dockerized Ollama LLM runtime.
- character-specific StyleTTS2 voices.
- Faster-Whisper speech input.
- RoBERTa GoEmotions routing.
- THA4 avatar rendering.
- runtime metrics and memory instrumentation.

The CI/CD design treats this as a desktop ML runtime, not a simple web app.

## Lightweight CI

Workflow:

```text
.github/workflows/ci.yml
```

The default CI uses [requirements-ci.txt](../requirements-ci.txt), not the full desktop/GPU dependency set.

It validates:

- syntax and undefined-name checks with flake8.
- formatting/import checks for CI-owned files.
- prompt contracts for all character prompts.
- TTS-safe response cleanup.
- streaming pipeline chunking behavior.
- emotion mapping and eval math.
- eval schema validation.
- promotion-gate behavior.
- baseline loading and scorecard rendering.
- runtime metrics logging.
- memory instrumentation contracts.
- release/launcher configuration contracts.

The test suite is designed to avoid importing the Windows GUI/audio/GPU stack in CI.

## Desktop Release Workflow

Workflow:

```text
.github/workflows/desktop-release.yml
```

Build scripts:

- [build_with_logs.ps1](../build_with_logs.ps1)
- [scripts/build_with_logs.ps1](../scripts/build_with_logs.ps1)
- [scripts/package_release.ps1](../scripts/package_release.ps1)

The release path builds a portable Windows package:

```text
dist/Vbot/Vbot.exe
release/artifacts/Vbot-[version]-windows-portable.zip
release/artifacts/Vbot-[version]-windows-portable.zip.sha256
release/artifacts/Vbot-[version]-release-manifest.json
release/logs/build-[version]-[timestamp].log
```

The manual workflow targets:

```text
runs-on: [self-hosted, Windows, X64]
```

This lets the release job use the same prepared desktop/GPU environment expected by the app.

## Model Evaluation Workflow

Workflow:

```text
.github/workflows/model-eval.yml
```

The model-eval workflow is manually dispatched and can run:

- environment/CUDA check.
- Ollama startup.
- emotion classifier gate.
- LLM benchmark.
- persona judge.
- TTS objective gates for all five characters.
- Markdown scorecard generation.
- artifact upload.

The workflow is intentionally separate from default CI because it depends on local model assets, Docker/Ollama, and GPU-capable inference.

## Evaluation Commands

| Area | Command | Artifact |
| --- | --- | --- |
| Emotion routing | `scripts/emotion_eval.py` | `asset/outputs/emotion_eval/*.json` |
| LLM behavior | `scripts/llm_benchmark.py` | `asset/outputs/llm_benchmark/*.json` |
| Persona scoring | `scripts/persona_judge.py` | `asset/outputs/llm_benchmark/persona_judged_*.json` |
| TTS objective metrics | `scripts/tts_objective_eval.py` | `asset/outputs/tts_objective/*.json` |
| Scorecard | `scripts/eval_summary.py` | Markdown summary |
| Runtime metrics | `scripts/metrics_report.py` | Console summary from JSONL metrics |

## Baseline Registry

Committed baselines live under:

```text
evaluation/baselines/
```

Current baseline types:

- `emotion_eval_baseline.json`
- `persona_judged_reference.json`
- `tts_objective_<Character>_baseline.json`

These artifacts define the current accepted model behavior for regression checks and scorecards.

## Model Cards

Current shipped model behavior is summarized in:

```text
docs/MODEL_CARDS.md
```

The model cards connect the runtime models to their evaluation results:

- shared `stheno` LLM through Ollama.
- shared GoEmotions classifier.
- five StyleTTS2 character voices.
- THA4 avatar runtime.

## Runtime Observability

Runtime conversation metrics are logged through:

```text
utils/runtime_metrics.py
```

The logger records per-turn metrics such as:

- character.
- emotion.
- LLM latency.
- TTS latency.
- audio duration.
- response word count.
- streaming chunk metrics when sentence-streaming TTS is enabled.

Metrics are stored under `asset/outputs/runtime_metrics/`, which is ignored from git.

## Optional Experiment Tracking

Evaluation scripts can log to MLflow through:

```text
scripts/eval_tracking.py
```

The JSON artifacts remain the source of truth. MLflow is a local indexing and inspection layer.

Local tracking outputs are ignored from git:

```text
mlflow.db
mlartifacts/
mlruns/
```

## Verification

Use the project conda environment:

```powershell
& "C:\Users\peepz\.conda\envs\vbot\python.exe" -m pytest tests/ -q
```

Current dev1 verification:

```text
280 passed
```
