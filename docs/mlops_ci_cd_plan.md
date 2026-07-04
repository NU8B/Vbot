# Vbot MLOps and CI/CD Implementation Plan

Last reviewed: 2026-07-01

## Goal

Add a practical CI/CD foundation to Vbot without pretending that the desktop GUI, CUDA runtime, model weights, and human evaluation workflows can all run on a small GitHub-hosted runner. The first target is a reliable CI gate on every push/PR. The second target is a manual release/build path for the Windows executable. Continuous training and model promotion come after those basics are trustworthy.

## Current Architecture Review

### Application Flow

1. User selects a VTuber avatar in the desktop UI.
2. Vbot initializes the selected character stack: avatar assets, StyleTTS2 voice model, RoBERTa emotion classifier, audio processor, and Ollama LLM handler.
3. User sends text or microphone input.
4. Speech-to-text uses Faster-Whisper for voice input.
5. Ollama generates the character response.
6. `utils.emotion_utils.EmotionHandler` classifies user/assistant text with `SamLowe/roberta-base-go_emotions`.
7. `utils.TTS_utils.InferenceHandler` chooses character-specific reference audio and StyleTTS2 alpha/beta/embedding parameters.
8. StyleTTS2 synthesizes speech, audio playback starts, and THA4 avatar emotion/speaking animation is updated.

### TTS Data and Training Pipeline

| Area | Status | Evidence | Notes |
| --- | --- | --- | --- |
| YouTube audio ingestion | Exists | `Data_prep/YT_dataset_maker.py` | Automated entry point for downloading source audio. |
| Vocal isolation | Exists | `Data_prep/audio_preprocessor/vocal_isolator.py` | Uses Demucs-style separation and resampling. |
| VAD, segmentation, quality gates | Exists | `Data_prep/audio_preprocessor/audio_segmenter_v2.py` | Has Silero VAD plus PESQ, STOI, ZCR, spectral, RMS, and duration thresholds. |
| Human segment review | Exists | `Data_prep/segment_reviewer/segment_reviewer.py` | Flask review app for approving/rejecting generated segments. |
| StyleTTS2 data prep | Exists | `Data_prep/data_StyleTTS2.py` | Normalizes text, phonemizes, prepares train/val lists. |
| StyleTTS2 training | Exists but manual | `StyleTTS2/train_finetune.py`, `StyleTTS2/train_finetune_accelerate.py`, `StyleTTS2/Configs/config_ft.yml`, `colab/StyleTTS_ft.ipynb` | Training logs to TensorBoard and saves checkpoints, but is not a repeatable CI/CT job yet. |
| Experiment tracking | Partial | `training/wandb_config.py` | Helper exists, but it is not wired into StyleTTS2 training scripts/notebook. |

### TTS Evaluation Pipeline

| Area | Status | Evidence | Notes |
| --- | --- | --- | --- |
| Human TTS model comparison | Exists | `new_tts_eval_form/flask_app.py` | Compares old/new model groups by emotion recognition accuracy and naturalness ratings. |
| Eval sample generation | Exists | `utils/TEST_multiple_model_inference.py`, `utils/MOS_EVAL_TTS.py` | Generates specific, generic, and dynamic emotion samples. Some older scripts need cleanup before CI use. |
| Objective model metrics | Exists in archive | `asset/old/benchmark/benchmark_TTS.py`, `asset/old/benchmark/benchmark_multiple_TTS.py` | Metrics include speaker similarity, PESQ, STOI, SDR, transcription accuracy, MCD, prosody similarity, and spectral convergence. |
| Evaluation result artifacts | Partial | JSON outputs under benchmark/eval apps | Needs a standard output schema and promotion thresholds. |

### LLM Evaluation Pipeline

| Area | Status | Evidence | Notes |
| --- | --- | --- | --- |
| Runtime LLM orchestration | Exists | `utils/ollama_utils.py` | Character prompts, Ollama calls, streaming/static response handling, emotion blending. |
| LLM benchmark | Exists in archive | `asset/old/benchmark/benchmark_LLM.py` | Measures personality consistency, topic relevance, emotional consistency, and reference similarity. It is hard-coded around Amelia and is not CI-safe. |
| Production LLM eval gate | Missing | N/A | Needs a light prompt-contract test in CI and a manual benchmark workflow for heavyweight local/HF models. |

### Emotion Classifier Pipeline

| Area | Status | Evidence | Notes |
| --- | --- | --- | --- |
| Emotion inference | Exists | `utils/emotion_utils.py` | Uses pretrained GoEmotions RoBERTa and maps fine-grained emotions to StyleTTS2 reference styles. |
| Per-character emotion tuning | Exists | `MODEL_PARAMS` in `utils/emotion_utils.py` | Character-specific alpha/beta/embedding settings exist for Amelia, Eveland, Gura, Shiori, and Wilson. |
| Emotion classifier fine-tuning | Missing | N/A | There is no training/fine-tune pipeline for the classifier itself. That is okay for now; evaluate the current pretrained classifier before adding training complexity. |

## Review Comments

1. The first implementation plan was directionally good but too broad. CI/CD should come before FastAPI, Docker Compose, Kubernetes, or CT.
2. The repo already had CI scaffolding, tests, and a WandB helper, but the initial CI would have been fragile on GitHub-hosted runners because it imported Windows GUI/audio modules and scanned archived scripts.
3. The default CI should not install the full `requirements.txt`. That file includes CUDA, Windows GUI, audio-driver, packaging, and model-training dependencies.
4. Archived benchmark code should not block the default CI gate. It should be cleaned or moved into a manual evaluation workflow later.
5. `BUILD_INSTRUCTIONS.md` and `QUICK_BUILD.txt` mention `build_with_logs.ps1`, but that script is not currently in the repo. That is a concrete CD task.
6. `training/wandb_config.py` is useful, but it does not create experiment tracking until the StyleTTS2 scripts or notebook call it.
7. `utils/MOS_EVAL_TTS.py` and some old inference scripts appear stale against the current `emotion_utils.py` API because they reference constants that are no longer exported. Treat them as cleanup targets before using them in automation.
8. The formal project report appendix still documents a source-based launch flow (`python Vbot.py`) even though `USER_GUIDE.md` and `BUILD_INSTRUCTIONS.md` describe a prebuilt `Vbot.exe` distribution. The report should be updated with a productization/deployment section before using it as portfolio evidence.

## CI/CD Tracking

| ID | Work Item | Status | Owner/Runner | Notes |
| --- | --- | --- | --- | --- |
| CI-1 | Add default GitHub Actions workflow | Done | GitHub-hosted Ubuntu | `.github/workflows/ci.yml` runs on push, PR, and manual dispatch. |
| CI-2 | Add CPU-only CI dependency set | Done | GitHub-hosted Ubuntu | `requirements-ci.txt` avoids the full CUDA/desktop environment. |
| CI-3 | Add lightweight unit/source-contract tests | Done | GitHub-hosted Ubuntu | Tests cover text normalization, TTS quality thresholds, core configs, and source contracts for desktop/audio/LLM modules. |
| CI-4 | Exclude vendor/archive/generated folders from syntax lint | Done | GitHub-hosted Ubuntu | Excludes `StyleTTS2`, `tha4`, `asset/old`, `.codex`, caches, build outputs, and Colab notebooks. |
| CI-5 | Add Flask eval-form tests | Todo | GitHub-hosted Ubuntu | Use Flask test client for `/`, `/submit`, and `/results` with temporary result/audio fixtures. |
| CI-6 | Add pure LLM prompt-contract tests | Todo | GitHub-hosted Ubuntu | Parse/validate `MODEL_PROMPTS` and character coverage without calling Ollama. |
| CI-7 | Add optional Windows smoke job | Todo | Windows runner | Import Windows GUI modules and check PyInstaller spec syntax. Keep non-blocking at first. |
| CD-1 | Add `scripts/build_with_logs.ps1` | Done | Local Windows or self-hosted Windows | Root wrapper `build_with_logs.ps1` calls the script under `scripts/`. |
| CD-2 | Add manual release-build workflow | Done | Self-hosted Windows recommended | `.github/workflows/desktop-release.yml` builds the portable zip on `[self-hosted, Windows, X64]`. |
| CD-3 | Upload release artifact/checksum | Done | Release workflow | Workflow uploads zip, `.sha256`, manifest, release notes, and build logs. Draft GitHub Release is optional. |
| CD-4 | Document the prebuilt executable in the formal report | Todo | Docs | Add a packaged-product flow: unzip/install, prerequisites, first-run model download, offline/local behavior, limitations, and build verification. |
| CD-5 | Decide installer vs portable zip | Done for desktop package | Product/release | The desktop package ships a reproducible portable zip. True installer/code signing is a later product polish phase. |
| CD-6 | Add launcher build mode | Done | Local Windows or self-hosted Windows | `vbot_launcher.spec` creates a real `Vbot.exe` launcher without freezing the whole ML runtime. |
| CD-7 | Harden full frozen bundle | Todo | Local Windows or self-hosted Windows | `vbot.spec` is still available, but PyInstaller can stall while analyzing ML packages. Treat as experimental until fixed. |
| CD-8 | Add true signed installer | Future | Product/release | Consider Inno Setup, NSIS, WiX, or MSIX after portable release is stable. |
| CT-1 | Standardize TTS benchmark result schema | Todo | Local GPU or self-hosted GPU | Normalize human eval and objective metrics into a single JSON schema. |
| CT-2 | Wire WandB into StyleTTS2 training | Todo | Training environment | Use existing `training/wandb_config.py` from `train_finetune*.py` or the Colab notebook. |
| CT-3 | Add manual model-evaluation workflow | Todo | Self-hosted GPU | Run objective TTS metrics and upload JSON/summary artifacts. |
| CT-4 | Add model promotion thresholds | Todo | Self-hosted GPU | Promote only if metrics beat baseline and human eval passes agreed thresholds. |

## Phase Plan

### Phase 1 - Reliable CI Foundation

Status: Done for the first pass.

Deliverables:

- `.github/workflows/ci.yml`
- `requirements-ci.txt`
- `tests/test_imports.py`
- `tests/test_data_pipeline.py`
- Portable source-contract checks for Windows/audio/LLM modules
- Local verification: syntax lint passed, Black/isort passed for CI-owned files, and pytest reported 49 passing tests

### Phase 2 - Evaluation Tests and Artifacts

Status: Next.

Tasks:

- Add Flask tests for `new_tts_eval_form/flask_app.py`.
- Create a small fixture-based test for the evaluation JSON aggregation logic.
- Move or wrap stale TTS eval scripts so they do not rely on removed constants.
- Define a single evaluation artifact schema:
  - model name
  - model version or Hugging Face repo id
  - dataset/sample set id
  - human emotion accuracy
  - average naturalness
  - speaker similarity
  - PESQ
  - STOI
  - transcription accuracy
  - MCD/prosody/spectral metrics where available

### Phase 3 - CD for Desktop Builds

Status: Desktop package implemented.

Tasks:

- Use `.\build_with_logs.ps1 -Version "v0.1.0"` for local release builds.
- Package Vbot as a portable zip containing `Vbot.exe`, app payload/assets, README, prerequisites, release notes, and build log.
- Use `.github/workflows/desktop-release.yml` for manual self-hosted Windows release builds.
- Keep the workflow self-hosted because the app still needs a prepared Windows/Python/CUDA/model environment, and full bundle mode needs even more local setup.
- Publish SHA256 checksum and JSON release manifest for every package.
- Update the project report so it no longer only describes the developer/source install path.

Reference doc: `docs/desktop_release_cd.md`

### Phase 4 - Experiment Tracking and Model Registry

Status: Planned.

Tasks:

- Wire `training/wandb_config.py` into `StyleTTS2/train_finetune.py` and/or `train_finetune_accelerate.py`.
- Log config, losses, validation loss, checkpoint metadata, and evaluation metrics.
- Keep training logs and benchmark summaries linked from the README.
- Avoid claiming a model registry until checkpoints are actually versioned as artifacts.

### Phase 5 - Continuous Training and Promotion

Status: Future.

Tasks:

- Create a manual CT script that chains ingestion, preprocessing, data prep, training, evaluation, and promotion.
- Run only on a self-hosted GPU or cloud GPU runner.
- Add promotion rules based on objective metrics plus human eval results.
- Keep CT manual until failures and cost are predictable.

### Phase 6 - API, Docker Compose, and Kubernetes

Status: Optional future portfolio extension.

Notes:

- FastAPI/model serving is useful but not required for the immediate CI/CD goal.
- Docker Compose is valuable after an API exists.
- Kubernetes should wait until the Docker/API path is real and deployable.

## Definition of Done

The CI/CD upgrade is genuinely useful when:

1. Every push/PR runs a green CI gate without requiring GPU, Docker Desktop, microphone access, Windows GUI libraries, or model downloads.
2. The human TTS eval form has automated tests around result submission and aggregation.
3. The Windows build path has one reproducible launcher script path and one manual workflow.
4. Heavy TTS/LLM/model evaluation runs are manual, artifact-producing, and do not block normal code review.
5. Training metrics and model comparison results are stored in a consistent place and can be explained in an interview.
