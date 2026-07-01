# Vbot Runtime Prerequisites

This release is a Windows portable desktop build. Extract the zip, open the `Vbot` folder, and run `Vbot.exe`.

## Required

- Windows 10/11, 64-bit
- 16 GB RAM minimum
- 10-15 GB free disk space for the app, bundled dependencies, cache, and local model downloads
- Internet connection on first run for local model downloads
- Current NVIDIA driver
- Python or Conda 3.10 environment with Vbot dependencies installed from `requirements.txt`

## Strongly Recommended

- NVIDIA GPU with 8 GB VRAM or more
- Headphones or speakers for TTS playback
- Microphone if you want voice input

## Required for Local LLM Chat

The current Vbot desktop release uses Ollama through Docker for the local LLM chat path.

Install:

- Docker Desktop
- WSL 2 backend enabled in Docker Desktop

Why this exists:

- Vbot is intentionally local-first for the current learning and evaluation workflow.
- We are not replacing the local LLM with a cloud API in this release.
- Docker keeps the Ollama runtime isolated and reproducible across Windows machines.

## First Run

On first launch, Vbot may:

- locate your Python/Conda runtime
- create cache folders
- start or create the Ollama container
- download the local LLM model
- initialize local TTS, emotion, STT, and avatar components

The first launch can be slow. Later launches should be faster because models and styles are cached.

If the launcher cannot find the correct environment, set `VBOT_PYTHON` to the full path of the intended runtime, for example:

```powershell
$env:VBOT_PYTHON = "C:\Users\YourName\miniconda3\envs\vbot\python.exe"
.\Vbot.exe
```

## Known Limitations

- The Level 1 launcher package does not bundle the full Python ML runtime.
- A future full bundle or installer can reduce manual setup, but it needs more PyInstaller hardening.
- The executable is not code-signed yet, so Windows SmartScreen may show a warning.
- This is a portable zip release, not a polished consumer installer.
- CPU-only use may be slow and is not the target experience.

## Privacy

The current release is designed around local model inference. User chat and audio do not need to be sent to a cloud LLM API for the default local workflow.
