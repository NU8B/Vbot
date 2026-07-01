# Vbot Level 1 Desktop CD

Last reviewed: 2026-07-01

## Purpose

Level 1 CD turns Vbot into a reproducible Windows desktop release artifact while keeping the AI stack local-first. This is not the future web app architecture and it does not replace local LLM work with API calls.

The release target is:

- a portable Windows zip
- containing `Vbot.exe`, the app payload/assets, user docs, prerequisites, release notes, and build log
- with a SHA256 checksum and JSON release manifest

This is enough for a serious portfolio artifact because it proves productization without hiding the real ML/runtime constraints.

## Current Runtime Reality

Vbot is a multimodal local AI desktop app:

- Text input and optional speech input
- Local Ollama LLM chat through Docker
- Local emotion classification
- Local StyleTTS2 speech synthesis
- Local THA4 avatar rendering
- Local audio playback and avatar animation

Docker Desktop is required for the current local Ollama LLM chat path. It is not a cloud API dependency; it is local infrastructure used to keep the Ollama runtime isolated and reproducible on Windows.

The Level 1 executable is a launcher build. It packages the app source/assets and starts Vbot through a prepared local Python/Conda 3.10 environment. This is less frictionless than a fully frozen commercial installer, but it avoids the current PyInstaller full-freeze failure mode where torch/transformers/CUDA analysis can stall for a long time.

## Level 1 CD Flow

```text
developer pushes code
        |
        v
CI quality gate
        |
        v
manual Windows release build
        |
        v
PyInstaller creates dist/Vbot/Vbot.exe launcher
        |
        v
package_release.ps1 creates portable zip
        |
        v
release artifacts: zip, sha256, manifest, notes, build log
```

## Build Modes

`Launcher` is the Level 1 default:

- builds quickly
- produces a real `Vbot.exe`
- packages app source/assets under the executable payload
- requires Python/Conda 3.10 plus `requirements.txt` on the target machine
- supports `VBOT_PYTHON=C:\path\to\python.exe` when the user wants to point at a specific environment

`Full` is experimental:

- uses the original `vbot.spec`
- attempts to freeze the ML/GUI/audio/CUDA runtime into `dist/Vbot`
- can be useful later, but currently needs hardening because PyInstaller may stall while analyzing heavyweight ML packages

## Local Release Command

From the repository root:

```powershell
.\build_with_logs.ps1 -Version "v0.1.0"
```

This defaults to:

```powershell
.\build_with_logs.ps1 -BuildMode Launcher -Version "v0.1.0"
```

Outputs:

```text
dist/Vbot/Vbot.exe
release/artifacts/Vbot-v0.1.0-windows-portable.zip
release/artifacts/Vbot-v0.1.0-windows-portable.zip.sha256
release/artifacts/Vbot-v0.1.0-release-manifest.json
release/artifacts/Vbot-v0.1.0-release-notes.md
release/logs/build-v0.1.0-[timestamp].log
```

Useful variants:

```powershell
.\build_with_logs.ps1 -SkipTests
.\build_with_logs.ps1 -NoClean
.\build_with_logs.ps1 -NoPackage
.\build_with_logs.ps1 -Version "v0.1.0"
.\build_with_logs.ps1 -BuildMode Full -Version "v0.1.0-full"
```

## GitHub Manual Release Workflow

Workflow:

```text
.github/workflows/desktop-release.yml
```

It is intentionally configured for:

```text
runs-on: [self-hosted, Windows, X64]
```

Reason: the real release build needs a prepared Windows GPU/build environment. GitHub-hosted Windows runners usually do not have the right CUDA, model cache, Docker, audio/GUI assumptions, or enough practical time for a multi-GB PyInstaller artifact.

The workflow can:

- run the build script
- package the portable zip
- upload release artifacts
- optionally create a draft GitHub Release

Self-hosted runner prerequisites:

- Python 3.10 environment with Vbot build dependencies installed
- PyInstaller available in that Python environment
- CUDA/NVIDIA driver setup matching the build machine if using `BuildMode Full`
- access to any local model/assets expected by `vbot.spec` if using `BuildMode Full`
- GitHub CLI (`gh`) only if using the optional draft GitHub Release step

## End-User Install Story

For Level 1, the user experience is:

1. Download `Vbot-[version]-windows-portable.zip`.
2. Extract it.
3. Read `PREREQUISITES.md`.
4. Install Python/Conda 3.10 and the dependencies from the packaged `requirements.txt`.
5. Install/start Docker Desktop with WSL 2 for local LLM chat.
6. Run `Vbot/Vbot.exe`.
7. Wait for first-run local model downloads/cache setup.

This is not as frictionless as a commercial installer, but it is an honest product artifact for a GPU-heavy local AI app.

## Why Not a True Installer Yet?

A true installer can come later with Inno Setup, NSIS, WiX, or MSIX. For now, it would mostly wrap the same constraints:

- Python/Conda dependency setup in Launcher mode
- large ML dependency bundle in Full mode
- Docker Desktop still required for local LLM chat
- first-run model downloads
- unsigned executable warnings unless code signing is added

The portable zip is simpler, easier to debug, and better for early portfolio iteration.

## Level 1 Done Criteria

- CI passes before release.
- `.\build_with_logs.ps1 -Version "..."` creates `dist/Vbot/Vbot.exe`.
- The package script creates zip, checksum, manifest, release notes, and build log.
- A Windows machine can extract the zip and launch `Vbot.exe` after installing prerequisites.
- Runtime limitations are documented instead of hidden.

## Future Upgrade Path

Level 2 desktop product:

- true installer
- code signing
- first-run prerequisite checks
- cleaner Docker/Ollama detection and recovery
- hardened full-freeze PyInstaller build or a managed embedded Python runtime
- optional native Ollama or llama.cpp backend to reduce Docker friction

Future web app:

- browser client
- backend model orchestration
- server-side GPU workers
- web avatar rendering or streamed avatar video
- evaluation and model promotion pipeline behind the service
