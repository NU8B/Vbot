# Vbot Desktop Package

Last reviewed: 2026-07-01

## Purpose

The desktop package turns Vbot into a reproducible Windows release artifact while keeping the AI stack on the user's machine.

The release target is:

- a portable Windows zip
- containing `Vbot.exe`, the app payload/assets, user docs, prerequisites, release notes, and build log
- with a SHA256 checksum and JSON release manifest

This is enough for a serious portfolio artifact because it proves productization around a real ML desktop runtime.

## Current Runtime Reality

Vbot is a multimodal local AI desktop app:

- Text input and optional speech input
- Local Ollama LLM chat through Docker
- Local emotion classification
- Local StyleTTS2 speech synthesis
- Local THA4 avatar rendering
- Local audio playback and avatar animation

Docker Desktop is required for the current local Ollama LLM chat path. It is not a cloud API dependency; it is local infrastructure used to keep the Ollama runtime isolated and reproducible on Windows.

The desktop package executable is a launcher build. It packages the app source/assets and starts Vbot through a prepared local Python/Conda 3.10 environment.

## Desktop Package Flow

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

`Launcher` is the default desktop package mode:

- builds quickly
- produces a real `Vbot.exe`
- packages app source/assets under the executable payload
- requires Python/Conda 3.10 plus `requirements.txt` on the target machine
- supports `VBOT_PYTHON=C:\path\to\python.exe` when the user wants to point at a specific environment

`Full` is the advanced bundle mode:

- uses the original `vbot.spec`
- attempts to freeze the ML/GUI/audio/CUDA runtime into `dist/Vbot`
- is available for deeper packaging work around the complete ML stack

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

For the desktop package, the user experience is:

1. Download `Vbot-[version]-windows-portable.zip`.
2. Extract it.
3. Read `PREREQUISITES.md`.
4. Install Python/Conda 3.10 and the dependencies from the packaged `requirements.txt`.
5. Install/start Docker Desktop with WSL 2 for local LLM chat.
6. Run `Vbot/Vbot.exe`.
7. Wait for first-run local model downloads/cache setup.

This gives the project a reproducible desktop distribution path for a GPU-heavy AI app.

## Desktop Package Done Criteria

- CI passes before release.
- `.\build_with_logs.ps1 -Version "..."` creates `dist/Vbot/Vbot.exe`.
- The package script creates zip, checksum, manifest, release notes, and build log.
- A Windows machine can extract the zip and launch `Vbot.exe` after installing prerequisites.
