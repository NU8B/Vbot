"""Lightweight Windows launcher for the Vbot desktop package.

This executable intentionally does not import torch, transformers, wx, audio
drivers, or the avatar stack. It validates the packaged payload and then
delegates the real app to a prepared local Python 3.10 environment.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

APP_ENTRYPOINT = "VbotSeamless.py"
SMOKE_FLAG = "--launcher-smoke"


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def bundle_dir() -> Path:
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def candidate_app_roots() -> List[Path]:
    roots: List[Path] = []

    if is_frozen():
        exe_dir = bundle_dir()
        roots.extend(
            [
                exe_dir / "app",
                Path(getattr(sys, "_MEIPASS", exe_dir)) / "app",
                exe_dir.parent / "app",
            ]
        )
    else:
        roots.append(bundle_dir())

    roots.append(Path.cwd())
    return unique_paths(roots)


def unique_paths(paths: Iterable[Path]) -> List[Path]:
    seen = set()
    unique: List[Path] = []
    for path in paths:
        key = str(path.resolve()) if path.exists() else str(path)
        key_lower = key.lower()
        if key_lower not in seen:
            seen.add(key_lower)
            unique.append(path)
    return unique


def find_app_root() -> Path:
    checked = []
    for root in candidate_app_roots():
        checked.append(str(root))
        if (root / APP_ENTRYPOINT).is_file() and (root / "utils").is_dir():
            return root.resolve()

    checked_text = "\n  - ".join(checked)
    message = "\n".join(
        [
            "Could not find the packaged Vbot app payload. Checked:",
            f"  - {checked_text}",
        ]
    )
    raise FileNotFoundError(message)


def split_env_command(value: str) -> List[str]:
    try:
        return shlex.split(value, posix=False)
    except ValueError:
        return [value]


def candidate_python_commands(app_root: Path) -> List[List[str]]:
    commands: List[List[str]] = []

    env_python = os.environ.get("VBOT_PYTHON")
    if env_python:
        commands.append(split_env_command(env_python))

    for root in [bundle_dir(), app_root, app_root.parent]:
        commands.append([str(root / ".venv" / "Scripts" / "python.exe")])
        commands.append([str(root / "venv" / "Scripts" / "python.exe")])

    if not is_frozen():
        commands.append([sys.executable])

    for name in ["python", "python3"]:
        path = shutil.which(name)
        if path:
            commands.append([path])

    py_launcher = shutil.which("py")
    if py_launcher:
        commands.append([py_launcher, "-3.10"])
        commands.append([py_launcher, "-3"])

    return unique_commands(commands)


def unique_commands(commands: Iterable[Sequence[str]]) -> List[List[str]]:
    seen = set()
    unique: List[List[str]] = []
    for command in commands:
        if not command:
            continue
        normalized = tuple(str(part) for part in command if str(part))
        if not normalized:
            continue
        key = "\0".join(normalized).lower()
        if key not in seen:
            seen.add(key)
            unique.append(list(normalized))
    return unique


def command_exists(command: Sequence[str]) -> bool:
    executable = command[0]
    return Path(executable).exists() or shutil.which(executable) is not None


def find_python_command(app_root: Path) -> Optional[List[str]]:
    for command in candidate_python_commands(app_root):
        if not command_exists(command):
            continue
        try:
            result = subprocess.run(
                list(command) + ["--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=20,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if result.returncode == 0:
            version = result.stdout.strip() or "Python"
            print(f"Using runtime Python: {version}")
            return list(command)
    return None


def prepend_pythonpath(env: dict, app_root: Path) -> None:
    existing = env.get("PYTHONPATH")
    if existing:
        env["PYTHONPATH"] = f"{app_root}{os.pathsep}{existing}"
    else:
        env["PYTHONPATH"] = str(app_root)


def format_command(command: Sequence[str]) -> str:
    return subprocess.list2cmdline([str(part) for part in command])


def run(argv: Optional[Sequence[str]] = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    smoke = SMOKE_FLAG in args
    args = [arg for arg in args if arg != SMOKE_FLAG]

    print("Vbot Desktop Package Launcher")
    print(f"Launcher directory: {bundle_dir()}")

    try:
        app_root = find_app_root()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 2

    print(f"App payload: {app_root}")
    entrypoint = app_root / APP_ENTRYPOINT
    python_command = find_python_command(app_root)
    if python_command is None:
        print("ERROR: No usable Python runtime was found.")
        print("Install Python/Conda 3.10 and Vbot dependencies.")
        print("Set VBOT_PYTHON if the launcher cannot find your environment.")
        return 3

    if smoke:
        print("Launcher smoke check passed.")
        return 0

    env = os.environ.copy()
    env["PROJECT_ROOT"] = str(app_root)
    prepend_pythonpath(env, app_root)

    command = python_command + [str(entrypoint)] + args
    print(f"Starting Vbot: {format_command(command)}")
    return subprocess.call(command, cwd=str(app_root), env=env)


if __name__ == "__main__":
    raise SystemExit(run())
