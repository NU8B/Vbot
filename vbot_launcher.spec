# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the Vbot desktop package launcher.

The full ML freeze is kept in vbot.spec. This launcher spec builds quickly and
packages the app payload without asking PyInstaller to analyze torch,
transformers, wxPython, CUDA, or the avatar runtime.
"""

from pathlib import Path

block_cipher = None
project_root = Path(".").resolve()


def project_file(relative_path, destination="app"):
    path = project_root / relative_path
    if path.exists():
        return [(str(path), destination)]
    return []


def project_tree(relative_path, destination):
    source_root = project_root / relative_path
    if not source_root.exists():
        return []

    records = []
    for path in source_root.rglob("*"):
        if not path.is_file():
            continue
        parts = set(path.parts)
        if "__pycache__" in parts or ".ipynb_checkpoints" in parts:
            continue
        if path.suffix in {".pyc", ".pyo"}:
            continue

        relative_parent = path.parent.relative_to(source_root)
        target = Path(destination) / relative_parent
        records.append((str(path), str(target).replace("\\", "/")))
    return records


datas = []
datas += project_file("VbotSeamless.py")
datas += project_file("Vbot.py")
datas += project_file("ChatBot.py")
datas += project_file("animation.py")
datas += project_file("requirements.txt")
datas += project_file("README.md")
datas += project_file("USER_GUIDE.md")
datas += project_file("PREREQUISITES.md")

datas += project_tree("utils", "app/utils")
datas += project_tree("StyleTTS2", "app/StyleTTS2")
datas += project_tree("tha4", "app/tha4")
datas += project_tree("asset/Background", "app/asset/Background")
datas += project_tree("asset/model", "app/asset/model")
datas += project_tree("asset/ref_sound", "app/asset/ref_sound")
datas += project_tree("recommender-avatar-profile", "app/recommender-avatar-profile")

if (project_root / "asset" / "icon.ico").exists():
    datas += [(str(project_root / "asset" / "icon.ico"), "app/asset")]


a = Analysis(
    ["scripts/vbot_launcher.py"],
    pathex=[str(project_root)],
    binaries=[],
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "torch",
        "torchaudio",
        "transformers",
        "wx",
        "OpenGL",
        "sklearn",
        "scipy",
        "librosa",
        "sounddevice",
        "pyaudio",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Vbot",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(project_root / "asset" / "icon.ico")
    if (project_root / "asset" / "icon.ico").exists()
    else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Vbot",
)
