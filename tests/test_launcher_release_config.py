import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_launcher_does_not_import_heavy_runtime_modules():
    source = (PROJECT_ROOT / "scripts" / "vbot_launcher.py").read_text()
    tree = ast.parse(source)
    imported_roots = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])

    heavy_modules = {
        "torch",
        "torchaudio",
        "transformers",
        "wx",
        "pyaudio",
        "sounddevice",
        "librosa",
        "StyleTTS2",
        "tha4",
    }
    assert imported_roots.isdisjoint(heavy_modules)


def test_launcher_spec_packages_app_payload_without_archives():
    spec = (PROJECT_ROOT / "vbot_launcher.spec").read_text()

    assert "scripts/vbot_launcher.py" in spec
    assert "VbotSeamless.py" in spec
    assert 'project_tree("utils", "app/utils")' in spec
    assert 'project_tree("StyleTTS2", "app/StyleTTS2")' in spec
    assert 'project_tree("tha4", "app/tha4")' in spec
    assert "asset/old" not in spec


def test_root_build_wrapper_passes_build_mode():
    wrapper = (PROJECT_ROOT / "build_with_logs.ps1").read_text()
    inner = (PROJECT_ROOT / "scripts" / "build_with_logs.ps1").read_text()

    assert '[ValidateSet("Launcher", "Full")]' in wrapper
    assert "-BuildMode $BuildMode" in wrapper
    assert '[ValidateSet("Launcher", "Full")]' in inner
    assert "vbot_launcher.spec" in inner
