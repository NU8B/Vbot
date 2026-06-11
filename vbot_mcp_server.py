import os
import sys
import subprocess
from pathlib import Path
from mcp.server.fastmcp import FastMCP

# Reconfigure stdout/stderr to use UTF-8 to prevent encoding issues on Windows
if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Initialize the MCP server for VBOT
mcp = FastMCP("Vbot-Server")

# Project Root setup
PROJECT_ROOT = Path(__file__).parent.resolve()

@mcp.tool()
def get_vbot_status() -> str:
    """
    Check the overall environment and status of the VBOT project.
    Verifies GPU availability (PyTorch), Docker container status, and critical project files.
    """
    lines = []
    lines.append("=== VBOT Environment Status ===")
    
    # 1. Check critical files and directories
    critical_items = {
        "Vbot.py": "Main application file",
        "VbotSeamless.py": "Seamless interaction entry point",
        "requirements.txt": "Python dependencies list",
        "hf_token.txt": "Hugging Face token file",
        "asset": "Assets directory (models, screenshots, etc.)",
        "StyleTTS2": "StyleTTS2 models directory",
    }
    
    lines.append("\n[File Checks]")
    for item, desc in critical_items.items():
        path = PROJECT_ROOT / item
        status = "[OK] Present" if path.exists() else "[MISSING] Missing"
        lines.append(f"- {item} ({desc}): {status}")

    # 2. Check Hugging Face token
    token_path = PROJECT_ROOT / "hf_token.txt"
    lines.append("\n[Hugging Face Integration]")
    if token_path.exists():
        try:
            token = token_path.read_text(encoding="utf-8").strip()
            if token:
                # Mask token for security
                masked = token[:6] + "..." + token[-4:] if len(token) > 10 else "invalid format"
                lines.append(f"- Token configured: [YES] Yes ({masked})")
            else:
                lines.append("- Token configured: [EMPTY] Empty file")
        except Exception as e:
            lines.append(f"- Token read error: [ERROR] {str(e)}")
    else:
        lines.append("- Token configured: [MISSING] hf_token.txt not found")

    # 3. Check GPU status
    lines.append("\n[Hardware/GPU Acceleration]")
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        lines.append(f"- PyTorch version: {torch.__version__}")
        lines.append(f"- CUDA available (GPU): {'[YES] Yes' if cuda_avail else '[NO] No'}")
        if cuda_avail:
            lines.append(f"- GPU Device: {torch.cuda.get_device_name(0)}")
            lines.append(f"- GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    except ImportError:
        lines.append("- PyTorch: [MISSING] Not installed or not loadable in current environment")
    except Exception as e:
        lines.append(f"- GPU check failed: [ERROR] {str(e)}")

    # 4. Check Docker Desktop status
    lines.append("\n[Docker Status]")
    try:
        # Run docker ps to see if Docker daemon is running
        res = subprocess.run(["docker", "ps", "--format", "{{.Names}} - {{.Status}}"], capture_output=True, text=True, timeout=5)
        if res.returncode == 0:
            lines.append("- Docker daemon: [RUNNING] Running")
            containers = res.stdout.strip().split("\n")
            containers = [c for c in containers if c]
            if containers:
                lines.append("- Running containers:")
                for c in containers:
                    lines.append(f"  * {c}")
            else:
                lines.append("- Running containers: None")
        else:
            lines.append("- Docker daemon: [ERROR] Running but returned error or not reachable")
    except subprocess.TimeoutExpired:
        lines.append("- Docker daemon: [TIMEOUT] Timeout checking docker (is Docker Desktop frozen?)")
    except FileNotFoundError:
        lines.append("- Docker command: [MISSING] 'docker' executable not found in PATH")
    except Exception as e:
        lines.append(f"- Docker check failed: [ERROR] {str(e)}")

    return "\n".join(lines)


@mcp.tool()
def list_avatars() -> str:
    """
    List available avatars configured in the VBOT asset/model folder.
    """
    model_path = PROJECT_ROOT / "asset" / "model"
    if not model_path.exists():
        return f"Error: Asset model path not found at: {model_path}"
    
    avatars = []
    try:
        for item in model_path.iterdir():
            if item.is_dir() and not item.name.startswith("."):
                # Check for standard model files if any
                avatars.append(item.name)
        
        if not avatars:
            return "No avatars found in asset/model/ directory."
        
        return "Available VBOT Avatars:\n" + "\n".join(f"- {name}" for name in sorted(avatars))
    except Exception as e:
        return f"Failed to list avatars: {str(e)}"


@mcp.tool()
def search_code(query: str, case_sensitive: bool = False) -> str:
    """
    Perform a simple case-insensitive search in Python files within the workspace.
    Useful for finding where specific classes or functions are defined or used.
    """
    results = []
    count = 0
    max_results = 25
    
    for root, dirs, files in os.walk(PROJECT_ROOT):
        # Exclude common directories to speed up
        dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", "build", "dist", "venv", "my-env")]
        
        for file in files:
            if file.endswith(".py"):
                file_path = Path(root) / file
                rel_path = file_path.relative_to(PROJECT_ROOT)
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        for idx, line in enumerate(f, 1):
                            match = query in line if case_sensitive else query.lower() in line.lower()
                            if match:
                                results.append(f"{rel_path}:L{idx}: {line.strip()}")
                                count += 1
                                if count >= max_results:
                                    break
                except Exception as e:
                    pass
            if count >= max_results:
                break
        if count >= max_results:
            break
            
    if not results:
        return f"No matches found for query: '{query}'"
    
    header = f"Found {count} matches (capped at {max_results}):\n"
    return header + "\n".join(results)


@mcp.tool()
def get_user_preferences() -> str:
    """
    Read the current user preferences and app settings for VBOT from ~/.vbot/user_preferences.json.
    """
    try:
        sys.path.append(str(PROJECT_ROOT))
        from utils.user_preferences import UserPreferences
        prefs = UserPreferences()
        import json
        return json.dumps(prefs.preferences, indent=2, ensure_ascii=False)
    except Exception as e:
        return f"Failed to read preferences: {str(e)}"


@mcp.tool()
def update_user_preference(key: str, value: str) -> str:
    """
    Update a specific VBOT user preference or app setting.
    Key should be in dot notation (e.g. 'app_settings.show_subtitles' or 'show_welcome_screen').
    Value can be parsed as boolean (true/false), integer, or string.
    """
    try:
        sys.path.append(str(PROJECT_ROOT))
        from utils.user_preferences import UserPreferences
        prefs = UserPreferences()
        
        # Convert types if possible
        import json
        try:
            parsed_val = json.loads(value)
        except json.JSONDecodeError:
            parsed_val = value  # Keep as string if it's not valid JSON
            
        prefs.set_preference(key, parsed_val)
        if prefs.save_preferences():
            return f"Successfully set preference '{key}' to: {parsed_val}"
        else:
            return "Failed to save preferences to file."
    except Exception as e:
        return f"Error updating preference: {str(e)}"


@mcp.tool()
def diagnose_audio_devices() -> str:
    """
    Diagnose the system audio hardware using PyAudio.
    Lists available microphone and speaker devices, and identifies issues.
    """
    try:
        import pyaudio
        p = pyaudio.PyAudio()
        info = p.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')
        
        lines = []
        lines.append("=== Audio Hardware Diagnostics ===")
        lines.append(f"Default Host API: {info.get('name')}")
        lines.append(f"Total Audio Devices Found: {numdevices}\n")
        
        lines.append("[Input Devices (Microphones)]")
        for i in range(0, numdevices):
            try:
                device_info = p.get_device_info_by_host_api_device_index(0, i)
                if device_info.get('maxInputChannels') > 0:
                    lines.append(f"- Index {i}: {device_info.get('name')}")
                    lines.append(f"  * Sample Rate: {int(device_info.get('defaultSampleRate'))}Hz")
                    lines.append(f"  * Channels: {device_info.get('maxInputChannels')}")
            except Exception:
                pass
                
        lines.append("\n[Output Devices (Speakers)]")
        for i in range(0, numdevices):
            try:
                device_info = p.get_device_info_by_host_api_device_index(0, i)
                if device_info.get('maxOutputChannels') > 0:
                    lines.append(f"- Index {i}: {device_info.get('name')}")
                    lines.append(f"  * Sample Rate: {int(device_info.get('defaultSampleRate'))}Hz")
                    lines.append(f"  * Channels: {device_info.get('maxOutputChannels')}")
            except Exception:
                pass
                
        p.terminate()
        return "\n".join(lines)
    except ImportError:
        return "PyAudio: [MISSING] PyAudio library is not installed in the environment."
    except Exception as e:
        return f"Audio diagnostics failed: [ERROR] {str(e)}"


@mcp.tool()
def run_vbot_test(test_name: str) -> str:
    """
    Run one of the pre-configured VBOT tests.
    test_name can be: 'welcome' (test_welcome_screen.py), 'seamless' (test_seamless.py), or 'recommender' (test_recommender_only.py).
    """
    test_files = {
        "welcome": "test_welcome_screen.py",
        "seamless": "test_seamless.py",
        "recommender": "test_recommender_only.py"
    }
    
    if test_name not in test_files:
        return f"Unknown test: '{test_name}'. Choose from: {', '.join(test_files.keys())}"
        
    test_file = test_files[test_name]
    path = PROJECT_ROOT / test_file
    if not path.exists():
        return f"Test file not found at: {path}"
        
    try:
        # Run test script using the current VBOT python environment interpreter
        python_exe = sys.executable
        res = subprocess.run([python_exe, str(path)], capture_output=True, text=True, timeout=30)
        
        output = []
        output.append(f"=== Running test {test_file} ===")
        output.append(f"Exit Code: {res.returncode}")
        if res.stdout:
            output.append("\n[Stdout]")
            output.append(res.stdout)
        if res.stderr:
            output.append("\n[Stderr]")
            output.append(res.stderr)
            
        return "\n".join(output)
    except subprocess.TimeoutExpired:
        return f"Test execution timed out after 30 seconds."
    except Exception as e:
        return f"Failed to execute test: {str(e)}"


@mcp.resource("vbot://readme")
def get_readme() -> str:
    """Get the primary README file contents for the VBOT project."""
    readme_path = PROJECT_ROOT / "README.md"
    if readme_path.exists():
        return readme_path.read_text(encoding="utf-8")
    return "README.md not found in project root."


@mcp.resource("vbot://build-instructions")
def get_build_instructions() -> str:
    """Get the build instructions for the VBOT project."""
    instructions_path = PROJECT_ROOT / "BUILD_INSTRUCTIONS.md"
    if instructions_path.exists():
        return instructions_path.read_text(encoding="utf-8")
    return "BUILD_INSTRUCTIONS.md not found in project root."


if __name__ == "__main__":
    # Start the MCP server using stdio transport (standard for local integration)
    mcp.run(transport="stdio")

