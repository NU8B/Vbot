# 🚀 Vbot Build Instructions

## Prerequisites

### Required Software:
1. **Anaconda/Miniconda** - Must be installed
2. **NVIDIA GPU Drivers** - For CUDA support
3. **Git** (optional) - For cloning repository

### System Requirements:
- Windows 10/11 (64-bit)
- NVIDIA GPU with CUDA support
- 16GB RAM minimum
- 50GB free disk space

---

## 📦 Step-by-Step Build Guide

### 1. Clone/Download Project
```powershell
cd C:\Users\[YourUsername]\Desktop\Project
git clone [repository-url]
# OR extract the Vbot folder from zip
```

### 2. Navigate to Project Directory
```powershell
cd C:\Users\[YourUsername]\Desktop\Project\college-project\senior-project\Vbot
```

### 3. Create Conda Environment
```powershell
# Create environment from requirements
conda create -n vbot python=3.10 -y
conda activate vbot
```

### 4. Install Dependencies
```powershell
# Install core packages
pip install -r requirements.txt

# Install additional packages needed for building
pip install pyinstaller pywin32-ctypes pywin32 wxPython PyOpenGL PyOpenGL-accelerate glfw gruut
```

### 5. Verify Environment
```powershell
# Test PyTorch installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

Expected output:
```
PyTorch: 2.5.1+cu121
CUDA Available: True
```

---

## 🔨 Building the Executable

### Option A: Using Build Script (Recommended)
```powershell
.\build_with_logs.ps1
```

This will:
- Run the fast test suite unless `-SkipTests` is provided
- Build the executable
- Save the full build log under `release/logs/`
- Package a portable release zip under `release/artifacts/`
- Write a SHA256 checksum and release manifest
- Take ~5-10 minutes

Useful options:
```powershell
# Build without running tests
.\build_with_logs.ps1 -SkipTests

# Fast rebuild without cleaning previous PyInstaller state
.\build_with_logs.ps1 -NoClean

# Build only, do not create release zip
.\build_with_logs.ps1 -NoPackage

# Set release version in artifact names
.\build_with_logs.ps1 -Version "v0.1.0"
```

### Option B: Direct Build
```powershell
# Clean previous builds
Remove-Item -Recurse -Force .\dist -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force .\build -ErrorAction SilentlyContinue

# Build
python -m PyInstaller vbot.spec --clean --noconfirm
```

---

## ✅ Testing the Build

### 1. Navigate to Output
```powershell
cd dist\Vbot
```

### 2. Run the Executable
```powershell
.\Vbot.exe
```

### Expected Behavior:
1. Launcher appears with system check
2. All checks pass (or warnings shown)
3. Main GUI window opens
4. Avatar loads successfully

---

## 🐛 Troubleshooting Common Issues

### Issue 1: "No module named 'unittest'"

**Cause:** PyInstaller not bundling standard library properly

**Fix:**
```powershell
# Go back to project root
cd C:\Users\[YourUsername]\Desktop\Project\college-project\senior-project\Vbot

# Rebuild with clean cache
python -m PyInstaller vbot.spec --clean --noconfirm --log-level=DEBUG > build_debug.log 2>&1

# Check if unittest is in the bundle
cd dist\Vbot\_internal
Get-ChildItem -Recurse -Filter "unittest*"
```

If unittest is not found, manually add it:
1. Find your Python installation: `where python`
2. Locate unittest folder: `C:\Users\[Username]\.conda\envs\vbot\Lib\unittest`
3. Copy entire `unittest` folder to `dist\Vbot\_internal\`

---

### Issue 2: "CUDA DLL not found"

**Fix:**
```powershell
# Verify CUDA installation
nvcc --version

# Check if CUDA DLLs exist
dir "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin\*.dll"
```

If missing, reinstall CUDA Toolkit 12.1 from NVIDIA website.

---

### Issue 3: "Docker not running"

**Fix:**
```powershell
# Start Docker Desktop
# Then restart Vbot.exe
```

Docker Desktop is required for the current local Ollama LLM chat path. Some non-LLM functionality may still open without Docker, but the main chatbot experience expects Docker to be running.

---

### Issue 4: Build Takes Too Long (>15 minutes)

**Cause:** Slow disk or analyzing too many modules

**Fix:**
```powershell
# Check if antivirus is scanning
# Add exclusion for project folder in Windows Defender

# OR use faster build (skips some optimizations)
python -m PyInstaller vbot.spec --noconfirm
```

---

### Issue 5: "wxPython not found" or GUI doesn't appear

**Fix:**
```powershell
# Reinstall wxPython in vbot environment
pip uninstall wxPython -y
pip install wxPython
```

---

## 📊 Build Output Structure

After successful build:
```
Vbot/
├── dist/
│   └── Vbot/                    ← Distributable folder
│       ├── Vbot.exe             ← Main executable (55-60 MB)
│       ├── _internal/           ← Dependencies (~2-3 GB)
│       │   ├── torch/
│       │   ├── StyleTTS2/
│       │   ├── tha4/
│       │   └── [Python libs]
│       ├── asset/               ← Models
│       └── cache/               ← Empty cache dir
```

**Total Size:** ~3-4 GB

---

## 🚚 Distribution

The Level 1 CD target is a portable zip, not a polished signed installer.

### What to Share:
```
Vbot-Distribution.zip containing:
├── Vbot/                        ← Entire dist\Vbot folder
│   ├── Vbot.exe
│   └── _internal/
├── README.md                    ← User instructions
├── PREREQUISITES.md             ← System requirements
├── RELEASE_NOTES.md             ← Release metadata and known limitations
└── build.log                    ← Build log, when packaged by script
```

The build script creates:
```
release/
├── artifacts/
│   ├── Vbot-[version]-windows-portable.zip
│   ├── Vbot-[version]-windows-portable.zip.sha256
│   └── Vbot-[version]-release-manifest.json
└── logs/
    └── build-[version]-[timestamp].log
```

### User Installation:
1. Extract `Vbot-Distribution.zip`
2. Install Docker Desktop with WSL 2 for local LLM chat
3. Install NVIDIA drivers
4. Run `Vbot.exe`
5. First launch downloads Ollama models (~4GB)

---

## 🔧 Development Build (for testing changes)

For faster iteration during development:

```powershell
# 1. Make code changes
# 2. Quick rebuild (no clean)
python -m PyInstaller vbot.spec --noconfirm

# 3. Test immediately
cd dist\Vbot
.\Vbot.exe
```

**Note:** Only use this for testing. Always do full clean build for distribution!

---

## Manual GitHub Release Workflow

The repo includes a manual workflow:

```
.github/workflows/desktop-release.yml
```

It is configured for a self-hosted Windows runner:

```
[self-hosted, Windows, X64]
```

Use this after the local build is stable. The runner must already have the Vbot Python/CUDA/build environment prepared. The workflow uploads the portable zip, checksum, manifest, release notes, and build log as workflow artifacts. It can optionally create a draft GitHub Release.

---

## 📝 Build Verification Checklist

Before distributing, verify:

- [ ] `Vbot.exe` runs without errors
- [ ] GUI appears correctly
- [ ] Avatar loads and animates
- [ ] Audio input/output works
- [ ] GPU is detected (check system info)
- [ ] StyleTTS2 generates speech
- [ ] File size is reasonable (~3-4 GB total)
- [ ] Tested on clean Windows system (if possible)

---

## 🆘 Getting Help

If build fails:

1. **Check build log:**
   ```powershell
   Get-Content build_log_*.txt | Select-String "ERROR:"
   ```

2. **Share specific error message** with team

3. **Common error patterns:**
   - `ModuleNotFoundError` → Missing hidden import in `vbot.spec`
   - `DLL not found` → Missing binary in `binaries` section
   - `CUDA error` → Driver/toolkit version mismatch

---

## ⚡ Quick Reference Commands

```powershell
# Activate environment
conda activate vbot

# Full clean build
Remove-Item -Recurse -Force .\dist, .\build -ErrorAction SilentlyContinue
python -m PyInstaller vbot.spec --clean --noconfirm

# Full release build and package
.\build_with_logs.ps1 -Version "v0.1.0"

# Test
cd dist\Vbot && .\Vbot.exe

# Check dependencies
pip list | Select-String "torch|wx|pyinstaller"

# View build size
Get-ChildItem .\dist\Vbot -Recurse | Measure-Object -Property Length -Sum | Select-Object @{Name="Size(GB)";Expression={[math]::Round($_.Sum/1GB,2)}}
```

---

## 📞 Contact

If you encounter issues not covered here:
- Check GitHub Issues
- Contact: [your-email]
- Discord: [discord-link]

---

**Last Updated:** 2025-10-10  
**Build System:** PyInstaller 6.16.0  
**Python Version:** 3.10.18  
**PyTorch Version:** 2.5.1+cu121
