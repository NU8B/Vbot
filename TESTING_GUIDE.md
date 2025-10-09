# Testing Guide for Vbot Distribution System

**Phase 1 Components Testing**  
**Estimated Time:** 15-30 minutes

---

## Prerequisites

1. Open PowerShell (normal user, NOT administrator yet)
2. Navigate to Vbot project directory:
   ```powershell
   cd "C:\Users\peepz\Desktop\Project\college-project\senior-project\Vbot"
   ```

---

## Test Suite

### Test 1: Python Launcher Test (2 minutes)
**Tests:** Path resolution, imports, system detection

```powershell
# Run the test script
python test_launcher.py
```

**Expected Output:**
```
============================================================
TEST 1: Import Launcher Modules
============================================================
✅ All launcher modules imported successfully

============================================================
TEST 2: Resource Path Functions
============================================================
Is Frozen: False
Base Path: C:\...\Vbot
Cache Path: C:\...\Vbot\cache
Output Path: C:\...\Vbot\asset\outputs
✅ All path functions work correctly

============================================================
TEST 3: Verify Critical Paths Exist
============================================================
✅ Base directory
✅ StyleTTS2 directory
✅ Asset directory
✅ Utils directory
✅ THA4 directory
✅ All critical paths verified

============================================================
TEST 4: System Requirements Check
============================================================
✅ OS           Windows 11 Build 22621
✅ RAM          32.0 GB available
✅ GPU          NVIDIA GeForce RTX 3070 (8.0 GB VRAM)
...
```

**What to check:**
- ✅ All 4 tests pass
- ✅ Paths are correct (showing your project directory)
- ✅ GPU is detected
- ⚠️ If any test fails, note the error

---

### Test 2: System Validation Script (3 minutes)
**Tests:** PowerShell execution, system checks

```powershell
# Run system validation
.\setup\validate_system.ps1
```

**Expected Output:**
```
============================================================
              Vbot System Validation
============================================================

ℹ️  Checking Windows version...
✅ Windows 11 Build 22621

ℹ️  Checking system RAM...
✅ RAM: 32.0 GB available

ℹ️  Checking disk space...
✅ Disk Space: 250.5 GB free

ℹ️  Checking for NVIDIA GPU...
✅ GPU: NVIDIA GeForce RTX 3070

ℹ️  Checking NVIDIA driver...
✅ NVIDIA Driver: 545.84

ℹ️  Checking Docker Desktop...
✅ Docker: Docker version 24.0.6
✅ Docker daemon is running

ℹ️  Checking Python installation...
✅ Python: Python 3.10.11

ℹ️  Checking administrator privileges...
⚠️  Not running as administrator

============================================================
VALIDATION SUMMARY
============================================================
Passed: 7 / 8 checks

⚠️  WARNINGS (Recommended to fix):
  • Some installation steps require administrator privileges

⚠️  System meets minimum requirements but has warnings.
============================================================
```

**What to check:**
- ✅ Script runs without errors
- ✅ GPU is detected with correct name
- ✅ CUDA is detected (if installed)
- ✅ Docker status is correct
- ⚠️ Warning about admin is expected (normal user mode)

**Try detailed mode:**
```powershell
.\setup\validate_system.ps1 -Detailed
```
(Shows more information like driver versions, GPU details)

---

### Test 3: CUDA/cuDNN Checker (5 minutes) ⚠️ IMPORTANT
**Tests:** CUDA Toolkit detection, cuDNN version validation

```powershell
# Run CUDA checker
.\setup\check_cuda.ps1
```

**Expected Output (if CUDA installed correctly):**
```
============================================================
         CUDA & cuDNN Validation
============================================================

ℹ️  Checking CUDA Toolkit installation...
✅ CUDA directory found: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA
  Installed versions:
    • v12.1
✅ CUDA 12.1 found at: C:\...\CUDA\v12.1
✅ nvcc.exe found

ℹ️  Checking cuDNN installation...
⚠️  CRITICAL: PyTorch 2.5.1+cu121 requires cuDNN v8.x (NOT v9.x)
✅ All cuDNN v8.x DLLs found (7 files)

ℹ️  Checking CUDA environment variables...
✅ CUDA_PATH: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1
✅ CUDA 12.1 bin directory is in PATH

============================================================
VALIDATION SUMMARY
============================================================
✅ CUDA 12.1 and cuDNN v8.x are properly installed!

Your system is ready for PyTorch 2.5.1+cu121
```

**Possible scenarios:**

**Scenario A - All Good:**
```
✅ CUDA 12.1 and cuDNN v8.x are properly installed!
```
→ Perfect! Continue to next test.

**Scenario B - cuDNN v9.x Detected (BAD!):**
```
❌ cuDNN v9.x detected! This WILL cause crashes!
Found v9.x DLLs:
  • cudnn_ops_infer64_9.dll
  • cudnn64_9.dll

❌ ISSUES FOUND:
  • cuDNN v9.x is installed but v8.9.7 is required
```
→ This is the issue you had before! The script catches it now.

**Scenario C - CUDA Not Installed:**
```
❌ CUDA Toolkit not found
❌ ISSUES FOUND:
  • CUDA Toolkit 12.1 must be installed

RECOMMENDED ACTIONS:
Download from: https://developer.nvidia.com/cuda-12-1-0-download-archive
```
→ Follow the instructions provided by the script.

**What to check:**
- ✅ CUDA 12.1 is detected
- ✅ cuDNN v8.x DLLs are found (NOT v9.x)
- ⚠️ If cuDNN v9.x is found, the script explains how to fix it

**Try detailed mode:**
```powershell
.\setup\check_cuda.ps1 -Detailed
```
(Shows individual DLL versions, header files, lib files)

---

### Test 4: Docker Installation Helper (2 minutes - info only)
**Tests:** Docker detection logic

```powershell
# Check what the Docker installer would do (doesn't install anything)
# This is safe to run - it just checks status
.\setup\install_docker.ps1
```

**Expected Output (if Docker already installed):**
```
============================================================
        Docker Desktop Installation Helper
============================================================

▶️  Step 1: Checking for existing Docker installation
✅ Docker is already installed: Docker version 24.0.6, build ed223bc
ℹ️  Checking Docker daemon status...
✅ Docker is running!

Docker Desktop is already installed and running.
No action needed.
```

**OR (if Docker not installed):**
```
ℹ️  Docker Desktop not found - proceeding with installation
▶️  Step 2: Validating Windows version
✅ Windows version check passed (Build 22621)
...
```

**What to check:**
- ✅ Correctly detects Docker presence/absence
- ✅ Would proceed with installation if needed
- ⚠️ Don't let it actually install if you already have Docker

**To stop if it tries to download:**
Press `Ctrl+C` to cancel

---

### Test 5: Main Setup Script (5 minutes - dry run)
**Tests:** Full orchestration flow

```powershell
# Run with quick mode (skips installations)
.\setup\setup_windows.ps1 -Quick -SkipDocker
```

**Expected Output:**
```
╔════════════════════════════════════════════════════════╗
║                                                        ║
║              Vbot Installation Wizard                  ║
║                                                        ║
╚════════════════════════════════════════════════════════╝

ℹ️  Project root: C:\...\Vbot
ℹ️  Setup scripts: C:\...\setup

━━━ Phase 1: System Validation ━━━
ℹ️  Checking if your system meets requirements...
[runs validate_system.ps1]
✅ System validation passed!

━━━ Phase 2: CUDA & cuDNN Validation ━━━
ℹ️  Checking CUDA Toolkit and cuDNN installation...
[runs check_cuda.ps1]
✅ CUDA and cuDNN are properly configured!

━━━ Phase 3: Docker Desktop ━━━
ℹ️  Skipping Docker check (--SkipDocker)

━━━ Phase 4: Python Dependencies ━━━
ℹ️  Quick mode: Skipping Python dependencies

━━━ Phase 5: Model Downloads ━━━
ℹ️  Vbot will download AI models on first run...
Total: ~7-8 GB download on first run

╔════════════════════════════════════════════════════════╗
║                                                        ║
║              Installation Complete!                    ║
║                                                        ║
╚════════════════════════════════════════════════════════╝

Next Steps:
1. Run Vbot:
   python vbot_launcher/launcher.py
...
```

**What to check:**
- ✅ All phases run in sequence
- ✅ Validations pass
- ✅ Clear output with progress indicators
- ✅ Provides next steps at the end

---

### Test 6: Launcher Pre-flight Check (3 minutes)
**Tests:** Launcher system checks before starting Vbot

```powershell
# This will run checks but NOT start Vbot yet
# Just see what happens - you can Ctrl+C after checks
python vbot_launcher/launcher.py
```

**Expected Output:**
```
============================================================
VBOT LAUNCHER
============================================================

ℹ️  Running in development mode
📁 Base path: C:\...\Vbot
💾 Cache path: C:\...\Vbot\cache
📤 Output path: C:\...\Vbot\asset\outputs

🔍 Checking system requirements...

============================================================
SYSTEM REQUIREMENTS CHECK
============================================================

✅ OS           Windows 11 Build 22621
✅ RAM          32.0 GB available
✅ DISK         250.5 GB free
✅ GPU          NVIDIA GeForce RTX 3070 (8.0 GB VRAM)
✅ CUDA         CUDA 12.1, cuDNN 8957
✅ DOCKER       Docker version 24.0.6
✅ PYTHON       Python 3.10.11

============================================================
✅ All checks passed! System is ready for Vbot.
✅ System check passed! Starting Vbot...

🚀 Loading Vbot modules...
```

**At this point:**
- If you want to actually start Vbot, let it continue
- If you just want to test the launcher checks, press `Ctrl+C`

**What to check:**
- ✅ Pre-flight checks run automatically
- ✅ All system checks pass
- ✅ Paths are displayed correctly
- ✅ Would proceed to load Vbot modules

---

### Test 7: Full Integration Test (Optional - 5-10 minutes)
**Tests:** Complete Vbot startup with new launcher

**Only if you want to fully test:**
```powershell
python vbot_launcher/launcher.py
```

**Let it run through:**
1. Pre-flight checks ✅
2. Module loading ✅
3. Docker/Ollama check ✅
4. Model initialization ✅
5. GUI appears ✅

**What to check:**
- ✅ Everything loads normally
- ✅ No path errors
- ✅ GUI works as before
- ✅ Can talk to AI and get responses

**To exit:** Close the GUI normally

---

## Test Results Checklist

Use this to track your testing:

```
[ ] Test 1: test_launcher.py runs successfully
    [ ] All 4 tests pass
    [ ] Paths look correct
    [ ] No import errors

[ ] Test 2: validate_system.ps1 runs successfully
    [ ] Shows system specs correctly
    [ ] GPU detected
    [ ] Docker status correct

[ ] Test 3: check_cuda.ps1 runs successfully
    [ ] CUDA 12.1 detected (if installed)
    [ ] cuDNN v8.x detected (if installed)
    [ ] No v9.x DLLs found
    [ ] OR shows clear instructions to fix

[ ] Test 4: install_docker.ps1 detects Docker correctly
    [ ] Recognizes if Docker installed
    [ ] Would proceed with install if needed

[ ] Test 5: setup_windows.ps1 orchestrates correctly
    [ ] All phases run in order
    [ ] Clear output
    [ ] Provides next steps

[ ] Test 6: launcher.py pre-flight checks work
    [ ] System validation runs
    [ ] Paths displayed correctly
    [ ] Would start Vbot normally

[ ] Test 7 (Optional): Full Vbot startup works
    [ ] No errors
    [ ] GUI appears
    [ ] Works normally
```

---

## Common Issues & Solutions

### Issue: PowerShell scripts won't run
**Error:** "cannot be loaded because running scripts is disabled"

**Solution:**
```powershell
# Run this once as Administrator
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Issue: Python not found
**Error:** "python is not recognized"

**Solution:**
```powershell
# Check Python installation
py --version

# Use 'py' instead of 'python'
py test_launcher.py
py vbot_launcher/launcher.py
```

### Issue: Module import errors in test_launcher.py
**Error:** "ModuleNotFoundError: No module named 'vbot_launcher'"

**Solution:**
```powershell
# Make sure you're in the project root
cd "C:\Users\peepz\Desktop\Project\college-project\senior-project\Vbot"

# Check current directory
pwd
# Should show: ...\Vbot (not ...\Vbot\vbot_launcher or other subdirectory)
```

### Issue: cuDNN v9.x detected
**This is the crash you experienced before!**

**Solution:** The script will show you instructions:
1. Download cuDNN v8.9.7 from NVIDIA
2. Extract the zip file
3. Copy to CUDA directory (as Administrator)

See the script output for detailed steps.

---

## Quick Test Sequence

**If you're short on time, run these 3 commands:**

```powershell
# 1. Quick test (2 min)
python test_launcher.py

# 2. System check (1 min)
.\setup\validate_system.ps1

# 3. CUDA check (1 min)
.\setup\check_cuda.ps1
```

**If all 3 pass → Everything is working!** ✅

---

## What Success Looks Like

### All Tests Passing:
```
✅ test_launcher.py: All 4 tests pass
✅ validate_system.ps1: 7-8 / 8 checks pass
✅ check_cuda.ps1: CUDA 12.1 + cuDNN v8.x detected
✅ setup_windows.ps1: All phases complete
✅ launcher.py: Pre-flight checks pass
```

### Ready for Phase 2:
If all tests pass, you're ready to:
- Build PyInstaller executable
- Create Windows installer
- Distribute to users

---

## Reporting Results

After testing, note:
1. **Which tests passed:** (all / some / none)
2. **Which tests failed:** (if any, with error messages)
3. **System configuration:**
   - Windows version
   - GPU model
   - CUDA version
   - cuDNN version
   - Docker status

**Then we can either:**
- Fix any issues found
- Move to Phase 2 (PyInstaller)
- Enhance the scripts further

---

**Ready to start testing?** Just run the commands above! 🚀
