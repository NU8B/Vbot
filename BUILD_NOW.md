# 🚀 BUILD VBOT INSTALLER NOW - Quick Guide

**Time to first installer: ~30 minutes**

---

## ⚡ Fast Track (3 Commands)

### Step 1: Install PyInstaller (30 seconds)
```powershell
pip install pyinstaller
```

### Step 2: Build Executable (15-30 minutes)

**Option A: Build with Logging (RECOMMENDED for debugging)**
```powershell
.\build_with_logs.ps1
```
This saves full build output to `build_log_[timestamp].txt`

**Option B: Direct Build**
```powershell
C:\Users\peepz\.conda\envs\vbot\python.exe -m PyInstaller vbot.spec --clean
```

### Step 3: Test It (2 minutes)
```powershell
cd dist\Vbot
.\Vbot.exe
```

**Done! You now have a distributable Vbot folder!**

---

## 📝 Build Logs

**If build fails, send me the log file:**
- Location: `build_log_YYYYMMDD_HHMMSS.txt`
- Contains: All warnings, errors, and build steps
- Usage: Copy/paste relevant sections for debugging

---

## 🎯 Automated Build (Even Easier!)

**Just run this one command:**
```powershell
.\build_installer.ps1
```

**What it does:**
1. ✅ Checks dependencies
2. ✅ Builds executable with PyInstaller
3. ✅ Shows build progress
4. ✅ Tests the output
5. ✅ Creates installer (if Inno Setup installed)

**Sits back and waits ~30 minutes!**

---

## 📋 What You'll Get

### After PyInstaller Build:
```
dist/
└── Vbot/
    ├── Vbot.exe           ← Main executable (30-50 MB)
    ├── *.dll              ← Dependencies (CUDA, Python, etc.)
    ├── asset/             ← Models and resources
    ├── StyleTTS2/         ← TTS system
    ├── tha4/              ← Avatar system
    └── cache/             ← Runtime cache
    
Total size: ~2-3 GB
```

**This entire folder is your distributable application!**

### After Inno Setup (Optional):
```
installer/Output/
└── VbotSetup.exe      ← Windows installer (~2-3 GB compressed)
```

**One-click installer for end users!**

---

## 🧪 Testing Steps

### Test 1: Quick Startup (2 min)
```powershell
cd dist\Vbot
.\Vbot.exe
```

**Expected:**
- ✅ Console window opens
- ✅ System checks run
- ✅ GUI appears
- ✅ Avatar loads

**If errors:**
- Check console output for missing DLLs
- Verify CUDA libraries are included
- Check logs in cache folder

### Test 2: Full Functionality (5 min)
```
1. Start Vbot.exe
2. Wait for GUI to load
3. Try voice input
4. Check if TTS works
5. Verify avatar animation
```

### Test 3: Fresh System (Optional)
```
1. Copy dist\Vbot folder to USB drive
2. Move to another PC (or VM)
3. Run Vbot.exe
4. Verify it works without dependencies
```

---

## ⚡ Build Options

### Option 1: Multi-File Build (RECOMMENDED)
```powershell
# Uses current vbot.spec configuration
python -m PyInstaller vbot.spec --clean
```

**Pros:**
- ✅ Faster startup (~10 seconds)
- ✅ Easier to debug
- ✅ Can update individual files

**Cons:**
- ⚠️ Multiple files to distribute (but we'll package them)

**Output:** `dist/Vbot/` folder with ~500-1000 files

---

### Option 2: Single-File Build
```powershell
# Modify vbot.spec: change EXE(onefile=False) to onefile=True
python -m PyInstaller vbot.spec --clean --onefile
```

**Pros:**
- ✅ Single .exe file (easier to share)
- ✅ Simpler distribution

**Cons:**
- ⚠️ Slower startup (30-60 seconds - extracts to temp)
- ⚠️ Larger file size
- ⚠️ Harder to debug

**Output:** Single `dist/Vbot.exe` file

**Recommendation:** Use multi-file, then package with Inno Setup installer!

---

## 🛠️ Build Script Options

### Basic Build
```powershell
.\build_installer.ps1
```

### Clean Build (Recommended First Time)
```powershell
.\build_installer.ps1 -Clean
```

### Just Build Executable (Skip Installer)
```powershell
.\build_installer.ps1 -SkipInstaller
```

### Just Build Installer (Already Have EXE)
```powershell
.\build_installer.ps1 -SkipBuild
```

---

## 🐛 Troubleshooting

### PyInstaller Command Not Found
**Error:** `pyinstaller is not recognized`
**Solution:** Use `python -m PyInstaller` instead of just `pyinstaller`
```powershell
python -m PyInstaller vbot.spec --clean
```

### Build Fails with Import Errors
**Solution:** Add missing modules to `hiddenimports` in `vbot.spec`

### Executable Crashes on Startup
**Solutions:**
1. Run with console window: Check vbot.spec has `console=True`
2. Check for missing DLLs: Look for "DLL not found" errors
3. Verify CUDA libraries: Check `binaries` section in vbot.spec

### CUDA/cuDNN Errors
**Solution:** Ensure cuDNN v8.x DLLs are in the CUDA_PATH
```powershell
# Check if DLLs exist
ls "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin\cudnn*.dll"
```

### Executable is Too Large
**Solutions:**
1. Remove unused models from `asset` folder before building
2. Add more packages to `excludes` in vbot.spec
3. Use compression in Inno Setup

### Missing Files in Build
**Solution:** Add to `datas` section in vbot.spec:
```python
datas = [
    ('your_missing_folder', 'destination_in_exe'),
]
```

---

## 📦 Size Optimization

### Current Build Size: ~2-3 GB

**What takes space:**
- PyTorch + CUDA: ~2 GB
- StyleTTS2 models: ~500 MB
- Transformers models: ~400 MB
- Other dependencies: ~500 MB

### To Reduce Size:

**1. Exclude Unused Models (Save ~1 GB)**
```powershell
# Before building, remove unused models from asset folder
# Keep only the models you need
```

**2. Use Model Download on First Run (Advanced)**
- Bundle small installer (~500 MB)
- Download models on first launch
- User downloads ~2 GB on first use

**3. Compress with Inno Setup**
- Inno Setup compresses well
- Can reduce installer by 30-40%

---

## 🎯 What's Included

### Your vbot.spec bundles:

✅ **Code:**
- Vbot.py (main application)
- vbot_launcher/* (launcher system)
- utils/* (all utilities)
- StyleTTS2/* (TTS system)
- tha4/* (avatar system)

✅ **Assets:**
- Models (Amelia character)
- Audio files
- Configuration files
- Icons

✅ **Dependencies:**
- PyTorch + CUDA
- Transformers
- Audio libraries
- GUI libraries
- All Python packages

✅ **CUDA Libraries:**
- CUDA 12.1 runtime DLLs
- cuDNN v8.x DLLs (IMPORTANT!)
- Other CUDA dependencies

❌ **NOT Included (User must have):**
- NVIDIA GPU drivers (user installs)
- Docker Desktop (for Ollama LLM)

---

## 🚀 Ready? Let's Build!

### Recommended First Build:

```powershell
# 1. Clean previous builds
.\build_installer.ps1 -Clean -SkipInstaller

# 2. Wait 15-30 minutes

# 3. Test the output
cd dist\Vbot
.\Vbot.exe

# 4. If it works, create installer
.\build_installer.ps1 -SkipBuild
```

---

## ⏱️ Time Estimates

| Step | First Time | Subsequent |
|------|-----------|------------|
| Install PyInstaller | 30 sec | - |
| Build executable | 15-30 min | 5-10 min |
| Test executable | 2 min | 2 min |
| Install Inno Setup | 5 min | - |
| Build installer | 5 min | 2 min |
| **TOTAL** | **30-45 min** | **10-15 min** |

---

## 📊 Success Checklist

After build completes:

```
[ ] PyInstaller finished without errors
[ ] dist/Vbot folder exists
[ ] Vbot.exe is present (~30-50 MB)
[ ] CUDA DLLs are in dist/Vbot folder
[ ] asset folder is included
[ ] StyleTTS2 folder is included
[ ] tha4 folder is included

[ ] Vbot.exe starts without crashes
[ ] System checks run
[ ] GUI appears
[ ] No missing DLL errors
[ ] No missing module errors

[ ] (Optional) Installer created
[ ] (Optional) Installer runs
[ ] (Optional) Installer completes
```

---

## 🎉 You're Ready!

**Run this now:**
```powershell
.\build_installer.ps1 -Clean
```

**Then go grab coffee for 30 minutes!** ☕

When you come back, you'll have a fully packaged Vbot ready to distribute! 🚀

---

## 📞 If You Need Help

**Common commands to try:**

```powershell
# Check PyInstaller version
python -m PyInstaller --version

# Check what's in dist folder
ls dist\Vbot

# Check exe size
(Get-Item dist\Vbot\Vbot.exe).Length / 1MB

# Test exe
cd dist\Vbot
.\Vbot.exe

# View build log
cat build\Vbot\warn-Vbot.txt
```

**Next: Check BUILD_TROUBLESHOOTING.md if issues occur**
