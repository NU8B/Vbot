# Vbot Quick Start Guide

## 🚀 For End Users (Recommended)

### Windows Automated Setup
```powershell
# Run PowerShell as Administrator
.\setup\setup_windows.ps1
```

This will:
- ✅ Validate your system
- ✅ Check CUDA and cuDNN
- ✅ Install Docker if needed
- ✅ Guide you through setup

### Launch Vbot
```powershell
# After setup completes
python vbot_launcher/launcher.py
```

---

## 🧪 For Testing

### Test Path Management
```powershell
python test_launcher.py
```

### Check System Requirements
```powershell
.\setup\validate_system.ps1
```

### Check CUDA/cuDNN
```powershell
.\setup\check_cuda.ps1
```

---

## 🔧 For Developers

### Current Development Mode
```powershell
# Still works exactly as before
python Vbot.py
```

### With Launcher (New)
```powershell
python vbot_launcher/launcher.py
```

### Manual Setup Steps
```powershell
# 1. Validate system
.\setup\validate_system.ps1 -Detailed

# 2. Check CUDA
.\setup\check_cuda.ps1 -Detailed

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run Vbot
python Vbot.py
```

---

## 📋 System Requirements

### Minimum
- Windows 10 Build 19041+ or Windows 11
- NVIDIA GPU with 8GB+ VRAM
- 16GB RAM
- 50GB free disk space
- CUDA Toolkit 12.1
- cuDNN v8.9.7 (NOT v9.x!)
- Docker Desktop
- Python 3.10.x

### Recommended
- Windows 11
- NVIDIA RTX 3070 or better
- 32GB RAM
- 100GB free SSD space

---

## ⚠️ Common Issues

### cuDNN Version Mismatch
**Error:** `Could not locate cudnn_ops_infer64_8.dll`

**Solution:** Run `.\setup\check_cuda.ps1` to verify cuDNN version

### Docker Not Running
**Error:** `Connection error - check if Ollama service is running`

**Solution:** Start Docker Desktop from Start Menu

### GPU Not Detected
**Error:** `No CUDA-capable GPU found`

**Solution:** 
1. Install NVIDIA drivers (545.84+)
2. Install CUDA Toolkit 12.1
3. Restart computer

---

## 📚 Documentation

- **Full Analysis:** `distribution_planning/DEEP_DIVE_ANALYSIS.md`
- **Implementation Log:** `distribution_planning/IMPLEMENTATION_COMPLETE_PHASE1.md`
- **Distribution Strategy:** `distribution_planning/distribution_strategy.md`
- **Challenges & Solutions:** `distribution_planning/challenges.md`

---

## 🆘 Getting Help

1. Run system check: `.\setup\validate_system.ps1 -Detailed`
2. Check CUDA: `.\setup\check_cuda.ps1 -Detailed`
3. Review: `distribution_planning/challenges.md`
4. Check logs in console output

---

## 🎯 What Works Now

✅ Path management for bundled executables  
✅ System requirements validation  
✅ CUDA/cuDNN version checking  
✅ Docker Desktop automation  
✅ Complete setup workflow  
✅ Development mode (unchanged)  

## 🔜 Coming Soon

- PyInstaller executable bundle
- Inno Setup Windows installer
- One-click installation
- Offline installer option

---

**Current Status:** Phase 1 Complete (85% of Week 1)  
**Last Updated:** 2025-10-02
