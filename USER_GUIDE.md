# 🎯 Vbot - User Guide

## Quick Start (Portable Windows Executable)

### Prerequisites
- Windows 10/11 (64-bit)
- NVIDIA GPU (GTX 1060 or better recommended)
- 16GB RAM minimum
- 10-15GB free disk space
- Internet connection on first run for local model downloads
- Python/Conda 3.10 environment with Vbot dependencies installed

---

## 🚀 Installation Steps

### 1. Extract the Package
```
Extract Vbot.zip to a location like:
C:\Program Files\Vbot\
or
C:\Users\[YourName]\Desktop\Vbot\
```

### 2. Install Docker Desktop (Required for Local LLM Chat)
1. Download: https://www.docker.com/products/docker-desktop
2. Install and restart computer
3. Enable the WSL 2 backend during setup
4. Start Docker Desktop before launching Vbot

### 3. First Run
```
Double-click Vbot.exe
```

**What happens:**
1. The launcher finds your Python/Conda runtime
2. The launcher starts the packaged Vbot app
3. Downloads required local models if needed (first time only, several GB)
4. Main window appears
5. You're ready to chat!

If the launcher cannot find the right Python environment, set `VBOT_PYTHON` to the full path of your environment's `python.exe`.

---

## 💬 Using Vbot

### Basic Chat
1. Type your message in the text box
2. Press Enter or click Send
3. Vbot responds with voice and text

### Voice Input
1. Click the microphone button (or press hotkey)
2. Speak your message
3. Release when done

### Changing Character
1. Click the character avatar
2. Select from available models (Amelia, etc.)
3. Voice and appearance change

---

## ⚙️ Settings

### Audio Settings
- Input Device: Select your microphone
- Output Device: Select speakers/headphones
- Volume: Adjust voice volume

### Model Settings
- TTS Voice: Change text-to-speech voice
- LLM Model: Select AI model (requires Docker Desktop in the current release)

---

## 🐛 Troubleshooting

### "No module named 'unittest'" Error

**Temporary Fix:**
1. Close Vbot
2. Open Command Prompt as Administrator
3. Navigate to Vbot folder:
   ```cmd
   cd "C:\Program Files\Vbot"
   cd _internal
   ```
4. Copy unittest from system Python:
   ```cmd
   xcopy "C:\Users\%USERNAME%\.conda\envs\vbot\Lib\unittest" ".\unittest\" /E /I /Y
   ```
   OR if Anaconda is in ProgramData:
   ```cmd
   xcopy "C:\ProgramData\Anaconda3\Lib\unittest" ".\unittest\" /E /I /Y
   ```
5. Run Vbot.exe again

---

### GPU Not Detected

**Fix:**
1. Update NVIDIA drivers: https://www.nvidia.com/download/index.aspx
2. Restart computer
3. Run Vbot again

---

### Docker Warning

The current full local chat path uses Ollama through Docker Desktop.

**Options:**
- **Install/start Docker:** Required for local LLM chat in this release
- **Continue without Docker:** Non-LLM functionality may still be usable, but the main chatbot experience will be limited

---

### Audio Issues

**No sound:**
1. Check volume in Vbot settings
2. Check Windows volume mixer
3. Try different output device in settings

**Can't hear me:**
1. Check microphone permissions in Windows
2. Select correct input device in Vbot
3. Test microphone in Windows settings

---

### Slow Performance

**Solutions:**
1. Close other GPU-heavy applications
2. Lower quality settings in Vbot
3. Ensure GPU is being used (check Task Manager > Performance > GPU)

---

## 📁 File Locations

### Application Files
```
Vbot\
├── Vbot.exe              ← Main application
└── _internal\
    └── app\              ← Packaged Vbot source, models, and resources
```

### User Data
```
C:\Users\[You]\AppData\Local\Vbot\
├── cache\                ← Cached models
└── config\               ← Settings

C:\Users\[You]\Documents\Vbot\
└── outputs\              ← Generated audio files
```

---

## 🔄 Updating Vbot

1. Download new version
2. Extract to new folder (or overwrite old)
3. Your settings and cache are preserved (separate location)

---

## ❓ FAQ

**Q: How much disk space does Vbot use?**
A: ~3-4GB for application + several GB for local models/cache. Keep 10-15GB free.

**Q: Can I run Vbot without internet?**
A: Yes after first setup for already-downloaded local models. First run needs internet.

**Q: Do I need a powerful GPU?**
A: Recommended but not required. Works on CPU (slower).

**Q: Is my data sent anywhere?**
A: The default goal is local inference. Chat/voice data does not need a cloud LLM API in this release.

**Q: Can I use different voices?**
A: Yes! Select from character models in the app.

---

## 🆘 Still Need Help?

1. Check error message in Vbot window
2. Look for log files in `cache\` folder
3. Contact support: [your-email]
4. GitHub Issues: [github-link]

---

**Version:** 1.0  
**Last Updated:** 2026-07-01
