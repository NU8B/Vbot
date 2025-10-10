# -*- mode: python ; coding: utf-8 -*-
# Vbot PyInstaller Build Specification
# This file configures how PyInstaller bundles Vbot into a standalone executable

import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# CRITICAL: Increase recursion limit for complex dependency analysis
sys.setrecursionlimit(sys.getrecursionlimit() * 5)

# Get project root
block_cipher = None
project_root = os.path.abspath('.')

# ============================================================================
# DATA FILES - Include all necessary assets and models
# ============================================================================

datas = [
    # Asset files
    ('asset', 'asset'),
    
    # Avatar profile images
    ('recommender-avatar-profile', 'recommender-avatar-profile'),
    
    # StyleTTS2 files
    ('StyleTTS2', 'StyleTTS2'),
    
    # THA4 avatar system
    ('tha4', 'tha4'),
    
    # Cache directory (empty, but structure needed)
    ('cache/.gitkeep', 'cache') if os.path.exists('cache/.gitkeep') else ('cache', 'cache'),
]

# Add torch/torchaudio data files if they exist
try:
    import torch
    torch_dir = os.path.dirname(torch.__file__)
    # Include CUDA DLLs and libraries
    datas += [(os.path.join(torch_dir, 'lib'), 'torch/lib')]
except ImportError:
    print("WARNING: torch not found - install dependencies first!")

# Collect transformers data files
try:
    datas += collect_data_files('transformers')
except Exception as e:
    print(f"WARNING: Could not collect transformers data: {e}")

# Collect wxPython data files
try:
    datas += collect_data_files('wx')
except Exception as e:
    print(f"WARNING: Could not collect wx data: {e}")

# Note: unittest will be handled via pathex in Analysis

# Collect other package data
for package in ['phonemizer', 'gruut', 'num2words', 'language_tags', 'segments', 'csvw']:
    try:
        datas += collect_data_files(package)
    except Exception:
        pass

# ============================================================================
# HIDDEN IMPORTS - Modules that PyInstaller might miss
# ============================================================================

hiddenimports = [
    # Core ML/AI packages
    'torch',
    'torch.nn',
    'torch.cuda',
    'torchaudio',
    'torchaudio.transforms',
    'transformers',
    'transformers.models',
    'transformers.models.whisper',
    
    # StyleTTS2 dependencies
    'monotonic_align',
    'phonemizer',
    'gruut',
    'num2words',
    'language_tags',
    'segments',
    'csvw',
    'einops',
    'einops_exts',
    
    # Audio processing
    'librosa',
    'soundfile',
    'sounddevice',
    'pydub',
    'pyaudio',
    
    # GUI
    'customtkinter',
    'PIL',
    'PIL._tkinter_finder',
    'wx',
    'wx._core',  # Internal wxPython module
    'win32gui',
    'win32con',
    'win32api',
    
    # THA4 avatar
    'OpenGL',
    'OpenGL.GL',
    'OpenGL.GLU',
    'glfw',
    
    # Docker/Ollama
    'docker',
    'requests',
    
    # Emotion detection & Scientific computing
    'sklearn',
    'sklearn.preprocessing',
    'scipy',
    'scipy.stats',
    'scipy.sparse',
    'numpy.testing',
    
    # Utilities
    'yaml',
    'json',
    'threading',
    'queue',
    'asyncio',
    'unittest',
    'unittest.mock',
    'unittest.case',
    
    # Performance
    'psutil',
    'gc',
]

# Collect all submodules for critical packages
for package in ['transformers', 'torch', 'torchaudio', 'unittest']:
    try:
        hiddenimports += collect_submodules(package)
    except Exception:
        pass

# ============================================================================
# BINARIES - External DLLs and libraries
# ============================================================================

binaries = []

# CUDA libraries (if available)
# Try multiple CUDA paths (user may have multiple versions or different env var)
cuda_paths_to_try = [
    os.environ.get('CUDA_PATH'),  # From environment variable
    r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1',  # Preferred version
    r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6',  # Alternative
    r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0',  # Alternative
]

cuda_path = None
for path in cuda_paths_to_try:
    if path and os.path.exists(path):
        cuda_bin = os.path.join(path, 'bin')
        if os.path.exists(cuda_bin):
            # Check if cuDNN DLLs exist here
            test_dll = os.path.join(cuda_bin, 'cudnn64_8.dll')
            if os.path.exists(test_dll):
                cuda_path = path
                print(f"Using CUDA from: {cuda_path}")
                break

if cuda_path:
    cuda_bin = os.path.join(cuda_path, 'bin')
    if os.path.exists(cuda_bin):
        # Include essential CUDA DLLs
        cuda_dlls = [
            'cublas64_12.dll',
            'cublasLt64_12.dll',
            'cudart64_12.dll',
            'cufft64_11.dll',
            'curand64_10.dll',
            'cusparse64_12.dll',
            'nvrtc64_120_0.dll',
            # cuDNN v8.x DLLs (CRITICAL!)
            'cudnn64_8.dll',
            'cudnn_ops_infer64_8.dll',
            'cudnn_ops_train64_8.dll',
            'cudnn_cnn_infer64_8.dll',
            'cudnn_cnn_train64_8.dll',
            'cudnn_adv_infer64_8.dll',
            'cudnn_adv_train64_8.dll',
        ]
        
        for dll in cuda_dlls:
            dll_path = os.path.join(cuda_bin, dll)
            if os.path.exists(dll_path):
                binaries.append((dll_path, '.'))
            else:
                print(f"WARNING: {dll} not found at {dll_path}")

# ============================================================================
# EXCLUDES - Remove unnecessary packages to reduce size
# ============================================================================

excludes = [
    # Development tools (minimal exclusions to avoid breaking dependencies)
    'pytest',
    'IPython',
    'jupyter',
    'notebook',
    
    # Unused matplotlib backends (but keep matplotlib itself as tha4 needs it)
    'matplotlib.backends.backend_gtk3',
    'matplotlib.backends.backend_qt5',
    
    # Unused GUI toolkits (but keep wx as it's needed for avatar)
    'PyQt5',
    'PyQt6',
    'PySide2',
    'PySide6',
    'gtk',
    
    # Notes on what we're keeping:
    # - unittest: needed by torch
    # - wx: needed for avatar system
    # - matplotlib: needed by tha4 avatar system
    # - numpy.testing: needed by scipy (used by transformers)
    # - pandas: would exclude but might break dependencies, keeping it in for safety
]

# ============================================================================
# ANALYSIS - Analyze the main script
# ============================================================================

a = Analysis(
    ['Vbot.py'],  # Main entry point
    pathex=[
        project_root,
        r'C:\ProgramData\Anaconda3\Lib',  # Add base Anaconda for unittest
    ],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['hook-unittest.py'],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# ============================================================================
# COLLECT FILES
# ============================================================================

pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=block_cipher
)

exe = EXE(
    pyz,
    a.scripts,
    [],  # Don't include everything in one file yet
    exclude_binaries=True,  # Separate binaries for easier debugging
    name='Vbot',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # Don't use UPX compression (can cause issues with ML models)
    console=True,  # Show console for debugging (change to False for release)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='asset/icon.ico' if os.path.exists('asset/icon.ico') else None,
)

# ============================================================================
# COLLECT DISTRIBUTION
# ============================================================================

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='Vbot',
)

# ============================================================================
# NOTES FOR BUILDING
# ============================================================================
"""
BUILD INSTRUCTIONS:

1. Install PyInstaller:
   pip install pyinstaller

2. Build the executable:
   pyinstaller vbot.spec

3. Output will be in:
   dist/Vbot/Vbot.exe

4. Test the executable:
   cd dist/Vbot
   ./Vbot.exe

5. For one-file build (slower startup, easier distribution):
   - Change 'onefile=False' to 'onefile=True' in EXE()
   - Remove COLLECT() section
   
SIZE OPTIMIZATION:
- Current config: ~2-3GB (includes all models and dependencies)
- One-file mode: Same size, but single .exe
- To reduce size: Exclude unused model files from 'asset' folder

TROUBLESHOOTING:
- If build fails: Check hiddenimports and datas
- If exe crashes: Run with console=True to see errors
- If CUDA errors: Verify CUDA DLLs are included in binaries
- If import errors: Add missing modules to hiddenimports
"""
