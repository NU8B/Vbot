# PyInstaller runtime hook for unittest module
# This ensures unittest can be imported from base Anaconda installation

import sys
import os

# Add base Anaconda Lib path for unittest
base_lib = r'C:\ProgramData\Anaconda3\Lib'
if os.path.exists(base_lib) and base_lib not in sys.path:
    sys.path.insert(0, base_lib)
    print(f"[Runtime Hook] Added {base_lib} to sys.path for unittest")
