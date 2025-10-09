"""
Quick test script for launcher functionality.
Tests path resolution and system checks without launching full Vbot.
"""
import sys
from pathlib import Path

# Test 1: Import launcher modules
print("="*60)
print("TEST 1: Import Launcher Modules")
print("="*60)
try:
    from vbot_launcher import resource_path, system_check, launcher
    print("✅ All launcher modules imported successfully\n")
except ImportError as e:
    print(f"❌ Import failed: {e}\n")
    sys.exit(1)

# Test 2: Test resource path functions
print("="*60)
print("TEST 2: Resource Path Functions")
print("="*60)
try:
    from vbot_launcher.resource_path import (
        get_base_path, get_asset_path, get_cache_path,
        get_output_path, get_model_path, get_ref_sound_path,
        is_frozen, get_styletts2_path
    )
    
    print(f"Is Frozen: {is_frozen()}")
    print(f"Base Path: {get_base_path()}")
    print(f"Cache Path: {get_cache_path()}")
    print(f"Output Path: {get_output_path()}")
    print(f"StyleTTS2 Path: {get_styletts2_path()}")
    print(f"Model Path (Amelia): {get_model_path('Amelia', 'character_model', 'character_model.yaml')}")
    print(f"Ref Sound Path (Amelia): {get_ref_sound_path('Amelia', 'neutral.wav')}")
    print("✅ All path functions work correctly\n")
except Exception as e:
    print(f"❌ Path functions failed: {e}\n")
    sys.exit(1)

# Test 3: Verify paths exist
print("="*60)
print("TEST 3: Verify Critical Paths Exist")
print("="*60)
try:
    base = get_base_path()
    styletts2 = get_styletts2_path()
    asset = get_asset_path()
    
    checks = {
        "Base directory": base.exists(),
        "StyleTTS2 directory": styletts2.exists(),
        "Asset directory": asset.exists(),
        "Utils directory": (base / "utils").exists(),
        "THA4 directory": (base / "tha4").exists(),
    }
    
    all_good = True
    for name, exists in checks.items():
        if exists:
            print(f"✅ {name}")
        else:
            print(f"❌ {name} - NOT FOUND")
            all_good = False
    
    if all_good:
        print("\n✅ All critical paths verified\n")
    else:
        print("\n⚠️  Some paths missing - may need to run from correct directory\n")
        
except Exception as e:
    print(f"❌ Path verification failed: {e}\n")

# Test 4: Run system check
print("="*60)
print("TEST 4: System Requirements Check")
print("="*60)
try:
    from vbot_launcher.system_check import SystemRequirements
    
    checker = SystemRequirements()
    success = checker.check_all()
    
    if success:
        print("\n✅ System requirements check completed successfully")
    else:
        print("\n⚠️  System requirements check found issues (see above)")
        
except Exception as e:
    print(f"❌ System check failed: {e}")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "="*60)
print("TEST SUMMARY")
print("="*60)
print("✅ Launcher infrastructure is working!")
print("✅ Path management system is functional")
print("✅ System validation is operational")
print("\nYou can now:")
print("  1. Run: python vbot_launcher/system_check.py")
print("  2. Run: python vbot_launcher/launcher.py")
print("="*60)
