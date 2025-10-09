"""
Main launcher for Vbot application.
Handles initialization, system checks, and application startup.
"""
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vbot_launcher.resource_path import (
    get_base_path, is_frozen, setup_python_path,
    get_cache_path, get_output_path
)
from vbot_launcher.system_check import SystemRequirements


def pre_flight_check():
    """Run system requirements check before launching."""
    print("="*60)
    print("VBOT LAUNCHER")
    print("="*60 + "\n")
    
    if is_frozen():
        print("ℹ️  Running as bundled application")
    else:
        print("ℹ️  Running in development mode")
    
    print(f"📁 Base path: {get_base_path()}")
    print(f"💾 Cache path: {get_cache_path()}")
    print(f"📤 Output path: {get_output_path()}\n")
    
    # Run system check
    checker = SystemRequirements()
    success = checker.check_all()
    
    if not success:
        print("\n❌ System requirements not met. Please fix the errors above.")
        input("\nPress Enter to exit...")
        sys.exit(1)
    
    print("\n✅ System check passed! Starting Vbot...\n")
    return True


def launch_vbot():
    """Launch the main Vbot application."""
    try:
        # Ensure Python path is set up
        setup_python_path()
        
        # Import and run main Vbot application
        print("🚀 Loading Vbot modules...")
        
        # Set environment variable for model selection if needed
        if not os.getenv("MODEL_NAME"):
            os.environ["MODEL_NAME"] = "Amelia"
        
        # Import main module
        import Vbot
        
        print("✅ Starting Vbot application...\n")
        
        # Run main function
        Vbot.main()
        
    except KeyboardInterrupt:
        print("\n\n👋 Vbot interrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error starting Vbot: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")
        sys.exit(1)


def main():
    """Main launcher entry point."""
    try:
        # Run pre-flight checks
        if not pre_flight_check():
            sys.exit(1)
        
        # Launch application
        launch_vbot()
        
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")
        sys.exit(1)


if __name__ == "__main__":
    main()
