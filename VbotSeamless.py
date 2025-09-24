"""
Vbot Seamless Launcher
New main entry point that provides a seamless experience from loading to chat
"""

import os
import sys
import argparse
import pyaudio

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from utils.seamless_interface import launch_seamless_vbot
from utils.performance_boost import memory_manager


def main():
    """Main entry point for seamless Vbot experience"""
    print("🚀 Starting Vbot Seamless Experience...")
    
    # Apply initial optimizations
    memory_manager.optimize_pytorch()
    print("💾 Memory optimizations applied")
    
    # Get default device index
    p = pyaudio.PyAudio()
    try:
        default_device_index = p.get_default_input_device_info()["index"]
    except IOError:
        default_device_index = None
    p.terminate()
    
    # Setup command-line argument parsing
    parser = argparse.ArgumentParser(description="Vbot Voice Assistant - Seamless Experience")
    parser.add_argument(
        "--device-index",
        type=int,
        default=default_device_index,
        help="Index of the audio input device to use.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="Maximum number of models to load simultaneously (default: 2)",
    )
    args = parser.parse_args()
    
    print("🎭 Launching seamless interface...")
    print("📋 This will:")
    print("   1. Load all avatar models in parallel")
    print("   2. Show welcome screen when ready")
    print("   3. Seamlessly transition to chat interface")
    print("   4. Allow instant avatar switching")
    
    try:
        # Launch the seamless interface
        launch_seamless_vbot(device_index=args.device_index)
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error launching Vbot: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
