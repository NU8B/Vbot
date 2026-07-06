"""
Vbot memory debug command.

Prints MemoryManager.get_memory_info() plus CUDA peak-allocation stats in a
readable form, so GPU sizing and hotswap behavior can be inspected without
launching the full app.

Usage:
    python scripts/memory_debug.py          # human-readable report
    python scripts/memory_debug.py --json   # machine-readable output
"""

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.performance_boost import MemoryManager


def collect_memory_report():
    """Extend get_memory_info() with peak stats useful for sizing GPUs."""
    import torch

    report = {"memory_info": MemoryManager.get_memory_info()}

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device)
        report["gpu"] = {
            "name": properties.name,
            "device_index": device,
            "peak_allocated_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
            "peak_reserved_gb": torch.cuda.max_memory_reserved(device) / 1024**3,
        }
    else:
        report["gpu"] = None

    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Print Vbot memory usage information")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a report")
    args = parser.parse_args(argv)

    report = collect_memory_report()

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    info = report["memory_info"]
    print("Vbot memory debug")
    print("=" * 40)
    print(f"System RAM used: {info['system_ram']:.1f}%")

    gpu_memory = info.get("gpu_memory")
    if gpu_memory and report["gpu"]:
        gpu = report["gpu"]
        print(f"GPU: {gpu['name']} (device {gpu['device_index']})")
        print(f"  allocated: {gpu_memory['allocated']:.2f} GB")
        print(f"  reserved:  {gpu_memory['cached']:.2f} GB")
        print(f"  total:     {gpu_memory['total']:.2f} GB")
        print(f"  peak allocated: {gpu['peak_allocated_gb']:.2f} GB")
        print(f"  peak reserved:  {gpu['peak_reserved_gb']:.2f} GB")
    else:
        print("GPU: CUDA not available")
    print("=" * 40)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
