"""
System requirements checker for Vbot.
Validates hardware, software, and dependency requirements.
"""
import os
import sys
import platform
import subprocess
import psutil
from pathlib import Path


class SystemRequirements:
    """Check system requirements for Vbot."""
    
    # Minimum requirements
    MIN_RAM_GB = 16
    MIN_VRAM_GB = 8
    MIN_FREE_SPACE_GB = 50
    REQUIRED_CUDA_VERSION = "12.1"
    REQUIRED_CUDNN_VERSION = "8.9.7"
    
    def __init__(self):
        self.checks = {}
        self.errors = []
        self.warnings = []
    
    def check_all(self):
        """Run all system checks."""
        print("🔍 Checking system requirements...\n")
        
        self.check_os()
        self.check_ram()
        self.check_disk_space()
        self.check_gpu()
        self.check_cuda()
        self.check_docker()
        self.check_python()
        
        return self.generate_report()
    
    def check_os(self):
        """Check operating system."""
        os_name = platform.system()
        os_version = platform.version()
        
        if os_name == "Windows":
            # Check Windows 10/11
            version_info = sys.getwindowsversion()
            build = version_info.build
            
            if build >= 19041:  # Windows 10 Build 19041 (May 2020 Update) or later
                self.checks['os'] = {'status': 'pass', 'message': f'Windows {version_info.major}.{version_info.minor} Build {build}'}
            else:
                self.checks['os'] = {'status': 'fail', 'message': f'Windows build {build} is too old. Need 19041+'}
                self.errors.append("Windows 10 Build 19041+ or Windows 11 required")
        else:
            self.checks['os'] = {'status': 'fail', 'message': f'{os_name} is not supported'}
            self.errors.append("Only Windows 10/11 is currently supported")
    
    def check_ram(self):
        """Check system RAM."""
        ram_gb = psutil.virtual_memory().total / (1024**3)
        
        if ram_gb >= self.MIN_RAM_GB:
            self.checks['ram'] = {'status': 'pass', 'message': f'{ram_gb:.1f} GB available'}
        else:
            self.checks['ram'] = {'status': 'fail', 'message': f'{ram_gb:.1f} GB (need {self.MIN_RAM_GB} GB)'}
            self.errors.append(f"Insufficient RAM: {ram_gb:.1f} GB (need {self.MIN_RAM_GB} GB)")
    
    def check_disk_space(self):
        """Check available disk space."""
        try:
            disk = psutil.disk_usage('C:\\')
            free_gb = disk.free / (1024**3)
            
            if free_gb >= self.MIN_FREE_SPACE_GB:
                self.checks['disk'] = {'status': 'pass', 'message': f'{free_gb:.1f} GB free'}
            else:
                self.checks['disk'] = {'status': 'warn', 'message': f'{free_gb:.1f} GB free (recommend {self.MIN_FREE_SPACE_GB} GB)'}
                self.warnings.append(f"Low disk space: {free_gb:.1f} GB (recommend {self.MIN_FREE_SPACE_GB} GB)")
        except Exception as e:
            self.checks['disk'] = {'status': 'unknown', 'message': f'Could not check: {e}'}
    
    def check_gpu(self):
        """Check NVIDIA GPU availability."""
        try:
            import torch
            
            if not torch.cuda.is_available():
                self.checks['gpu'] = {'status': 'fail', 'message': 'No CUDA-capable GPU found'}
                self.errors.append("NVIDIA GPU with CUDA support required")
                return
            
            gpu_name = torch.cuda.get_device_name(0)
            gpu_count = torch.cuda.device_count()
            
            # Try to get VRAM (may fail on some systems)
            try:
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
                if vram_gb >= self.MIN_VRAM_GB:
                    self.checks['gpu'] = {'status': 'pass', 'message': f'{gpu_name} ({vram_gb:.1f} GB VRAM)'}
                else:
                    self.checks['gpu'] = {'status': 'fail', 'message': f'{gpu_name} ({vram_gb:.1f} GB VRAM, need {self.MIN_VRAM_GB} GB)'}
                    self.errors.append(f"Insufficient VRAM: {vram_gb:.1f} GB (need {self.MIN_VRAM_GB} GB)")
            except:
                self.checks['gpu'] = {'status': 'pass', 'message': f'{gpu_name} (VRAM check unavailable)'}
                self.warnings.append("Could not verify VRAM amount")
                
        except ImportError:
            self.checks['gpu'] = {'status': 'unknown', 'message': 'PyTorch not installed yet'}
            self.warnings.append("GPU check will be performed after installation")
    
    def check_cuda(self):
        """Check CUDA and cuDNN installation."""
        try:
            import torch
            
            if torch.cuda.is_available():
                cuda_version = torch.version.cuda
                cudnn_version = torch.backends.cudnn.version()
                
                # Check CUDA version
                if cuda_version and cuda_version.startswith("12.1"):
                    cuda_status = 'pass'
                else:
                    cuda_status = 'warn'
                    self.warnings.append(f"CUDA {cuda_version} detected, expected 12.1")
                
                # Check cuDNN version
                cudnn_str = str(cudnn_version)
                cudnn_major = cudnn_str[0]
                
                if cudnn_major == "8":
                    cudnn_status = 'pass'
                    self.checks['cuda'] = {
                        'status': cuda_status,
                        'message': f'CUDA {cuda_version}, cuDNN {cudnn_version}'
                    }
                else:
                    self.checks['cuda'] = {
                        'status': 'fail',
                        'message': f'cuDNN {cudnn_version} (need v8.x, found v{cudnn_major}.x)'
                    }
                    self.errors.append(f"cuDNN version mismatch: v{cudnn_major}.x found, need v8.9.7")
            else:
                self.checks['cuda'] = {'status': 'fail', 'message': 'CUDA not available'}
                self.errors.append("CUDA Toolkit 12.1 required")
                
        except ImportError:
            self.checks['cuda'] = {'status': 'unknown', 'message': 'PyTorch not installed yet'}
    
    def check_docker(self):
        """Check Docker Desktop installation."""
        try:
            # Try to run docker command
            result = subprocess.run(
                ['docker', '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                version = result.stdout.strip()
                self.checks['docker'] = {'status': 'pass', 'message': version}
                
                # Check if Docker daemon is running
                result = subprocess.run(
                    ['docker', 'ps'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if result.returncode != 0:
                    self.checks['docker'] = {'status': 'warn', 'message': 'Docker installed but not running'}
                    self.warnings.append("Docker Desktop is not running - please start it")
            else:
                self.checks['docker'] = {'status': 'fail', 'message': 'Docker command failed'}
                self.errors.append("Docker Desktop required")
                
        except FileNotFoundError:
            self.checks['docker'] = {'status': 'fail', 'message': 'Docker not installed'}
            self.errors.append("Docker Desktop not installed")
        except subprocess.TimeoutExpired:
            self.checks['docker'] = {'status': 'warn', 'message': 'Docker check timed out'}
            self.warnings.append("Docker check timed out - Docker may not be running")
        except Exception as e:
            self.checks['docker'] = {'status': 'unknown', 'message': f'Could not check: {e}'}
    
    def check_python(self):
        """Check Python version."""
        version = sys.version_info
        version_str = f"{version.major}.{version.minor}.{version.micro}"
        
        if version.major == 3 and version.minor == 10:
            self.checks['python'] = {'status': 'pass', 'message': f'Python {version_str}'}
        else:
            self.checks['python'] = {'status': 'warn', 'message': f'Python {version_str} (recommend 3.10.x)'}
            self.warnings.append(f"Python {version_str} detected, 3.10.x recommended")
    
    def generate_report(self):
        """Generate system check report."""
        print("\n" + "="*60)
        print("SYSTEM REQUIREMENTS CHECK")
        print("="*60 + "\n")
        
        # Print all checks
        for check_name, result in self.checks.items():
            status = result['status']
            message = result['message']
            
            if status == 'pass':
                symbol = '✅'
            elif status == 'fail':
                symbol = '❌'
            elif status == 'warn':
                symbol = '⚠️'
            else:
                symbol = '❓'
            
            print(f"{symbol} {check_name.upper():12} {message}")
        
        print("\n" + "="*60)
        
        # Print errors
        if self.errors:
            print("\n❌ ERRORS (Must fix before proceeding):")
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. {error}")
        
        # Print warnings
        if self.warnings:
            print("\n⚠️  WARNINGS (Recommended to fix):")
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. {warning}")
        
        if not self.errors and not self.warnings:
            print("\n✅ All checks passed! System is ready for Vbot.")
        elif not self.errors:
            print("\n⚠️  System meets minimum requirements but has warnings.")
        else:
            print("\n❌ System does not meet minimum requirements.")
        
        print("="*60 + "\n")
        
        return len(self.errors) == 0


def main():
    """Run system check as standalone script."""
    checker = SystemRequirements()
    success = checker.check_all()
    
    if not success:
        print("\n🔧 Please fix the errors above before installing Vbot.")
        sys.exit(1)
    else:
        print("\n✅ System check complete! You can proceed with installation.")
        sys.exit(0)


if __name__ == "__main__":
    main()
