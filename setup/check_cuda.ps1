# Vbot CUDA and cuDNN Checker
# Validates CUDA Toolkit 12.1 and cuDNN v8.9.7 installation
# Critical: PyTorch 2.5.1+cu121 requires cuDNN v8.x (NOT v9.x)

param(
    [switch]$Detailed,
    [switch]$FixMode
)

$ErrorActionPreference = "Continue"

# Color functions
function Write-Success { param($Message) Write-Host "✅ $Message" -ForegroundColor Green }
function Write-Error { param($Message) Write-Host "❌ $Message" -ForegroundColor Red }
function Write-Warning { param($Message) Write-Host "⚠️  $Message" -ForegroundColor Yellow }
function Write-Info { param($Message) Write-Host "ℹ️  $Message" -ForegroundColor Cyan }

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "         CUDA & cuDNN Validation" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

$issues = @()
$cudaOk = $false
$cudnnOk = $false

# Check 1: CUDA Toolkit Installation
Write-Info "Checking CUDA Toolkit installation..."
$cudaPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"

if (Test-Path $cudaPath) {
    Write-Success "CUDA directory found: $cudaPath"
    
    # List all CUDA versions
    $versions = Get-ChildItem $cudaPath -Directory
    Write-Host "  Installed versions:" -ForegroundColor Gray
    foreach ($ver in $versions) {
        Write-Host "    • $($ver.Name)" -ForegroundColor Gray
    }
    
    # Check for v12.1 specifically
    $cuda121 = Join-Path $cudaPath "v12.1"
    if (Test-Path $cuda121) {
        Write-Success "CUDA 12.1 found at: $cuda121"
        $cudaOk = $true
        
        if ($Detailed) {
            $nvccPath = Join-Path $cuda121 "bin\nvcc.exe"
            if (Test-Path $nvccPath) {
                Write-Success "nvcc.exe found"
                try {
                    $nvccVersion = & $nvccPath --version 2>&1 | Select-String "release"
                    Write-Host "  $nvccVersion" -ForegroundColor Gray
                } catch {}
            }
        }
    } else {
        Write-Error "CUDA 12.1 not found (required for PyTorch 2.5.1+cu121)"
        $issues += "CUDA Toolkit 12.1 is required but not installed"
    }
} else {
    Write-Error "CUDA Toolkit not found at: $cudaPath"
    $issues += "CUDA Toolkit 12.1 must be installed"
}

# Check 2: cuDNN Installation (CRITICAL CHECK)
Write-Host ""
Write-Info "Checking cuDNN installation..."
Write-Warning "CRITICAL: PyTorch 2.5.1+cu121 requires cuDNN v8.x (NOT v9.x)"

if ($cudaOk) {
    $cuda121 = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"
    $cudnnBinPath = Join-Path $cuda121 "bin"
    
    # Check for cuDNN DLLs
    $cudnnDlls = @(
        "cudnn_ops_infer64_8.dll",
        "cudnn_ops_train64_8.dll",
        "cudnn_adv_infer64_8.dll",
        "cudnn_adv_train64_8.dll",
        "cudnn_cnn_infer64_8.dll",
        "cudnn_cnn_train64_8.dll",
        "cudnn64_8.dll"
    )
    
    $foundDlls = @()
    $missingDlls = @()
    
    foreach ($dll in $cudnnDlls) {
        $dllPath = Join-Path $cudnnBinPath $dll
        if (Test-Path $dllPath) {
            $foundDlls += $dll
            
            if ($Detailed) {
                $fileInfo = Get-Item $dllPath
                $version = $fileInfo.VersionInfo.FileVersion
                Write-Host "  ✓ $dll $(if ($version) { "(v$version)" })" -ForegroundColor Gray
            }
        } else {
            $missingDlls += $dll
        }
    }
    
    # Check for v9.x DLLs (WRONG VERSION)
    $cudnn9Dlls = Get-ChildItem $cudnnBinPath -Filter "cudnn*64_9.dll" -ErrorAction SilentlyContinue
    if ($cudnn9Dlls) {
        Write-Error "cuDNN v9.x detected! This WILL cause crashes!"
        Write-Error "Found v9.x DLLs:"
        foreach ($dll in $cudnn9Dlls) {
            Write-Host "  • $($dll.Name)" -ForegroundColor Red
        }
        $issues += "cuDNN v9.x is installed but v8.9.7 is required (version mismatch)"
        $cudnnOk = $false
    }
    
    if ($foundDlls.Count -eq $cudnnDlls.Count) {
        Write-Success "All cuDNN v8.x DLLs found ($($foundDlls.Count) files)"
        $cudnnOk = $true
    } elseif ($foundDlls.Count -gt 0) {
        Write-Warning "Some cuDNN v8.x DLLs found but not all"
        Write-Host "  Found: $($foundDlls.Count) / $($cudnnDlls.Count)" -ForegroundColor Yellow
        if ($missingDlls.Count -gt 0) {
            Write-Host "  Missing:" -ForegroundColor Yellow
            foreach ($dll in $missingDlls) {
                Write-Host "    • $dll" -ForegroundColor Yellow
            }
        }
        $issues += "Incomplete cuDNN v8.x installation"
    } else {
        Write-Error "No cuDNN v8.x DLLs found in $cudnnBinPath"
        $issues += "cuDNN v8.9.7 is not installed"
    }
    
    # Check include and lib folders
    if ($Detailed) {
        Write-Host ""
        Write-Info "Checking cuDNN include and lib directories..."
        
        $includePath = Join-Path $cuda121 "include\cudnn*.h"
        $includeFiles = Get-ChildItem $includePath -ErrorAction SilentlyContinue
        if ($includeFiles) {
            Write-Success "cuDNN header files found ($($includeFiles.Count) files)"
        } else {
            Write-Warning "cuDNN header files not found"
        }
        
        $libPath = Join-Path $cuda121 "lib\x64\cudnn*.lib"
        $libFiles = Get-ChildItem $libPath -ErrorAction SilentlyContinue
        if ($libFiles) {
            Write-Success "cuDNN library files found ($($libFiles.Count) files)"
        } else {
            Write-Warning "cuDNN library files not found"
        }
    }
}

# Check 3: Environment Variables
Write-Host ""
Write-Info "Checking CUDA environment variables..."

$cudaPathEnv = [Environment]::GetEnvironmentVariable("CUDA_PATH", "Machine")
if ($cudaPathEnv) {
    Write-Success "CUDA_PATH: $cudaPathEnv"
    
    if ($Detailed) {
        $cudaPathV12 = [Environment]::GetEnvironmentVariable("CUDA_PATH_V12_1", "Machine")
        if ($cudaPathV12) {
            Write-Success "CUDA_PATH_V12_1: $cudaPathV12"
        }
    }
} else {
    Write-Warning "CUDA_PATH environment variable not set"
    $issues += "CUDA_PATH environment variable should be set"
}

# Check PATH
$pathEnv = [Environment]::GetEnvironmentVariable("PATH", "Machine")
if ($pathEnv -like "*CUDA*v12.1*bin*") {
    Write-Success "CUDA 12.1 bin directory is in PATH"
} else {
    Write-Warning "CUDA 12.1 bin directory may not be in PATH"
}

# Summary
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "VALIDATION SUMMARY" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

if ($cudaOk -and $cudnnOk) {
    Write-Success "CUDA 12.1 and cuDNN v8.x are properly installed!"
    Write-Host ""
    Write-Host "Your system is ready for PyTorch 2.5.1+cu121" -ForegroundColor Green
    exit 0
} else {
    if ($issues.Count -gt 0) {
        Write-Host ""
        Write-Host "❌ ISSUES FOUND:" -ForegroundColor Red
        foreach ($issue in $issues) {
            Write-Host "  • $issue" -ForegroundColor Red
        }
    }
    
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Yellow
    Write-Host "RECOMMENDED ACTIONS" -ForegroundColor Yellow
    Write-Host "============================================================" -ForegroundColor Yellow
    
    if (-not $cudaOk) {
        Write-Host ""
        Write-Host "CUDA Toolkit 12.1 Installation:" -ForegroundColor Cyan
        Write-Host "  1. Download from: https://developer.nvidia.com/cuda-12-1-0-download-archive" -ForegroundColor Gray
        Write-Host "  2. Select: Windows > x86_64 > 10 > exe (local)" -ForegroundColor Gray
        Write-Host "  3. Run installer with default options" -ForegroundColor Gray
        Write-Host "  4. Restart computer after installation" -ForegroundColor Gray
    }
    
    if (-not $cudnnOk) {
        Write-Host ""
        Write-Host "cuDNN v8.9.7 Installation (CRITICAL):" -ForegroundColor Cyan
        Write-Host "  ⚠️  WARNING: Do NOT use pip to install cuDNN!" -ForegroundColor Yellow
        Write-Host "  ⚠️  pip may install v9.x which causes crashes!" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "  Manual Installation Steps:" -ForegroundColor Cyan
        Write-Host "  1. Download cuDNN v8.9.7 for CUDA 12.x:" -ForegroundColor Gray
        Write-Host "     https://developer.nvidia.com/rdp/cudnn-archive" -ForegroundColor Gray
        Write-Host "  2. Extract the downloaded zip file" -ForegroundColor Gray
        Write-Host "  3. Open PowerShell as Administrator" -ForegroundColor Gray
        Write-Host "  4. Copy files to CUDA directory:" -ForegroundColor Gray
        Write-Host "     Copy-Item -Recurse -Force 'cudnn-*\bin\*' 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin\'" -ForegroundColor Gray
        Write-Host "     Copy-Item -Recurse -Force 'cudnn-*\include\*' 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\include\'" -ForegroundColor Gray
        Write-Host "     Copy-Item -Recurse -Force 'cudnn-*\lib\*' 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\lib\'" -ForegroundColor Gray
    }
    
    Write-Host ""
    Write-Host "After fixing issues, run this script again to verify." -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
    exit 1
}
