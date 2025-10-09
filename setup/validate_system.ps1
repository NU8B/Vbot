# Vbot System Validation Script
# Checks if system meets minimum requirements before installation

param(
    [switch]$Detailed,
    [switch]$Silent
)

$ErrorActionPreference = "Continue"

# Color functions
function Write-Success { param($Message) Write-Host "✅ $Message" -ForegroundColor Green }
function Write-Error { param($Message) Write-Host "❌ $Message" -ForegroundColor Red }
function Write-Warning { param($Message) Write-Host "⚠️  $Message" -ForegroundColor Yellow }
function Write-Info { param($Message) Write-Host "ℹ️  $Message" -ForegroundColor Cyan }

# Banner
if (-not $Silent) {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "              Vbot System Validation" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host ""
}

$errors = @()
$warnings = @()
$passed = 0
$total = 0

# Check 1: Windows Version
$total++
Write-Info "Checking Windows version..."
try {
    $os = Get-CimInstance Win32_OperatingSystem
    $build = [int]$os.BuildNumber
    
    if ($build -ge 19041) {
        Write-Success "Windows $($os.Caption) (Build $build)"
        $passed++
    } else {
        Write-Error "Windows Build $build is too old (need 19041+)"
        $errors += "Windows 10 Build 19041+ or Windows 11 required"
    }
    
    if ($Detailed) {
        Write-Host "  Version: $($os.Version)" -ForegroundColor Gray
        Write-Host "  Architecture: $($os.OSArchitecture)" -ForegroundColor Gray
    }
} catch {
    Write-Error "Could not check Windows version: $_"
    $errors += "Windows version check failed"
}

# Check 2: RAM
$total++
Write-Info "Checking system RAM..."
try {
    $ram = (Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB
    
    if ($ram -ge 16) {
        Write-Success "RAM: $([math]::Round($ram, 1)) GB available"
        $passed++
    } else {
        Write-Error "RAM: $([math]::Round($ram, 1)) GB (need 16 GB minimum)"
        $errors += "Insufficient RAM: $([math]::Round($ram, 1)) GB (need 16 GB)"
    }
} catch {
    Write-Warning "Could not check RAM: $_"
    $warnings += "RAM check failed"
}

# Check 3: Disk Space
$total++
Write-Info "Checking disk space..."
try {
    $disk = Get-PSDrive C | Select-Object Free
    $freeGB = $disk.Free / 1GB
    
    if ($freeGB -ge 50) {
        Write-Success "Disk Space: $([math]::Round($freeGB, 1)) GB free"
        $passed++
    } else {
        Write-Warning "Disk Space: $([math]::Round($freeGB, 1)) GB free (recommend 50 GB)"
        $warnings += "Low disk space: $([math]::Round($freeGB, 1)) GB (recommend 50 GB)"
    }
} catch {
    Write-Warning "Could not check disk space: $_"
}

# Check 4: NVIDIA GPU
$total++
Write-Info "Checking for NVIDIA GPU..."
try {
    $gpu = Get-CimInstance Win32_VideoController | Where-Object { $_.Name -like "*NVIDIA*" }
    
    if ($gpu) {
        Write-Success "GPU: $($gpu.Name)"
        $passed++
        
        if ($Detailed) {
            Write-Host "  Driver Version: $($gpu.DriverVersion)" -ForegroundColor Gray
            Write-Host "  Status: $($gpu.Status)" -ForegroundColor Gray
        }
    } else {
        Write-Error "No NVIDIA GPU found"
        $errors += "NVIDIA GPU with CUDA support required"
    }
} catch {
    Write-Error "Could not check GPU: $_"
    $errors += "GPU check failed"
}

# Check 5: NVIDIA Driver
$total++
Write-Info "Checking NVIDIA driver..."
try {
    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    
    if ($nvidiaSmi) {
        $driverInfo = & nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>$null
        if ($driverInfo) {
            Write-Success "NVIDIA Driver: $driverInfo"
            $passed++
        } else {
            Write-Warning "NVIDIA driver found but nvidia-smi failed"
            $warnings += "Could not verify NVIDIA driver version"
        }
    } else {
        Write-Warning "nvidia-smi not found in PATH"
        $warnings += "NVIDIA driver may not be installed or not in PATH"
    }
} catch {
    Write-Warning "Could not check NVIDIA driver: $_"
}

# Check 6: Docker Desktop
$total++
Write-Info "Checking Docker Desktop..."
try {
    $docker = Get-Command docker -ErrorAction SilentlyContinue
    
    if ($docker) {
        $version = docker --version 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Docker: $version"
            $passed++
            
            # Check if Docker is running
            $running = docker ps 2>$null
            if ($LASTEXITCODE -eq 0) {
                Write-Success "Docker daemon is running"
            } else {
                Write-Warning "Docker installed but not running"
                $warnings += "Docker Desktop is not running - please start it"
            }
        } else {
            Write-Warning "Docker command found but failed"
            $warnings += "Docker may not be properly installed"
        }
    } else {
        Write-Warning "Docker Desktop not installed"
        $warnings += "Docker Desktop required for Ollama LLM"
    }
} catch {
    Write-Warning "Could not check Docker: $_"
}

# Check 7: Python
$total++
Write-Info "Checking Python installation..."
try {
    $python = Get-Command python -ErrorAction SilentlyContinue
    
    if ($python) {
        $version = & python --version 2>&1
        if ($version -match "Python (\d+)\.(\d+)\.(\d+)") {
            $major = [int]$Matches[1]
            $minor = [int]$Matches[2]
            
            if ($major -eq 3 -and $minor -eq 10) {
                Write-Success "Python: $version"
                $passed++
            } else {
                Write-Warning "Python: $version (recommend 3.10.x)"
                $warnings += "Python $version detected, 3.10.x recommended"
            }
        }
    } else {
        Write-Warning "Python not installed or not in PATH"
        $warnings += "Python 3.10.x required"
    }
} catch {
    Write-Warning "Could not check Python: $_"
}

# Check 8: Administrator Privileges
$total++
Write-Info "Checking administrator privileges..."
try {
    $isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    
    if ($isAdmin) {
        Write-Success "Running with administrator privileges"
        $passed++
    } else {
        Write-Warning "Not running as administrator"
        $warnings += "Some installation steps require administrator privileges"
    }
} catch {
    Write-Warning "Could not check admin privileges: $_"
}

# Summary
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "VALIDATION SUMMARY" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Passed: $passed / $total checks" -ForegroundColor $(if ($passed -eq $total) { "Green" } else { "Yellow" })

if ($errors.Count -gt 0) {
    Write-Host ""
    Write-Host "❌ ERRORS (Must fix before proceeding):" -ForegroundColor Red
    $errors | ForEach-Object { Write-Host "  • $_" -ForegroundColor Red }
}

if ($warnings.Count -gt 0) {
    Write-Host ""
    Write-Host "⚠️  WARNINGS (Recommended to fix):" -ForegroundColor Yellow
    $warnings | ForEach-Object { Write-Host "  • $_" -ForegroundColor Yellow }
}

if ($errors.Count -eq 0 -and $warnings.Count -eq 0) {
    Write-Host ""
    Write-Success "All checks passed! System is ready for Vbot installation."
    Write-Host "============================================================" -ForegroundColor Cyan
    exit 0
} elseif ($errors.Count -eq 0) {
    Write-Host ""
    Write-Warning "System meets minimum requirements but has warnings."
    Write-Host "============================================================" -ForegroundColor Cyan
    exit 0
} else {
    Write-Host ""
    Write-Error "System does not meet minimum requirements."
    Write-Host "============================================================" -ForegroundColor Cyan
    exit 1
}
