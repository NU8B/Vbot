# Vbot Main Setup Script for Windows
# Orchestrates the entire installation process

param(
    [switch]$SkipValidation,
    [switch]$SkipDocker,
    [switch]$Quick
)

$ErrorActionPreference = "Continue"

# Color functions
function Write-Success { param($Message) Write-Host "✅ $Message" -ForegroundColor Green }
function Write-Error { param($Message) Write-Host "❌ $Message" -ForegroundColor Red }
function Write-Warning { param($Message) Write-Host "⚠️  $Message" -ForegroundColor Yellow }
function Write-Info { param($Message) Write-Host "ℹ️  $Message" -ForegroundColor Cyan }
function Write-Step { param($Message) Write-Host "`n━━━ $Message ━━━" -ForegroundColor Yellow }

# Banner
Clear-Host
Write-Host ""
Write-Host "╔════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║                                                        ║" -ForegroundColor Cyan
Write-Host "║              Vbot Installation Wizard                  ║" -ForegroundColor Cyan
Write-Host "║                                                        ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Get script directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir

Write-Info "Project root: $projectRoot"
Write-Info "Setup scripts: $scriptDir"
Write-Host ""

# Check Administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Warning "Not running as Administrator"
    Write-Host "Some installation steps may require administrator privileges." -ForegroundColor Yellow
    Write-Host ""
    $continue = Read-Host "Continue anyway? (y/N)"
    if ($continue -ne 'y' -and $continue -ne 'Y') {
        Write-Info "Please restart PowerShell as Administrator and run again."
        exit 1
    }
}

# Phase 1: System Validation
if (-not $SkipValidation) {
    Write-Step "Phase 1: System Validation"
    Write-Info "Checking if your system meets requirements..."
    Write-Host ""
    
    $validateScript = Join-Path $scriptDir "validate_system.ps1"
    if (Test-Path $validateScript) {
        & $validateScript
        $validationResult = $LASTEXITCODE
        
        if ($validationResult -ne 0) {
            Write-Host ""
            Write-Error "System validation failed!"
            Write-Host ""
            Write-Host "Please fix the errors above before proceeding." -ForegroundColor Yellow
            Write-Host "You can run validation separately:" -ForegroundColor Cyan
            Write-Host "  .\setup\validate_system.ps1 -Detailed" -ForegroundColor Gray
            Write-Host ""
            $force = Read-Host "Continue anyway? (NOT RECOMMENDED) (y/N)"
            if ($force -ne 'y' -and $force -ne 'Y') {
                exit 1
            }
        } else {
            Write-Success "System validation passed!"
        }
    } else {
        Write-Warning "Validation script not found, skipping..."
    }
} else {
    Write-Info "Skipping system validation (--SkipValidation)"
}

# Phase 2: CUDA/cuDNN Check
Write-Step "Phase 2: CUDA & cuDNN Validation"
Write-Info "Checking CUDA Toolkit and cuDNN installation..."
Write-Host ""

$cudaScript = Join-Path $scriptDir "check_cuda.ps1"
if (Test-Path $cudaScript) {
    & $cudaScript
    $cudaResult = $LASTEXITCODE
    
    if ($cudaResult -ne 0) {
        Write-Host ""
        Write-Error "CUDA/cuDNN validation failed!"
        Write-Host ""
        Write-Warning "CRITICAL: Vbot requires CUDA 12.1 and cuDNN v8.9.7"
        Write-Host ""
        Write-Host "Please follow the instructions above to install:" -ForegroundColor Yellow
        Write-Host "  1. CUDA Toolkit 12.1" -ForegroundColor Gray
        Write-Host "  2. cuDNN v8.9.7 (manually, NOT via pip!)" -ForegroundColor Gray
        Write-Host ""
        Write-Host "After installing, run this script again." -ForegroundColor Cyan
        Write-Host ""
        $continue = Read-Host "Continue anyway? (NOT RECOMMENDED) (y/N)"
        if ($continue -ne 'y' -and $continue -ne 'Y') {
            exit 1
        }
    } else {
        Write-Success "CUDA and cuDNN are properly configured!"
    }
} else {
    Write-Warning "CUDA check script not found, skipping..."
}

# Phase 3: Docker Installation
if (-not $SkipDocker) {
    Write-Step "Phase 3: Docker Desktop"
    Write-Info "Checking Docker Desktop installation..."
    Write-Host ""
    
    $docker = Get-Command docker -ErrorAction SilentlyContinue
    if ($docker) {
        $version = docker --version 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Docker is installed: $version"
            
            # Check if running
            $dockerRunning = docker ps 2>$null
            if ($LASTEXITCODE -eq 0) {
                Write-Success "Docker is running!"
            } else {
                Write-Warning "Docker is installed but not running"
                Write-Host ""
                Write-Host "Please start Docker Desktop:" -ForegroundColor Yellow
                Write-Host "  1. Open Docker Desktop from Start Menu" -ForegroundColor Gray
                Write-Host "  2. Wait for it to fully start" -ForegroundColor Gray
                Write-Host "  3. Run this script again" -ForegroundColor Gray
                Write-Host ""
                $wait = Read-Host "Press Enter when Docker is running..."
            }
        }
    } else {
        Write-Warning "Docker Desktop is not installed"
        Write-Host ""
        $install = Read-Host "Install Docker Desktop now? (Y/n)"
        if ($install -ne 'n' -and $install -ne 'N') {
            $dockerScript = Join-Path $scriptDir "install_docker.ps1"
            if (Test-Path $dockerScript) {
                & $dockerScript
                if ($LASTEXITCODE -ne 0) {
                    Write-Error "Docker installation failed!"
                    exit 1
                }
            } else {
                Write-Error "Docker installation script not found!"
                Write-Host "Please install Docker Desktop manually:" -ForegroundColor Yellow
                Write-Host "  https://www.docker.com/products/docker-desktop/" -ForegroundColor Gray
                exit 1
            }
        } else {
            Write-Warning "Skipping Docker installation"
            Write-Host "Note: Vbot requires Docker for the Ollama LLM!" -ForegroundColor Yellow
        }
    }
} else {
    Write-Info "Skipping Docker check (--SkipDocker)"
}

# Phase 4: Python Dependencies
Write-Step "Phase 4: Python Dependencies"
Write-Info "Installing Python packages..."
Write-Host ""

$requirementsFile = Join-Path $projectRoot "requirements.txt"
if (Test-Path $requirementsFile) {
    Write-Info "Found requirements.txt"
    Write-Warning "This will take 5-15 minutes (downloading ~8GB of packages)..."
    Write-Host ""
    
    if (-not $Quick) {
        $install = Read-Host "Install Python dependencies now? (Y/n)"
        if ($install -eq 'n' -or $install -eq 'N') {
            Write-Info "Skipping Python dependencies installation"
            Write-Warning "You will need to run 'pip install -r requirements.txt' manually"
        } else {
            Write-Info "Installing packages..."
            try {
                & python -m pip install --upgrade pip
                & python -m pip install -r $requirementsFile
                Write-Success "Python dependencies installed!"
            } catch {
                Write-Error "Failed to install dependencies: $_"
                Write-Host ""
                Write-Host "Please try manually:" -ForegroundColor Yellow
                Write-Host "  python -m pip install -r requirements.txt" -ForegroundColor Gray
            }
        }
    } else {
        Write-Info "Quick mode: Skipping Python dependencies"
    }
} else {
    Write-Warning "requirements.txt not found at: $requirementsFile"
}

# Phase 5: Model Downloads Info
Write-Step "Phase 5: Model Downloads"
Write-Info "Vbot will download AI models on first run..."
Write-Host ""
Write-Host "First run will download:" -ForegroundColor Cyan
Write-Host "  • StyleTTS2 models (~2GB)" -ForegroundColor Gray
Write-Host "  • Ollama LLM model (~5GB)" -ForegroundColor Gray
Write-Host "  • Whisper model (~400MB)" -ForegroundColor Gray
Write-Host "  • Emotion classifier (~500MB)" -ForegroundColor Gray
Write-Host ""
Write-Host "Total: ~7-8 GB download on first run" -ForegroundColor Yellow
Write-Host "Time: 10-30 minutes depending on internet speed" -ForegroundColor Yellow
Write-Host ""

# Final Summary
Write-Host ""
Write-Host "╔════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║                                                        ║" -ForegroundColor Green
Write-Host "║              Installation Complete!                    ║" -ForegroundColor Green
Write-Host "║                                                        ║" -ForegroundColor Green
Write-Host "╚════════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""

Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Run Vbot:" -ForegroundColor Yellow
Write-Host "   python vbot_launcher/launcher.py" -ForegroundColor Gray
Write-Host ""
Write-Host "2. OR test the launcher:" -ForegroundColor Yellow
Write-Host "   python test_launcher.py" -ForegroundColor Gray
Write-Host ""
Write-Host "3. OR test system check:" -ForegroundColor Yellow
Write-Host "   python vbot_launcher/system_check.py" -ForegroundColor Gray
Write-Host ""

Write-Info "First run will download models (may take 10-30 minutes)"
Write-Info "Subsequent runs will be much faster (<1 minute)"
Write-Host ""
Write-Host "For help or issues, see:" -ForegroundColor Cyan
Write-Host "  distribution_planning/challenges.md" -ForegroundColor Gray
Write-Host ""
Write-Host "════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

exit 0
