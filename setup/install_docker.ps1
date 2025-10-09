# Vbot Docker Desktop Installation Helper
# Automates Docker Desktop installation and configuration for Windows

param(
    [switch]$SkipWSL,
    [switch]$SkipDownload,
    [string]$InstallerPath
)

$ErrorActionPreference = "Stop"

# Requires Administrator
if (-not ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "❌ This script requires Administrator privileges!" -ForegroundColor Red
    Write-Host "Please run PowerShell as Administrator and try again." -ForegroundColor Yellow
    exit 1
}

# Color functions
function Write-Success { param($Message) Write-Host "✅ $Message" -ForegroundColor Green }
function Write-Error { param($Message) Write-Host "❌ $Message" -ForegroundColor Red }
function Write-Warning { param($Message) Write-Host "⚠️  $Message" -ForegroundColor Yellow }
function Write-Info { param($Message) Write-Host "ℹ️  $Message" -ForegroundColor Cyan }
function Write-Step { param($Message) Write-Host "`n▶️  $Message" -ForegroundColor Yellow }

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "        Docker Desktop Installation Helper" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check if Docker is already installed
Write-Step "Step 1: Checking for existing Docker installation"
$docker = Get-Command docker -ErrorAction SilentlyContinue

if ($docker) {
    $version = docker --version 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Docker is already installed: $version"
        
        # Check if it's running
        Write-Info "Checking Docker daemon status..."
        $running = docker ps 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Docker is running!"
            Write-Host ""
            Write-Host "Docker Desktop is already installed and running." -ForegroundColor Green
            Write-Host "No action needed." -ForegroundColor Green
            exit 0
        } else {
            Write-Warning "Docker installed but not running"
            Write-Host ""
            Write-Host "Please start Docker Desktop manually:" -ForegroundColor Yellow
            Write-Host "  1. Press Windows key" -ForegroundColor Gray
            Write-Host "  2. Search for 'Docker Desktop'" -ForegroundColor Gray
            Write-Host "  3. Click to start" -ForegroundColor Gray
            Write-Host "  4. Wait for it to fully start (icon in system tray)" -ForegroundColor Gray
            exit 0
        }
    }
}

Write-Info "Docker Desktop not found - proceeding with installation"

# Step 2: Check Windows version
Write-Step "Step 2: Validating Windows version"
$os = Get-CimInstance Win32_OperatingSystem
$build = [int]$os.BuildNumber

if ($build -ge 19041) {
    Write-Success "Windows version check passed (Build $build)"
} else {
    Write-Error "Windows Build $build is too old"
    Write-Host "Docker Desktop requires Windows 10 Build 19041+ or Windows 11" -ForegroundColor Red
    exit 1
}

# Step 3: Enable WSL 2 (if not skipped)
if (-not $SkipWSL) {
    Write-Step "Step 3: Enabling WSL 2"
    Write-Info "Checking if WSL is installed..."
    
    $wslFeature = Get-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux -ErrorAction SilentlyContinue
    
    if ($wslFeature.State -eq "Enabled") {
        Write-Success "WSL feature is already enabled"
    } else {
        Write-Info "Enabling WSL feature (requires restart)..."
        try {
            Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux -NoRestart -ErrorAction Stop
            Write-Success "WSL feature enabled"
            $needRestart = $true
        } catch {
            Write-Error "Failed to enable WSL: $_"
            Write-Host "Please enable WSL manually:" -ForegroundColor Yellow
            Write-Host "  wsl --install" -ForegroundColor Gray
            exit 1
        }
    }
    
    # Check Virtual Machine Platform
    $vmFeature = Get-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform -ErrorAction SilentlyContinue
    
    if ($vmFeature.State -eq "Enabled") {
        Write-Success "Virtual Machine Platform is already enabled"
    } else {
        Write-Info "Enabling Virtual Machine Platform..."
        try {
            Enable-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform -NoRestart -ErrorAction Stop
            Write-Success "Virtual Machine Platform enabled"
            $needRestart = $true
        } catch {
            Write-Warning "Could not enable Virtual Machine Platform: $_"
        }
    }
    
    # Set WSL 2 as default
    Write-Info "Setting WSL 2 as default version..."
    try {
        & wsl --set-default-version 2 2>&1 | Out-Null
        Write-Success "WSL 2 set as default"
    } catch {
        Write-Warning "Could not set WSL 2 as default (may need to run 'wsl --install' manually)"
    }
} else {
    Write-Info "Skipping WSL setup (--SkipWSL specified)"
}

# Step 4: Download Docker Desktop
Write-Step "Step 4: Downloading Docker Desktop"

if ($InstallerPath -and (Test-Path $InstallerPath)) {
    Write-Success "Using provided installer: $InstallerPath"
    $installerPath = $InstallerPath
} elseif ($SkipDownload) {
    Write-Error "Installer path not provided and --SkipDownload specified"
    exit 1
} else {
    $downloadUrl = "https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe"
    $installerPath = "$env:TEMP\DockerDesktopInstaller.exe"
    
    Write-Info "Downloading from: $downloadUrl"
    Write-Info "Saving to: $installerPath"
    Write-Warning "This may take several minutes (400+ MB download)..."
    
    try {
        # Use BITS transfer for better reliability
        Import-Module BitsTransfer -ErrorAction SilentlyContinue
        if (Get-Module BitsTransfer) {
            Start-BitsTransfer -Source $downloadUrl -Destination $installerPath -DisplayName "Docker Desktop" -Description "Downloading Docker Desktop Installer"
        } else {
            # Fallback to Invoke-WebRequest
            $ProgressPreference = 'SilentlyContinue'
            Invoke-WebRequest -Uri $downloadUrl -OutFile $installerPath -UseBasicParsing
            $ProgressPreference = 'Continue'
        }
        Write-Success "Download completed!"
    } catch {
        Write-Error "Download failed: $_"
        Write-Host ""
        Write-Host "Please download Docker Desktop manually:" -ForegroundColor Yellow
        Write-Host "  1. Visit: https://www.docker.com/products/docker-desktop/" -ForegroundColor Gray
        Write-Host "  2. Download Docker Desktop for Windows" -ForegroundColor Gray
        Write-Host "  3. Run this script again with: -InstallerPath 'path\to\installer.exe'" -ForegroundColor Gray
        exit 1
    }
}

# Step 5: Install Docker Desktop
Write-Step "Step 5: Installing Docker Desktop"
Write-Info "Running installer (this will take a few minutes)..."
Write-Warning "DO NOT close this window!"

try {
    # Run installer silently
    $process = Start-Process -FilePath $installerPath -ArgumentList "install --quiet --accept-license" -Wait -PassThru -NoNewWindow
    
    if ($process.ExitCode -eq 0) {
        Write-Success "Docker Desktop installed successfully!"
    } elseif ($process.ExitCode -eq 3010) {
        Write-Success "Docker Desktop installed (restart required)"
        $needRestart = $true
    } else {
        Write-Error "Installation failed with exit code: $($process.ExitCode)"
        exit 1
    }
} catch {
    Write-Error "Installation error: $_"
    exit 1
}

# Step 6: Post-installation
Write-Step "Step 6: Post-installation configuration"

if ($needRestart) {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Yellow
    Write-Host "⚠️  RESTART REQUIRED" -ForegroundColor Yellow
    Write-Host "============================================================" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Windows features were enabled that require a restart." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Please:" -ForegroundColor Cyan
    Write-Host "  1. Save all your work" -ForegroundColor Gray
    Write-Host "  2. Restart your computer" -ForegroundColor Gray
    Write-Host "  3. After restart, start Docker Desktop from Start Menu" -ForegroundColor Gray
    Write-Host "  4. Wait for Docker to fully start" -ForegroundColor Gray
    Write-Host "  5. Run Vbot installation again" -ForegroundColor Gray
    Write-Host ""
    
    $restart = Read-Host "Restart now? (y/N)"
    if ($restart -eq 'y' -or $restart -eq 'Y') {
        Write-Info "Restarting computer in 10 seconds..."
        Start-Sleep -Seconds 10
        Restart-Computer -Force
    }
} else {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host "✅ Installation Complete!" -ForegroundColor Green
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "  1. Start Docker Desktop from Start Menu" -ForegroundColor Gray
    Write-Host "  2. Accept the Docker Subscription Service Agreement" -ForegroundColor Gray
    Write-Host "  3. Wait for Docker to fully start (icon in system tray)" -ForegroundColor Gray
    Write-Host "  4. You can now proceed with Vbot installation" -ForegroundColor Gray
    Write-Host ""
    Write-Host "To verify Docker is running:" -ForegroundColor Cyan
    Write-Host "  docker ps" -ForegroundColor Gray
    Write-Host ""
}

# Cleanup
if (-not $InstallerPath -and (Test-Path $installerPath)) {
    Write-Info "Cleaning up installer..."
    try {
        Remove-Item $installerPath -Force -ErrorAction SilentlyContinue
    } catch {
        # Ignore cleanup errors
    }
}

Write-Host "============================================================" -ForegroundColor Cyan
exit 0
