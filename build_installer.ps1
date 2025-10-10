# Vbot Installer Build Script
# Builds PyInstaller executable and creates Inno Setup installer

param(
    [switch]$SkipBuild,      # Skip PyInstaller build
    [switch]$SkipInstaller,  # Skip Inno Setup
    [switch]$Clean,          # Clean build directories first
    [switch]$OneFile         # Build single-file executable (slower startup)
)

$ErrorActionPreference = "Stop"

# Colors
function Write-Success { param($Message) Write-Host "✅ $Message" -ForegroundColor Green }
function Write-Error { param($Message) Write-Host "❌ $Message" -ForegroundColor Red }
function Write-Warning { param($Message) Write-Host "⚠️  $Message" -ForegroundColor Yellow }
function Write-Info { param($Message) Write-Host "ℹ️  $Message" -ForegroundColor Cyan }
function Write-Step { param($Message) Write-Host "`n▶️  $Message" -ForegroundColor Yellow }

Write-Host ""
Write-Host "╔════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║                                                        ║" -ForegroundColor Cyan
Write-Host "║              Vbot Installer Builder                    ║" -ForegroundColor Cyan
Write-Host "║                                                        ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Get project root
$projectRoot = $PSScriptRoot
Write-Info "Project root: $projectRoot"

# Step 0: Clean if requested
if ($Clean) {
    Write-Step "Step 0: Cleaning build directories"
    
    $dirsToClean = @('build', 'dist')
    foreach ($dir in $dirsToClean) {
        $fullPath = Join-Path $projectRoot $dir
        if (Test-Path $fullPath) {
            Write-Info "Removing $dir..."
            Remove-Item $fullPath -Recurse -Force
            Write-Success "Cleaned $dir"
        }
    }
}

# Step 1: Check dependencies
Write-Step "Step 1: Checking dependencies"

# Check Python
try {
    $pythonVersion = python --version 2>&1
    Write-Success "Python: $pythonVersion"
} catch {
    Write-Error "Python not found! Please install Python 3.10"
    exit 1
}

# Check PyInstaller
Write-Info "Checking PyInstaller..."
$pyinstallerCheck = python -c "import PyInstaller; print(PyInstaller.__version__)" 2>&1

if ($LASTEXITCODE -ne 0) {
    Write-Warning "PyInstaller not installed"
    $install = Read-Host "Install PyInstaller now? (Y/n)"
    if ($install -ne 'n' -and $install -ne 'N') {
        Write-Info "Installing PyInstaller..."
        pip install pyinstaller
        Write-Success "PyInstaller installed"
    } else {
        Write-Error "PyInstaller required to build executable"
        exit 1
    }
} else {
    Write-Success "PyInstaller: v$pyinstallerCheck"
}

# Check if vbot.spec exists
$specFile = Join-Path $projectRoot "vbot.spec"
if (-not (Test-Path $specFile)) {
    Write-Error "vbot.spec not found! Cannot build without spec file."
    exit 1
}
Write-Success "Build spec found: vbot.spec"

# Step 2: Build PyInstaller executable
if (-not $SkipBuild) {
    Write-Step "Step 2: Building PyInstaller executable"
    Write-Warning "This will take 15-30 minutes on first build..."
    Write-Info "Building from: vbot_launcher/launcher.py"
    Write-Host ""
    
    # Build
    Write-Info "Running: python -m PyInstaller vbot.spec"
    $startTime = Get-Date
    
    try {
        python -m PyInstaller vbot.spec --clean
        
        if ($LASTEXITCODE -eq 0) {
            $endTime = Get-Date
            $duration = ($endTime - $startTime).TotalMinutes
            Write-Success "Build completed in $([math]::Round($duration, 1)) minutes"
            
            # Check output
            $exePath = Join-Path $projectRoot "dist\Vbot\Vbot.exe"
            if (Test-Path $exePath) {
                $exeSize = (Get-Item $exePath).Length / 1MB
                Write-Success "Executable created: $exePath"
                Write-Info "Executable size: $([math]::Round($exeSize, 1)) MB"
                
                # Check dist folder size
                $distPath = Join-Path $projectRoot "dist\Vbot"
                $distSize = (Get-ChildItem $distPath -Recurse | Measure-Object -Property Length -Sum).Sum / 1GB
                Write-Info "Total distribution size: $([math]::Round($distSize, 2)) GB"
            } else {
                Write-Error "Build succeeded but executable not found at: $exePath"
                exit 1
            }
        } else {
            Write-Error "PyInstaller build failed!"
            Write-Host "Check the build output above for errors." -ForegroundColor Yellow
            exit 1
        }
    } catch {
        Write-Error "Build error: $_"
        exit 1
    }
} else {
    Write-Info "Skipping PyInstaller build (--SkipBuild)"
}

# Step 3: Test the executable
Write-Step "Step 3: Quick executable test"

$exePath = Join-Path $projectRoot "dist\Vbot\Vbot.exe"
if (Test-Path $exePath) {
    Write-Info "Testing executable startup..."
    
    # Just check if it starts (will fail if dependencies missing)
    Write-Warning "This is a quick test - full testing requires running the exe manually"
    Write-Info "To test manually:"
    Write-Host "  cd dist\Vbot" -ForegroundColor Gray
    Write-Host "  .\Vbot.exe" -ForegroundColor Gray
    Write-Host ""
    
    $test = Read-Host "Do you want to test the executable now? (y/N)"
    if ($test -eq 'y' -or $test -eq 'Y') {
        Write-Info "Starting Vbot.exe (close it after verifying it loads)..."
        Start-Process $exePath -WorkingDirectory (Join-Path $projectRoot "dist\Vbot")
        Write-Success "Executable started! Check if it loads correctly, then close it."
        Read-Host "Press Enter after closing Vbot to continue..."
    }
} else {
    Write-Warning "Executable not found - skipping test"
}

# Step 4: Create Inno Setup installer
if (-not $SkipInstaller) {
    Write-Step "Step 4: Creating Inno Setup installer"
    
    # Check if Inno Setup is installed
    $innoPath = "C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
    if (-not (Test-Path $innoPath)) {
        Write-Warning "Inno Setup not found at: $innoPath"
        Write-Host ""
        Write-Host "To create an installer, you need Inno Setup:" -ForegroundColor Yellow
        Write-Host "  1. Download from: https://jrsoftware.org/isdl.php" -ForegroundColor Gray
        Write-Host "  2. Install Inno Setup 6" -ForegroundColor Gray
        Write-Host "  3. Run this script again" -ForegroundColor Gray
        Write-Host ""
        
        $download = Read-Host "Open Inno Setup download page? (Y/n)"
        if ($download -ne 'n' -and $download -ne 'N') {
            Start-Process "https://jrsoftware.org/isdl.php"
        }
        
        Write-Info "Skipping installer creation for now"
        Write-Info "Run again with Inno Setup installed to create installer"
    } else {
        Write-Success "Inno Setup found"
        
        # Check for installer script
        $issFile = Join-Path $projectRoot "installer\vbot_installer.iss"
        if (Test-Path $issFile) {
            Write-Info "Compiling installer..."
            
            try {
                & $innoPath $issFile
                
                if ($LASTEXITCODE -eq 0) {
                    Write-Success "Installer created successfully!"
                    
                    # Find the installer
                    $installerPath = Join-Path $projectRoot "installer\Output\VbotSetup.exe"
                    if (Test-Path $installerPath) {
                        $installerSize = (Get-Item $installerPath).Length / 1GB
                        Write-Success "Installer: $installerPath"
                        Write-Info "Installer size: $([math]::Round($installerSize, 2)) GB"
                    }
                } else {
                    Write-Error "Inno Setup compilation failed"
                }
            } catch {
                Write-Error "Installer creation error: $_"
            }
        } else {
            Write-Warning "Installer script not found: $issFile"
            Write-Info "Create installer script first, then run again"
        }
    }
} else {
    Write-Info "Skipping Inno Setup installer (--SkipInstaller)"
}

# Final summary
Write-Host ""
Write-Host "╔════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║                                                        ║" -ForegroundColor Green
Write-Host "║              Build Complete!                           ║" -ForegroundColor Green
Write-Host "║                                                        ║" -ForegroundColor Green
Write-Host "╚════════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""

Write-Host "📦 Outputs:" -ForegroundColor Cyan
Write-Host ""

$exePath = Join-Path $projectRoot "dist\Vbot\Vbot.exe"
if (Test-Path $exePath) {
    Write-Host "  Executable:" -ForegroundColor Yellow
    Write-Host "    $exePath" -ForegroundColor Gray
    Write-Host ""
}

$installerPath = Join-Path $projectRoot "installer\Output\VbotSetup.exe"
if (Test-Path $installerPath) {
    Write-Host "  Installer:" -ForegroundColor Yellow
    Write-Host "    $installerPath" -ForegroundColor Gray
    Write-Host ""
}

Write-Host "🧪 Next Steps:" -ForegroundColor Cyan
Write-Host ""
Write-Host "  1. Test executable:" -ForegroundColor Yellow
Write-Host "     cd dist\Vbot" -ForegroundColor Gray
Write-Host "     .\Vbot.exe" -ForegroundColor Gray
Write-Host ""
Write-Host "  2. Test installer:" -ForegroundColor Yellow
Write-Host "     .\installer\Output\VbotSetup.exe" -ForegroundColor Gray
Write-Host ""
Write-Host "  3. Test on clean VM or another PC" -ForegroundColor Yellow
Write-Host ""

Write-Host "════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

exit 0
