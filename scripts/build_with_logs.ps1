[CmdletBinding()]
param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$Python = "python",
    [string]$Version = "",
    [ValidateSet("Launcher", "Full")]
    [string]$BuildMode = "Launcher",
    [switch]$NoClean,
    [switch]$SkipTests,
    [switch]$NoPackage
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-FullPath {
    param([Parameter(Mandatory = $true)][string]$Path)
    return [System.IO.Path]::GetFullPath($Path)
}

function Test-IsSubPath {
    param(
        [Parameter(Mandatory = $true)][string]$Child,
        [Parameter(Mandatory = $true)][string]$Parent
    )
    $childFull = Get-FullPath $Child
    $parentFull = (Get-FullPath $Parent).TrimEnd([System.IO.Path]::DirectorySeparatorChar) + [System.IO.Path]::DirectorySeparatorChar
    return $childFull.StartsWith($parentFull, [System.StringComparison]::OrdinalIgnoreCase)
}

function Remove-ProjectDirectory {
    param([Parameter(Mandatory = $true)][string]$Path)
    $fullPath = Get-FullPath $Path
    if (-not (Test-IsSubPath -Child $fullPath -Parent $script:ProjectRootFull)) {
        throw "Refusing to remove path outside project root: $fullPath"
    }
    if (Test-Path -LiteralPath $fullPath) {
        Write-Host "Removing $fullPath"
        Remove-Item -LiteralPath $fullPath -Recurse -Force
    }
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][scriptblock]$Command
    )
    Write-Host ""
    Write-Host "== $Label =="
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE"
    }
}

function Get-GitValue {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    try {
        $value = (& git @Arguments 2>$null)
        if ($LASTEXITCODE -eq 0) {
            return ($value | Select-Object -First 1)
        }
    } catch {
        return $null
    }
    return $null
}

$script:ProjectRootFull = Get-FullPath $ProjectRoot

$specName = if ($BuildMode -eq "Full") { "vbot.spec" } else { "vbot_launcher.spec" }
$specPath = Join-Path $script:ProjectRootFull $specName
if (-not (Test-Path -LiteralPath $specPath)) {
    throw "Project root does not contain ${specName}: $script:ProjectRootFull"
}

Set-Location -LiteralPath $script:ProjectRootFull

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
if ([string]::IsNullOrWhiteSpace($Version)) {
    $commit = Get-GitValue -Arguments @("rev-parse", "--short", "HEAD")
    if ([string]::IsNullOrWhiteSpace($commit)) {
        $Version = $timestamp
    } else {
        $Version = "dev-$commit"
    }
}
$Version = ($Version -replace "[^A-Za-z0-9_.-]", "-")

$releaseRoot = Join-Path $script:ProjectRootFull "release"
$logDir = Join-Path $releaseRoot "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$logPath = Join-Path $logDir "build-$Version-$timestamp.log"
$transcriptStarted = $false

try {
    Start-Transcript -Path $logPath -Force | Out-Null
    $transcriptStarted = $true

    Write-Host "Vbot desktop release build"
    Write-Host "Project root: $script:ProjectRootFull"
    Write-Host "Version: $Version"
    Write-Host "Build mode: $BuildMode"
    Write-Host "Build started: $((Get-Date).ToString('o'))"

    Invoke-Checked "Python version" { & $Python --version }
    Invoke-Checked "PyInstaller availability" { & $Python -m PyInstaller --version }

    Write-Host ""
    Write-Host "== Environment summary =="
    & $Python -c "import sys; print('Python executable:', sys.executable)"
    if ($BuildMode -eq "Full") {
        & $Python -c "import torch; print('Torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
    } else {
        Write-Host "Launcher mode does not import the ML runtime during packaging."
    }

    if (-not $SkipTests) {
        if (Test-Path -LiteralPath (Join-Path $script:ProjectRootFull "tests")) {
            Invoke-Checked "Fast test suite" { & $Python -m pytest tests -q --tb=short }
        } else {
            Write-Host "No tests directory found; skipping tests."
        }
    } else {
        Write-Host "Skipping tests because -SkipTests was provided."
    }

    if (-not $NoClean) {
        Remove-ProjectDirectory -Path (Join-Path $script:ProjectRootFull "build")
        Remove-ProjectDirectory -Path (Join-Path $script:ProjectRootFull "dist\Vbot")
    } else {
        Write-Host "Skipping clean because -NoClean was provided."
    }

    Invoke-Checked "PyInstaller build ($specName)" {
        if ($NoClean) {
            & $Python -m PyInstaller $specName --noconfirm
        } else {
            & $Python -m PyInstaller $specName --clean --noconfirm
        }
    }

    $distDir = Join-Path $script:ProjectRootFull "dist\Vbot"
    $exePath = Join-Path $distDir "Vbot.exe"
    if (-not (Test-Path -LiteralPath $exePath)) {
        throw "Build finished but Vbot.exe was not found at $exePath"
    }

    $sizeBytes = (Get-ChildItem -LiteralPath $distDir -Recurse -File | Measure-Object -Property Length -Sum).Sum
    $sizeGb = [math]::Round($sizeBytes / 1GB, 2)
    Write-Host ""
    Write-Host "Build output: $distDir"
    Write-Host "Build size: $sizeGb GB"

    if (-not $NoPackage) {
        $packageScript = Join-Path $PSScriptRoot "package_release.ps1"
        if (-not (Test-Path -LiteralPath $packageScript)) {
            throw "Packaging script not found: $packageScript"
        }
        & $packageScript -ProjectRoot $script:ProjectRootFull -Version $Version -DistDir $distDir -BuildLogPath $logPath -BuildMode $BuildMode
        if ($LASTEXITCODE -ne 0) {
            throw "Packaging failed with exit code $LASTEXITCODE"
        }
    } else {
        Write-Host "Skipping packaging because -NoPackage was provided."
    }

    Write-Host ""
    Write-Host "Build completed successfully."
    Write-Host "Build log: $logPath"
} finally {
    if ($transcriptStarted) {
        Stop-Transcript | Out-Null
    }
}
