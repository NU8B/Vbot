[CmdletBinding()]
param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$DistDir = "",
    [string]$Version = "",
    [string]$BuildLogPath = "",
    [string]$OutputDir = "",
    [switch]$KeepStaging
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
        Remove-Item -LiteralPath $fullPath -Recurse -Force
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

function Copy-RequiredFile {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Destination
    )
    if (-not (Test-Path -LiteralPath $Source)) {
        throw "Required release file is missing: $Source"
    }
    Copy-Item -LiteralPath $Source -Destination $Destination -Force
}

$script:ProjectRootFull = Get-FullPath $ProjectRoot
Set-Location -LiteralPath $script:ProjectRootFull

if ([string]::IsNullOrWhiteSpace($DistDir)) {
    $DistDir = Join-Path $script:ProjectRootFull "dist\Vbot"
}
$DistDir = Get-FullPath $DistDir

if ([string]::IsNullOrWhiteSpace($Version)) {
    $commit = Get-GitValue -Arguments @("rev-parse", "--short", "HEAD")
    if ([string]::IsNullOrWhiteSpace($commit)) {
        $Version = Get-Date -Format "yyyyMMdd-HHmmss"
    } else {
        $Version = "dev-$commit"
    }
}
$Version = ($Version -replace "[^A-Za-z0-9_.-]", "-")

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $script:ProjectRootFull "release\artifacts"
}
$OutputDir = Get-FullPath $OutputDir

$exePath = Join-Path $DistDir "Vbot.exe"
if (-not (Test-Path -LiteralPath $exePath)) {
    throw "Vbot.exe not found. Build first, then package. Expected: $exePath"
}

$releaseRoot = Join-Path $script:ProjectRootFull "release"
$stagingBase = Join-Path $releaseRoot "staging"
$stagingRoot = Join-Path $stagingBase "Vbot-$Version-windows-portable"
$stagedAppDir = Join-Path $stagingRoot "Vbot"

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
New-Item -ItemType Directory -Force -Path $stagingBase | Out-Null
Remove-ProjectDirectory -Path $stagingRoot
New-Item -ItemType Directory -Force -Path $stagingRoot | Out-Null

Write-Host "Creating release staging directory: $stagingRoot"
Copy-Item -LiteralPath $DistDir -Destination $stagedAppDir -Recurse -Force

Copy-RequiredFile -Source (Join-Path $script:ProjectRootFull "USER_GUIDE.md") -Destination (Join-Path $stagingRoot "README.md")
Copy-RequiredFile -Source (Join-Path $script:ProjectRootFull "PREREQUISITES.md") -Destination (Join-Path $stagingRoot "PREREQUISITES.md")

if (-not [string]::IsNullOrWhiteSpace($BuildLogPath) -and (Test-Path -LiteralPath $BuildLogPath)) {
    Copy-Item -LiteralPath $BuildLogPath -Destination (Join-Path $stagingRoot "build.log") -Force
}

$commit = Get-GitValue -Arguments @("rev-parse", "HEAD")
$branch = Get-GitValue -Arguments @("rev-parse", "--abbrev-ref", "HEAD")
$builtAt = (Get-Date).ToUniversalTime().ToString("o")

$releaseNotesPath = Join-Path $stagingRoot "RELEASE_NOTES.md"
@"
# Vbot $Version

Package type: Windows portable zip
Built at (UTC): $builtAt
Git commit: $commit
Git branch: $branch

## Runtime model policy

This release keeps the AI stack local-first. It does not switch to a cloud LLM API.

- StyleTTS2 voice synthesis runs locally.
- THA4 avatar rendering runs locally.
- Emotion classification runs locally.
- Ollama LLM chat currently requires Docker Desktop and WSL 2.
- First run may download local model assets.

## Known limitations

- The package is large because it bundles ML, GUI, avatar, and audio dependencies.
- The executable is not code-signed yet, so Windows SmartScreen may warn users.
- This is a portable package, not a polished consumer installer.
"@ | Set-Content -LiteralPath $releaseNotesPath -Encoding UTF8

$artifactReleaseNotesPath = Join-Path $OutputDir "Vbot-$Version-release-notes.md"
Copy-Item -LiteralPath $releaseNotesPath -Destination $artifactReleaseNotesPath -Force

$artifactName = "Vbot-$Version-windows-portable.zip"
$artifactPath = Join-Path $OutputDir $artifactName
if (Test-Path -LiteralPath $artifactPath) {
    Remove-Item -LiteralPath $artifactPath -Force
}

Write-Host "Creating portable zip: $artifactPath"
$sevenZip = Get-Command 7z -ErrorAction SilentlyContinue
if ($sevenZip) {
    Push-Location -LiteralPath $stagingRoot
    try {
        & $sevenZip.Source a -tzip -mx=1 $artifactPath ".\*" | Write-Host
        if ($LASTEXITCODE -ne 0) {
            throw "7-Zip failed with exit code $LASTEXITCODE"
        }
    } finally {
        Pop-Location
    }
} else {
    Compress-Archive -Path (Join-Path $stagingRoot "*") -DestinationPath $artifactPath -CompressionLevel Fastest -Force
}

$artifactHash = Get-FileHash -LiteralPath $artifactPath -Algorithm SHA256
$exeHash = Get-FileHash -LiteralPath $exePath -Algorithm SHA256
$artifactSize = (Get-Item -LiteralPath $artifactPath).Length
$distSize = (Get-ChildItem -LiteralPath $DistDir -Recurse -File | Measure-Object -Property Length -Sum).Sum

$checksumPath = Join-Path $OutputDir "$artifactName.sha256"
"$($artifactHash.Hash)  $artifactName" | Set-Content -LiteralPath $checksumPath -Encoding ASCII

$manifest = [ordered]@{
    app = "Vbot"
    version = $Version
    package_type = "windows_portable_zip"
    built_at_utc = $builtAt
    git_commit = $commit
    git_branch = $branch
    artifact_name = $artifactName
    artifact_path = $artifactPath
    artifact_sha256 = $artifactHash.Hash
    artifact_size_bytes = $artifactSize
    dist_size_bytes = $distSize
    exe_sha256 = $exeHash.Hash
    runtime_requirements = @(
        "Windows 10/11 64-bit",
        "NVIDIA GPU and current NVIDIA driver recommended",
        "Docker Desktop with WSL 2 required for current local Ollama LLM chat",
        "Internet connection required on first run for local model downloads"
    )
}

$manifestPath = Join-Path $OutputDir "Vbot-$Version-release-manifest.json"
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $manifestPath -Encoding UTF8

Write-Host ""
Write-Host "Release package created:"
Write-Host "Artifact: $artifactPath"
Write-Host "SHA256:   $($artifactHash.Hash)"
Write-Host "Manifest: $manifestPath"

if (-not $KeepStaging) {
    Remove-ProjectDirectory -Path $stagingRoot
}
