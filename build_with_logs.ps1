[CmdletBinding()]
param(
    [string]$Python = "python",
    [string]$Version = "",
    [ValidateSet("Launcher", "Full")]
    [string]$BuildMode = "Launcher",
    [switch]$NoClean,
    [switch]$SkipTests,
    [switch]$NoPackage
)

$scriptPath = Join-Path $PSScriptRoot "scripts\build_with_logs.ps1"
if (-not (Test-Path -LiteralPath $scriptPath)) {
    throw "Missing build script: $scriptPath"
}

& $scriptPath `
    -ProjectRoot $PSScriptRoot `
    -Python $Python `
    -Version $Version `
    -BuildMode $BuildMode `
    -NoClean:$NoClean `
    -SkipTests:$SkipTests `
    -NoPackage:$NoPackage

exit $LASTEXITCODE
