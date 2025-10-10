# Build Vbot with Full Logging
# Saves all output to build_log.txt for debugging

$ErrorActionPreference = "Continue"

# Setup
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = "build_log_$timestamp.txt"
$projectRoot = $PSScriptRoot

Write-Host ""
Write-Host "╔════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║         Building Vbot with Logging Enabled            ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""
Write-Host "📝 Log file: $logFile" -ForegroundColor Yellow
Write-Host "⏱️  Started: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
Write-Host ""

# Log header
$logHeader = @"
╔════════════════════════════════════════════════════════╗
║              Vbot Build Log                            ║
╚════════════════════════════════════════════════════════╝

Build Started: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
Project Root: $projectRoot

"@

$logHeader | Out-File $logFile -Encoding UTF8

# Build command
$pythonPath = "C:\Users\peepz\.conda\envs\vbot\python.exe"
$buildCmd = "$pythonPath -m PyInstaller vbot.spec --clean"

Write-Host "🔨 Running: $buildCmd" -ForegroundColor Green
Write-Host "   (This will take 5-10 minutes...)" -ForegroundColor Gray
Write-Host ""

# Run build and capture output
try {
    # Execute and capture both stdout and stderr
    & $pythonPath -m PyInstaller vbot.spec --clean --noconfirm 2>&1 | Tee-Object -Append -FilePath $logFile | Out-Null
    
    $exitCode = $LASTEXITCODE
    
    # Log completion
    $logFooter = @"

╔════════════════════════════════════════════════════════╗
║              Build Completed                           ║
╚════════════════════════════════════════════════════════╝

Build Finished: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
Exit Code: $exitCode

"@
    
    $logFooter | Out-File $logFile -Append -Encoding UTF8
    
    Write-Host ""
    if ($exitCode -eq 0) {
        Write-Host "✅ Build completed successfully!" -ForegroundColor Green
        Write-Host "📁 Output: dist\Vbot\Vbot.exe" -ForegroundColor Cyan
    } else {
        Write-Host "❌ Build failed with exit code: $exitCode" -ForegroundColor Red
        Write-Host "📝 Check $logFile for details" -ForegroundColor Yellow
    }
    
    Write-Host ""
    Write-Host "📝 Full log saved to: $logFile" -ForegroundColor Yellow
    Write-Host ""
    
    # Show log file location
    $fullLogPath = Join-Path $projectRoot $logFile
    Write-Host "Log file location:" -ForegroundColor Cyan
    Write-Host "  $fullLogPath" -ForegroundColor Gray
    Write-Host ""
    
    # Extract and show important info
    Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host "BUILD SUMMARY" -ForegroundColor Yellow
    Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Cyan
    
    # Count warnings and errors
    $warnings = (Select-String -Path $logFile -Pattern "WARNING:" | Measure-Object).Count
    $errors = (Select-String -Path $logFile -Pattern "ERROR:" | Measure-Object).Count
    
    Write-Host "Warnings: $warnings" -ForegroundColor $(if ($warnings -gt 0) {"Yellow"} else {"Green"})
    Write-Host "Errors: $errors" -ForegroundColor $(if ($errors -gt 0) {"Red"} else {"Green"})
    
    # Show critical warnings/errors
    if ($errors -gt 0) {
        Write-Host ""
        Write-Host "🔴 Critical Errors Found:" -ForegroundColor Red
        Select-String -Path $logFile -Pattern "ERROR:" | Select-Object -First 10 | ForEach-Object {
            Write-Host "  $_" -ForegroundColor Red
        }
    }
    
    if ($warnings -gt 10) {
        Write-Host ""
        Write-Host "⚠️  Top Warnings:" -ForegroundColor Yellow
        Select-String -Path $logFile -Pattern "WARNING:" | Select-Object -First 5 | ForEach-Object {
            Write-Host "  $_" -ForegroundColor Yellow
        }
        Write-Host "  ... and $($warnings - 5) more warnings" -ForegroundColor Gray
    }
    
    Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host ""
    
    if ($exitCode -eq 0) {
        Write-Host "🧪 Next Steps:" -ForegroundColor Cyan
        Write-Host "  1. Test: cd dist\Vbot && .\Vbot.exe" -ForegroundColor Gray
        Write-Host "  2. If errors occur, send me: $logFile" -ForegroundColor Gray
    }
    
} catch {
    Write-Host ""
    Write-Host "❌ Build failed with exception:" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    
    $errorLog = @"

EXCEPTION OCCURRED:
$($_.Exception.Message)

Stack Trace:
$($_.ScriptStackTrace)

"@
    $errorLog | Out-File $logFile -Append -Encoding UTF8
}

Write-Host ""
Write-Host "════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

exit $exitCode
