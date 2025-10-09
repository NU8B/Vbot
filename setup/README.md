# Vbot Setup Scripts

Automated setup scripts for Windows installation.

## Scripts

### `setup_windows.ps1`
Main installation script that orchestrates the entire setup process.

**Usage:**
```powershell
# Run as Administrator
.\setup\setup_windows.ps1
```

### `validate_system.ps1`
Validates system requirements before installation.

**Usage:**
```powershell
.\setup\validate_system.ps1
```

### `install_docker.ps1`
Automates Docker Desktop installation and configuration.

**Usage:**
```powershell
# Run as Administrator
.\setup\install_docker.ps1
```

### `check_cuda.ps1`
Checks CUDA and cuDNN installation status.

**Usage:**
```powershell
.\setup\check_cuda.ps1
```

## Requirements

- Windows 10 Build 19041+ or Windows 11
- Administrator privileges (for some scripts)
- PowerShell 5.1 or later

## Installation Order

1. Run `validate_system.ps1` to check requirements
2. Run `setup_windows.ps1` as Administrator
3. Follow on-screen instructions
4. Restart computer if prompted

## Troubleshooting

See [distribution_planning/challenges.md](../distribution_planning/challenges.md) for common issues.
