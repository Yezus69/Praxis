# Clean reinstall of the NVIDIA driver to fix the broken WSL GPU paravirtualization
# component (dxgkio_query_adapter_info Ioctl -2 / CUDA NO_DEVICE). Uses the installer
# Chocolatey already extracted. Run as Administrator.
$logPath = 'C:\Users\Asav\source\repos\Praxis\driver_update.log'
Start-Transcript -Path $logPath -Force | Out-Null
$ErrorActionPreference = 'Continue'
Write-Host "=== clean driver reinstall started $(Get-Date) ==="

$setup = 'C:\Users\Asav\AppData\Local\Temp\chocolatey\nvidiainstall\setup.exe'
if (-not (Test-Path $setup)) {
    Write-Host "ERROR: cached installer not found at $setup"
} else {
    Write-Host "Running clean install: $setup -s -clean -noreboot"
    # -s silent, -clean wipes old driver state first, -noreboot so we control the reboot.
    $p = Start-Process -FilePath $setup -ArgumentList '-s','-clean','-noreboot' -Wait -PassThru
    Write-Host "SETUP_EXIT=$($p.ExitCode)"
}

Write-Host "=== post-install driver ==="
try { & nvidia-smi --query-gpu=name,driver_version --format=csv,noheader } catch { Write-Host "nvidia-smi: $_" }
Write-Host "=== clean reinstall finished $(Get-Date) ==="
Stop-Transcript | Out-Null
