# Elevated NVIDIA driver update for Praxis (run as Administrator).
# Logs everything to driver_update.log so the orchestrator can monitor progress.
$logPath = 'C:\Users\Asav\source\repos\Praxis\driver_update.log'
Start-Transcript -Path $logPath -Force | Out-Null
$ErrorActionPreference = 'Continue'
Write-Host "=== driver update started $(Get-Date) ==="

# Show current driver for the record.
try { & nvidia-smi --query-gpu=name,driver_version --format=csv,noheader } catch {}

# Install Chocolatey if missing.
if (-not (Get-Command choco -ErrorAction SilentlyContinue)) {
    Write-Host "Installing Chocolatey..."
    Set-ExecutionPolicy Bypass -Scope Process -Force
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.SecurityProtocolType]::Tls12
    Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
} else {
    Write-Host "Chocolatey already present."
}

$choco = "$env:ProgramData\chocolatey\bin\choco.exe"

Write-Host "=== choco info nvidia-display-driver (version that will be installed) ==="
& $choco info nvidia-display-driver

Write-Host "=== installing nvidia-display-driver (downloads ~800MB, installs silently) ==="
& $choco install nvidia-display-driver -y --no-progress --ignore-checksums
Write-Host "CHOCO_EXIT=$LASTEXITCODE"

Write-Host "=== post-install driver ==="
try { & nvidia-smi --query-gpu=name,driver_version --format=csv,noheader } catch { Write-Host "nvidia-smi failed (display may be resetting): $_" }

Write-Host "=== driver update finished $(Get-Date) ==="
Stop-Transcript | Out-Null
