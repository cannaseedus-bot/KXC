# install_bridge_service.ps1
# Create a Windows service that runs the bridge_wrapper_service.ps1 using sc.exe
param(
    [string]$ServiceName = 'BridgeWrapperService'
)
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$wrapper = Join-Path $scriptDir 'bridge_wrapper_service.ps1'
if (-not (Test-Path $wrapper)) { Write-Error "Wrapper script not found: $wrapper"; exit 1 }
$pwsh = Join-Path $env:WINDIR 'System32\WindowsPowerShell\v1.0\powershell.exe'
$binPath = '"' + $pwsh + '" -NoProfile -ExecutionPolicy Bypass -File "' + $wrapper + '"'
Write-Output "Creating service $ServiceName with binPath: $binPath"
# Remove existing service if present
try { sc.exe delete $ServiceName } catch {}
Start-Sleep -Seconds 1
$create = sc.exe create $ServiceName binPath= $binPath start= auto
Write-Output $create
Start-Sleep -Seconds 1
# Start service
sc.exe start $ServiceName
Write-Output "Service $ServiceName created and started (if permissions allowed)."