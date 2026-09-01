<#
PowerShell Bridge for Win2D Brain Integration
Replaces bridge.ts with a PowerShell-based bridge that delegates to the Python runner (bridge_py_runner.py)
Usage:
  .\bridge.ps1 send '{"id":"r1","prompt":"Hello"}'    # send a single request JSON and print JSON response
  .\bridge.ps1 start-server                                 # start the python websocket server in background
  .\bridge.ps1 stop-server                                  # stop the python websocket server

This script uses a temp file to pass JSON to the Python runner for robustness.
#>
param(
    [string]$Command = "send",
    [string]$Payload = "",
    [int]$TimeoutSec = 30
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = "python"
$Runner = Join-Path $ScriptDir "bridge_py_runner.py"
$ServerScript = Join-Path $ScriptDir "bridge_server.py"

function Start-BridgeServer {
    if (-not (Test-Path $ServerScript)) {
        Write-Error "bridge_server.py not found at $ServerScript"
        return 1
    }
    Write-Output "Starting Python bridge server..."
    $proc = Start-Process -FilePath $Python -ArgumentList @($ServerScript) -WindowStyle Hidden -PassThru
    Start-Sleep -Seconds 1
    Write-Output "Started (PID=$($proc.Id))"
}

function Stop-BridgeServer {
    $procs = Get-Process -Name python -ErrorAction SilentlyContinue | Where-Object { $_.Path -ne $null -and $_.StartInfo -ne $null }
    foreach ($p in $procs) {
        try {
            $cmdline = (Get-CimInstance Win32_Process -Filter "ProcessId=$($p.Id)").CommandLine
            if ($cmdline -and $cmdline -like "*bridge_server.py*") {
                Write-Output "Stopping python bridge (PID=$($p.Id))"
                Stop-Process -Id $p.Id -Force
            }
        } catch {
            # ignore
        }
    }
}

function Send-Request {
    param([string]$JsonPayload)
    if (-not (Test-Path $Runner)) {
        Write-Error "Runner script not found: $Runner"
        return @{ status = 'error'; message = 'runner missing' } | ConvertTo-Json
    }
    # Write payload to temp file to avoid quoting issues
    $tmp = [System.IO.Path]::GetTempFileName()
    try {
        Set-Content -Path $tmp -Value $JsonPayload -Encoding UTF8
        $args = @($Runner, $tmp)
        $start = Get-Date
        $out = & $Python @args 2>&1
        $elapsed = (Get-Date) - $start
        if ($LASTEXITCODE -ne 0) {
            # Use explicit variable expansion and formatting to avoid parser issues
            $msg = [string]::Format('Runner exited with code {0}: {1}', ${LASTEXITCODE}, $out)
            Write-Error $msg
            return @{ status = 'error'; message = $out } | ConvertTo-Json
        }
        return $out
    } finally {
        Remove-Item -Path $tmp -ErrorAction SilentlyContinue
    }
}

switch ($Command.ToLower()) {
    'start-server' { Start-BridgeServer; exit 0 }
    'stop-server'  { Stop-BridgeServer; exit 0 }
    'send' {
        if (-not $Payload) { Write-Error 'No payload provided'; exit 2 }
        $resp = Send-Request -JsonPayload $Payload
        Write-Output $resp
        exit 0
    }
    'send-file' {
        if (-not $Payload) { Write-Error 'No payload file provided'; exit 2 }
        if (-not (Test-Path $Payload)) { Write-Error "Payload file not found: $Payload"; exit 2 }
        $json = Get-Content -Path $Payload -Raw -ErrorAction Stop
        $resp = Send-Request -JsonPayload $json
        Write-Output $resp
        exit 0
    }
    default {
        Write-Output "Unknown command: $Command"
        Write-Output "Usage: bridge.ps1 [start-server|stop-server|send|send-file] '<json-payload-or-file>'"
        exit 1
    }
}
