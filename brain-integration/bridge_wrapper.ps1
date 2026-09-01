# bridge_wrapper.ps1
# Hardened HTTP listener that accepts POST JSON and invokes bridge.ps1 send-file '<tmpfile>'
# - Requires BRIDGE_TOKEN (Bearer) if set in environment
# - Respects ORIGIN whitelist via BRIDGE_ORIGIN_WHITELIST (comma-separated)
# - Auto-starts bridge_server.py (WebSocket) if not already running
# Runs on http://127.0.0.1:3170/bridge/

$ErrorActionPreference = 'Stop'
$prefix = "http://127.0.0.1:3170/bridge/"

# Security configuration
$BRIDGE_TOKEN = $env:BRIDGE_TOKEN
$ORIGIN_WHITELIST = $env:BRIDGE_ORIGIN_WHITELIST
if (-not $ORIGIN_WHITELIST) { $ORIGIN_WHITELIST = "http://localhost,http://127.0.0.1" }
$allowedOrigins = $ORIGIN_WHITELIST.Split(',') | ForEach-Object { $_.Trim() }

Add-Type -AssemblyName System.Net.Http
$listener = New-Object System.Net.HttpListener
$listener.Prefixes.Add($prefix)
$listener.Start()
Write-Output "[bridge_wrapper] Listening on $prefix"

# Try to ensure the WS server is running (bridge_server.py)
$scriptDir = $PSScriptRoot
$wsScript = Join-Path $scriptDir 'bridge_server.py'
$wsProc = $null
function Ensure-WsServer {
    if (-not (Test-Path $wsScript)) { return }
    $running = Get-Process -Name python -ErrorAction SilentlyContinue | Where-Object {
        try { (Get-CimInstance Win32_Process -Filter "ProcessId=$($_.Id)").CommandLine -like "*bridge_server.py*" } catch { $false }
    }
    if ($running) {
        Write-Output "[bridge_wrapper] bridge_server.py already running (PID=$(($running | Select-Object -First 1).Id))"
    } else {
        try {
            Write-Output "[bridge_wrapper] Starting bridge_server.py..."
            $py = 'python'
            # Prefer venv if present in current workspace or up the tree
            $maybeVenv = Join-Path $scriptDir '..\..\venv\Scripts\python.exe'
            if (-not (Test-Path $maybeVenv)) { $maybeVenv = Join-Path $scriptDir '..\..\..\venv\Scripts\python.exe' }
            if (-not (Test-Path $maybeVenv)) { $maybeVenv = Join-Path $scriptDir '..\..\..\..\venv\Scripts\python.exe' }
            $maybeVenv = (Resolve-Path $maybeVenv -ErrorAction SilentlyContinue).ProviderPath 2>$null
            if ($maybeVenv -and (Test-Path $maybeVenv)) { $py = $maybeVenv }
            $wsProc = Start-Process -FilePath $py -ArgumentList @($wsScript) -WorkingDirectory $scriptDir -WindowStyle Hidden -PassThru
            Start-Sleep -Seconds 1
            Write-Output "[bridge_wrapper] Started bridge_server.py (PID=$($wsProc.Id))"
        } catch {
            Write-Output "[bridge_wrapper] Failed to start bridge_server.py: $($_.Exception.Message)" -ForegroundColor Yellow
        }
    }
}

# Attempt to start WS server on wrapper startup
Ensure-WsServer

while ($true) {
    try {
        $context = $listener.GetContext()
        $req = $context.Request
        $res = $context.Response

        # Provide a simple health endpoint at /bridge/health
        $path = $req.Url.AbsolutePath
        if ($req.HttpMethod -eq 'GET' -and ($path -like '*/health' -or $path -eq '/bridge/health')) {
            $statusObj = @{ status = 'ok'; ws_running = ($wsProc -ne $null -and ($wsProc.HasExited -ne $true)) }
            $b = [System.Text.Encoding]::UTF8.GetBytes(($statusObj | ConvertTo-Json))
            $res.ContentType = 'application/json'
            $res.ContentLength64 = $b.Length
            $res.OutputStream.Write($b,0,$b.Length)
            $res.StatusCode = 200
            $res.Close()
            continue
        }

        if ($req.HttpMethod -ne 'POST') {
            $res.StatusCode = 405
            $res.StatusDescription = 'Method Not Allowed'
            $bytes = [System.Text.Encoding]::UTF8.GetBytes('{"error":"Method Not Allowed"}')
            $res.ContentType = 'application/json'
            $res.ContentLength64 = $bytes.Length
            $res.OutputStream.Write($bytes,0,$bytes.Length)
            $res.Close()
            continue
        }

        # Authorization check (if BRIDGE_TOKEN is set)
        if ($BRIDGE_TOKEN) {
            $auth = $req.Headers['Authorization']
            if (-not $auth -or -not $auth.StartsWith('Bearer ')) {
                $res.StatusCode = 401
                $res.StatusDescription = 'Unauthorized'
                $b = [System.Text.Encoding]::UTF8.GetBytes('{"error":"Missing or invalid Authorization header"}')
                $res.ContentType = 'application/json'
                $res.ContentLength64 = $b.Length
                $res.OutputStream.Write($b,0,$b.Length)
                $res.Close()
                continue
            }
            $token = $auth.Substring(7)
            if ($token -ne $BRIDGE_TOKEN) {
                $res.StatusCode = 401
                $res.StatusDescription = 'Unauthorized'
                $b = [System.Text.Encoding]::UTF8.GetBytes('{"error":"Invalid token"}')
                $res.ContentType = 'application/json'
                $res.ContentLength64 = $b.Length
                $res.OutputStream.Write($b,0,$b.Length)
                $res.Close()
                continue
            }
        }

        # Origin check
        $origin = $req.Headers['Origin']
        if (-not $origin -or ($allowedOrigins -notcontains $origin)) {
            $res.StatusCode = 403
            $res.StatusDescription = 'Forbidden'
            $b = [System.Text.Encoding]::UTF8.GetBytes('{"error":"Origin not allowed"}')
            $res.ContentType = 'application/json'
            $res.ContentLength64 = $b.Length
            $res.OutputStream.Write($b,0,$b.Length)
            $res.Close()
            continue
        }

        $reader = New-Object System.IO.StreamReader($req.InputStream, $req.ContentEncoding)
        $body = $reader.ReadToEnd()

        # Prepare bridge invocation
        $bridgePath = Join-Path $PSScriptRoot 'bridge.ps1'
        if (-not (Test-Path $bridgePath)) {
            $errJson = @{ error = "bridge.ps1 not found at $bridgePath" } | ConvertTo-Json
            $b = [System.Text.Encoding]::UTF8.GetBytes($errJson)
            $res.ContentType = 'application/json'
            $res.ContentLength64 = $b.Length
            $res.OutputStream.Write($b,0,$b.Length)
            $res.StatusCode = 500
            $res.Close()
            continue
        }

        # Write body to temp file and invoke bridge.ps1 with send-file to avoid quoting issues
        $tmpFile = [System.IO.Path]::GetTempFileName()
        try {
            Set-Content -Path $tmpFile -Value $body -Encoding UTF8
            # Invoke bridge.ps1 directly to avoid subprocess quoting issues
            try {
                $output = & $bridgePath 'send-file' $tmpFile 2>&1
                $stdout = ($output | Where-Object { $_ -ne $null }) -join "`n"
                $stderr = ""
            } catch {
                $stdout = ""
                $stderr = $_.Exception.Message
            }
        } finally {
            Remove-Item -Path $tmpFile -ErrorAction SilentlyContinue
        }

        $outObj = @{ stdout = $stdout; stderr = $stderr }
        $jsonOut = $outObj | ConvertTo-Json -Depth 5
        $buffer = [System.Text.Encoding]::UTF8.GetBytes($jsonOut)

        $res.ContentType = 'application/json'
        $res.ContentLength64 = $buffer.Length
        $res.OutputStream.Write($buffer,0,$buffer.Length)
        $res.StatusCode = 200
        $res.Close()

    } catch {
        try {
            $errObj = @{ error = $_.Exception.Message }
            $errJson = $errObj | ConvertTo-Json
            $b = [System.Text.Encoding]::UTF8.GetBytes($errJson)
            if ($res) {
                $res.ContentType = 'application/json'
                $res.ContentLength64 = $b.Length
                $res.OutputStream.Write($b,0,$b.Length)
                $res.StatusCode = 500
                $res.Close()
            }
        } catch { }
    }
}
