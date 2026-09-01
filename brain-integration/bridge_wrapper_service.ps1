# bridge_wrapper_service.ps1
# Hardened bridge wrapper with Prometheus metrics and simple rate limiter
# Runs on http://127.0.0.1:3170/bridge/

$ErrorActionPreference = 'Stop'
$prefix = "http://127.0.0.1:3170/bridge/"

# Configuration
$BRIDGE_TOKEN = $env:BRIDGE_TOKEN
$ORIGIN_WHITELIST = $env:BRIDGE_ORIGIN_WHITELIST
if (-not $ORIGIN_WHITELIST) { $ORIGIN_WHITELIST = "http://localhost,http://127.0.0.1" }
$allowedOrigins = $ORIGIN_WHITELIST.Split(',') | ForEach-Object { $_.Trim() }
$RATE_LIMIT_PER_MIN = [int]($env:BRIDGE_RATE_LIMIT -as [int])
if (-not $RATE_LIMIT_PER_MIN -or $RATE_LIMIT_PER_MIN -le 0) { $RATE_LIMIT_PER_MIN = 120 }
$RATE_REFILL_PER_SEC = $RATE_LIMIT_PER_MIN / 60.0

# Metrics
$global:metrics = @{ requests_total = 0; requests_allowed = 0; requests_denied = 0; request_latency_sum = 0.0 }

# Rate limiter state
$global:buckets = @{}

function Allow-Request($key) {
    $now = Get-Date
    if (-not $global:buckets.ContainsKey($key)) {
        $global:buckets[$key] = @{ tokens = $RATE_LIMIT_PER_MIN; last = $now }
    }
    $entry = $global:buckets[$key]
    $elapsed = ($now - $entry.last).TotalSeconds
    if ($elapsed -gt 0) {
        $refill = $elapsed * $RATE_REFILL_PER_SEC
        $entry.tokens = [Math]::Min($RATE_LIMIT_PER_MIN, $entry.tokens + $refill)
        $entry.last = $now
    }
    if ($entry.tokens -ge 1) {
        $entry.tokens = $entry.tokens - 1
        return $true
    }
    return $false
}

Add-Type -AssemblyName System.Net.Http
$listener = New-Object System.Net.HttpListener
$listener.Prefixes.Add($prefix)
$listener.Start()
Write-Output "[bridge_wrapper_service] Listening on $prefix"

# WS server autostart (same as before)
$scriptDir = $PSScriptRoot
$wsScript = Join-Path $scriptDir 'bridge_server.py'
function Ensure-WsServer {
    if (-not (Test-Path $wsScript)) { return }
    # Detect existing bridge_server.py processes
    $existing = Get-Process -Name python -ErrorAction SilentlyContinue | Where-Object {
        try { (Get-CimInstance Win32_Process -Filter "ProcessId=$($_.Id)" -ErrorAction SilentlyContinue).CommandLine -like '*bridge_server.py*' } catch { $false }
    }
    if ($existing) { $global:ws_running = $true; return }
    try {
        $py = $env:BRIDGE_PYTHON
        if (-not $py -or -not (Test-Path $py)) {
            $maybeVenv = Join-Path $scriptDir '..\..\..\venv\Scripts\python.exe'
            $resolved = (Resolve-Path $maybeVenv -ErrorAction SilentlyContinue).ProviderPath 2>$null
            if ($resolved -and (Test-Path $resolved)) { $py = $resolved }
        }
        if (-not $py -or -not (Test-Path $py)) { $py = 'python' }
        Start-Process -FilePath $py -ArgumentList @($wsScript) -WorkingDirectory $scriptDir -WindowStyle Hidden -PassThru | Out-Null
        Start-Sleep -Seconds 1
        $existing = Get-Process -Name python -ErrorAction SilentlyContinue | Where-Object {
            try { (Get-CimInstance Win32_Process -Filter "ProcessId=$($_.Id)" -ErrorAction SilentlyContinue).CommandLine -like '*bridge_server.py*' } catch { $false }
        }
        $global:ws_running = ($existing -ne $null)
        if ($global:ws_running) { Write-Output "[bridge_wrapper_service] Started bridge_server.py" }
    } catch {
        $global:ws_running = $false
        Write-Output "[bridge_wrapper_service] Failed to start bridge_server.py: $($_.Exception.Message)" -ForegroundColor Yellow
    }
}


Ensure-WsServer

while ($true) {
    try {
        $context = $listener.GetContext()
        $req = $context.Request
        $res = $context.Response
        $ip = $context.Request.RemoteEndPoint.Address.ToString()

        # Metrics and health endpoints
        $path = $req.Url.AbsolutePath
        if ($req.HttpMethod -eq 'GET' -and ($path -like '*/health' -or $path -eq '/bridge/health')) {
            # Determine whether the WS bridge is running. Prefer the global flag, fall back to process, port, TCP connect, or pid-file checks.
            $ws_running = $false
            if ($global:ws_running) {
                $ws_running = $true
            } else {
                try {
                    $procCheck = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -and ($_.CommandLine -match 'bridge_server.py' -or $_.CommandLine -match 'bridge_server') }
                    if ($procCheck) { $ws_running = $true }
                } catch {}
                if (-not $ws_running) {
                    try {
                        $conn = Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue
                        if ($conn) { $ws_running = $true }
                    } catch {}
                }
                if (-not $ws_running) {
                    try {
                        $tcp = New-Object System.Net.Sockets.TcpClient
                        $async = $tcp.BeginConnect('127.0.0.1',8765,$null,$null)
                        $ok = $async.AsyncWaitHandle.WaitOne(500)
                        if ($ok -and $tcp.Connected) { $ws_running = $true; $tcp.Close() } else { try { $tcp.Close() } catch {} }
                    } catch {}
                }
                if (-not $ws_running) {
                    # Fallback: check for pid file created by admin/launcher. If present, assume bridge is running.
                    $pidFile = 'C:\\public_html\\MX2LM\\bridge_server.pid'
                    if (Test-Path $pidFile) { $ws_running = $true }
                }
            }
            $statusObj = @{ status = 'ok'; ws_running = $ws_running }
            $b = [System.Text.Encoding]::UTF8.GetBytes(($statusObj | ConvertTo-Json))
            $res.ContentType = 'application/json'
            $res.ContentLength64 = $b.Length
            $res.OutputStream.Write($b,0,$b.Length)
            $res.StatusCode = 200
            $res.Close()
            continue
        }
        if ($req.HttpMethod -eq 'GET' -and ($path -like '*/metrics' -or $path -eq '/bridge/metrics')) {
            $lines = @()
            $lines += "# HELP bridge_requests_total Total number of requests received"
            $lines += "# TYPE bridge_requests_total counter"
            $lines += "bridge_requests_total $($global:metrics.requests_total)"
            $lines += "# HELP bridge_requests_allowed Number of allowed requests"
            $lines += "# TYPE bridge_requests_allowed counter"
            $lines += "bridge_requests_allowed $($global:metrics.requests_allowed)"
            $lines += "# HELP bridge_requests_denied Number of denied requests"
            $lines += "# TYPE bridge_requests_denied counter"
            $lines += "bridge_requests_denied $($global:metrics.requests_denied)"
            $body = [System.Text.Encoding]::UTF8.GetBytes(([string]::Join("`n", $lines)))
            $res.ContentType = 'text/plain; version=0.0.4'
            $res.ContentLength64 = $body.Length
            $res.OutputStream.Write($body,0,$body.Length)
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

        $global:metrics.requests_total += 1
        if (-not (Allow-Request $ip)) {
            $global:metrics.requests_denied += 1
            $res.StatusCode = 429
            $res.StatusDescription = 'Too Many Requests'
            $b = [System.Text.Encoding]::UTF8.GetBytes('{"error":"rate_limited"}')
            $res.ContentType = 'application/json'
            $res.ContentLength64 = $b.Length
            $res.OutputStream.Write($b,0,$b.Length)
            $res.Close()
            continue
        }

        $global:metrics.requests_allowed += 1

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
                $start = Get-Date
                $output = & $bridgePath 'send-file' $tmpFile 2>&1
                $elapsed = (Get-Date) - $start
                $stdout = ($output | Where-Object { $_ -ne $null }) -join "`n"
                $stderr = ""
                $global:metrics.request_latency_sum += $elapsed.TotalSeconds
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

