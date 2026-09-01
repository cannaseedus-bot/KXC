[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$DriverDir = Split-Path -Parent $MyInvocation.MyCommand.Path
[Environment]::CurrentDirectory = $DriverDir

$typeDef = @'
using System;
using System.Runtime.InteropServices;
public class DagDll {
    [DllImport("dag.dll", CallingConvention=CallingConvention.Cdecl)]
    public static extern IntPtr dag_create();
    [DllImport("dag.dll", CallingConvention=CallingConvention.Cdecl)]
    public static extern void dag_destroy(IntPtr ctx);
    [DllImport("dag.dll", CallingConvention=CallingConvention.Cdecl)]
    public static extern IntPtr dag_schedule_json(string tasksJson, IntPtr errorBuf, int errorBufSize);
    [DllImport("dag.dll", CallingConvention=CallingConvention.Cdecl)]
    public static extern void dag_free_string(IntPtr str);
}
'@

try {
    Add-Type -TypeDefinition $typeDef -ErrorAction Stop
} catch {
    Write-Error "Failed to compile P/Invoke wrapper: $($_.Exception.Message)"
    throw
}

function Invoke-DagSchedule([string]$Json) {
    $ptr = [DagDll]::dag_schedule_json($Json, [IntPtr]::Zero, 0)
    $result = [Runtime.InteropServices.Marshal]::PtrToStringAnsi($ptr)
    [DagDll]::dag_free_string($ptr)
    return $result
}

Write-Host '[test_dag_dll] Test 1: simple chain'
$json = '[{"id":"b"},{"id":"a","dependsOn":["b"]}]'
$r = Invoke-DagSchedule $json
Write-Host $r

Write-Host '[test_dag_dll] Test 2: cycle detection'
$json2 = '[{"id":"a","dependsOn":["b"]},{"id":"b","dependsOn":["a"]}]'
$r2 = Invoke-DagSchedule $json2
Write-Host $r2

Write-Host '[test_dag_dll] Test 3: missing dependency'
$json3 = '[{"id":"a","dependsOn":["missing"]}]'
$r3 = Invoke-DagSchedule $json3
Write-Host $r3

Write-Host '[test_dag_dll] done'
