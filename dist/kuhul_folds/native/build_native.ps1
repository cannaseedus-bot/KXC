# Build script: tries g++ then cl
$cwd = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $cwd
$src = Join-Path $cwd 'native_glyph_engine.cpp'
$out = Join-Path $cwd 'native_glyph_engine.exe'

if (Get-Command g++ -ErrorAction SilentlyContinue) {
    g++ -O2 -std=c++17 -o $out $src
    if ($LASTEXITCODE -eq 0) { Write-Output "Built with g++: $out"; Pop-Location; exit 0 }
}

# Try MSVC cl if available
if (Get-Command cl.exe -ErrorAction SilentlyContinue) {
    cl.exe /EHsc /std:c++17 /Fe:$out $src
    if ($LASTEXITCODE -eq 0) { Write-Output "Built with cl: $out"; Pop-Location; exit 0 }
}

Write-Error "No supported compiler found. Install g++ or Visual Studio Build Tools.";
Pop-Location
exit 1
