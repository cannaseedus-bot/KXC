@echo off
setlocal
:: build_all.bat — compile all 6 KHANARY driver DLLs with MSVC
::
:: Prerequisites: VS 2022/2026 BuildTools with C++ workload
:: Run from a VS x64 Native Tools command prompt (vcvars64)
::
:: Output: 6 .dll files in drivers/
::
::   khanary_driver.dll        — TaskEngine + DAG + provider dispatch
::   khanary_glyph_driver.dll  — 12 phase/fold glyphs + 13 compute lanes
::   kuhul_engine_driver.dll   — model loading + Atomic DOM + chat
::   gl_infer_driver.dll       — OpenGL 4.3 compute shader inference
::   qwen_infer_driver.dll     — Qwen 1.8B D3D11 inference
::   native_glyph_engine.dll   — K'UHUL glyph IPC engine (Powernaut source)
::

set "ROOT=%~dp0"
set "OUT=%ROOT%"

echo ============================================================
echo  KHANARY Driver Build — 6 DLLs
echo ============================================================
echo.

:: Verify cl.exe is available
where cl.exe >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] cl.exe not found. Run this from a VS x64 Native Tools command prompt.
    echo         Or: call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    exit /b 1
)

set "FAILED=0"

:: ── khanary_driver.dll ────────────────────────────────────────
echo [1/6] khanary_driver.dll ^(TaskEngine + DAG^)
cl /nologo /LD /EHsc /O2 /Fe:"%OUT%\khanary_driver.dll" ^
  "%ROOT%khanary_driver.cpp" "%ROOT%DAG.cpp" /I"%ROOT%" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo   FAILED — check khanary_driver.cpp + DAG.cpp
    set "FAILED=1"
) else (
    echo   OK
)

:: ── khanary_glyph_driver.dll ──────────────────────────────────
echo [2/6] khanary_glyph_driver.dll ^(25-entry glyph+lane registry^)
cl /nologo /LD /EHsc /O2 /Fe:"%OUT%\khanary_glyph_driver.dll" ^
  "%ROOT%khanary_glyph_driver.cpp" /I"%ROOT%" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo   FAILED — check khanary_glyph_driver.cpp
    set "FAILED=1"
) else (
    echo   OK
)

:: ── kuhul_engine_driver.dll ───────────────────────────────────
echo [3/6] kuhul_engine_driver.dll ^(Atomic DOM + chat^)
cl /nologo /LD /EHsc /O2 /Fe:"%OUT%\kuhul_engine_driver.dll" ^
  "%ROOT%kuhul_engine_driver.cpp" /I"%ROOT%" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo   FAILED — check kuhul_engine_driver.cpp
    set "FAILED=1"
) else (
    echo   OK
)

:: ── gl_infer_driver.dll ───────────────────────────────────────
echo [4/6] gl_infer_driver.dll ^(OpenGL 4.3 backend^)
cl /nologo /LD /EHsc /O2 /Fe:"%OUT%\gl_infer_driver.dll" ^
  "%ROOT%gl_infer_driver.cpp" /I"%ROOT%" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo   FAILED — check gl_infer_driver.cpp
    set "FAILED=1"
) else (
    echo   OK
)

:: ── qwen_infer_driver.dll ─────────────────────────────────────
echo [5/6] qwen_infer_driver.dll ^(Qwen 1.8B D3D11 backend^)
cl /nologo /LD /EHsc /O2 /Fe:"%OUT%\qwen_infer_driver.dll" ^
  "%ROOT%qwen_infer_driver.cpp" /I"%ROOT%" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo   FAILED — check qwen_infer_driver.cpp
    set "FAILED=1"
) else (
    echo   OK
)

:: ── native_glyph_engine.dll ───────────────────────────────────
echo [6/6] native_glyph_engine.dll ^(K'UHUL glyph IPC^)
cl /nologo /LD /EHsc /O2 /Fe:"%OUT%\native_glyph_engine.dll" ^
  "%ROOT%native_glyph_engine.cpp" "%ROOT%kuhul_glyph_api.cpp" ^
  "%ROOT%glyph_backend_abi.cpp" /I"%ROOT%" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo   FAILED — check native_glyph_engine.cpp + glyph_backend_abi.cpp
    set "FAILED=1"
) else (
    echo   OK
)

echo.
if "%FAILED%"=="1" (
    echo Build completed with errors. Check failed drivers above.
    exit /b 1
)

echo ============================================================
echo  All 6 drivers compiled successfully.
echo ============================================================
dir /b "%OUT%\*.dll" 2>nul
echo.
echo Copy to dist:
echo   copy drivers\*.dll dist\khanary-server\
echo   START-SERVERS
echo.
endlocal
