@echo off
setlocal
::
:: build_drivers.bat — compile all Khanary native driver DLLs with MSVC BuildTools 2022
::
:: Run from inside a VS x64 dev shell, or let this script call vcvars64.bat.
::

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 (
    echo [build_drivers] ERROR: vcvars64.bat not found. Install VS 2022 BuildTools C++ workload.
    exit /b 1
)

set CLFLAGS=/nologo /std:c++17 /EHsc /O2 /LD /I.

echo [build_drivers] Building driver DLLs in %CD%

cl %CLFLAGS% khanary_driver.cpp DAG.cpp /Fe:khanary_driver.dll
if errorlevel 1 (echo [build_drivers] khanary_driver.dll FAILED & exit /b 1)
echo [build_drivers] khanary_driver.dll OK

cl %CLFLAGS% khanary_glyph_driver.cpp /Fe:khanary_glyph_driver.dll
if errorlevel 1 (echo [build_drivers] khanary_glyph_driver.dll FAILED & exit /b 1)
echo [build_drivers] khanary_glyph_driver.dll OK

cl %CLFLAGS% kuhul_engine_driver.cpp /Fe:kuhul_engine_driver.dll
if errorlevel 1 (echo [build_drivers] kuhul_engine_driver.dll FAILED & exit /b 1)
echo [build_drivers] kuhul_engine_driver.dll OK

cl %CLFLAGS% gl_infer_driver.cpp /Fe:gl_infer_driver.dll
if errorlevel 1 (echo [build_drivers] gl_infer_driver.dll FAILED & exit /b 1)
echo [build_drivers] gl_infer_driver.dll OK

cl %CLFLAGS% qwen_infer_driver.cpp /Fe:qwen_infer_driver.dll
if errorlevel 1 (echo [build_drivers] qwen_infer_driver.dll FAILED & exit /b 1)
echo [build_drivers] qwen_infer_driver.dll OK

echo [build_drivers] All driver DLLs built successfully.
endlocal
