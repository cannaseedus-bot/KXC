@echo off
setlocal
:: build_dag_dll.bat — standalone DAG scheduler DLL
::
:: Output: dag.dll, dag.lib, dag.exp in drivers/
::

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 (
    echo [build_dag_dll] ERROR: vcvars64.bat not found. Install VS 2022 BuildTools C++ workload.
    exit /b 1
)

set ROOT=%~dp0
if "%ROOT:~-1%"=="\" set ROOT=%ROOT:~0,-1%

cd /d "%ROOT%"

echo [build_dag_dll] Building dag.dll in %CD%

cl /nologo /LD /EHsc /O2 /std:c++17 /Fe:dag.dll dag_abi.cpp DAG.cpp /I"%ROOT%"
if errorlevel 1 (
    echo [build_dag_dll] FAILED
    exit /b 1
)

echo [build_dag_dll] OK: dag.dll built
dir /b dag.dll dag.lib dag.exp 2>nul
endlocal
