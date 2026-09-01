@echo off
:: build.cmd -- rebuild gpt2_trainer.exe
:: Double-click or run from any directory.
:: Kills running trainer first (releases exe lock), then compiles.

taskkill /F /IM gpt2_trainer.exe >nul 2>&1

cd /d "C:\Users\canna\.ASX.cpp\trainer"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1

cl /O2 /std:c++17 /EHsc ^
    /I"C:\Users\canna\_khanary_inspect\dist\xvm-d3d12\src" ^
    /I"C:\Users\canna\.ASX.cpp\trainer" ^
    /I"C:\Users\canna\_khanary_inspect\desktop\semantic_engine\include" ^
    gpt2_trainer.cpp gpt2_train_main.cpp ^
    "C:\Users\canna\_khanary_inspect\dist\xvm-d3d12\src\d3d11_engine.cpp" ^
    /Fe:gpt2_trainer.exe ^
    /link d3d11.lib dxgi.lib d3dcompiler.lib

echo.
if exist gpt2_trainer.exe (
    echo [ok] gpt2_trainer.exe rebuilt.
) else (
    echo [FAIL] build failed -- check errors above.
)
pause
