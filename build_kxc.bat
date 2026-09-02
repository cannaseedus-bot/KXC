@echo off
setlocal

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 ( echo vcvars64 failed & exit /b 1 )

set SRC=%~dp0
if "%SRC:~-1%"=="\" set SRC=%SRC:~0,-1%

if not exist "%SRC%\build_kxc_src" mkdir "%SRC%\build_kxc_src"
pushd "%SRC%\build_kxc_src"

cmake .. -G Ninja -DKXC_NATIVE_GLYPH=OFF -DKXC_KDML=OFF -DCMAKE_BUILD_TYPE=Release
if errorlevel 1 ( popd & echo cmake configure failed & exit /b 1 )

cmake --build . --target kxc
if errorlevel 1 ( popd & echo cmake build failed & exit /b 1 )

popd
echo.
echo === DONE: kxc.exe built ===
copy /Y "%SRC%\build_kxc_src\kxc.exe" "%SRC%\bin\kxc.exe"
echo Deployed to bin\kxc.exe
endlocal
