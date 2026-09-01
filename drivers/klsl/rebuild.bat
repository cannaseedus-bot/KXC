@echo off
REM Bootstrap MSVC x64 environment and rebuild klslc.exe
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cd /d "C:\Users\canna\_khanary_inspect\drivers\klsl\build"
"C:\Users\canna\scoop\apps\ninja\1.13.2\ninja.exe"
