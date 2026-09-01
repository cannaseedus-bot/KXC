@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cl /O2 /std:c++17 /EHsc /DKUHUL_GLYPH_BACKEND_BUILD native_glyph_engine.cpp glyph_backend_abi.cpp kuhul_glyph_api.cpp /Fe:native_glyph_engine.dll /LD
