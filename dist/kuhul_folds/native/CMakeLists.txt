cmake_minimum_required(VERSION 3.10)
project(native_glyph_engine LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(native_glyph_engine SHARED
    glyph_backend_abi.cpp
    kuhul_glyph_api.cpp
)
target_compile_definitions(native_glyph_engine PRIVATE KUHUL_GLYPH_BACKEND_BUILD)

add_executable(native_glyph_engine_cli native_glyph_engine.cpp glyph_backend_abi.cpp kuhul_glyph_api.cpp)
target_compile_definitions(native_glyph_engine_cli PRIVATE KUHUL_GLYPH_BACKEND_BUILD)

# Keep the original ABI-only DLL for back-compat
add_library(native_glyph_engine_abi SHARED glyph_backend_abi.cpp)
target_compile_definitions(native_glyph_engine_abi PRIVATE KUHUL_GLYPH_BACKEND_BUILD)
