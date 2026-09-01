cmake_minimum_required(VERSION 3.20)
project(kxc CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Shared include paths ───────────────────────────────────────────────────────
# OpenCL headers (CL/cl.h etc.) — used by emit_opencl and native_glyph_engine
set(OPENCL_HEADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/khanary-llama-build/opencl-headers")
if(EXISTS "${OPENCL_HEADERS_DIR}/CL/cl.h")
    include_directories("${OPENCL_HEADERS_DIR}")
    message(STATUS "KXC: OpenCL headers → ${OPENCL_HEADERS_DIR}")
else()
    message(STATUS "KXC: OpenCL headers not found — emit_opencl will use system CL/")
endif()

# onnxruntime-genai 0.15.2 headers (ort_genai.h / ort_genai_c.h)
set(ORT_GENAI_INCLUDE
    "${CMAKE_CURRENT_SOURCE_DIR}/dist/WindowsML/onnxruntime-genai-0.15.2-win-x64-winml/onnxruntime-genai-0.15.2-win-x64-cuda/include")
if(EXISTS "${ORT_GENAI_INCLUDE}/ort_genai.h")
    include_directories("${ORT_GENAI_INCLUDE}")
    message(STATUS "KXC: ort_genai headers → ${ORT_GENAI_INCLUDE}")
endif()

# ── KXC compiler ──────────────────────────────────────────────────────────────
file(GLOB KXC_SOURCES src/*.cpp)
add_executable(kxc ${KXC_SOURCES})
target_include_directories(kxc PRIVATE src)
if(MSVC)
    target_compile_options(kxc PRIVATE /W3 /utf-8)
endif()
install(TARGETS kxc RUNTIME DESTINATION bin)

# ── native_glyph_engine — fold ABI + 12-lane glyph registry ──────────────────
# Builds: native_glyph_engine.dll (full), native_glyph_engine_abi.dll (ABI-only),
#         native_glyph_engine_cli executable.
option(KXC_NATIVE_GLYPH "Build native_glyph_engine DLL and CLI" ON)
if(KXC_NATIVE_GLYPH)
    add_subdirectory(dist/kuhul_folds/native)
endif()

# ── kdml — K'UHUL IR → DirectML lowering library ──────────────────────────────
# Builds: kuhul_ir (static), kuhul_conv_example, kuhul_ir_test.
option(KXC_KDML "Build kuhul_ir DirectML lowering library" ON)
if(KXC_KDML)
    add_subdirectory(kdml)
endif()

# NOTE: bridges/ggml-xcfe is excluded from this build.
# It calls ggml_add_backend_library() which only exists inside the ggml CMake
# scope. Build it through llama-build.bat (adds -DGGML_XCFE=ON to ggml's cmake).
