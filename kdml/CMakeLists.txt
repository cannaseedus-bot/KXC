# CMakeLists.txt - K'UHUL v0.1 Compiler Build System
# Frozen Reference Implementation for Layer 6 Compiler
# Date: 2026-07-12

cmake_minimum_required(VERSION 3.16)
project(Kuhul VERSION 0.1.0 LANGUAGES CXX)

# ===== C++ Standard =====
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ===== Build Flags =====
if(MSVC)
    add_compile_options(/W4 /WX /permissive-)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Werror)
endif()

# ===== Include Directories =====
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# ===== Main IR Library =====
add_library(kuhul_ir
    src/lowering_to_dml.cpp
)

target_include_directories(kuhul_ir
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# ===== DirectML Integration (Optional) =====
# Uncomment when DirectML SDK is available
# find_package(DirectML REQUIRED)
# target_link_libraries(kuhul_ir PRIVATE DirectML::DirectML)

# ===== Example Executables =====

# Conv2D + ReLU + Pooling Example
add_executable(kuhul_conv_example
    examples/conv_example.cpp
)

target_link_libraries(kuhul_conv_example
    kuhul_ir
)

# ===== Unit Tests =====
enable_testing()

# Test runner (basic validation)
add_executable(kuhul_ir_test
    tests/test_ir_types.cpp
)

target_link_libraries(kuhul_ir_test
    kuhul_ir
)

add_test(NAME ir_types COMMAND kuhul_ir_test)

# ===== Installation =====
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/kuhul
    DESTINATION include
)

install(TARGETS kuhul_ir
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# ===== Summary =====
message(STATUS "========================================")
message(STATUS "K'UHUL v0.1 Compiler Build Configuration")
message(STATUS "========================================")
message(STATUS "Version:       ${CMAKE_PROJECT_VERSION}")
message(STATUS "C++ Standard:  ${CMAKE_CXX_STANDARD}")
message(STATUS "Build Type:    ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler:      ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "")
message(STATUS "Targets:")
message(STATUS "  Library:     kuhul_ir")
message(STATUS "  Example:     kuhul_conv_example")
message(STATUS "  Tests:       kuhul_ir_test")
message(STATUS "")
message(STATUS "Build with: cmake --build . --config Release")
message(STATUS "========================================")
