cmake_minimum_required(VERSION 3.14)
project(klslc LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(klslc klsl_main.cpp klsl_compiler.cpp)

if (MSVC)
    target_compile_options(klslc PRIVATE /utf-8 /W3 /O2)
else()
    target_compile_options(klslc PRIVATE -O2 -Wall)
endif()

install(TARGETS klslc RUNTIME DESTINATION bin)
