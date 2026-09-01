cmake_minimum_required(VERSION 3.20)
project(kxc CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB KXC_SOURCES src/*.cpp)
add_executable(kxc ${KXC_SOURCES})
target_include_directories(kxc PRIVATE src)

if(MSVC)
    target_compile_options(kxc PRIVATE /W3 /utf-8)
endif()

install(TARGETS kxc RUNTIME DESTINATION bin)
