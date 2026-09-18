cmake_minimum_required(VERSION 3.22.1)
project(dinov2
        DESCRIPTION "Inference DINOv2 in plain C/C++ with ggml"
        HOMEPAGE_URL "https://github.com/lavaman131/dinov2.cpp")

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif ()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Set the compiler flags, modify here if needed!
# If doesn't work, try removing -ffast-math and -fno-finite-math-only
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -ffast-math -fno-finite-math-only")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math -fno-finite-math-only")

# deps
add_subdirectory(ggml)

# dinov2 executable
add_executable(inference inference.cpp dinov2.cpp src/image.cpp)
target_link_libraries(inference PUBLIC ggml)
target_include_directories(inference PUBLIC . ${CMAKE_CURRENT_SOURCE_DIR})

option(BUILD_QUANTIZE "Build the quantization executable" ON)
if (BUILD_QUANTIZE)
    add_executable(quantize quantize.cpp dinov2.cpp src/image.cpp)
    target_link_libraries(quantize PUBLIC ggml)
    target_include_directories(quantize PUBLIC .)
endif ()