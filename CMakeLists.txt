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

option(DINOV2_FATAL_WARNINGS "Enable -Wall -Wextra -Wpedantic -Werror (MSVC: /W4 /WX)" OFF)
option(DINOV2_BUILD_TESTS "Build doctest unit tests" ON)

# deps
add_subdirectory(ggml)

# per-target release optimization flags (kept off global CMAKE_*_FLAGS so that
# sanitizer/Debug builds of ggml and this project are unaffected)
if (NOT MSVC)
    set(DINOV2_RELEASE_FLAGS -O3 -ffast-math -fno-finite-math-only)
else ()
    set(DINOV2_RELEASE_FLAGS)
endif ()

function(dinov2_target_flags target)
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target} PRIVATE
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:${DINOV2_RELEASE_FLAGS}>)
        if (DINOV2_FATAL_WARNINGS)
            target_compile_options(${target} PRIVATE
                -Wall -Wextra -Wpedantic -Werror
                # pre-existing code + vendored stb headers are not warning-clean
                -Wno-missing-field-initializers -Wno-unused-parameter
                -Wno-unused-function -Wno-deprecated-declarations -Wno-c99-extensions
                -Wno-unused-but-set-variable -Wno-sign-compare -Wno-unused-variable)
        endif ()
    elseif (MSVC)
        if (DINOV2_FATAL_WARNINGS)
            target_compile_options(${target} PRIVATE /W4 /WX)
        endif ()
    endif ()
endfunction()

# dinov2 executable
add_executable(inference inference.cpp dinov2.cpp src/image.cpp)
target_link_libraries(inference PUBLIC ggml)
target_include_directories(inference PUBLIC . ${CMAKE_CURRENT_SOURCE_DIR})
dinov2_target_flags(inference)

option(BUILD_QUANTIZE "Build the quantization executable" ON)
if (BUILD_QUANTIZE)
    add_executable(quantize quantize.cpp dinov2.cpp src/image.cpp)
    target_link_libraries(quantize PUBLIC ggml)
    target_include_directories(quantize PUBLIC .)
    dinov2_target_flags(quantize)
endif ()

# tests
if (DINOV2_BUILD_TESTS)
    enable_testing()
    add_executable(test_image tests/test_image.cpp src/image.cpp)
    target_include_directories(test_image PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src tests)
    dinov2_target_flags(test_image)
    add_test(NAME test_image COMMAND test_image)
endif ()
