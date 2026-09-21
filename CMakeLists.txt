cmake_minimum_required(VERSION 3.22.1)
project(dinov2
        VERSION 0.3.0
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
    target_compile_definitions(${target} PRIVATE DINOV2_VERSION="${PROJECT_VERSION}")
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
add_executable(dinov2-cli dinov2-cli.cpp dinov2.cpp src/image.cpp)
target_link_libraries(dinov2-cli PUBLIC ggml)
target_include_directories(dinov2-cli PUBLIC . ${CMAKE_CURRENT_SOURCE_DIR})
dinov2_target_flags(dinov2-cli)

# tests
if (DINOV2_BUILD_TESTS)
    enable_testing()
    add_executable(test_image tests/test_image.cpp src/image.cpp)
    target_include_directories(test_image PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src tests)
    dinov2_target_flags(test_image)
    add_test(NAME test_image COMMAND test_image)

    # dinov2 library tests: pure-function coverage for dino_hparams math,
    # interpolate_pos_embed, dino_preprocess fallback. Links ggml because
    # interpolate_pos_embed / dino_preprocess don't need it but the test
    # binary is built against the same source as dinov2-cli for cohesion.
    add_executable(test_dinov2 tests/test_dinov2.cpp dinov2.cpp src/image.cpp)
    target_link_libraries(test_dinov2 PUBLIC ggml)
    target_include_directories(test_dinov2 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        tests)
    dinov2_target_flags(test_dinov2)
    add_test(NAME test_dinov2 COMMAND test_dinov2)
endif ()
