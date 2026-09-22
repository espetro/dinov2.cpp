cmake_minimum_required(VERSION 3.22.1)
project(dinov2
        VERSION 0.4.0
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
option(DINOV2_BUILD_CLI "Build the dinov2-cli executable" ON)
option(DINOV2_BUILD_C_API "Build the pure C API (include/dinov2.h) into libdinov2" ON)

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

# dinov2 library: engine (dinov2.cpp) + image/preprocessing (src/image.cpp).
# The public C API in include/dinov2.h is compiled in when
# DINOV2_BUILD_C_API is on; the internal C++ header stays at the repo root
# and is not installed.
add_library(dinov2 dinov2.cpp src/image.cpp)
target_link_libraries(dinov2 PUBLIC ggml)
# repo root first: both include/dinov2.h and the internal dinov2.h exist, and
# the internal header must win bare "dinov2.h" quoted includes
target_include_directories(dinov2 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/include)
dinov2_target_flags(dinov2)
if (DINOV2_BUILD_C_API)
    target_sources(dinov2 PRIVATE src/dinov2-c.cpp)
endif ()
if (BUILD_SHARED_LIBS)
    # DINOV2_SHARED is public so consumers get dllimport; DINOV2_BUILD marks
    # the library's own TU for dllexport. WINDOWS_EXPORT_ALL_SYMBOLS lets the
    # CLI and tests still reach the internal C++ symbols in shared builds.
    target_compile_definitions(dinov2 PUBLIC DINOV2_SHARED PRIVATE DINOV2_BUILD)
    set_target_properties(dinov2 PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif ()

# dinov2 executable
if (DINOV2_BUILD_CLI)
    add_executable(dinov2-cli dinov2-cli.cpp)
    target_link_libraries(dinov2-cli PRIVATE dinov2)
    dinov2_target_flags(dinov2-cli)
endif ()

# tests
if (DINOV2_BUILD_TESTS)
    enable_testing()
    add_executable(test_image tests/test_image.cpp src/image.cpp)
    target_include_directories(test_image PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src tests)
    dinov2_target_flags(test_image)
    add_test(NAME test_image COMMAND test_image)

    # dinov2 library tests: pure-function coverage for dino_hparams math,
    # interpolate_pos_embed, dino_preprocess, and dino_predict against the
    # synthetic TinyModel fixture.
    add_executable(test_dinov2 tests/test_dinov2.cpp)
    target_link_libraries(test_dinov2 PRIVATE dinov2)
    target_include_directories(test_dinov2 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        tests)
    dinov2_target_flags(test_dinov2)
    add_test(NAME test_dinov2 COMMAND test_dinov2)

    if (DINOV2_BUILD_CLI)
        add_executable(test_cli tests/test_cli.cpp)
        target_link_libraries(test_cli PRIVATE ggml)
        dinov2_target_flags(test_cli)
        add_test(NAME test_cli COMMAND test_cli $<TARGET_FILE:dinov2-cli>)
    endif ()
endif ()
