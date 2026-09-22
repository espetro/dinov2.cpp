# Tier-2 Emscripten target. Reachable only with -DDINOV2_BUILD_WASM=ON, which
# the root CMakeLists already gates on the Emscripten toolchain; the guard
# below protects against a direct add_subdirectory(wasm).
if (NOT EMSCRIPTEN)
    message(FATAL_ERROR "wasm/ can only be built with the Emscripten toolchain")
endif ()

add_executable(dinov2-wasm dinov2-wasm.cpp)
target_link_libraries(dinov2-wasm PRIVATE dinov2 embind)
dinov2_target_flags(dinov2-wasm)

# ES6 module (import Dinov2Module from './dinov2-wasm.js'), separate .wasm
# artifact so browsers stream-compile it. web+worker only: this is a browser
# artifact; node is left out of the shipped ENVIRONMENT on purpose but can be
# re-added for smoke tests (-DDINOV2_WASM_ENVIRONMENT=web,worker,node).
set(DINOV2_WASM_ENVIRONMENT "web,worker" CACHE STRING
    "emscripten -sENVIRONMENT value for the dinov2-wasm target")
target_link_options(dinov2-wasm PRIVATE
    -sALLOW_MEMORY_GROWTH=1
    -sMAXIMUM_MEMORY=4GB
    -sMODULARIZE=1
    -sEXPORT_ES6=1
    -sEXPORT_NAME=Dinov2Module
    -sENVIRONMENT=${DINOV2_WASM_ENVIRONMENT})
set_target_properties(dinov2-wasm PROPERTIES SUFFIX ".js")

# --- multithreaded variant (not shipped in v1) -----------------------------
# ggml can drive a pthread pool under emscripten, but SharedArrayBuffer-backed
# threads require the page to be cross-origin isolated:
#   Cross-Origin-Opener-Policy: same-origin
#   Cross-Origin-Embedder-Policy: require-corp
# To experiment: add -pthread to the compile and link flags of dinov2-wasm and
# of ggml (CMAKE_C_FLAGS/CMAKE_CXX_FLAGS), add -sPTHREAD_POOL_SIZE=<n> to
# target_link_options above ('worker' is already in ENVIRONMENT), lift the
# n_threads=1 pin in wasm/dinov2-wasm.cpp, and serve wasm/ with the COOP/COEP
# headers above. Keep it a separate build flavor if it ever ships.
