// Emscripten binding layer over the pure C API in include/dinov2.h.
// Compiled only when DINOV2_BUILD_WASM=ON under the Emscripten toolchain.
//
// JS surface (see wasm/main.js for usage):
//   Module.version()                                  -> string
//   Module.loadModel(Uint8Array ggufBytes)            -> model handle (>0), 0 on failure
//   Module.modelInfo(handle)                          -> {hiddenSize, patchSize, nRegisterTokens,
//                                                        hasClassifier, nClasses} or null
//   Module.encode(handle, pixels, width, height, channels) -> result object:
//        pixels: Uint8Array/Uint8ClampedArray with width*height*channels bytes,
//        channels 3 (RGB8) or 4 (RGBA8, alpha dropped). On success:
//        {ok: true, cls: Float32Array, pooled: Float32Array,
//         patches: Float32Array, gridW, gridH, nPatches, hiddenSize}
//        On failure: {ok: false, status: <dino_status int>}
//        All returned typed arrays are copies owned by JS; they stay valid
//        across later encode() calls.
//   Module.freeModel(handle)                          -> void
//
// No C++ exceptions are thrown (emscripten builds may disable them); every
// failure is reported as a 0 handle, null, or {ok:false} result. Single-
// threaded by design (no -pthread): each encode() blocks the calling thread.
// Cosine similarity and friends live in JS, not here.

#include "dinov2.h"

#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace {

struct wasm_entry {
    dino_model *model = nullptr;
    dino_ctx   *ctx   = nullptr;
};

std::unordered_map<int, wasm_entry> g_entries;
int                                 g_next_handle = 1;

// Bulk-copy a JS ArrayBufferView into a fresh std::vector via
// Uint8Array.set() on a wasm-memory view: one memcpy, no per-element traffic.
std::vector<uint8_t> copy_view_bytes(const emscripten::val &view, size_t n) {
    std::vector<uint8_t> out(n);
    emscripten::val(emscripten::typed_memory_view(n, out.data())).call<void>("set", view);
    return out;
}

// Copy n floats into a JS-owned Float32Array (new Float32Array(view) copies).
emscripten::val f32_array(const float *data, size_t n) {
    return emscripten::val::global("Float32Array").new_(emscripten::val(emscripten::typed_memory_view(n, data)));
}

wasm_entry *find_entry(int handle) {
    const auto it = g_entries.find(handle);
    return it != g_entries.end() ? &it->second : nullptr;
}

emscripten::val error_result(dino_status status) {
    emscripten::val err = emscripten::val::object();
    err.set("ok", false);
    err.set("status", (int)status);
    return err;
}

std::string wasm_version() {
    return dino_version();
}

int wasm_load_model(const emscripten::val &bytes) {
    if (bytes.isNull() || bytes.isUndefined()) {
        return 0;
    }
    const size_t size = bytes["byteLength"].as<size_t>();
    if (size == 0) {
        return 0;
    }
    const std::vector<uint8_t> buffer = copy_view_bytes(bytes, size);

    dino_model_params mparams = dino_model_default_params();
    dino_model       *model   = dino_model_load_from_buffer(buffer.data(), buffer.size(), mparams);
    if (!model) {
        return 0;
    }

    // Single-threaded wasm v1: pin n_threads so the backend never tries to
    // fan out (ggml is built without a pthread pool here anyway).
    dino_ctx_params cparams = dino_ctx_default_params();
    cparams.n_threads       = 1;
    dino_ctx *ctx           = dino_init_from_model(model, cparams);
    if (!ctx) {
        dino_model_free(model);
        return 0;
    }

    const int handle = g_next_handle++;
    g_entries.emplace(handle, wasm_entry{model, ctx});
    return handle;
}

emscripten::val wasm_model_info(int handle) {
    const wasm_entry *e = find_entry(handle);
    if (!e) {
        return emscripten::val::null();
    }
    emscripten::val info = emscripten::val::object();
    info.set("hiddenSize", dino_model_hidden_size(e->model));
    info.set("patchSize", dino_model_patch_size(e->model));
    info.set("nRegisterTokens", dino_model_n_register_tokens(e->model));
    info.set("hasClassifier", dino_model_has_classifier(e->model));
    info.set("nClasses", dino_model_n_classes(e->model));
    return info;
}

emscripten::val wasm_encode(int handle, const emscripten::val &pixels, int width, int height, int channels) {
    wasm_entry *e = find_entry(handle);
    if (!e || width <= 0 || height <= 0 || (channels != 3 && channels != 4) || pixels.isNull() ||
        pixels.isUndefined()) {
        return error_result(DINO_STATUS_INVALID_ARGUMENT);
    }
    const size_t need = (size_t)width * height * (size_t)channels;
    if (pixels["byteLength"].as<size_t>() < need) {
        return error_result(DINO_STATUS_INVALID_ARGUMENT);
    }

    // The C API takes interleaved RGB8 only; drop alpha here when RGBA comes
    // in from canvas ImageData.
    std::vector<uint8_t> rgb((size_t)width * height * 3);
    if (channels == 3) {
        emscripten::val(emscripten::typed_memory_view(rgb.size(), rgb.data())).call<void>("set", pixels);
    } else {
        const std::vector<uint8_t> rgba = copy_view_bytes(pixels, need);
        for (size_t i = 0, n = (size_t)width * height; i < n; ++i) {
            rgb[i * 3 + 0] = rgba[i * 4 + 0];
            rgb[i * 3 + 1] = rgba[i * 4 + 1];
            rgb[i * 3 + 2] = rgba[i * 4 + 2];
        }
    }

    dino_image      image{rgb.data(), width, height, 0};
    dino_run_params run = dino_run_default_params();
    run.classify        = false; // feature mode; classifier heads are rare in the published GGUFs
    run.l2_normalize    = true;  // embeddings arrive unit-length, cosine = dot

    const dino_status status = dino_encode(e->ctx, &image, 1, run);
    if (status != DINO_STATUS_SUCCESS || dino_output_n_images(e->ctx) < 1) {
        return error_result(status);
    }

    const uint32_t hidden     = dino_model_hidden_size(e->model);
    int32_t        n_patches  = 0;
    int32_t        grid_w     = 0;
    int32_t        grid_h     = 0;
    const float   *cls        = dino_output_cls(e->ctx, 0);
    const float   *pooled     = dino_output_pooled(e->ctx, 0);
    const float   *patch_data = dino_output_patches(e->ctx, 0, &n_patches, &grid_w, &grid_h);

    emscripten::val result = emscripten::val::object();
    result.set("ok", true);
    result.set("cls", cls ? f32_array(cls, hidden) : emscripten::val::null());
    result.set("pooled", pooled ? f32_array(pooled, 2 * (size_t)hidden) : emscripten::val::null());
    result.set("patches", patch_data ? f32_array(patch_data, (size_t)n_patches * hidden) : emscripten::val::null());
    result.set("gridW", grid_w);
    result.set("gridH", grid_h);
    result.set("nPatches", n_patches);
    result.set("hiddenSize", (int)hidden);
    return result;
}

void wasm_free_model(int handle) {
    const auto it = g_entries.find(handle);
    if (it == g_entries.end()) {
        return;
    }
    if (it->second.ctx) {
        dino_free(it->second.ctx);
    }
    if (it->second.model) {
        dino_model_free(it->second.model);
    }
    g_entries.erase(it);
}

} // namespace

EMSCRIPTEN_BINDINGS(dinov2_wasm) {
    emscripten::function("version", &wasm_version);
    emscripten::function("loadModel", &wasm_load_model);
    emscripten::function("modelInfo", &wasm_model_info);
    emscripten::function("encode", &wasm_encode);
    emscripten::function("freeModel", &wasm_free_model);
}
