#define CRT_SECURE_NO_DEPRECATE // Disables ridiculous "unsafe" warnings on Windows

#include "dinov2.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "gguf.h"
#include "src/image.h"
#include <regex>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <cinttypes>
#include <algorithm>
#include <charconv>
#include <iostream>

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#if defined(_MSC_VER)
#pragma warning(disable : 4244 4267) // possible loss of data
#endif

#ifndef DINOV2_VERSION
#define DINOV2_VERSION "dev"
#endif

uint32_t dino_hparams::n_enc_head_dim() const {
    return hidden_size / num_attention_heads;
}

uint32_t dino_hparams::n_img_size() const {
    return img_size;
}

uint32_t dino_hparams::n_patch_size() const {
    return patch_size;
}

uint32_t dino_hparams::n_img_embd() const {
    return n_img_size() / n_patch_size();
}

uint32_t get_val_u32(const struct gguf_context *ctx, const char *key) {
    const int64_t key_id = gguf_find_key(ctx, key);
    assert(key_id >= 0);
    return gguf_get_val_u32(ctx, key_id);
}

const char *get_val_str(const struct gguf_context *ctx, const char *key) {
    const int64_t key_id = gguf_find_key(ctx, key);
    assert(key_id >= 0);
    return gguf_get_val_str(ctx, key_id);
}

static std::optional<uint32_t> get_val_u32_optional(const struct gguf_context *ctx, const char *key) {
    const int64_t key_id = gguf_find_key(ctx, key);
    if (key_id < 0) {
        return std::nullopt;
    }
    return gguf_get_val_u32(ctx, key_id);
}

//
// Helpers
//

// L2-normalize a contiguous span in place; zero vectors are left unchanged.
static void l2_normalize_span(float *data, size_t n) {
    double norm_sq = 0.0;
    for (size_t i = 0; i < n; ++i) {
        norm_sq += (double)data[i] * data[i];
    }
    if (norm_sq == 0.0) {
        return;
    }
    const float inv_norm = (float)(1.0 / std::sqrt(norm_sq));
    for (size_t i = 0; i < n; ++i) {
        data[i] *= inv_norm;
    }
}

void l2_normalize(std::vector<float> &v) {
    l2_normalize_span(v.data(), v.size());
}

ImageF dino_classify_preprocess(const Image &img, const dino_hparams &params) {
    (void)params;
    // HF BitImageProcessor recipe: shortest-edge 256 + center crop 224.
    return preprocess_resize_crop(img, 256, 224);
}

ImageF dino_preprocess(const Image &img, const dino_hparams &params) {
    const auto patch = static_cast<int>(params.patch_size);
    const auto new_w = ((img.nx + patch - 1) / patch) * patch;
    const auto new_h = ((img.ny + patch - 1) / patch) * patch;

    // 1) resize bicubic and 2) scale to [0,1] + channel-wise standardization (RGB)
    ImageF image = preprocess_for_dinov2(img, params.patch_size);
    if (image.nx != new_w || image.ny != new_h) {
        // fallback: exact resize to the expected multiple of patch_size
        Image resized = resize_bicubic(img, new_w, new_h);
        image.nx      = new_w;
        image.ny      = new_h;
        image.c       = 3;
        image.data.resize((size_t)new_w * new_h * 3);
        for (size_t i = 0; i < image.data.size(); i += 3) {
            image.data[i + 0] = (resized.data[i + 0] / 255.0f - IMAGENET_DEFAULT_MEAN[0]) / IMAGENET_DEFAULT_STD[0];
            image.data[i + 1] = (resized.data[i + 1] / 255.0f - IMAGENET_DEFAULT_MEAN[1]) / IMAGENET_DEFAULT_STD[1];
            image.data[i + 2] = (resized.data[i + 2] / 255.0f - IMAGENET_DEFAULT_MEAN[2]) / IMAGENET_DEFAULT_STD[2];
        }
    }
    return image;
}

ImageF dino_feature_preprocess(const Image &img, const dino_hparams &hparams, const dino_params &params) {
    switch (params.preprocess_mode) {
    case dino_preprocess_mode::hf:
        // same HF recipe as classification: shortest edge 256, center crop 224
        return preprocess_resize_crop(img, 256, 224);
    case dino_preprocess_mode::crop518:
        // fixed 518x518 grid; every input shares dims, so batches always match
        return preprocess_resize_crop(img, 518, 518);
    case dino_preprocess_mode::bounded:
    default: {
        // one bicubic resample straight to the bounded, patch-aligned dims
        const ImgSize target = dino_feature_output_size(img, hparams, params);
        return preprocess_resize_normalized(img, target.width, target.height);
    }
    }
}

ImgSize dino_feature_output_size(const Image &img, const dino_hparams &hparams, const dino_params &params) {
    switch (params.preprocess_mode) {
    case dino_preprocess_mode::hf:
        return {224, 224};
    case dino_preprocess_mode::crop518:
        return {518, 518};
    case dino_preprocess_mode::bounded:
    default: {
        int nx = img.nx;
        int ny = img.ny;
        if (!params.no_resize && std::min(nx, ny) > DINO_FEATURE_SHORT_EDGE) {
            const float scale = (float)DINO_FEATURE_SHORT_EDGE / (float)std::min(nx, ny);
            nx                = std::max((int)std::lround(nx * scale), 1);
            ny                = std::max((int)std::lround(ny * scale), 1);
        }
        const int p = (int)hparams.patch_size;
        return {((nx + p - 1) / p) * p, ((ny + p - 1) / p) * p};
    }
    }
}

std::vector<float> interpolate_pos_embed(const ImgSize       img_size,
                                         const float        *pos_embed_data, // Input data shouldn't be modified
                                         const dino_hparams &hparams) {
    // --- Calculate New Grid Dimensions ---
    const int     h_new           = img_size.height / (int)hparams.patch_size;
    const int     w_new           = img_size.width / (int)hparams.patch_size;
    const int64_t num_patches_new = (int64_t)h_new * w_new;

    // --- Calculate Original Grid Dimensions ---
    // The loader guarantees the stored table is a square [M*M + 1, hidden]
    // grid with M = img_size/patch_size (embeddings.position_embeddings shape
    // check), so the source grid side is n_img_embd().
    const int M         = (int)hparams.n_img_embd();
    const int hidden_sz = (int)hparams.hidden_size;

    // --- Early Return Check ---
    // Only an exact grid match may skip interpolation: an equal patch count
    // under a different aspect still needs resampling.
    if (h_new == M && w_new == M) {
        const size_t total_elements = (size_t)(M * M + 1) * hidden_sz;
        return {pos_embed_data, pos_embed_data + total_elements};
    }

    // --- Prepare Output Vector ---
    std::vector<float> pos_embed_new((size_t)(num_patches_new + 1) * hidden_sz);

    // --- Step 1: Copy CLS token embedding directly ---
    std::copy(pos_embed_data, pos_embed_data + hidden_sz, pos_embed_new.data());

    // --- Step 2: Interpolate Patch Embeddings ---
    // The patch rows are a [M*M, hidden] row-major block, which is exactly
    // the interleaved-channel layout resize_planes expects, so one call with
    // channels = hidden replaces the old per-channel gather/resize/scatter.
    // TODO: honor hparams.interpolation when GGUFs carry that key (the
    // converter does not write it today); only bicubic is implemented.
    std::vector<float> patches = resize_planes(pos_embed_data + hidden_sz, M, M, w_new, h_new, hidden_sz);
    std::copy(patches.begin(), patches.end(), pos_embed_new.data() + hidden_sz);

    return pos_embed_new;
}

// load the model's weights from a file following the ggml format(gguf)
bool dino_model_load(const ImgSize img_size, const std::string &fname, dino_model &model, const dino_params &params) {
    (void)img_size; // the graph derives its dims from the actual input, not the load-time hint
    fprintf(stderr, "%s: loading model from '%s' - please wait\n", __func__, fname.c_str());
#ifdef GGML_USE_CUDA
    fprintf(stderr, "%s: using CUDA backend\n", __func__);
    model.backend = ggml_backend_cuda_init(0); // init device 0
    if (!model.backend) {
        fprintf(stderr, "%s: ggml_backend_cuda_init() failed\n", __func__);
    }
#endif

#ifdef GGML_USE_METAL
    fprintf(stderr, "%s: using Metal backend\n", __func__);
    model.backend = ggml_backend_metal_init();
    if (!model.backend) {
        fprintf(stderr, "%s: ggml_backend_metal_init() failed\n", __func__);
    }
#endif

    // if there aren't GPU Backends fallback to CPU backend
    if (!model.backend) {
        model.backend = ggml_backend_cpu_init();
        if (!model.backend) {
            fprintf(stderr, "%s: ggml_backend_cpu_init() failed\n", __func__);
            return false;
        }
        ggml_backend_cpu_set_n_threads(model.backend, params.n_threads);
    }

    // Releases the gguf metadata context, the scratch tensor context and any
    // partially initialized model state when the load fails; on success the
    // guard is dismissed and ownership moves to `model`.
    struct load_guard {
        dino_model   &model;
        gguf_context *gguf_ctx = nullptr;
        ggml_context *tmp_ctx  = nullptr;
        bool          ok       = false;

        ~load_guard() {
            gguf_free(gguf_ctx);
            if (tmp_ctx) {
                ggml_free(tmp_ctx);
            }
            if (ok) {
                return;
            }
            model.tensors.clear();
            // same teardown order as the CLI's free_model
            if (model.ctx) {
                ggml_free(model.ctx);
                model.ctx = nullptr;
            }
            if (model.buffer) {
                ggml_backend_buffer_free(model.buffer);
                model.buffer = nullptr;
            }
            if (model.backend) {
                ggml_backend_free(model.backend);
                model.backend = nullptr;
            }
        }
    };
    load_guard guard{model};

    struct ggml_context    *tmp_ctx     = nullptr;
    struct gguf_init_params gguf_params = {
        /*.no_alloc   =*/false,
        /*.ctx        =*/&tmp_ctx,
    };
    gguf_context *gguf_ctx = gguf_init_from_file(fname.c_str(), gguf_params);
    guard.gguf_ctx         = gguf_ctx;
    guard.tmp_ctx          = tmp_ctx;
    if (!gguf_ctx) {
        fprintf(stderr, "%s: gguf_init_from_file() failed\n", __func__);
        return false;
    }

    // required metadata keys fail cleanly instead of aborting inside gguf
    const auto required_u32 = [&](const char *key, uint32_t &out) {
        const int64_t key_id = gguf_find_key(gguf_ctx, key);
        if (key_id < 0) {
            fprintf(stderr, "error: gguf missing required key '%s'\n", key);
            return false;
        }
        out = gguf_get_val_u32(gguf_ctx, key_id);
        return true;
    };

    // load hparams
    // override defaults
    auto &hparams = model.hparams;
    if (!required_u32("hidden_size", hparams.hidden_size) ||
        !required_u32("num_hidden_layers", hparams.num_hidden_layers) ||
        !required_u32("num_attention_heads", hparams.num_attention_heads) ||
        !required_u32("patch_size", hparams.patch_size) || !required_u32("img_size", hparams.img_size) ||
        !required_u32("ftype", hparams.ftype)) {
        return false;
    }

    // sanity-check the hparams before they drive any division or sizing math
    if (hparams.patch_size == 0 || hparams.img_size == 0 || hparams.img_size % hparams.patch_size != 0 ||
        hparams.num_attention_heads == 0 || hparams.hidden_size == 0 ||
        hparams.hidden_size % hparams.num_attention_heads != 0 || hparams.hidden_size % 4 != 0) {
        fprintf(stderr,
                "%s: invalid gguf hparams: hidden_size=%u num_attention_heads=%u patch_size=%u img_size=%u "
                "(requires patch_size > 0, img_size %% patch_size == 0, hidden_size %% num_attention_heads == 0, "
                "hidden_size %% 4 == 0)\n",
                __func__, hparams.hidden_size, hparams.num_attention_heads, hparams.patch_size, hparams.img_size);
        return false;
    }

    const auto num_register_tokens = get_val_u32_optional(gguf_ctx, "num_register_tokens");
    const bool has_register_tokens = ggml_get_tensor(tmp_ctx, "embeddings.register_tokens") != nullptr;
    if (has_register_tokens && !num_register_tokens) {
        fprintf(stderr, "%s: GGUF has embeddings.register_tokens but is missing num_register_tokens metadata\n",
                __func__);
        return false;
    }
    if (has_register_tokens != (num_register_tokens && *num_register_tokens > 0)) {
        fprintf(stderr, "%s: GGUF register-token metadata and embeddings.register_tokens tensor are inconsistent\n",
                __func__);
        return false;
    }
    // Backbone-only converters may omit this metadata because zero registers is
    // the ordinary DINOv2 default. Keep loading feature mode in that case.
    hparams.num_register_tokens = num_register_tokens.value_or(0);
    if (has_register_tokens && hparams.num_register_tokens > 0) {
        const ggml_tensor *register_tensor = ggml_get_tensor(tmp_ctx, "embeddings.register_tokens");
        const bool         shape_matches   = register_tensor->ne[0] == hparams.hidden_size &&
                                   register_tensor->ne[1] == hparams.num_register_tokens &&
                                   register_tensor->ne[2] == 1 && register_tensor->ne[3] == 1;
        if (!shape_matches) {
            fprintf(stderr,
                    "%s: embeddings.register_tokens has shape [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64
                    "] but expected "
                    "[%u, %u, 1, 1] (hidden_size, num_register_tokens, singleton trailing dimensions)\n",
                    __func__, register_tensor->ne[0], register_tensor->ne[1], register_tensor->ne[2],
                    register_tensor->ne[3], hparams.hidden_size, hparams.num_register_tokens);
            return false;
        }
    }

    const int32_t qntvr = hparams.ftype / GGML_QNT_VERSION_FACTOR;

    fprintf(stderr, "%s: hidden_size            = %u\n", __func__, hparams.hidden_size);
    fprintf(stderr, "%s: num_hidden_layers      = %u\n", __func__, hparams.num_hidden_layers);
    fprintf(stderr, "%s: num_register_tokens    = %u\n", __func__, hparams.num_register_tokens);
    fprintf(stderr, "%s: num_attention_heads    = %u\n", __func__, hparams.num_attention_heads);
    fprintf(stderr, "%s: patch_size             = %u\n", __func__, hparams.patch_size);
    fprintf(stderr, "%s: img_size               = %u\n", __func__, hparams.img_size);
    fprintf(stderr, "%s: ftype                  = %u\n", __func__, hparams.ftype);
    fprintf(stderr, "%s: qntvr                  = %d\n", __func__, qntvr);

    // num_classes is plain metadata; read it unconditionally so a model
    // loaded with classify=false still carries the real class count if a
    // later call enables classification (the default is 1000, which would
    // mis-size the probs read for any other count).
    const auto num_classes = get_val_u32_optional(gguf_ctx, "num_classes");
    if (num_classes && *num_classes > 0) {
        hparams.num_classes = *num_classes;
    }

    if (params.classify) {
        if (!num_classes || *num_classes == 0) {
            fprintf(stderr,
                    "%s: classification requested but GGUF has no non-zero num_classes metadata; "
                    "backbone-only models support feature mode only\n",
                    __func__);
            return false;
        }
        fprintf(stderr, "%s: num_classes            = %u\n", __func__, hparams.num_classes);

        const auto has_tensor = [&](const char *name) { return ggml_get_tensor(tmp_ctx, name) != nullptr; };
        if (!has_tensor("classifier.weight") || !has_tensor("classifier.bias")) {
            fprintf(stderr,
                    "%s: classification requested but GGUF is missing classifier.weight or classifier.bias; "
                    "backbone-only models support feature mode only\n",
                    __func__);
            return false;
        }

        // Read id2label dictionary into an ordered map. A classifier without
        // labels is not safe to present as a classification-capable model.
        for (uint32_t i = 0; i < hparams.num_classes; ++i) {
            const std::string key = std::to_string(i);
            if (gguf_find_key(gguf_ctx, key.c_str()) < 0) {
                fprintf(stderr, "%s: classification GGUF is missing label metadata for class %u\n", __func__, i);
                return false;
            }
            model.hparams.id2label[static_cast<int>(i)] = get_val_str(gguf_ctx, key.c_str());
        }
    }

    hparams.ftype %= GGML_QNT_VERSION_FACTOR;

    // fail fast on tensors the compute graph dereferences unconditionally,
    // rather than aborting on a map miss in build_graph
    const auto require_tensor = [&](const std::string &name) {
        if (ggml_get_tensor(tmp_ctx, name.c_str()) == nullptr) {
            fprintf(stderr, "error: gguf missing required tensor '%s'\n", name.c_str());
            return false;
        }
        return true;
    };
    for (const char *name :
         {"embeddings.cls_token", "embeddings.position_embeddings", "embeddings.patch_embeddings.projection.weight",
          "embeddings.patch_embeddings.projection.bias", "layernorm.weight", "layernorm.bias"}) {
        if (!require_tensor(name)) {
            return false;
        }
    }
    for (uint32_t il = 0; il < hparams.num_hidden_layers; ++il) {
        const std::string base = "encoder.layer." + std::to_string(il) + ".";
        for (const char *suffix :
             {"norm1.weight", "norm1.bias", "attention.attention.qkv.weight", "attention.attention.qkv.bias",
              "attention.output.dense.weight", "attention.output.dense.bias", "layer_scale1.lambda1", "norm2.weight",
              "norm2.bias", "layer_scale2.lambda1"}) {
            if (!require_tensor(base + suffix)) {
                return false;
            }
        }
        // the FFN variant is decided per layer by tensor presence (plain fc1/fc2
        // MLP or swiglu weights_in/weights_out)
        static const char *const mlp_suffixes[4] = {"mlp.fc1.weight", "mlp.fc1.bias", "mlp.fc2.weight", "mlp.fc2.bias"};
        static const char *const swiglu_suffixes[4] = {"mlp.weights_in.weight", "mlp.weights_in.bias",
                                                       "mlp.weights_out.weight", "mlp.weights_out.bias"};
        const bool         has_swiglu = ggml_get_tensor(tmp_ctx, (base + "mlp.weights_in.weight").c_str()) != nullptr;
        const char *const *mlp_list   = has_swiglu ? swiglu_suffixes : mlp_suffixes;
        for (int i = 0; i < 4; ++i) {
            if (!require_tensor(base + mlp_list[i])) {
                return false;
            }
        }
    }

    // the position table must be a square [M*M + 1, hidden] F32 grid where
    // M = img_size/patch_size; register tokens are stored separately. This is
    // also the invariant interpolate_pos_embed relies on to derive the source
    // grid (it only receives the data pointer, not the tensor).
    const int64_t      pos_rows   = (int64_t)hparams.n_img_embd() * hparams.n_img_embd() + 1;
    const ggml_tensor *pos_embeds = ggml_get_tensor(tmp_ctx, "embeddings.position_embeddings");
    if (pos_embeds->type != GGML_TYPE_F32 || pos_embeds->ne[0] != (int64_t)hparams.hidden_size ||
        pos_embeds->ne[1] != pos_rows || pos_embeds->ne[2] != 1 || pos_embeds->ne[3] != 1) {
        fprintf(stderr,
                "%s: embeddings.position_embeddings has shape [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64
                "] (%s) but expected [%u, %" PRId64 ", 1, 1] (f32; hidden_size, (img_size/patch_size)^2 + 1)\n",
                __func__, pos_embeds->ne[0], pos_embeds->ne[1], pos_embeds->ne[2], pos_embeds->ne[3],
                ggml_type_name(pos_embeds->type), hparams.hidden_size, pos_rows);
        return false;
    }

    const int num_tensors = gguf_get_n_tensors(gguf_ctx);

    struct ggml_init_params model_params = ggml_init_params{
        /*.mem_size   =*/ggml_tensor_overhead() * (size_t)num_tensors,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    model.ctx = ggml_init(model_params);
    if (!model.ctx) {
        fprintf(stderr, "%s: ggml_init() failed\n", __func__);
        return false;
    }
    for (int i = 0; i < num_tensors; i++) {
        const char         *name = gguf_get_tensor_name(gguf_ctx, i);
        struct ggml_tensor *src  = ggml_get_tensor(tmp_ctx, name);
        struct ggml_tensor *dst  = ggml_dup_tensor(model.ctx, src);
        ggml_set_name(dst, name);
        model.tensors[name] = dst;
        // std::cout << "i: " << i << ", name: " << name << ", type: " << ggml_type_name(dst->type) << std::endl;
    }

    model.buffer = ggml_backend_alloc_ctx_tensors(model.ctx, model.backend);
    if (!model.buffer) {
        fprintf(stderr, "%s: failed to allocate model buffer on the backend\n", __func__);
        return false;
    }
    // copy tensors from main memory to backend; tmp_ctx must stay alive until
    // every ggml_backend_tensor_set has read the source bytes
    for (struct ggml_tensor *cur = ggml_get_first_tensor(model.ctx); cur != nullptr;
         cur                     = ggml_get_next_tensor(model.ctx, cur)) {
        struct ggml_tensor *src    = ggml_get_tensor(tmp_ctx, ggml_get_name(cur));
        size_t              n_size = ggml_nbytes(src);
        ggml_backend_tensor_set(cur, ggml_get_data(src), 0, n_size);
    }

    guard.ok = true;
    return true;
}

// DINOv2 Encoder

struct ggml_tensor *attn(struct ggml_tensor *cur, const float scale, const int il, struct ggml_context *ctx_cgraph,
                         const dino_model &model, const dino_params &params) {
    const uint32_t num_attention_heads = model.hparams.num_attention_heads;
    const uint32_t n_enc_head_dim      = model.hparams.n_enc_head_dim();
    const uint32_t hidden_size         = model.hparams.hidden_size;
    const int64_t  W                   = cur->ne[1];
    const int64_t  H                   = cur->ne[2];
    const int64_t  total_patches       = W * H;

    // self-attention

    const std::string base_layer_name = "encoder.layer." + std::to_string(il);

    cur = ggml_mul_mat(ctx_cgraph, model.tensors.at(base_layer_name + ".attention.attention.qkv.weight"), cur);
    cur = ggml_add_inplace(ctx_cgraph, cur, model.tensors.at(base_layer_name + ".attention.attention.qkv.bias"));

    // split qkv into separate tensors
    const int B = cur->ne[3];

    cur = ggml_reshape_4d(ctx_cgraph, cur, hidden_size, 3, W * H, B);
    cur = ggml_cont(ctx_cgraph, ggml_permute(ctx_cgraph, cur, 0, 3, 1, 2));

    struct ggml_tensor *Q =
        ggml_view_3d(ctx_cgraph, cur, hidden_size, W * H, B, cur->nb[1], cur->nb[2], 0 * cur->nb[3]);
    Q = ggml_reshape_4d(ctx_cgraph, Q, n_enc_head_dim, num_attention_heads, W * H, B);
    Q = ggml_cont(ctx_cgraph, ggml_permute(ctx_cgraph, Q, 0, 2, 1, 3));

    struct ggml_tensor *K =
        ggml_view_3d(ctx_cgraph, cur, hidden_size, W * H, B, cur->nb[1], cur->nb[2], 1 * cur->nb[3]);
    K = ggml_reshape_4d(ctx_cgraph, K, n_enc_head_dim, num_attention_heads, W * H, B);
    K = ggml_cont(ctx_cgraph, ggml_permute(ctx_cgraph, K, 0, 2, 1, 3));

    struct ggml_tensor *V =
        ggml_view_3d(ctx_cgraph, cur, hidden_size, W * H, B, cur->nb[1], cur->nb[2], 2 * cur->nb[3]);
    V = ggml_reshape_4d(ctx_cgraph, V, n_enc_head_dim, num_attention_heads, W * H, B);

    // std::cout << "K type " << ggml_type_name(K->type) << std::endl;

    if (params.enable_flash_attn) {
        // Only the head dim (ne[0]) may need padding, and it is
        // n_enc_head_dim, not hidden_size. The KV seq dim needs no padding at
        // all: ggml_flash_attn_ext only requires ggml_can_mul_mat(k, q)
        // (ggml.c:5506), the CPU kernel iterates an arbitrary KV length (the
        // tiled path pads the KV tail with -inf internally, ops.cpp), and
        // Metal pads KV internally (flash_attn_ext_pad). Padding K/V with
        // zeros and no mask let each padded key vote exp(0) into the softmax
        // denominator while its zero V row diluted the numerator.
        const int64_t head_dim_to_pad = GGML_PAD(n_enc_head_dim, 4) - n_enc_head_dim;

        // Q-seq padding is kept: the extra output rows are trimmed below
        const int64_t total_patches_to_pad = GGML_PAD(total_patches, 32) - total_patches;

        V = ggml_cont(ctx_cgraph, ggml_permute(ctx_cgraph, V, 0, 2, 1, 3));

        if (head_dim_to_pad > 0 || total_patches_to_pad > 0) {
            Q = ggml_pad(ctx_cgraph, Q, (int)head_dim_to_pad, (int)total_patches_to_pad, 0, 0);
        }
        if (head_dim_to_pad > 0) {
            K = ggml_pad(ctx_cgraph, K, (int)head_dim_to_pad, 0, 0, 0);
            V = ggml_pad(ctx_cgraph, V, (int)head_dim_to_pad, 0, 0, 0);
        }

        const ggml_type dtype = model.tensors.at(base_layer_name + ".attention.attention.qkv.weight")->type;

        K = ggml_cast(ctx_cgraph, K, dtype);
        V = ggml_cast(ctx_cgraph, V, dtype);

        struct ggml_tensor *KQV = ggml_flash_attn_ext(ctx_cgraph, Q, K, V, nullptr, scale, 0.0f, 0.0f);
        // trim the padded head-dim entries (ne[0]) and the padded query rows
        // (ne[2]) the pads introduced
        KQV = ggml_view_4d(ctx_cgraph, KQV, n_enc_head_dim, KQV->ne[1], KQV->ne[2] - total_patches_to_pad, KQV->ne[3],
                           KQV->nb[1], KQV->nb[2], KQV->nb[3], 0);

        // the unpad view is non-contiguous across the batch dim when
        // total_patches_to_pad > 0, so materialize before the reshape
        cur = ggml_reshape_4d(ctx_cgraph, ggml_cont(ctx_cgraph, KQV), hidden_size, W, H, B);
    } else {
        // keep Q/K/V 4D: ggml_mul_mat batches over dims 2 (heads) and 3 (batch)
        // independently, which a merged (B * heads) dim 2 could not express
        V                      = ggml_cont(ctx_cgraph, ggml_permute(ctx_cgraph, V, 1, 2, 0, 3)); // transposed
        struct ggml_tensor *KQ = ggml_mul_mat(ctx_cgraph, K, Q);

        // attention weights
        struct ggml_tensor *KQ_soft_max = ggml_soft_max_ext(ctx_cgraph, KQ, nullptr, scale, 0.0f);

        struct ggml_tensor *KQV = ggml_mul_mat(ctx_cgraph, V, KQ_soft_max);

        cur = ggml_reshape_4d(ctx_cgraph, ggml_cont(ctx_cgraph, ggml_permute(ctx_cgraph, KQV, 0, 2, 1, 3)), hidden_size,
                              W, H, B);
    }

    cur = ggml_mul_mat(ctx_cgraph, model.tensors.at(base_layer_name + ".attention.output.dense.weight"), cur);
    cur = ggml_add_inplace(ctx_cgraph, cur, model.tensors.at(base_layer_name + ".attention.output.dense.bias"));

    return cur;
}

struct ggml_tensor *mlp(struct ggml_tensor *cur, const int il, struct ggml_context *ctx_cgraph, const dino_model &model,
                        const dino_params &params) {
    const std::string base_layer_name = "encoder.layer." + std::to_string(il);
    // fully connected layer
    cur = ggml_mul_mat(ctx_cgraph, model.tensors.at(base_layer_name + ".mlp.fc1.weight"), cur);
    cur = ggml_add_inplace(ctx_cgraph, cur, model.tensors.at(base_layer_name + ".mlp.fc1.bias"));

    // GELU activation
    cur = ggml_gelu(ctx_cgraph, cur);

    // projection
    cur = ggml_mul_mat(ctx_cgraph, model.tensors.at(base_layer_name + ".mlp.fc2.weight"), cur);
    cur = ggml_add_inplace(ctx_cgraph, cur, model.tensors.at(base_layer_name + ".mlp.fc2.bias"));
    return cur;
}

struct ggml_tensor *swiglu_ffn(struct ggml_tensor *cur, const int il, struct ggml_context *ctx_cgraph,
                               const dino_model &model, const dino_params &params) {
    const std::string base_layer_name = "encoder.layer." + std::to_string(il);
    // fully connected layer
    cur = ggml_mul_mat(ctx_cgraph, model.tensors.at(base_layer_name + ".mlp.weights_in.weight"), cur);
    cur = ggml_add_inplace(ctx_cgraph, cur, model.tensors.at(base_layer_name + ".mlp.weights_in.bias"));

    int64_t ne0    = cur->ne[0] / 2;
    int64_t ne1    = cur->ne[1];
    int64_t ne2    = cur->ne[2];
    int64_t ne3    = cur->ne[3];
    size_t  nb0    = cur->nb[0];
    size_t  nb1    = cur->nb[1];
    size_t  nb2    = cur->nb[2];
    size_t  nb3    = cur->nb[3];
    size_t  offset = nb0 * ne0;

    struct ggml_tensor *cur1 = ggml_view_4d(ctx_cgraph, cur, ne0, ne1, ne2, ne3, nb1, nb2, nb3, 0);

    struct ggml_tensor *cur2 = ggml_view_4d(ctx_cgraph, cur, ne0, ne1, ne2, ne3, nb1, nb2, nb3, offset);

    // SILU activation
    cur = ggml_mul_inplace(ctx_cgraph, ggml_silu_inplace(ctx_cgraph, ggml_cont(ctx_cgraph, cur1)), cur2);

    // projection
    cur = ggml_mul_mat(ctx_cgraph, model.tensors.at(base_layer_name + ".mlp.weights_out.weight"), cur);
    cur = ggml_add_inplace(ctx_cgraph, cur, model.tensors.at(base_layer_name + ".mlp.weights_out.bias"));
    return cur;
}

void forward_features(const ImgSize img_size, struct ggml_cgraph *graph, struct ggml_context *ctx_cgraph,
                      const dino_model &model, const dino_params &params) {
    const uint32_t hidden_size         = model.hparams.hidden_size;
    const uint32_t num_hidden_layers   = model.hparams.num_hidden_layers;
    const uint32_t n_enc_head_dim      = model.hparams.n_enc_head_dim();
    const uint32_t num_register_tokens = model.hparams.num_register_tokens;
    const int      h0                  = img_size.height / model.hparams.patch_size;
    const int      w0                  = img_size.width / model.hparams.patch_size;
    const int64_t  num_patches         = (int64_t)h0 * w0;
    const int64_t  n_batch             = params.n_batch;

    const float scale = 1.0f / sqrtf(static_cast<float>(n_enc_head_dim));
    // (W, H, C, B)
    // (518, 518, 3, n_batch)
    struct ggml_tensor *input =
        ggml_new_tensor_4d(ctx_cgraph, GGML_TYPE_F32, img_size.width, img_size.height, 3, n_batch);
    ggml_set_name(input, "input");

    // patch embedding
    // (37, 37, 768, 1)
    // std::cout << "patch embed " << enc.patch_embed_w->ne[0] << std::endl;
    struct ggml_tensor *cur =
        ggml_conv_2d_sk_p0(ctx_cgraph, model.tensors.at("embeddings.patch_embeddings.projection.weight"), input);

    // std::cout << ggml_type_name(tensor->type) << std::endl;

    cur = ggml_add_inplace(
        ctx_cgraph, ggml_repeat(ctx_cgraph, model.tensors.at("embeddings.patch_embeddings.projection.bias"), cur),
        cur); // (37, 37, 768, 1)

    cur = ggml_cont(ctx_cgraph, ggml_permute(ctx_cgraph, cur, 1, 2, 0, 3)); // (37, 768, 37, 1)
    //
    // std::cout << "cur shape " << cur->ne[0] << ", " << cur->ne[1] << ", " << cur->ne[2] << ", " << cur->ne[3]
    //         << std::endl;

    //
    // add positional embedding
    // cur dim     : 768  37  37  1
    // enc.pe dim  : 768  1370  1  1

    // std::cout << "cur shape " << cur->ne[0] << ", " << cur->ne[1] << ", " << cur->ne[2] << ", " << cur->ne[3]
    //         << std::endl;
    //
    // reshape patch embeddings from (768  37  37  B) to (768  1369  1  B)
    cur = ggml_reshape_4d(ctx_cgraph, cur, hidden_size, num_patches, 1, n_batch);

    // the positional table is shared across batch elements; ggml_add
    // broadcasts dim 3 (ne[3] == 1) over n_batch
    struct ggml_tensor *pos_embed_fixed =
        ggml_new_tensor_3d(ctx_cgraph, model.tensors.at("embeddings.position_embeddings")->type,
                           model.hparams.hidden_size, num_patches + 1, 1);

    ggml_set_name(pos_embed_fixed, "pos_embed_fixed");

    // prepend the CLS token to every batch element
    struct ggml_tensor *cls_token_b =
        ggml_repeat_4d(ctx_cgraph, model.tensors.at("embeddings.cls_token"), hidden_size, 1, 1, n_batch);
    cur = ggml_concat(ctx_cgraph, cls_token_b, cur, 1);

    cur = ggml_add_inplace(ctx_cgraph, cur, pos_embed_fixed);

    if (num_register_tokens > 0) {
        struct ggml_tensor *cls_token =
            ggml_view_4d(ctx_cgraph, cur, hidden_size, 1, 1, n_batch, cur->nb[1], cur->nb[2], cur->nb[3], 0);
        struct ggml_tensor *patch_tokens = ggml_view_4d(ctx_cgraph, cur, cur->ne[0], cur->ne[1] - 1, cur->ne[2],
                                                        cur->ne[3], cur->nb[1], cur->nb[2], cur->nb[3], cur->nb[1]);
        // register tokens sit between CLS and the patch tokens, repeated per batch element
        struct ggml_tensor *reg_tokens_b = ggml_repeat_4d(ctx_cgraph, model.tensors.at("embeddings.register_tokens"),
                                                          hidden_size, num_register_tokens, 1, n_batch);
        struct ggml_tensor *cls_reg      = ggml_concat(ctx_cgraph, cls_token, reg_tokens_b, 1);
        cur                              = ggml_concat(ctx_cgraph, cls_reg, patch_tokens, 1);
    }

    struct ggml_tensor *inpL = cur;
    //
    // loop over layers
    for (int il = 0; il < num_hidden_layers; ++il) {
        // norm 1
        {
            cur = ggml_norm(ctx_cgraph, inpL, model.hparams.eps);

            // cur = w * cur + b
            cur = ggml_mul_inplace(ctx_cgraph, cur,
                                   model.tensors.at("encoder.layer." + std::to_string(il) + ".norm1.weight"));
            cur = ggml_add_inplace(ctx_cgraph, cur,
                                   model.tensors.at("encoder.layer." + std::to_string(il) + ".norm1.bias"));
        }

        // std::cout << cur->ne[0] << ", " << cur->ne[1] << ", " << cur->ne[2] << ", " << cur->ne[3] << std::endl;

        // self attn
        cur = attn(cur, scale, il, ctx_cgraph, model, params);

        cur = ggml_mul_inplace(ctx_cgraph, cur,
                               model.tensors.at("encoder.layer." + std::to_string(il) + ".layer_scale1.lambda1"));

        // add skip connection
        cur = ggml_add_inplace(ctx_cgraph, cur, inpL);

        struct ggml_tensor *inpFF = cur;

        // feed-forward network
        {
            // norm 2
            {
                cur = ggml_norm(ctx_cgraph, inpFF, model.hparams.eps);

                // cur = w * cur + b
                cur = ggml_mul_inplace(ctx_cgraph, cur,
                                       model.tensors.at("encoder.layer." + std::to_string(il) + ".norm2.weight"));
                cur = ggml_add_inplace(ctx_cgraph, cur,
                                       model.tensors.at("encoder.layer." + std::to_string(il) + ".norm2.bias"));
            }

            // std::cout << "cur shape " << cur->ne[0] << ", " << cur->ne[1] << ", " << cur->ne[2] << ", " << cur->ne[3]
            //         << std::endl;
            //
            // std::cout << "mlp.fc1 size " << model.tensors.at("encoder.layer." + std::to_string(il) +
            // ".mlp.fc1.weight")
            //         ->ne[0] << ", "
            //         << model.tensors.at("encoder.layer." + std::to_string(il) + ".mlp.fc1.weight")->ne[1] << ", "
            //         << model.tensors.at("encoder.layer." + std::to_string(il) + ".mlp.fc1.weight")->ne[2] << ", "
            //         << model.tensors.at("encoder.layer." + std::to_string(il) + ".mlp.fc1.weight")->ne[3] <<
            //         std::endl;

            // the FFN variant is decided by tensor presence, not a layer-count
            // heuristic (the == 40 guess misclassifies any 40-layer non-SwiGLU
            // model and any non-40-layer SwiGLU one)
            if (model.tensors.count("encoder.layer." + std::to_string(il) + ".mlp.weights_in.weight") > 0) {
                cur = swiglu_ffn(cur, il, ctx_cgraph, model, params);
            } else {
                cur = mlp(cur, il, ctx_cgraph, model, params);
            }
            cur = ggml_mul_inplace(ctx_cgraph, cur,
                                   model.tensors.at("encoder.layer." + std::to_string(il) + ".layer_scale2.lambda1"));
        }

        inpL = ggml_add_inplace(ctx_cgraph, cur, inpFF);
    }

    cur = inpL;

    // layer normalization
    {
        cur = ggml_norm_inplace(ctx_cgraph, cur, model.hparams.eps);

        // cur = w * cur + b
        cur = ggml_mul_inplace(ctx_cgraph, cur, model.tensors.at("layernorm.weight"));
        cur = ggml_add_inplace(ctx_cgraph, cur, model.tensors.at("layernorm.bias"));
    }

    // get the output of cls token at index 0, for every batch element;
    // ggml_cont materializes the strided view so the output is a dense
    // (hidden, 1, 1, B) block, sliceable per image as b * hidden_size
    struct ggml_tensor *cls_token = ggml_cont(ctx_cgraph, ggml_view_4d(ctx_cgraph, cur, hidden_size, 1, 1, cur->ne[3],
                                                                       cur->nb[1], cur->nb[2], cur->nb[3], 0));

    ggml_set_output(cls_token);
    ggml_set_name(cls_token, "cls_token");
    ggml_build_forward_expand(graph, cls_token);

    int64_t ne1    = cur->ne[1] - 1;
    size_t  offset = cur->nb[1];
    // patch_tokens always excludes the cls + register tokens, in both feature
    // and classify mode; classification pooling matches HF, which pools over
    // patch tokens only (sequence_output[:, 1 + num_register_tokens:])
    ne1 -= num_register_tokens;
    offset *= (num_register_tokens + 1);

    // same cont treatment as cls_token: dense (hidden, n_patches, 1, B)
    struct ggml_tensor *patch_tokens =
        ggml_cont(ctx_cgraph, ggml_view_4d(ctx_cgraph, cur, cur->ne[0], ne1, cur->ne[2], cur->ne[3], cur->nb[1],
                                           cur->nb[2], cur->nb[3], offset));

    ggml_set_output(patch_tokens);
    ggml_set_name(patch_tokens, "patch_tokens");
    ggml_build_forward_expand(graph, patch_tokens);
}

void forward_head(const ImgSize img_size, struct ggml_cgraph *graph, struct ggml_context *ctx_cgraph,
                  const dino_model &model, const dino_params &params) {
    struct ggml_tensor *cls_token    = ggml_graph_get_tensor(graph, "cls_token");
    struct ggml_tensor *patch_tokens = ggml_graph_get_tensor(graph, "patch_tokens");
    // classification head

    struct ggml_tensor *pooled_patch_tokens =
        ggml_sum_rows(ctx_cgraph, ggml_cont(ctx_cgraph, ggml_permute(ctx_cgraph, patch_tokens, 1, 0, 2, 3)));
    // divide by the actual pooled token count, not the model's native grid
    // (they differ when the input isn't the GGUF-declared img_size)
    pooled_patch_tokens =
        ggml_scale_inplace(ctx_cgraph, pooled_patch_tokens, 1.0f / static_cast<float>(patch_tokens->ne[1]));

    struct ggml_tensor *cur =
        ggml_concat(ctx_cgraph, cls_token, ggml_permute(ctx_cgraph, pooled_patch_tokens, 1, 0, 2, 3), 0);

    // projection
    cur = ggml_mul_mat(ctx_cgraph, model.tensors.at("classifier.weight"), cur);
    cur = ggml_add_inplace(ctx_cgraph, cur, model.tensors.at("classifier.bias"));

    // softmax
    ggml_tensor *probs = ggml_soft_max(ctx_cgraph, cur);
    //
    ggml_set_output(probs);
    ggml_set_name(probs, "probs");

    ggml_build_forward_expand(graph, probs);
}

struct ggml_cgraph *build_graph(const ImgSize img_size, struct ggml_context *ctx_cgraph, const dino_model &model,
                                const dino_params &params, const size_t graph_size) {
    const auto &hparams = model.hparams;

    // a 40-layer model emits ~2k nodes, past GGML_DEFAULT_GRAPH_SIZE (2048)
    struct ggml_cgraph *gf = ggml_new_graph_custom(ctx_cgraph, graph_size, false);

    forward_features(img_size, gf, ctx_cgraph, model, params);

    if (params.classify) {
        forward_head(img_size, gf, ctx_cgraph, model, params);
    }

    return gf;
}

bool dino_batch_size_valid(int64_t n) {
    return n >= 1 && n <= (int64_t)DINO_MAX_BATCH;
}

bool write_embeddings_binary(const std::string &path, const dino_output &output, uint32_t hidden_size,
                             uint32_t patch_count, uint32_t grid_w, uint32_t grid_h, bool include_patches,
                             bool normalized, std::string &error) {
    if (!output.cls_token || output.cls_token->size() != hidden_size) {
        error = "CLS vector has an unexpected length";
        return false;
    }
    if (!output.pooled || output.pooled->size() != static_cast<size_t>(2) * hidden_size) {
        error = "pooled vector has an unexpected length";
        return false;
    }
    if (include_patches &&
        (!output.patch_tokens || output.patch_tokens->size() != static_cast<size_t>(patch_count) * hidden_size)) {
        error = "patch vectors have an unexpected length";
        return false;
    }

    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) {
        error = "cannot open output file";
        return false;
    }

    const uint32_t flags     = (include_patches ? 1u : 0u) | (normalized ? 2u : 0u);
    auto           write_u16 = [&](uint16_t value) {
        const unsigned char bytes[2] = {static_cast<unsigned char>(value & 0xffu),
                                        static_cast<unsigned char>((value >> 8) & 0xffu)};
        file.write(reinterpret_cast<const char *>(bytes), sizeof(bytes));
    };
    auto write_u32 = [&](uint32_t value) {
        const unsigned char bytes[4] = {
            static_cast<unsigned char>(value & 0xffu), static_cast<unsigned char>((value >> 8) & 0xffu),
            static_cast<unsigned char>((value >> 16) & 0xffu), static_cast<unsigned char>((value >> 24) & 0xffu)};
        file.write(reinterpret_cast<const char *>(bytes), sizeof(bytes));
    };
    auto write_float = [&](float value) {
        uint32_t bits = 0;
        static_assert(sizeof(bits) == sizeof(value));
        std::memcpy(&bits, &value, sizeof(bits));
        write_u32(bits);
    };

    const char magic[8] = {'D', '2', 'E', 'M', 'B', '\0', '\0', '\0'};
    file.write(magic, sizeof(magic));
    write_u16(2);
    write_u16(40);
    write_u32(hidden_size);
    write_u32(2 * hidden_size);
    write_u32(include_patches ? patch_count : 0);
    write_u32(flags);
    write_u32(0);
    write_u32(include_patches ? grid_w : 0);
    write_u32(include_patches ? grid_h : 0);

    for (float value : *output.cls_token) {
        write_float(value);
    }
    for (float value : *output.pooled) {
        write_float(value);
    }
    if (include_patches) {
        for (float value : *output.patch_tokens) {
            write_float(value);
        }
    }

    if (!file) {
        error = "write failed";
        return false;
    }
    file.flush();
    if (!file) {
        error = "flush failed";
        return false;
    }
    file.close();
    if (!file) {
        error = "close failed";
        return false;
    }
    return true;
}

void print_usage(FILE *out, int argc, char **argv, const dino_params &params) {
    fprintf(out, "usage: %s [options]\n", argv[0]);
    fprintf(out, "\n");
    fprintf(out, "Model:\n");
    fprintf(out, "  -m FNAME, --model     model path (default: %s)\n", params.model.c_str());
    fprintf(out, "  -fa, --flash_attn     enable flash attention, less accurate (default: off)\n");
    fprintf(out, "  -t N, --threads       number of threads to use during computation, 1 or greater (default: %u)\n",
            params.n_threads);
    fprintf(out, "\n");
    fprintf(out, "Input:\n");
    fprintf(out, "  -i FNAME, --inp       input image file; repeat or comma-separate for several\n");
    fprintf(out, "                        (default: %s)\n",
            params.fnames_inp.empty() ? "" : params.fnames_inp.front().c_str());
    fprintf(out, "  -s N, --seed          accepted for compatibility; has no effect (default: %d)\n", params.seed);
    fprintf(out, "  --batch N             max images per forward pass; inputs run in chunks of N\n");
    fprintf(out, "                        (default: %u, max: %u)\n", params.n_batch, DINO_MAX_BATCH);
    fprintf(out, "\n");
    fprintf(out, "Preprocessing (feature mode only; rejected with -c):\n");
    fprintf(out, "  --preprocess MODE     bounded (default): resize shortest edge to %d when larger;\n",
            DINO_FEATURE_SHORT_EDGE);
    fprintf(out, "                        hf: shortest edge 256 + center crop 224 (HF recipe);\n");
    fprintf(out, "                        crop518: shortest edge 518 + center crop 518 (fixed 37x37 grid)\n");
    fprintf(out, "  --no-resize           bounded mode: keep native resolution (still capped)\n");
    fprintf(out, "  --max-tokens N        hard cap on patch tokens per image, 0 disables\n");
    fprintf(out, "                        (default: 4 * (%d/patch)^2 from the model's patch size)\n",
            DINO_FEATURE_SHORT_EDGE);
    fprintf(out, "\n");
    fprintf(out, "Output modes:\n");
    fprintf(out, "  -c, --classify        classify each input image and print top-k labels (default: off)\n");
    fprintf(out, "  -k N, --topk          top k classes to print, 1 through model class count (default: %u)\n",
            params.topk);
    fprintf(out, "  --print-embeddings    emit embeddings JSON on stdout, one object per input image (JSONL)\n");
    fprintf(out, "  --embeddings-binary   write preview binary embeddings to -o (unstable format)\n");
    fprintf(out, "  --print-patch-tokens  include per-patch token vectors in the embedding output\n");
    fprintf(out, "  --l2-normalize        L2-normalize emitted embedding vectors\n");
    fprintf(out, "  -o FNAME, --out       write PCA output to FNAME, or binary embeddings file/directory\n");
    fprintf(out, "                        output when used with --embeddings-binary\n");
    fprintf(out, "\n");
    fprintf(out, "Benchmark:\n");
    fprintf(out, "  --bench               enable bench loop (default repeats=5, warmup=1); skips PCA image output\n");
    fprintf(out,
            "  --bench-runs N        number of timed runs, 1 or greater (overrides default 5 when --bench is set)\n");
    fprintf(out, "  --bench-warmup N      number of warmup runs, 0 or greater (default: %u)\n", params.bench_warmup);
    fprintf(out, "  --bench-json          emit one JSON object per line to stdout instead of markdown row\n");
    fprintf(out, "\n");
    fprintf(out, "Misc:\n");
    fprintf(out, "  -h, --help            show this help message and exit\n");
    fprintf(out, "  --version             print version and exit\n");
    fprintf(out, "\n");
    fprintf(out, "Workflows:\n");
    fprintf(out, "  dinov2-cli -m model.gguf -i img.jpg -c                        # classify: top-k labels\n");
    fprintf(out, "  dinov2-cli -m model.gguf -i img.jpg --print-embeddings        # embeddings JSON on stdout\n");
    fprintf(out, "  dinov2-cli -m model.gguf -i img.jpg --print-embeddings --print-patch-tokens\n");
    fprintf(out, "                                                                # + per-patch tokens\n");
    fprintf(out, "  dinov2-cli -m model.gguf -i a.jpg -i b.jpg --batch 2 --print-embeddings\n");
    fprintf(out, "                                                                # batch: one JSON line per image\n");
    fprintf(out, "  dinov2-cli -m model.gguf -i img.jpg -o pca.png                # PCA viz of patch features\n");
    fprintf(out, "  dinov2-cli -m model.gguf -i img.jpg --bench --bench-json      # benchmark, JSON lines\n");
    fprintf(out, "\n");
    fprintf(out, "docs: https://raw.githubusercontent.com/espetro/dinov2.cpp/main/docs/cli.md\n");
}

// Append a comma-separated list of paths to out; tokens are whitespace-trimmed
// and empty tokens are dropped (same convention as parity_check.py's --image).
static void append_csv_paths(std::vector<std::string> &out, const std::string &value) {
    size_t pos = 0;
    while (pos <= value.size()) {
        const size_t      comma = value.find(',', pos);
        const std::string tok   = value.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        const size_t      first = tok.find_first_not_of(" \t\r\n");
        const size_t      last  = tok.find_last_not_of(" \t\r\n");
        if (first != std::string::npos) {
            out.push_back(tok.substr(first, last - first + 1));
        }
        if (comma == std::string::npos) {
            break;
        }
        pos = comma + 1;
    }
}

template <typename T> static bool parse_integer(const char *value, T &result) {
    const char *first = value;
    if (*first == '+') {
        ++first;
        if (*first == '+' || *first == '-') {
            return false;
        }
    } else if (*first == '-' && (first[1] == '+' || first[1] == '-')) {
        return false;
    }
    if (*first == '\0') {
        return false;
    }

    const char *last = value + std::strlen(value);
    T           parsed{};
    const auto  conversion = std::from_chars(first, last, parsed, 10);
    if (conversion.ec != std::errc() || conversion.ptr != last) {
        return false;
    }
    result = parsed;
    return true;
}

[[noreturn]] static void numeric_parse_error(const char *option, const char *value, const char *range, int argc,
                                             char **argv, const dino_params &params) {
    fprintf(stderr, "error: %s has invalid value '%s' (expected %s)\n", option, value, range);
    print_usage(stderr, argc, argv, params);
    exit(1);
}

bool dino_params_parse(int argc, char **argv, dino_params &params) {
    // consume argv[++i] as the value for a flag; a trailing flag with no
    // value is a usage error, not a read past argv[argc - 1]
    auto next_value = [&](int &i) -> const char * {
        if (i + 1 >= argc) {
            fprintf(stderr, "error: %s requires a value\n", argv[i]);
            print_usage(stderr, argc, argv, params);
            exit(1);
        }
        return argv[++i];
    };

    // the first -i replaces the default image; later -i flags append
    bool first_inp      = true;
    bool preprocess_set = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-s" || arg == "--seed") {
            const char *value = next_value(i);
            int32_t     parsed;
            if (!parse_integer(value, parsed)) {
                numeric_parse_error(arg.c_str(), value, "a signed 32-bit integer", argc, argv, params);
            }
            params.seed = parsed;
        } else if (arg == "-m" || arg == "--model") {
            params.model = next_value(i);
        } else if (arg == "-i" || arg == "--inp") {
            if (first_inp) {
                params.fnames_inp.clear();
                first_inp = false;
            }
            append_csv_paths(params.fnames_inp, next_value(i));
        } else if (arg == "--batch") {
            const char *value = next_value(i);
            int32_t     parsed;
            if (!parse_integer(value, parsed) || !dino_batch_size_valid(parsed)) {
                const std::string range = "an integer from 1 through " + std::to_string(DINO_MAX_BATCH);
                numeric_parse_error(arg.c_str(), value, range.c_str(), argc, argv, params);
            }
            params.n_batch = static_cast<uint32_t>(parsed);
        } else if (arg == "-o" || arg == "--out") {
            params.image_out = next_value(i);
        } else if (arg == "-t" || arg == "--threads") {
            const char *value = next_value(i);
            int32_t     parsed;
            if (!parse_integer(value, parsed) || parsed <= 0) {
                numeric_parse_error(arg.c_str(), value, "a positive 32-bit integer", argc, argv, params);
            }
            params.n_threads = static_cast<uint32_t>(parsed);
        } else if (arg == "-k" || arg == "--topk") {
            const char *value = next_value(i);
            int32_t     parsed;
            if (!parse_integer(value, parsed) || parsed <= 0) {
                numeric_parse_error(arg.c_str(), value, "a positive 32-bit integer", argc, argv, params);
            }
            params.topk = static_cast<uint32_t>(parsed);
        } else if (arg == "-fa" || arg == "--flash_attn") {
            params.enable_flash_attn = true;
        } else if (arg == "-c" || arg == "--classify") {
            params.classify = true;
        } else if (arg == "--bench") {
            // --bench alone: enable bench loop with the default repeat count (5).
            // --bench-runs N below overrides this if the user supplies a count.
            if (params.bench_repeats == 0) {
                params.bench_repeats = 5;
            }
        } else if (arg == "--bench-runs") {
            const char *value = next_value(i);
            int32_t     parsed;
            if (!parse_integer(value, parsed) || parsed <= 0) {
                numeric_parse_error(arg.c_str(), value, "a positive 32-bit integer", argc, argv, params);
            }
            params.bench_repeats = static_cast<uint32_t>(parsed);
        } else if (arg == "--bench-warmup") {
            const char *value = next_value(i);
            int32_t     parsed;
            if (!parse_integer(value, parsed) || parsed < 0) {
                numeric_parse_error(arg.c_str(), value, "a non-negative 32-bit integer", argc, argv, params);
            }
            params.bench_warmup = static_cast<uint32_t>(parsed);
        } else if (arg == "--bench-json") {
            params.bench_json = true;
        } else if (arg == "--print-embeddings") {
            params.print_embeddings = true;
        } else if (arg == "--embeddings-binary") {
            params.embeddings_binary = true;
        } else if (arg == "--print-patch-tokens") {
            params.print_patch_tokens = true;
        } else if (arg == "--l2-normalize") {
            params.l2_normalize = true;
        } else if (arg == "--preprocess") {
            const std::string value = next_value(i);
            if (value == "bounded") {
                params.preprocess_mode = dino_preprocess_mode::bounded;
            } else if (value == "hf") {
                params.preprocess_mode = dino_preprocess_mode::hf;
            } else if (value == "crop518") {
                params.preprocess_mode = dino_preprocess_mode::crop518;
            } else {
                fprintf(stderr, "error: %s has invalid value '%s' (expected one of: bounded, hf, crop518)\n",
                        arg.c_str(), value.c_str());
                print_usage(stderr, argc, argv, params);
                exit(1);
            }
            preprocess_set = true;
        } else if (arg == "--no-resize") {
            params.no_resize = true;
        } else if (arg == "--max-tokens") {
            const char *value = next_value(i);
            int32_t     parsed;
            if (!parse_integer(value, parsed) || parsed < 0) {
                numeric_parse_error(arg.c_str(), value, "a non-negative 32-bit integer", argc, argv, params);
            }
            params.max_tokens = parsed;
        } else if (arg == "--version") {
            fprintf(stdout, "dinov2-cli %s\n", DINOV2_VERSION);
            exit(0);
        } else if (arg == "-h" || arg == "--help") {
            print_usage(stdout, argc, argv, params);
            exit(0);
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            print_usage(stderr, argc, argv, params);
            exit(1);
        }
    }

    if (params.embeddings_binary && params.image_out.empty()) {
        fprintf(stderr, "error: --embeddings-binary requires -o PATH\n");
        print_usage(stderr, argc, argv, params);
        exit(1);
    }
    if (params.embeddings_binary && params.classify) {
        fprintf(stderr, "error: --embeddings-binary cannot be combined with --classify\n");
        print_usage(stderr, argc, argv, params);
        exit(1);
    }
    if (params.embeddings_binary && params.bench_repeats != 0) {
        fprintf(stderr, "error: --embeddings-binary cannot be combined with --bench\n");
        print_usage(stderr, argc, argv, params);
        exit(1);
    }
    if (params.no_resize && params.preprocess_mode != dino_preprocess_mode::bounded) {
        fprintf(stderr, "error: --no-resize only applies to --preprocess bounded\n");
        print_usage(stderr, argc, argv, params);
        exit(1);
    }
    if (params.classify && !params.image_out.empty()) {
        fprintf(stderr, "error: -o/--out writes feature-mode output (PCA or binary) and cannot be combined with -c\n");
        print_usage(stderr, argc, argv, params);
        exit(1);
    }
    if (params.classify && (preprocess_set || params.no_resize)) {
        fprintf(stderr, "error: --preprocess/--no-resize are feature-mode flags and cannot be combined with -c\n");
        print_usage(stderr, argc, argv, params);
        exit(1);
    }

    return true;
}

std::vector<dino_output> dino_predict(const dino_model &model, const std::vector<ImageF> &imgs,
                                      const dino_params &params, ggml_gallocr_t allocr) {
    if (imgs.empty()) {
        fprintf(stderr, "%s: no input images\n", __func__);
        return {};
    }
    if (imgs.size() > params.n_batch) {
        fprintf(stderr, "%s: %zu images exceed n_batch = %u\n", __func__, imgs.size(), params.n_batch);
        return {};
    }
    // a single graph is built for the whole batch, so every image must share
    // the same dimensions (preprocessing decides the graph's input size)
    const int nx = imgs[0].nx;
    const int ny = imgs[0].ny;
    for (const ImageF &img : imgs) {
        if (img.nx != nx || img.ny != ny) {
            fprintf(stderr, "%s: batch images must share dimensions (%dx%d vs %dx%d)\n", __func__, img.nx, img.ny, nx,
                    ny);
            return {};
        }
        // the planar deinterleave below indexes data[i * 3 + c]
        if (img.c != 3 || img.data.size() != (size_t)img.nx * img.ny * 3) {
            fprintf(stderr, "%s: expected a 3-channel float image of %dx%d (%zu values), got c=%d, %zu values\n",
                    __func__, img.nx, img.ny, (size_t)img.nx * img.ny * 3, img.c, img.data.size());
            return {};
        }
    }

    // the graph batch dimension is the number of images actually provided
    dino_params batch_params  = params;
    batch_params.n_batch      = (uint32_t)imgs.size();
    const size_t  n_batch     = imgs.size();
    const size_t  hidden_size = model.hparams.hidden_size;
    const size_t  npix        = (size_t)nx * ny;
    const int64_t num_patches = (int64_t)(ny / (int)model.hparams.patch_size) * (nx / (int)model.hparams.patch_size);

    // graph size derived from the layer count: each encoder layer emits
    // ~35-45 nodes (a bit more on the flash path), plus ~50 fixed nodes for
    // patch embedding, the token glue, and the output heads. 64 per layer
    // leaves comfortable headroom without the fixed 8192-node pool. The same
    // value sizes the ctx_cgraph tensor pool and the cgraph node capacity;
    // a 40-layer model exceeds GGML_DEFAULT_GRAPH_SIZE (2048), so both must
    // use the custom-size ggml entry points.
    const size_t graph_size = (size_t)model.hparams.num_hidden_layers * 64 + 128;

    struct ggml_init_params params0 = {
        /*.mem_size   =*/ggml_tensor_overhead() * graph_size + ggml_graph_overhead_custom(graph_size, false),
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true, // the tensors will be allocated later by ggml_gallocr_alloc_graph()
    };
    struct ggml_context *ctx_cgraph = ggml_init(params0);
    if (!ctx_cgraph) {
        fprintf(stderr, "%s: ggml_init() failed\n", __func__);
        return {};
    }
    struct ggml_cgraph *gf = build_graph({nx, ny}, ctx_cgraph, model, batch_params, graph_size);

    if (!ggml_gallocr_alloc_graph(allocr, gf)) {
        fprintf(stderr,
                "%s: failed to allocate compute graph for a %d x %d input (%lld patch tokens); "
                "reduce input size or use --preprocess crop518 / --max-tokens\n",
                __func__, nx, ny, (long long)num_patches);
        ggml_free(ctx_cgraph);
        return {};
    }

    struct ggml_tensor *input = ggml_graph_get_tensor(gf, "input");

    // Convert interleaved RGB to planar RGB layout (image is already RGB,
    // no BGR swap needed), one 3*npix block per batch element
    std::vector<float> planar(npix * 3 * n_batch);
    for (size_t b = 0; b < n_batch; ++b) {
        float *dst = planar.data() + b * npix * 3;
        for (size_t i = 0; i < npix; ++i) {
            dst[0 * npix + i] = imgs[b].data[i * 3 + 0]; // R
            dst[1 * npix + i] = imgs[b].data[i * 3 + 1]; // G
            dst[2 * npix + i] = imgs[b].data[i * 3 + 2]; // B
        }
    }

    ggml_backend_tensor_set(input, planar.data(), 0, ggml_nbytes(input));

    const struct ggml_tensor *pos_embed = ggml_get_tensor(model.ctx, "embeddings.position_embeddings");

    // read the table through the backend: ->data is a device pointer on CUDA
    // and only valid for CPU/Metal buffers. One tensor_get per predict keeps
    // this a local change; caching an f32 copy in dino_model would avoid the
    // per-call copy but needs a new public member (PR B territory).
    std::vector<float> pos_embed_host(ggml_nelements(pos_embed));
    ggml_backend_tensor_get(pos_embed, pos_embed_host.data(), 0, ggml_nbytes(pos_embed));

    const std::vector<float> pos_embed_fixed_data =
        interpolate_pos_embed({nx, ny}, pos_embed_host.data(), model.hparams);

    struct ggml_tensor *pos_embed_fixed = ggml_graph_get_tensor(gf, "pos_embed_fixed");

    ggml_backend_tensor_set(pos_embed_fixed, pos_embed_fixed_data.data(), 0, ggml_nbytes(pos_embed_fixed));

    if (ggml_backend_graph_compute(model.backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "%s: ggml_backend_graph_compute() failed\n", __func__);
        ggml_free(ctx_cgraph);
        return {};
    }

    std::vector<dino_output> outputs(n_batch);

    // cls_token is marked as an output unconditionally by forward_features;
    // read it in both classify and feature modes. It is a dense
    // (hidden, 1, 1, B) block: image b's vector starts at b * hidden_size.
    // ggml_backend_tensor_get is backend-agnostic; ->data/ggml_get_data_f32
    // would be device pointers on GPU backends.
    std::vector<float> cls_buf((size_t)hidden_size * n_batch);
    ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "cls_token"), cls_buf.data(), 0, cls_buf.size() * sizeof(float));
    const float *cls_data = cls_buf.data();

    if (params.classify) {
        // probs is a dense (num_classes, 1, 1, B) block
        std::vector<float> probs_buf((size_t)model.hparams.num_classes * n_batch);
        ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "probs"), probs_buf.data(), 0,
                                probs_buf.size() * sizeof(float));
        const float *probs_data = probs_buf.data();
        for (size_t b = 0; b < n_batch; ++b) {
            dino_output &output = outputs[b];
            output.cls_token    = std::vector<float>(cls_data + b * hidden_size, cls_data + (b + 1) * hidden_size);

            const float                       *img_probs = probs_data + b * model.hparams.num_classes;
            std::vector<std::pair<float, int>> predictions;
            // store probability and index
            for (int i = 0; i < model.hparams.num_classes; ++i) {
                predictions.emplace_back(img_probs[i], i);
            }

            // sort in descending order
            std::sort(predictions.begin(), predictions.end(),
                      [](const std::pair<float, int> &a, const std::pair<float, int> &b) { return a.first > b.first; });

            // top k predictions: class indices in preds, probabilities in pred_scores.
            // Label printing is left to the caller (dino_predict must not write to stdout).
            const uint32_t        topk = std::min(params.topk, (uint32_t)predictions.size());
            std::vector<uint32_t> preds(topk);
            std::vector<float>    scores(topk);
            for (uint32_t i = 0; i < topk; ++i) {
                preds[i]  = static_cast<uint32_t>(predictions[i].second);
                scores[i] = predictions[i].first;
            }

            output.preds       = std::move(preds);
            output.pred_scores = std::move(scores);
        }
    } else {
        // patch_tokens is a dense (hidden, num_patches, 1, B) block; each
        // image's region is a contiguous num_patches * hidden_size slice
        std::vector<float> patch_tokens_buf((size_t)num_patches * hidden_size * n_batch);
        ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "patch_tokens"), patch_tokens_buf.data(), 0,
                                patch_tokens_buf.size() * sizeof(float));
        const float *patch_tokens_data = patch_tokens_buf.data();
        for (size_t b = 0; b < n_batch; ++b) {
            dino_output &output = outputs[b];
            output.cls_token    = std::vector<float>(cls_data + b * hidden_size, cls_data + (b + 1) * hidden_size);

            const float *img_patches = patch_tokens_data + b * (size_t)num_patches * hidden_size;
            output.patch_tokens      = std::vector<float>(img_patches, img_patches + (size_t)num_patches * hidden_size);

            // pooled = [cls_token || mean(patch_tokens)], 2*hidden floats, cls first
            std::vector<float> pooled(2 * hidden_size, 0.0f);
            std::copy(output.cls_token->begin(), output.cls_token->end(), pooled.begin());
            float *mean = pooled.data() + hidden_size;
            for (int64_t p = 0; p < num_patches; ++p) {
                const float *row = img_patches + (size_t)p * hidden_size;
                for (size_t d = 0; d < hidden_size; ++d) {
                    mean[d] += row[d];
                }
            }
            for (size_t d = 0; d < hidden_size; ++d) {
                mean[d] /= (float)num_patches;
            }
            output.pooled = std::move(pooled);
        }
    }

    if (params.l2_normalize) {
        for (dino_output &output : outputs) {
            if (output.cls_token) {
                l2_normalize(*output.cls_token);
            }
            if (output.pooled) {
                l2_normalize(*output.pooled);
            }
            if (output.patch_tokens) {
                // normalize each patch row independently
                const size_t n_rows = output.patch_tokens->size() / hidden_size;
                for (size_t r = 0; r < n_rows; ++r) {
                    l2_normalize_span(output.patch_tokens->data() + r * hidden_size, hidden_size);
                }
            }
        }
    }

    // free memory
    ggml_free(ctx_cgraph);

    return outputs;
}

std::unique_ptr<dino_output> dino_predict(const dino_model &model, const ImageF &img, const dino_params &params,
                                          ggml_gallocr_t allocr) {
    std::vector<dino_output> outputs = dino_predict(model, std::vector<ImageF>{img}, params, allocr);
    if (outputs.empty()) {
        return nullptr;
    }
    return std::make_unique<dino_output>(std::move(outputs[0]));
}
