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
    // 1) shortest-edge resize preserving aspect ratio (HF BitImageProcessor
    //    parity): scale so the shorter side becomes 256.
    constexpr int short_edge = 256;
    const float   scale      = (float)short_edge / (float)std::min(img.nx, img.ny);
    const int     new_w      = std::max((int)std::lround(img.nx * scale), 1);
    const int     new_h      = std::max((int)std::lround(img.ny * scale), 1);
    Image         image      = resize_bicubic(img, new_w, new_h);

    constexpr int crop_size = 224;
    // clamp >= 0 for safety (min dimension is 256 > 224 by construction)
    const int offset_w = std::max((image.nx - crop_size) / 2, 0);
    const int offset_h = std::max((image.ny - crop_size) / 2, 0);

    // 2) center crop
    Image cropped;
    cropped.nx = crop_size;
    cropped.ny = crop_size;
    cropped.c  = 3;
    cropped.data.resize((size_t)crop_size * crop_size * 3);
    for (int y = 0; y < crop_size; ++y) {
        const uint8_t *src_row = &image.data[((size_t)(offset_h + y) * image.nx + offset_w) * 3];
        std::memcpy(&cropped.data[(size_t)y * crop_size * 3], src_row, (size_t)crop_size * 3);
    }

    // 3) convert to float, scale to [0,1] and channel-wise standardization (RGB)
    ImageF out;
    out.nx = crop_size;
    out.ny = crop_size;
    out.c  = 3;
    out.data.resize((size_t)crop_size * crop_size * 3);
    for (size_t i = 0; i < cropped.data.size(); i += 3) {
        out.data[i + 0] = (cropped.data[i + 0] / 255.0f - IMAGENET_DEFAULT_MEAN[0]) / IMAGENET_DEFAULT_STD[0];
        out.data[i + 1] = (cropped.data[i + 1] / 255.0f - IMAGENET_DEFAULT_MEAN[1]) / IMAGENET_DEFAULT_STD[1];
        out.data[i + 2] = (cropped.data[i + 2] / 255.0f - IMAGENET_DEFAULT_MEAN[2]) / IMAGENET_DEFAULT_STD[2];
    }
    return out;
}

ImageF dino_preprocess(const Image &img, const dino_hparams &params) {
    const auto new_w = (img.nx / params.patch_size + 1) * params.patch_size;
    const auto new_h = (img.ny / params.patch_size + 1) * params.patch_size;

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

std::vector<float> interpolate_pos_embed(const ImgSize       img_size,
                                         const float        *pos_embed_data, // Input data shouldn't be modified
                                         const dino_hparams &hparams) {
    // --- Calculate New Grid Dimensions ---
    const int h_new           = img_size.height / hparams.patch_size;
    const int w_new           = img_size.width / hparams.patch_size;
    const int num_patches_new = h_new * w_new;

    // --- Calculate Original Grid Dimensions ---
    const int M                = hparams.n_img_embd(); // Original grid side length
    const int h_orig           = M;
    const int w_orig           = M;
    const int num_patches_orig = h_orig * w_orig;     // N = M*M
    const int hidden_sz        = hparams.hidden_size; // Alias for clarity

    // --- Early Return Check ---
    if (num_patches_new == num_patches_orig) {
        const size_t total_elements = (size_t)(num_patches_orig + 1) * hidden_sz;
        return {pos_embed_data, pos_embed_data + total_elements};
    }

    // --- Prepare Output Vector ---
    const size_t       total_elements_new = (size_t)(num_patches_new + 1) * hidden_sz;
    std::vector<float> pos_embed_new(total_elements_new);

    // --- Step 1: Copy CLS token embedding directly ---
    // The first hidden_sz elements are the CLS token.
    std::copy(pos_embed_data, pos_embed_data + hidden_sz, pos_embed_new.data());

    // --- Step 2: Interpolate Patch Embeddings (Dimension by Dimension) ---
    // Although data is [N, H], we process H slices of [N] shaped spatially.
    for (int c = 0; c < hidden_sz; ++c) {
        // Create a 2D grid for the *original* patches for the current hidden dimension 'c'.
        std::vector<float> src_grid((size_t)h_orig * w_orig);

        // Gather data for the c-th dimension from all original patches.
        for (int i = 0; i < num_patches_orig; ++i) {
            const int y_orig = i / w_orig;
            const int x_orig = i % w_orig;

            // Index for the c-th component of the i-th patch embedding.
            // (i+1) because the first "row" (index 0) is the CLS token.
            size_t input_idx                           = (size_t)(i + 1) * hidden_sz + c;
            src_grid[(size_t)y_orig * w_orig + x_orig] = pos_embed_data[input_idx];
        }

        // Resize the 2D grid for the current dimension.
        std::vector<float> dst_grid = resize_bicubic_f32(src_grid.data(), w_orig, h_orig, w_new, h_new);

        // Scatter the interpolated data back into the new embedding vector.
        for (int i = 0; i < num_patches_new; ++i) {
            const int y_new = i / w_new;
            const int x_new = i % w_new;

            // Index for the c-th component of the i-th *new* patch embedding.
            // (i+1) because the first "row" (index 0) is the CLS token.
            size_t output_idx         = (size_t)(i + 1) * hidden_sz + c;
            pos_embed_new[output_idx] = dst_grid[(size_t)y_new * w_new + x_new];
        }
    }

    return pos_embed_new;
}

// load the model's weights from a file following the ggml format(gguf)
bool dino_model_load(const ImgSize img_size, const std::string &fname, dino_model &model, const dino_params &params) {
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
        ggml_backend_cpu_set_n_threads(model.backend, params.n_threads);
    }

    struct ggml_context    *tmp_ctx     = nullptr;
    struct gguf_init_params gguf_params = {
        /*.no_alloc   =*/false,
        /*.ctx        =*/&tmp_ctx,
    };
    gguf_context *gguf_ctx = gguf_init_from_file(fname.c_str(), gguf_params);
    if (!gguf_ctx) {
        fprintf(stderr, "%s: gguf_init_from_file() failed\n", __func__);
        return false;
    }

    // load hparams
    // override defaults
    auto &hparams               = model.hparams;
    hparams.hidden_size         = get_val_u32(gguf_ctx, std::string("hidden_size").c_str());
    hparams.num_hidden_layers   = get_val_u32(gguf_ctx, std::string("num_hidden_layers").c_str());
    hparams.num_attention_heads = get_val_u32(gguf_ctx, std::string("num_attention_heads").c_str());

    hparams.patch_size          = get_val_u32(gguf_ctx, std::string("patch_size").c_str());
    hparams.img_size            = get_val_u32(gguf_ctx, std::string("img_size").c_str());
    hparams.ftype               = get_val_u32(gguf_ctx, std::string("ftype").c_str());
    hparams.num_register_tokens = get_val_u32(gguf_ctx, std::string("num_register_tokens").c_str());

    const int32_t qntvr = hparams.ftype / GGML_QNT_VERSION_FACTOR;

    fprintf(stderr, "%s: hidden_size            = %d\n", __func__, hparams.hidden_size);
    fprintf(stderr, "%s: num_hidden_layers      = %d\n", __func__, hparams.num_hidden_layers);
    fprintf(stderr, "%s: num_register_tokens    = %d\n", __func__, hparams.num_register_tokens);
    fprintf(stderr, "%s: num_attention_heads    = %d\n", __func__, hparams.num_attention_heads);
    fprintf(stderr, "%s: patch_size             = %d\n", __func__, hparams.patch_size);
    fprintf(stderr, "%s: img_size               = %d\n", __func__, hparams.img_size);
    fprintf(stderr, "%s: ftype                  = %d\n", __func__, hparams.ftype);
    fprintf(stderr, "%s: qntvr                  = %d\n", __func__, qntvr);

    if (params.classify) {
        hparams.num_classes = get_val_u32(gguf_ctx, std::string("num_classes").c_str());
        fprintf(stderr, "%s: num_classes            = %d\n", __func__, hparams.num_classes);
        // read id2label dictionary into an ordered map (sort of an OrderedDict)
        int num_labels = get_val_u32(gguf_ctx, std::string("num_classes").c_str());
        for (int i = 0; i < num_labels; ++i) {
            model.hparams.id2label[i] = get_val_str(gguf_ctx, std::to_string(i).c_str());
        }
    }

    hparams.ftype %= GGML_QNT_VERSION_FACTOR;

    int num_tensors = gguf_get_n_tensors(gguf_ctx) + 1; // +1 for new_pos_embed

    // std::cout << "patch size " << hparams.patch_size << std::endl;

    const int new_w = (img_size.width / model.hparams.patch_size + 1) * model.hparams.patch_size;
    const int new_h = (img_size.height / model.hparams.patch_size + 1) * model.hparams.patch_size;

    const int h0                = new_h / hparams.patch_size;
    const int w0                = new_w / hparams.patch_size;
    const int num_patches       = h0 * w0;
    const int model_num_patches = hparams.n_img_embd() * hparams.n_img_embd();

    const int offset = std::max(num_patches - model_num_patches, 0);

    struct ggml_init_params model_params = ggml_init_params{
        /*.mem_size   =*/ggml_tensor_overhead() * num_tensors + offset,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    model.ctx = ggml_init(model_params);
    for (int i = 0; i < num_tensors - 1; i++) {
        const char         *name = gguf_get_tensor_name(gguf_ctx, i);
        struct ggml_tensor *src  = ggml_get_tensor(tmp_ctx, name);
        struct ggml_tensor *dst  = ggml_dup_tensor(model.ctx, src);
        ggml_set_name(dst, name);
        model.tensors[name] = dst;
        // std::cout << "i: " << i << ", name: " << name << ", type: " << ggml_type_name(dst->type) << std::endl;
    }

    gguf_free(gguf_ctx);

    model.buffer = ggml_backend_alloc_ctx_tensors(model.ctx, model.backend);
    // copy tensors from main memory to backend
    for (struct ggml_tensor *cur = ggml_get_first_tensor(model.ctx); cur != nullptr;
         cur                     = ggml_get_next_tensor(model.ctx, cur)) {
        struct ggml_tensor *src    = ggml_get_tensor(tmp_ctx, ggml_get_name(cur));
        size_t              n_size = ggml_nbytes(src);
        ggml_backend_tensor_set(cur, ggml_get_data(src), 0, n_size);
    }

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
        const int64_t total_patches_padding = GGML_PAD(total_patches, 32);
        const int64_t total_patches_to_pad  = total_patches_padding - total_patches;

        const int64_t hidden_size_padding = GGML_PAD(hidden_size, 4);
        const int64_t hidden_size_to_pad  = hidden_size_padding - hidden_size;

        V = ggml_cont(ctx_cgraph, ggml_permute(ctx_cgraph, V, 0, 2, 1, 3));

        Q = ggml_pad(ctx_cgraph, Q, hidden_size_to_pad, total_patches_to_pad, 0, 0);

        K = ggml_pad(ctx_cgraph, K, hidden_size_to_pad, total_patches_to_pad, 0, 0);

        V = ggml_pad(ctx_cgraph, V, hidden_size_to_pad, total_patches_to_pad, 0, 0);

        const ggml_type dtype = model.tensors.at(base_layer_name + ".attention.attention.qkv.weight")->type;

        K = ggml_cast(ctx_cgraph, K, dtype);
        V = ggml_cast(ctx_cgraph, V, dtype);

        struct ggml_tensor *KQV = ggml_flash_attn_ext(ctx_cgraph, Q, K, V, nullptr, scale, 0.0f, 0.0f);
        KQV = ggml_view_4d(ctx_cgraph, KQV, KQV->ne[0], KQV->ne[1], KQV->ne[2] - total_patches_to_pad, KQV->ne[3],
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
    const int      num_patches         = h0 * w0;
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

            if (model.hparams.num_hidden_layers == 40) {
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
                                const dino_params &params) {
    const auto &hparams = model.hparams;

    struct ggml_cgraph *gf = ggml_new_graph(ctx_cgraph);

    forward_features(img_size, gf, ctx_cgraph, model, params);

    if (params.classify) {
        forward_head(img_size, gf, ctx_cgraph, model, params);
    }

    return gf;
}

bool dino_batch_size_valid(int64_t n) {
    return n >= 1 && n <= (int64_t)DINO_MAX_BATCH;
}

void print_usage(FILE *out, int argc, char **argv, const dino_params &params) {
    fprintf(out, "usage: %s [options]\n", argv[0]);
    fprintf(out, "\n");
    fprintf(out, "Model:\n");
    fprintf(out, "  -m FNAME, --model     model path (default: %s)\n", params.model.c_str());
    fprintf(out, "  -fa, --flash_attn     enable flash attention, less accurate (default: off)\n");
    fprintf(out, "  -t N, --threads       number of threads to use during computation (default: %d)\n",
            params.n_threads);
    fprintf(out, "\n");
    fprintf(out, "Input:\n");
    fprintf(out, "  -i FNAME, --inp       input image file (default: %s)\n", params.fname_inp.c_str());
    fprintf(out, "  -s N, --seed          RNG seed (default: %d)\n", params.seed);
    fprintf(out, "  --batch N             max images per forward pass (default: %d, max: %d)\n", params.n_batch,
            DINO_MAX_BATCH);
    fprintf(out, "\n");
    fprintf(out, "Output modes:\n");
    fprintf(out, "  -c, --classify        classify the image and print top-k labels (default: off)\n");
    fprintf(out, "  -k N, --topk          top k classes to print (default: %d)\n", params.topk);
    fprintf(out, "  --print-embeddings    emit one JSON object on stdout with cls/pooled embeddings\n");
    fprintf(out, "  --print-patch-tokens  include per-patch token vectors in the JSON output\n");
    fprintf(out, "  --l2-normalize        L2-normalize emitted embedding vectors\n");
    fprintf(out, "  -o FNAME, --out       write PCA visualization of patch features to FNAME (default: off)\n");
    fprintf(out, "\n");
    fprintf(out, "Benchmark:\n");
    fprintf(out, "  --bench               enable bench loop (default repeats=5, warmup=1); skips PCA image output\n");
    fprintf(out, "  --bench-runs N        number of timed runs (overrides default 5 when --bench is set)\n");
    fprintf(out, "  --bench-warmup N      number of warmup runs discarded before timing (default: %u)\n",
            params.bench_warmup);
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
    fprintf(out, "  dinov2-cli -m model.gguf -i img.jpg -o pca.png                # PCA viz of patch features\n");
    fprintf(out, "  dinov2-cli -m model.gguf -i img.jpg --bench --bench-json      # benchmark, JSON lines\n");
    fprintf(out, "\n");
    fprintf(out, "docs: https://raw.githubusercontent.com/espetro/dinov2.cpp/main/docs/cli.md\n");
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

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-s" || arg == "--seed") {
            params.seed = std::stoi(next_value(i));
        } else if (arg == "-m" || arg == "--model") {
            params.model = next_value(i);
        } else if (arg == "-i" || arg == "--inp") {
            params.fname_inp = next_value(i);
        } else if (arg == "--batch") {
            const long v = std::stol(next_value(i));
            if (!dino_batch_size_valid(v)) {
                fprintf(stderr, "error: --batch must be between 1 and %u, got %ld\n", DINO_MAX_BATCH, v);
                print_usage(stderr, argc, argv, params);
                exit(1);
            }
            params.n_batch = (uint32_t)v;
        } else if (arg == "-o" || arg == "--out") {
            params.image_out = next_value(i);
        } else if (arg == "-t" || arg == "--threads") {
            params.n_threads = std::stoi(next_value(i));
        } else if (arg == "-k" || arg == "--topk") {
            params.topk = std::stoi(next_value(i));
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
            params.bench_repeats = std::stoi(next_value(i));
        } else if (arg == "--bench-warmup") {
            params.bench_warmup = std::stoi(next_value(i));
        } else if (arg == "--bench-json") {
            params.bench_json = true;
        } else if (arg == "--print-embeddings") {
            params.print_embeddings = true;
        } else if (arg == "--print-patch-tokens") {
            params.print_patch_tokens = true;
        } else if (arg == "--l2-normalize") {
            params.l2_normalize = true;
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
    }

    // the graph batch dimension is the number of images actually provided
    dino_params batch_params = params;
    batch_params.n_batch     = (uint32_t)imgs.size();
    const size_t n_batch     = imgs.size();
    const size_t hidden_size = model.hparams.hidden_size;
    const size_t npix        = (size_t)nx * ny;
    const int    num_patches = (ny / (int)model.hparams.patch_size) * (nx / (int)model.hparams.patch_size);

    struct ggml_init_params params0 = {
        /*.mem_size   =*/ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead(),
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true, // the tensors will be allocated later by ggml_gallocr_alloc_graph()
    };
    struct ggml_context *ctx_cgraph = ggml_init(params0);
    struct ggml_cgraph  *gf         = build_graph({nx, ny}, ctx_cgraph, model, batch_params);

    ggml_gallocr_alloc_graph(allocr, gf);

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

    const std::vector<float> pos_embed_fixed_data =
        interpolate_pos_embed({nx, ny}, (float *)(pos_embed->data), model.hparams);

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
    const float *cls_data = ggml_get_data_f32(ggml_graph_get_tensor(gf, "cls_token"));

    if (params.classify) {
        // probs is a dense (num_classes, 1, 1, B) block
        const float *probs_data = ggml_get_data_f32(ggml_graph_get_tensor(gf, "probs"));
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
        const float *patch_tokens_data = ggml_get_data_f32(ggml_graph_get_tensor(gf, "patch_tokens"));
        for (size_t b = 0; b < n_batch; ++b) {
            dino_output &output = outputs[b];
            output.cls_token    = std::vector<float>(cls_data + b * hidden_size, cls_data + (b + 1) * hidden_size);

            const float *img_patches = patch_tokens_data + b * (size_t)num_patches * hidden_size;
            output.patch_tokens      = std::vector<float>(img_patches, img_patches + (size_t)num_patches * hidden_size);

            // pooled = [cls_token || mean(patch_tokens)], 2*hidden floats, cls first
            std::vector<float> pooled(2 * hidden_size, 0.0f);
            std::copy(output.cls_token->begin(), output.cls_token->end(), pooled.begin());
            float *mean = pooled.data() + hidden_size;
            for (int p = 0; p < num_patches; ++p) {
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
