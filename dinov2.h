#pragma once

#include "ggml.h"
#include "ggml-alloc.h"
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <cinttypes>
#include <cstdio>
#include <optional>
#include <memory>
#include <thread>
#include "src/image.h"

struct ImgSize {
    int width  = 0;
    int height = 0;
};

constexpr float IMAGENET_DEFAULT_MEAN[3] = {0.485f, 0.456f, 0.406f};
constexpr float IMAGENET_DEFAULT_STD[3]  = {0.229f, 0.224f, 0.225f};

uint32_t get_val_u32(const struct gguf_context *ctx, const char *key);

const char *get_val_str(const struct gguf_context *ctx, const char *key);

// L2-normalize a vector in place; zero vectors are left unchanged.
void l2_normalize(std::vector<float> &v);

struct dino_hparams {
    uint32_t                   hidden_size         = 768;
    uint32_t                   num_hidden_layers   = 12;
    uint32_t                   num_attention_heads = 12;
    uint32_t                   num_classes         = 1000;
    uint32_t                   num_register_tokens = 0;
    uint32_t                   patch_size          = 8;
    uint32_t                   img_size            = 224;
    uint32_t                   ftype               = 1;
    float                      eps                 = 1e-6f;
    std::string                interpolation       = "bicubic";
    std::map<int, std::string> id2label;

    uint32_t n_enc_head_dim() const;

    uint32_t n_img_size() const;

    uint32_t n_patch_size() const;

    uint32_t n_img_embd() const;
};

struct dino_model {
    dino_hparams                                hparams;
    struct ggml_context                        *ctx     = nullptr;
    ggml_backend_t                              backend = nullptr;
    ggml_backend_buffer_t                       buffer  = nullptr;
    std::map<std::string, struct ggml_tensor *> tensors;
};

// Maximum accepted value for dino_params::n_batch / the --batch flag.
constexpr uint32_t DINO_MAX_BATCH = 64;

// Valid range for a batch size: 1 <= n <= DINO_MAX_BATCH.
bool dino_batch_size_valid(int64_t n);

// Feature-mode preprocessing recipes selectable via --preprocess.
//   bounded: resize so the shortest edge is DINO_FEATURE_SHORT_EDGE when
//            larger, then align to patch multiples (default)
//   hf:      HF AutoImageProcessor recipe, shortest edge 256 + center crop 224
//   crop518: shortest edge 518 + center crop 518x518 (fixed grid, batch-safe)
enum class dino_preprocess_mode { bounded, hf, crop518 };

// Shortest-edge bound for the bounded preprocessing mode.
constexpr int DINO_FEATURE_SHORT_EDGE = 518;

// Default --max-tokens: 4x the square grid at the 518 bound for the model's
// patch size (patch 14 gives 4 * 37 * 37 = 5476).
inline uint64_t dino_default_max_tokens(uint32_t patch_size) {
    const uint64_t side = ((uint64_t)DINO_FEATURE_SHORT_EDGE + patch_size - 1) / patch_size;
    return 4 * side * side;
}

struct dino_params {
    int32_t     seed               = 42; // unused: no RNG in inference; kept for API compatibility
    uint32_t    topk               = 5;
    uint32_t    n_batch            = 1; // max images per forward pass
    bool        enable_flash_attn  = false;
    uint32_t    n_threads          = std::min(4u, std::thread::hardware_concurrency());
    bool        classify           = false;
    bool        print_embeddings   = false;           // print cls + pooled embeddings (parsing added separately)
    bool        embeddings_binary  = false;           // write preview binary embeddings to -o
    bool        print_patch_tokens = false;           // print per-patch embeddings (parsing added separately)
    bool        l2_normalize       = false;           // L2-normalize emitted embedding vectors
    std::string model              = "../model.gguf"; // model path
    // input image paths; -i repeats or comma-separates to add more than one.
    // Images are forwarded to dino_predict in chunks of n_batch.
    std::vector<std::string> fnames_inp = {"../assets/tench.jpg"};
    std::string              image_out  = ""; // output of pca visualization (if used; a directory for multi-input)
    // Benchmark controls. bench_repeats=0 disables the bench loop (legacy single-shot path).
    // --bench with no count sets bench_repeats to 5 (the default for one-shot "is it faster").
    uint32_t bench_repeats = 0;
    uint32_t bench_warmup  = 1;
    bool     bench_json    = false;
    // Feature-mode preprocessing; ignored (rejected) with -c.
    dino_preprocess_mode preprocess_mode = dino_preprocess_mode::bounded;
    bool                 no_resize       = false; // bounded mode: keep native resolution
    // Hard cap on patch tokens per image after preprocessing. -1: default
    // (dino_default_max_tokens from the model's patch size); 0 disables.
    int64_t max_tokens = -1;
};

struct ggml_tensor *attn(struct ggml_tensor *cur, const float scale, int il, struct ggml_context *ctx_cgraph,
                         const dino_model &model, const dino_params &params);

struct ggml_tensor *mlp(struct ggml_tensor *cur, int il, struct ggml_context *ctx_cgraph, const dino_model &model,
                        const dino_params &params);

struct ggml_tensor *swiglu_ffn(struct ggml_tensor *cur, int il, struct ggml_context *ctx_cgraph,
                               const dino_model &model, const dino_params &params);

void forward_features(ImgSize img_size, struct ggml_cgraph *graph, struct ggml_context *ctx_cgraph,
                      const dino_model &model, const dino_params &params);

void forward_head(ImgSize img_size, struct ggml_cgraph *graph, struct ggml_context *ctx_cgraph, const dino_model &model,
                  const dino_params &params);

struct dino_output {
    std::optional<std::vector<uint32_t>> preds;        // top-k class indices (classify mode)
    std::optional<std::vector<float>>    pred_scores;  // top-k probabilities, parallel to preds (classify mode)
    std::optional<std::vector<float>>    cls_token;    // hidden_size floats (both modes)
    std::optional<std::vector<float>>    pooled;       // [cls_token || mean(patch_tokens)], 2*hidden (feature mode)
    std::optional<std::vector<float>>    patch_tokens; // n_patches x hidden_size, row-major (feature mode)
};

ImageF dino_classify_preprocess(const Image &img, const dino_hparams &params);

ImageF dino_preprocess(const Image &img, const dino_hparams &params);

// Feature-mode preprocessing: dispatches on params.preprocess_mode
// (bounded / hf / crop518; --no-resize applies to bounded only).
ImageF dino_feature_preprocess(const Image &img, const dino_hparams &hparams, const dino_params &params);

// Output dimensions dino_feature_preprocess will produce for img, without
// doing the work. Used to apply --max-tokens before preprocessing.
ImgSize dino_feature_output_size(const Image &img, const dino_hparams &hparams, const dino_params &params);

bool dino_model_load(ImgSize img_size, const std::string &fname, dino_model &model, const dino_params &params);

std::vector<float> interpolate_pos_embed(ImgSize img_size, const float *pos_embed_data, const dino_hparams &hparams);

// graph_size is the cgraph node capacity; it must cover every node the
// encoder emits (see dino_predict for the per-layer sizing expression).
struct ggml_cgraph *build_graph(ImgSize img_size, struct ggml_context *ctx_cgraph, const dino_model &model,
                                const dino_params &params, size_t graph_size);

// Batch inference: runs the model on up to params.n_batch preprocessed images
// and returns one dino_output per image, in input order. All images must share
// the same dimensions (they are packed into a single graph whose batch
// dimension is imgs.size()). Returns an empty vector on failure.
std::vector<dino_output> dino_predict(const dino_model &model, const std::vector<ImageF> &imgs,
                                      const dino_params &params, ggml_gallocr_t allocr);

// Single-image convenience wrapper around the batch form.
std::unique_ptr<dino_output> dino_predict(const dino_model &model, const ImageF &img, const dino_params &params,
                                          ggml_gallocr_t allocr);

void print_usage(FILE *out, int argc, char **argv, const dino_params &params);

// Write the intentionally unstable version-2 D2EMB preview format.
// grid_w/grid_h are the patch-grid dimensions (zero when patches are absent).
// The vectors are expected to already have the requested normalization applied.
bool write_embeddings_binary(const std::string &path, const dino_output &output, uint32_t hidden_size,
                             uint32_t patch_count, uint32_t grid_w, uint32_t grid_h, bool include_patches,
                             bool normalized, std::string &error);

bool dino_params_parse(int argc, char **argv, dino_params &params);
