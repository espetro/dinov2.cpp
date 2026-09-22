#pragma once

// Internal engine header for libdinov2: used by dinov2.cpp, dinov2-cli.cpp,
// src/dinov2-c.cpp and the unit tests. Not installed; the stable C API lives
// in include/dinov2.h.

#include "ggml.h"
#include "ggml-backend.h"
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
    // set at load: classifier.weight + classifier.bias present and num_classes > 0
    bool has_classifier = false;
};

// Maximum accepted value for dino_ctx_options::n_batch / the --batch flag and
// the public dino_encode() image count.
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

// Model-load settings (the public dino_model_params mirrors this).
struct dino_model_options {
    bool        require_classifier = false; // fail load unless the GGUF carries a classifier head + labels
    std::string device;                     // "" = auto (ggml_backend_init_best); else a ggml device name
};

// Initialize a compute backend through the ggml registry. Loads dynamic
// backends lazily (ggml_backend_load_all) when the registry is empty, then
// returns ggml_backend_init_by_name(device_name) for a non-empty name or
// ggml_backend_init_best() otherwise. Returns nullptr on failure.
ggml_backend_t dino_backend_init(const char *device_name);

// Context settings: compute capacity + feature-mode preprocessing recipe
// (the public dino_ctx_params mirrors this).
struct dino_ctx_options {
    uint32_t             n_threads         = std::min(4u, std::thread::hardware_concurrency());
    uint32_t             n_batch           = 1; // max images packed into one forward pass
    bool                 enable_flash_attn = false;
    dino_preprocess_mode preprocess_mode   = dino_preprocess_mode::bounded;
    bool                 no_resize         = false; // bounded mode only
    // Hard cap on patch tokens per image after preprocessing. -1: default
    // (dino_default_max_tokens from the model's patch size); 0 disables.
    int64_t max_tokens = -1;
};

// Per-encode settings (the public dino_run_params mirrors this).
struct dino_run_options {
    bool     classify     = false; // run the classifier head; implies the 224x224 recipe
    uint32_t topk         = 5;
    bool     l2_normalize = false; // normalize cls/pooled/patch vectors before storing
};

struct ggml_tensor *attn(struct ggml_tensor *cur, const float scale, int il, struct ggml_context *ctx_cgraph,
                         const dino_model &model, const dino_ctx_options &options);

struct ggml_tensor *mlp(struct ggml_tensor *cur, int il, struct ggml_context *ctx_cgraph, const dino_model &model);

struct ggml_tensor *swiglu_ffn(struct ggml_tensor *cur, int il, struct ggml_context *ctx_cgraph,
                               const dino_model &model);

void forward_features(ImgSize img_size, struct ggml_cgraph *graph, struct ggml_context *ctx_cgraph,
                      const dino_model &model, const dino_ctx_options &options);

void forward_head(ImgSize img_size, struct ggml_cgraph *graph, struct ggml_context *ctx_cgraph,
                  const dino_model &model);

struct dino_output {
    std::optional<std::vector<uint32_t>> preds;        // top-k class indices (classify mode)
    std::optional<std::vector<float>>    pred_scores;  // top-k probabilities, parallel to preds (classify mode)
    std::optional<std::vector<float>>    cls_token;    // hidden_size floats (both modes)
    std::optional<std::vector<float>>    pooled;       // [cls_token || mean(patch_tokens)], 2*hidden (feature mode)
    std::optional<std::vector<float>>    patch_tokens; // n_patches x hidden_size, row-major (feature mode)
    int32_t                              grid_w = 0;   // patch grid dims of this image's
    int32_t                              grid_h = 0;   // preprocessed input
};

ImageF dino_classify_preprocess(const Image &img, const dino_hparams &params);

ImageF dino_preprocess_padded(const Image &img, const dino_hparams &params);

// Feature-mode preprocessing: dispatches on options.preprocess_mode
// (bounded / hf / crop518; --no-resize applies to bounded only).
ImageF dino_feature_preprocess(const Image &img, const dino_hparams &hparams, const dino_ctx_options &options);

// Output dimensions dino_feature_preprocess will produce for img, without
// doing the work. Used to apply the max_tokens cap before preprocessing.
ImgSize dino_feature_output_size(const Image &img, const dino_hparams &hparams, const dino_ctx_options &options);

bool dino_model_load(const std::string &fname, dino_model &model, const dino_model_options &options);

// Same loader fed from an in-memory GGUF image; the bytes are copied during
// the load so the caller may release data on return.
bool dino_model_load_buffer(const void *data, size_t size, dino_model &model, const dino_model_options &options);

// Streaming reader for dino_model_load_callback: read up to `len` bytes at
// `offset` into `output`, return the number of bytes read. Same contract as
// gguf_reader_callback_t.
using dino_reader_fn = size_t (*)(void *userdata, void *output, uint64_t offset, size_t len);

// Same loader fed through a streaming read callback (mmap-friendly).
bool dino_model_load_callback(dino_reader_fn read, void *userdata, dino_model &model,
                              const dino_model_options &options);

// Release the ggml context, backend buffer and backend held by model.
void dino_model_unload(dino_model &model);

std::vector<float> interpolate_pos_embed(ImgSize img_size, const float *pos_embed_data, const dino_hparams &hparams);

// graph_size is the cgraph node capacity; it must cover every node the
// encoder emits (see dino_predict for the per-layer sizing expression).
struct ggml_cgraph *build_graph(ImgSize img_size, struct ggml_context *ctx_cgraph, const dino_model &model,
                                const dino_ctx_options &options, bool classify, size_t graph_size);

// Failure categories recorded in dino_ctx::last_status by dino_predict; the
// public dino_status enum in include/dinov2.h mirrors these codes.
enum class dino_errc {
    ok,
    invalid_argument,
    alloc_failed,
    compute_failed,
};

// Inference context: owns the backend scheduler (which owns the graph
// allocator), the options applied to every run, and the outputs of the most
// recent predict call. Create one per model after dino_model_load. Not
// thread-safe; concurrent runs need one context each.
struct dino_ctx {
    const dino_model    *model = nullptr; // borrowed; must outlive the ctx
    dino_ctx_options     options;
    ggml_backend_sched_t sched = nullptr;
    // owned fallback for ops the model's backend cannot run; only set when
    // model.backend is not CPU (ggml_backend_sched requires a CPU tail)
    ggml_backend_t           cpu_fallback = nullptr;
    std::vector<dino_output> last_outputs; // owned by the ctx; overwritten on each predict
    // failure category of the last predict; meaningful only when
    // last_outputs is empty after a call
    dino_errc last_status = dino_errc::ok;
};

bool dino_ctx_init(dino_ctx &ctx, const dino_model &model, const dino_ctx_options &options);

void dino_ctx_free(dino_ctx &ctx);

// Batch inference: runs the model on up to ctx.options.n_batch preprocessed
// images and stores one dino_output per image, in input order, into
// ctx.last_outputs. All images must share the same dimensions (they are packed
// into a single graph whose batch dimension is imgs.size()). The returned
// reference is invalidated by the next dino_predict call on ctx; an empty
// vector signals failure.
const std::vector<dino_output> &dino_predict(const dino_model &model, dino_ctx &ctx, const std::vector<ImageF> &imgs,
                                             const dino_run_options &run);

// Single-image convenience wrapper around the batch form. Returns nullptr on
// failure, otherwise a pointer into ctx.last_outputs.
const dino_output *dino_predict(const dino_model &model, dino_ctx &ctx, const ImageF &img, const dino_run_options &run);
