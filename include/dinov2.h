#ifndef DINOV2_H
#define DINOV2_H

// Public pure-C API for libdinov2. Opaque handles only: no ggml/gguf types
// leak across the boundary. Compilable with gcc -std=c11. Unstable for the
// 0.4.x line; see docs/stability.md.
//
// Threading: a dino_ctx is single-threaded; drive dino_encode calls on it
// from one thread at a time. Distinct contexts are independent and may run
// concurrently, even on the same model. The dino_model_* getters and
// dino_model_label may be called from any thread while the model is alive.

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef DINOV2_SHARED
#ifdef _WIN32
#ifdef DINOV2_BUILD
#define DINO_API __declspec(dllexport)
#else
#define DINO_API __declspec(dllimport)
#endif
#else
#define DINO_API __attribute__((visibility("default")))
#endif
#else
#define DINO_API
#endif

#ifdef __GNUC__
#define DINO_DEPRECATED(func, hint) func __attribute__((deprecated(hint)))
#elif defined(_MSC_VER)
#define DINO_DEPRECATED(func, hint) __declspec(deprecated(hint)) func
#else
#define DINO_DEPRECATED(func, hint) func
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define DINO_MAX_BATCH 64

// opaque handles; the C++ definitions live in the internal engine header
typedef struct dino_model dino_model;
typedef struct dino_ctx   dino_ctx;

typedef enum dino_status {
    DINO_STATUS_SUCCESS          = 0,
    DINO_STATUS_ERROR            = 1, // unspecified internal failure
    DINO_STATUS_INVALID_ARGUMENT = 2, // null pointers, bad dims/stride, n_images out of range,
                                      // classify topk < 1 or > n_classes
    DINO_STATUS_ALLOC_FAILED    = 3,  // compute graph buffer allocation failed
    DINO_STATUS_COMPUTE_FAILED  = 4,  // backend graph compute failed
    DINO_STATUS_NO_CLASSIFIER   = 5,  // classify requested on a backbone-only model
    DINO_STATUS_TOO_MANY_TOKENS = 6,  // an image exceeded the ctx max_tokens cap
} dino_status;

typedef enum dino_preprocess {
    DINO_PREPROCESS_BOUNDED = 0, // shortest edge <= 518, patch-aligned (CLI default)
    DINO_PREPROCESS_HF      = 1, // HF recipe: shortest edge 256 + center crop 224
    DINO_PREPROCESS_CROP518 = 2, // fixed 518x518 grid
} dino_preprocess;

// streaming read callback for dino_model_load_from_callback:
// read up to `len` bytes at `offset` into `output`, return bytes read
typedef size_t (*dino_reader_callback_t)(void *userdata, void *output, uint64_t offset, size_t len);

typedef struct dino_model_params {
    // NULL or "" = auto-select the best device; otherwise a ggml device name
    // matched case-insensitively, e.g. "CPU", "MTL0" (first Metal device),
    // "CUDA0". Names come from ggml_backend_dev_name(); C++ callers can list
    // them via ggml_backend_dev_count()/ggml_backend_dev_get(), and debug
    // builds log every registered device to stderr during backend loading.
    const char *device;
    bool        require_classifier; // fail load unless the GGUF carries a classifier head + labels
} dino_model_params;

typedef struct dino_ctx_params {
    int32_t         n_threads; // applied via ggml_backend_set_n_threads proc address
    int32_t         n_batch;   // max images packed into one forward pass (1..DINO_MAX_BATCH)
    bool            flash_attn;
    dino_preprocess preprocess; // feature-mode recipe; classify ignores it
    bool            no_resize;  // bounded mode only: keep native resolution
    int64_t         max_tokens; // -1: default cap (4 * (518/patch)^2); 0: disabled
} dino_ctx_params;

typedef struct dino_run_params {
    bool    classify;     // run the classifier head and fill topk outputs
    int32_t topk;         // number of top classes stored per image when classify is set
    bool    l2_normalize; // L2-normalize cls/pooled/patch vectors before storing
} dino_run_params;

// one image of a batch: interleaved RGB8, row-major
typedef struct dino_image {
    const uint8_t *pixels; // required
    int32_t        width;  // > 0
    int32_t        height; // > 0
    int32_t        stride; // bytes per row; 0 means tightly packed (width * 3)
} dino_image;

DINO_API const char *dino_version(void);

// Load every available backend into the ggml registry. Optional: model load
// calls this lazily on first use. Safe to call more than once.
DINO_API void dino_backend_init(void);

DINO_API struct dino_model_params dino_model_default_params(void);
DINO_API struct dino_ctx_params   dino_ctx_default_params(void);
DINO_API struct dino_run_params   dino_run_default_params(void);

// Load a GGUF model. Returns NULL on failure (details on stderr).
DINO_API dino_model *dino_model_load_from_file(const char *path, struct dino_model_params params);
DINO_API dino_model *dino_model_load_from_buffer(const void *data, size_t size, struct dino_model_params params);
DINO_API dino_model *dino_model_load_from_callback(dino_reader_callback_t read, void *userdata,
                                                   struct dino_model_params params);
DINO_API void        dino_model_free(dino_model *model);

// Model metadata; 0/NULL when absent. label() returns NULL for classes
// without a label; the returned string is borrowed and stays valid until
// dino_model_free().
DINO_API uint32_t    dino_model_hidden_size(const dino_model *model);
DINO_API uint32_t    dino_model_patch_size(const dino_model *model);
DINO_API uint32_t    dino_model_n_register_tokens(const dino_model *model);
DINO_API bool        dino_model_has_classifier(const dino_model *model);
DINO_API uint32_t    dino_model_n_classes(const dino_model *model);
DINO_API const char *dino_model_label(const dino_model *model, uint32_t class_index);

// Create an inference context on a model. The model is borrowed and must
// outlive the context. Returns NULL on invalid params.
DINO_API dino_ctx *dino_init_from_model(dino_model *model, struct dino_ctx_params params);
DINO_API void      dino_free(dino_ctx *ctx);

// Run a batch of raw RGB8 images through the encoder. All preprocessing
// (resize, crop, ImageNet normalization) happens inside the library.
// n_images must be 1..DINO_MAX_BATCH. Feature mode groups consecutive
// same-sized images into chunks of up to ctx n_batch; classify mode always
// produces 224x224 inputs, so the whole call is one chunk when n_batch allows.
// Results are stored in the ctx; use the dino_output_* accessors. Every call
// replaces the ctx's outputs: on any return other than DINO_STATUS_SUCCESS
// the previous outputs are gone and the accessors report 0 images.
DINO_API enum dino_status dino_encode(dino_ctx *ctx, const struct dino_image *images, int32_t n_images,
                                      struct dino_run_params params);

// Output accessors. Returned pointers are borrowed, point into storage owned by
// ctx, and are invalidated by the next dino_encode or by dino_free.
DINO_API int32_t dino_output_n_images(const dino_ctx *ctx);
// cls embedding: hidden_size floats per image
DINO_API const float *dino_output_cls(const dino_ctx *ctx, int32_t index);
// pooled embedding [cls || mean(patches)]: 2 * hidden_size floats; NULL in classify mode
DINO_API const float *dino_output_pooled(const dino_ctx *ctx, int32_t index);
// patch tokens: grid_w * grid_h * hidden_size floats, row-major; NULL in classify mode.
// All out params may be NULL.
DINO_API const float *dino_output_patches(const dino_ctx *ctx, int32_t index, int32_t *n_patches, int32_t *grid_w,
                                          int32_t *grid_h);
// classify mode: writes the top-k index/probability arrays, returns k. 0/NULL otherwise.
DINO_API int32_t dino_output_topk(const dino_ctx *ctx, int32_t index, const uint32_t **indices, const float **probs);

#ifdef __cplusplus
}
#endif

#endif // DINOV2_H
