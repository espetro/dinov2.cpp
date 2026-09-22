// Public C API implementation: the only translation unit that sees both the
// public header (include/dinov2.h) and the internal engine header
// (../dinov2.h). Converts between the public by-value PODs and the internal
// options structs, packs raw RGB8 input into Image/ImageF, chunks batches,
// and maps failures to dino_status codes.

// Internal header first: the public header defines the DINO_MAX_BATCH macro,
// which would rewrite the internal constexpr of the same name if included
// earlier. After both are included, DINO_MAX_BATCH expands to the public 64,
// which equals the internal cap.
#include "../dinov2.h"
#include "include/dinov2.h"

#include <cstdio>
#include <cstring>
#include <new>
#include <vector>

#ifndef DINOV2_VERSION
#define DINOV2_VERSION "dev"
#endif

namespace {

dino_model_options to_model_options(const dino_model_params &params) {
    dino_model_options options;
    options.require_classifier = params.require_classifier;
    if (params.device) {
        options.device = params.device;
    }
    return options;
}

// Convert a public dino_ctx_params; returns false on invalid values.
bool to_ctx_options(const dino_ctx_params &params, dino_ctx_options &options) {
    if (params.n_threads < 1 || !dino_batch_size_valid(params.n_batch) || params.max_tokens < -1) {
        return false;
    }
    options.n_threads         = (uint32_t)params.n_threads;
    options.n_batch           = (uint32_t)params.n_batch;
    options.enable_flash_attn = params.flash_attn;
    options.no_resize         = params.no_resize;
    options.max_tokens        = params.max_tokens;
    switch (params.preprocess) {
    case DINO_PREPROCESS_BOUNDED:
        options.preprocess_mode = dino_preprocess_mode::bounded;
        break;
    case DINO_PREPROCESS_HF:
        options.preprocess_mode = dino_preprocess_mode::hf;
        break;
    case DINO_PREPROCESS_CROP518:
        options.preprocess_mode = dino_preprocess_mode::crop518;
        break;
    default:
        return false;
    }
    return true;
}

// Map the failure category recorded by dino_predict onto the public codes.
dino_status predict_status(const dino_ctx &ctx) {
    switch (ctx.last_status) {
    case dino_errc::invalid_argument:
        return DINO_STATUS_INVALID_ARGUMENT;
    case dino_errc::alloc_failed:
        return DINO_STATUS_ALLOC_FAILED;
    case dino_errc::compute_failed:
        return DINO_STATUS_COMPUTE_FAILED;
    case dino_errc::ok:
    default:
        return DINO_STATUS_ERROR;
    }
}

const dino_output *output_at(const dino_ctx *ctx, int32_t index) {
    if (!ctx || index < 0 || (size_t)index >= ctx->last_outputs.size()) {
        return nullptr;
    }
    return &ctx->last_outputs[(size_t)index];
}

} // namespace

extern "C" {

const char *dino_version(void) {
    return DINOV2_VERSION;
}

void dino_backend_init(void) {
    // resolve to the internal overload: loads dynamic backends once and
    // initializes the best device; the backend itself is dropped here (this
    // only warms the registry for later loads)
    ggml_backend_t backend = ::dino_backend_init(nullptr);
    if (backend) {
        ggml_backend_free(backend);
    }
}

struct dino_model_params dino_model_default_params(void) {
    struct dino_model_params params;
    params.device             = nullptr;
    params.require_classifier = false;
    return params;
}

struct dino_ctx_params dino_ctx_default_params(void) {
    const dino_ctx_options options;
    struct dino_ctx_params params;
    params.n_threads  = (int32_t)options.n_threads;
    params.n_batch    = (int32_t)options.n_batch;
    params.flash_attn = options.enable_flash_attn;
    params.preprocess = DINO_PREPROCESS_BOUNDED;
    params.no_resize  = options.no_resize;
    params.max_tokens = options.max_tokens;
    return params;
}

struct dino_run_params dino_run_default_params(void) {
    const dino_run_options options;
    struct dino_run_params params;
    params.classify     = options.classify;
    params.topk         = (int32_t)options.topk;
    params.l2_normalize = options.l2_normalize;
    return params;
}

dino_model *dino_model_load_from_file(const char *path, struct dino_model_params params) {
    if (!path) {
        fprintf(stderr, "%s: path is NULL\n", __func__);
        return nullptr;
    }
    dino_model *model = new (std::nothrow) dino_model;
    if (!model) {
        return nullptr;
    }
    if (!dino_model_load(path, *model, to_model_options(params))) {
        delete model;
        return nullptr;
    }
    return model;
}

dino_model *dino_model_load_from_buffer(const void *data, size_t size, struct dino_model_params params) {
    if (!data || size == 0) {
        fprintf(stderr, "%s: empty model buffer\n", __func__);
        return nullptr;
    }
    dino_model *model = new (std::nothrow) dino_model;
    if (!model) {
        return nullptr;
    }
    if (!dino_model_load_buffer(data, size, *model, to_model_options(params))) {
        delete model;
        return nullptr;
    }
    return model;
}

dino_model *dino_model_load_from_callback(dino_reader_callback_t read, void *userdata,
                                          struct dino_model_params params) {
    if (!read) {
        fprintf(stderr, "%s: read callback is NULL\n", __func__);
        return nullptr;
    }
    dino_model *model = new (std::nothrow) dino_model;
    if (!model) {
        return nullptr;
    }
    if (!dino_model_load_callback(read, userdata, *model, to_model_options(params))) {
        delete model;
        return nullptr;
    }
    return model;
}

void dino_model_free(dino_model *model) {
    if (!model) {
        return;
    }
    dino_model_unload(*model);
    delete model;
}

uint32_t dino_model_hidden_size(const dino_model *model) {
    return model ? model->hparams.hidden_size : 0;
}

uint32_t dino_model_patch_size(const dino_model *model) {
    return model ? model->hparams.patch_size : 0;
}

uint32_t dino_model_n_register_tokens(const dino_model *model) {
    return model ? model->hparams.num_register_tokens : 0;
}

bool dino_model_has_classifier(const dino_model *model) {
    return model ? model->has_classifier : false;
}

uint32_t dino_model_n_classes(const dino_model *model) {
    return model && model->has_classifier ? model->hparams.num_classes : 0;
}

const char *dino_model_label(const dino_model *model, uint32_t class_index) {
    if (!model) {
        return nullptr;
    }
    const auto it = model->hparams.id2label.find((int)class_index);
    return it != model->hparams.id2label.end() ? it->second.c_str() : nullptr;
}

dino_ctx *dino_init_from_model(dino_model *model, struct dino_ctx_params params) {
    if (!model || !model->backend) {
        return nullptr;
    }
    dino_ctx_options options;
    if (!to_ctx_options(params, options)) {
        fprintf(stderr, "%s: invalid ctx params\n", __func__);
        return nullptr;
    }
    dino_ctx *ctx = new (std::nothrow) dino_ctx;
    if (!ctx) {
        return nullptr;
    }
    if (!dino_ctx_init(*ctx, *model, options)) {
        delete ctx;
        return nullptr;
    }
    return ctx;
}

void dino_free(dino_ctx *ctx) {
    if (!ctx) {
        return;
    }
    dino_ctx_free(*ctx);
    delete ctx;
}

enum dino_status dino_encode(dino_ctx *ctx, const struct dino_image *images, int32_t n_images,
                             struct dino_run_params params) {
    if (!ctx || !ctx->model || !ctx->sched) {
        return DINO_STATUS_INVALID_ARGUMENT;
    }
    const dino_model &model = *ctx->model;
    if (!images || n_images < 1 || n_images > (int32_t)DINO_MAX_BATCH) {
        return DINO_STATUS_INVALID_ARGUMENT;
    }
    if (params.topk < 1) {
        return DINO_STATUS_INVALID_ARGUMENT;
    }
    if (params.classify) {
        if (!model.has_classifier) {
            return DINO_STATUS_NO_CLASSIFIER;
        }
        if ((uint32_t)params.topk > model.hparams.num_classes) {
            return DINO_STATUS_INVALID_ARGUMENT;
        }
    }

    // validate the records and copy into tightly packed Image buffers
    std::vector<Image> imgs((size_t)n_images);
    for (int32_t i = 0; i < n_images; ++i) {
        const dino_image &in     = images[i];
        const int64_t     stride = in.stride == 0 ? (int64_t)in.width * 3 : in.stride;
        if (!in.pixels || in.width <= 0 || in.height <= 0 || stride < (int64_t)in.width * 3) {
            return DINO_STATUS_INVALID_ARGUMENT;
        }
        Image &img = imgs[(size_t)i];
        img.nx     = in.width;
        img.ny     = in.height;
        img.c      = 3;
        img.data.resize((size_t)in.width * in.height * 3);
        for (int32_t y = 0; y < in.height; ++y) {
            std::memcpy(img.data.data() + (size_t)y * in.width * 3, in.pixels + (size_t)y * stride,
                        (size_t)in.width * 3);
        }
    }

    // preprocess; the max_tokens cap applies to feature mode only, checked on
    // the prospective output size before the (expensive) resize
    const int64_t       token_limit = ctx->options.max_tokens >= 0
                                          ? ctx->options.max_tokens
                                          : (int64_t)dino_default_max_tokens(model.hparams.patch_size);
    std::vector<ImageF> imgs_f((size_t)n_images);
    for (int32_t i = 0; i < n_images; ++i) {
        if (token_limit > 0 && !params.classify) {
            const ImgSize out_size  = dino_feature_output_size(imgs[(size_t)i], model.hparams, ctx->options);
            const int64_t n_patches = (int64_t)(out_size.height / (int)model.hparams.patch_size) *
                                      (out_size.width / (int)model.hparams.patch_size);
            if (n_patches > token_limit) {
                fprintf(stderr, "error: image %d yields %lld patch tokens after preprocessing (limit %lld)\n", (int)i,
                        (long long)n_patches, (long long)token_limit);
                return DINO_STATUS_TOO_MANY_TOKENS;
            }
        }
        imgs_f[(size_t)i] = params.classify ? dino_classify_preprocess(imgs[(size_t)i], model.hparams)
                                            : dino_feature_preprocess(imgs[(size_t)i], model.hparams, ctx->options);
    }

    // one graph requires equal dims: group consecutive same-sized images into
    // chunks of at most ctx n_batch
    const dino_run_options   run{params.classify, (uint32_t)params.topk, params.l2_normalize};
    std::vector<dino_output> all;
    all.reserve((size_t)n_images);
    for (size_t s = 0; s < imgs_f.size();) {
        size_t e = s + 1;
        while (e < imgs_f.size() && e - s < ctx->options.n_batch && imgs_f[e].nx == imgs_f[s].nx &&
               imgs_f[e].ny == imgs_f[s].ny) {
            ++e;
        }
        const std::vector<ImageF>       chunk(imgs_f.begin() + (ptrdiff_t)s, imgs_f.begin() + (ptrdiff_t)e);
        const std::vector<dino_output> &outs = dino_predict(model, *ctx, chunk, run);
        if (outs.empty()) {
            return predict_status(*ctx);
        }
        all.insert(all.end(), outs.begin(), outs.end());
        s = e;
    }
    ctx->last_outputs = std::move(all);
    return DINO_STATUS_SUCCESS;
}

int32_t dino_output_n_images(const dino_ctx *ctx) {
    return ctx ? (int32_t)ctx->last_outputs.size() : 0;
}

const float *dino_output_cls(const dino_ctx *ctx, int32_t index) {
    const dino_output *out = output_at(ctx, index);
    return out && out->cls_token ? out->cls_token->data() : nullptr;
}

const float *dino_output_pooled(const dino_ctx *ctx, int32_t index) {
    const dino_output *out = output_at(ctx, index);
    return out && out->pooled ? out->pooled->data() : nullptr;
}

const float *dino_output_patches(const dino_ctx *ctx, int32_t index, int32_t *n_patches, int32_t *grid_w,
                                 int32_t *grid_h) {
    const dino_output *out = output_at(ctx, index);
    if (!out || !out->patch_tokens) {
        return nullptr;
    }
    if (grid_w) {
        *grid_w = out->grid_w;
    }
    if (grid_h) {
        *grid_h = out->grid_h;
    }
    if (n_patches) {
        *n_patches = out->grid_w * out->grid_h;
    }
    return out->patch_tokens->data();
}

int32_t dino_output_topk(const dino_ctx *ctx, int32_t index, const uint32_t **indices, const float **probs) {
    const dino_output *out = output_at(ctx, index);
    if (!out || !out->preds || !out->pred_scores) {
        return 0;
    }
    if (indices) {
        *indices = out->preds->data();
    }
    if (probs) {
        *probs = out->pred_scores->data();
    }
    return (int32_t)out->preds->size();
}

} // extern "C"
