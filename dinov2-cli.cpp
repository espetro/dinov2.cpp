#define CRT_SECURE_NO_DEPRECATE // disables "unsafe" warnings on Windows
#include "dinov2.h"
#include "ggml.h"
#include "src/image.h"
#include "ggml-alloc.h"
#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>
#if defined(_MSC_VER)
#pragma comment(lib, "psapi.lib")
#endif
#else
#include <sys/resource.h>
#endif
#include <algorithm>
#include <cassert>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ggml-backend.h"

#if defined(_MSC_VER)
#pragma warning(disable : 4244 4267) // possible loss of data
#endif

// Extract a short model label from a GGUF path: e.g. "models/dinov2-vit-base-patch14/model.f16.gguf"
// -> "dinov2-vit-base-patch14". Falls back to the full path if nothing matches.
static std::string model_label_from_path(const std::string &path) {
    std::string       label  = path;
    const std::string needle = "dinov2-vit-";
    const std::size_t pos    = label.find(needle);
    if (pos != std::string::npos) {
        std::size_t end = label.find('/', pos);
        if (end == std::string::npos) {
            end = label.find('.', pos);
        }
        label = label.substr(pos, end - pos);
    }
    return label;
}

// Escape '"' and '\' so a string can be embedded inside a JSON string literal.
static std::string json_escape(const std::string &s) {
    std::string out;
    out.reserve(s.size());
    for (const char c : s) {
        if (c == '"' || c == '\\') {
            out += '\\';
        }
        out += c;
    }
    return out;
}

// Look up a class label, falling back to the numeric index when the model has no label for it.
static std::string class_label(const dino_model &model, uint32_t idx) {
    const auto it = model.hparams.id2label.find(static_cast<int>(idx));
    return it != model.hparams.id2label.end() ? it->second : std::to_string(idx);
}

static void print_float_array(const std::vector<float> &v) {
    for (size_t i = 0; i < v.size(); ++i) {
        if (i > 0) {
            fprintf(stdout, ",");
        }
        fprintf(stdout, "%.6g", static_cast<double>(v[i]));
    }
}

// Per-image PCA output path for multi-input runs: <out_dir>/<input-stem>.pca.png.
static std::string pca_out_path(const std::string &out_dir, const std::string &input_path) {
    const std::string stem = std::filesystem::path(input_path).stem().string();
    return (std::filesystem::path(out_dir) / (stem + ".pca.png")).string();
}

static std::string binary_out_path(const std::string &out_path, const std::string &input_path, size_t index,
                                   bool single_input) {
    if (single_input) {
        return out_path;
    }
    std::string stem = std::filesystem::path(input_path).stem().string();
    for (char &c : stem) {
        if (!std::isalnum(static_cast<unsigned char>(c)) && c != '-' && c != '_') {
            c = '_';
        }
    }
    if (stem.empty()) {
        stem = "input";
    }
    return (std::filesystem::path(out_path) / (std::to_string(index) + "-" + stem + ".d2e")).string();
}

// Emit one JSON object on stdout with whatever output fields are set:
// cls (both modes), pooled (feature mode), topk (classify mode),
// patches (feature mode + --print-patch-tokens). With multiple inputs the
// caller prints one such line per image (JSONL).
static void print_embeddings_json(const dino_params &params, const dino_model &model, const ImageF &img_f, size_t index,
                                  const std::string &image_path, const dino_output &output) {
    const int grid_w    = img_f.nx / model.hparams.patch_size;
    const int grid_h    = img_f.ny / model.hparams.patch_size;
    const int n_patches = grid_h * grid_w;
    fprintf(stdout,
            "{\"model\":\"%s\",\"index\":%zu,\"image\":\"%s\",\"n_patches\":%d,"
            "\"grid\":{\"h\":%d,\"w\":%d},\"hidden\":%u",
            json_escape(model_label_from_path(params.model)).c_str(), index, json_escape(image_path).c_str(), n_patches,
            grid_h, grid_w, model.hparams.hidden_size);
    if (output.cls_token) {
        fprintf(stdout, ",\"cls\":[");
        print_float_array(*output.cls_token);
        fprintf(stdout, "]");
    }
    if (output.pooled) {
        fprintf(stdout, ",\"pooled\":[");
        print_float_array(*output.pooled);
        fprintf(stdout, "]");
    }
    if (params.classify && output.preds) {
        fprintf(stdout, ",\"topk\":[");
        for (size_t i = 0; i < output.preds->size(); ++i) {
            if (i > 0) {
                fprintf(stdout, ",");
            }
            const uint32_t idx  = (*output.preds)[i];
            const float    prob = (*output.pred_scores)[i];
            fprintf(stdout, "{\"idx\":%u,\"label\":\"%s\",\"prob\":%.6f}", idx,
                    json_escape(class_label(model, idx)).c_str(), static_cast<double>(prob));
        }
        fprintf(stdout, "]");
    }
    if (params.print_patch_tokens && output.patch_tokens) {
        fprintf(stdout, ",\"patches\":[");
        print_float_array(*output.patch_tokens);
        fprintf(stdout, "]");
    }
    fprintf(stdout, "}\n");
    fflush(stdout);
}

// main function
int main(int argc, char **argv) {
    ggml_time_init();
    dino_params params;
    dino_model  model;

    if (dino_params_parse(argc, argv, params) == false) {
        return 1;
    }

    if (params.embeddings_binary) {
        fprintf(stderr,
                "%s: warning: --embeddings-binary is an intentionally unstable preview format; "
                "no compatibility guarantees are provided\n",
                __func__);
    }

    if (params.fnames_inp.empty()) {
        fprintf(stderr, "%s: no input images; pass at least one -i FNAME\n", __func__);
        return 1;
    }

    // nothing-to-do guard: no output mode selected
    if (!params.classify && !params.print_embeddings && !params.embeddings_binary && params.image_out.empty() &&
        params.bench_repeats == 0) {
        fprintf(stderr,
                "%s: nothing to do: no output mode selected; choose one of:\n"
                "  -c, --classify       print top-k classification labels\n"
                "  --print-embeddings   emit embeddings JSON to stdout\n"
                "  -o FNAME, --out      write PCA output or binary embeddings file/directory\n"
                "docs: https://raw.githubusercontent.com/espetro/dinov2.cpp/main/docs/cli.md\n",
                __func__);
        return 1;
    }

    fprintf(stderr, "%s: seed = %d\n", __func__, params.seed);

    // load every input image
    std::vector<Image> imgs;
    imgs.reserve(params.fnames_inp.size());
    ImgSize max_img_size{0, 0};
    for (const std::string &path : params.fnames_inp) {
        Image img = load_image(path);
        if (img.data.empty()) {
            fprintf(stderr, "%s: failed to load image from '%s'\n", __func__, path.c_str());
            return 1;
        }
        fprintf(stderr, "%s: loaded image '%s' (%d x %d)\n", __func__, path.c_str(), img.nx, img.ny);
        max_img_size.width  = std::max(max_img_size.width, img.nx);
        max_img_size.height = std::max(max_img_size.height, img.ny);
        imgs.push_back(std::move(img));
    }

    // load the model
    if (!dino_model_load(max_img_size, params.model, model, params)) {
        fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
        fprintf(stderr,
                "%s: hint: download a model with:\n"
                "  hf download dinov2-cpp-core/dinov2-small-gguf --local-dir models\n"
                "docs: https://raw.githubusercontent.com/espetro/dinov2.cpp/main/docs/cli.md\n",
                __func__);
        return 1;
    }
    if (params.classify && params.topk > model.hparams.num_classes) {
        fprintf(stderr, "%s: --topk (%u) cannot exceed the model's %u classes\n", __func__, params.topk,
                model.hparams.num_classes);
        ggml_free(model.ctx);
        ggml_backend_buffer_free(model.buffer);
        ggml_backend_free(model.backend);
        return 1;
    }

    // error paths after this point must release the model (the Metal backend
    // aborts at process exit if its buffers were never freed)
    auto free_model = [&]() {
        ggml_free(model.ctx);
        ggml_backend_buffer_free(model.buffer);
        ggml_backend_free(model.backend);
    };

    // preprocess every input; classify mode always yields 224x224 crops
    std::vector<ImageF> imgs_f;
    imgs_f.reserve(imgs.size());
    const int64_t token_limit =
        params.max_tokens >= 0 ? params.max_tokens : (int64_t)dino_default_max_tokens(model.hparams.patch_size);
    for (size_t i = 0; i < imgs.size(); ++i) {
        const Image &img = imgs[i];
        // hard cap on patch tokens, applied on the prospective preprocessed
        // size so oversize inputs fail before the (expensive) resize and
        // before any graph is constructed
        if (token_limit > 0 && !params.classify) {
            const ImgSize out_size  = dino_feature_output_size(img, model.hparams, params);
            const int64_t n_patches = (int64_t)(out_size.height / (int)model.hparams.patch_size) *
                                      (out_size.width / (int)model.hparams.patch_size);
            if (n_patches > token_limit) {
                fprintf(stderr,
                        "error: image '%s' yields %lld patch tokens after preprocessing (limit %lld). "
                        "Use a smaller input, --preprocess crop518, or raise --max-tokens.\n",
                        params.fnames_inp[i].c_str(), (long long)n_patches, (long long)token_limit);
                free_model();
                return 1;
            }
        }
        ImageF img_f = params.classify ? dino_classify_preprocess(img, model.hparams)
                                       : dino_feature_preprocess(img, model.hparams, params);
        fprintf(stderr, "%s: preprocessed image '%s' (%d x %d)\n", __func__, params.fnames_inp[i].c_str(), img_f.nx,
                img_f.ny);
        imgs_f.push_back(std::move(img_f));
    }
    std::vector<Image>().swap(imgs);

    // one graph covers a whole chunk, so all images inside an n_batch-sized
    // chunk must share dimensions; different chunks may differ
    for (size_t s = 0; s < imgs_f.size(); s += params.n_batch) {
        const size_t e = std::min(s + (size_t)params.n_batch, imgs_f.size());
        for (size_t i = s + 1; i < e; ++i) {
            if (imgs_f[i].nx != imgs_f[s].nx || imgs_f[i].ny != imgs_f[s].ny) {
                fprintf(stderr,
                        "%s: batch chunk mixes input sizes: '%s' (%d x %d) and '%s' (%d x %d); "
                        "images in the same batch must share dimensions - group same-size inputs "
                        "together or use -c (classify preprocesses every input to 224 x 224)\n",
                        __func__, params.fnames_inp[s].c_str(), imgs_f[s].nx, imgs_f[s].ny,
                        params.fnames_inp[i].c_str(), imgs_f[i].nx, imgs_f[i].ny);
                free_model();
                return 1;
            }
        }
    }

    // With multiple inputs, -o names a directory for per-image PCA or binary files;
    // create it up front so a bad path fails before any compute.
    if (params.fnames_inp.size() > 1 && !params.image_out.empty() && !params.classify && params.bench_repeats == 0) {
        std::error_code ec;
        std::filesystem::create_directories(params.image_out, ec);
        const bool output_is_directory = std::filesystem::is_directory(params.image_out, ec);
        if (ec || !output_is_directory) {
            fprintf(stderr, "%s: failed to create output directory '%s'%s%s\n", __func__, params.image_out.c_str(),
                    ec ? ": " : ".", ec ? ec.message().c_str() : "");
            free_model();
            return 1;
        }
    }

    // prepare for graph computation, memory allocation and results processing
    {
        ggml_backend_synchronize(model.backend);
        ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
        if (!allocr) {
            fprintf(stderr, "%s: failed to create graph allocator\n", __func__);
            free_model();
            return 1;
        }

        if (params.bench_repeats == 0) {
            // Single-shot path: run the inputs through dino_predict in chunks
            // of n_batch and emit per-image outputs in input order.
            for (size_t s = 0; s < imgs_f.size(); s += params.n_batch) {
                const size_t              e       = std::min(s + (size_t)params.n_batch, imgs_f.size());
                const std::vector<ImageF> chunk   = {imgs_f.begin() + (ptrdiff_t)s, imgs_f.begin() + (ptrdiff_t)e};
                const int64_t             t0      = ggml_time_ms();
                std::vector<dino_output>  outputs = dino_predict(model, chunk, params, allocr);
                ggml_backend_synchronize(model.backend);
                const int64_t dt_ms = ggml_time_ms() - t0;
                fprintf(stderr, "%s: graph computation took %lld ms\n", __func__, dt_ms);

                if (outputs.empty()) {
                    ggml_gallocr_free(allocr);
                    free_model();
                    return 1;
                }

                for (size_t b = 0; b < outputs.size(); ++b) {
                    const size_t       idx        = s + b;
                    const dino_output &output     = outputs[b];
                    const std::string &image_path = params.fnames_inp[idx];

                    if (params.print_embeddings) {
                        print_embeddings_json(params, model, imgs_f[idx], idx, image_path, output);
                    }
                    if (params.embeddings_binary) {
                        const std::string out_path =
                            binary_out_path(params.image_out, image_path, idx, params.fnames_inp.size() == 1);
                        std::string error;
                        if (!write_embeddings_binary(out_path, output, model.hparams.hidden_size,
                                                     static_cast<uint32_t>((imgs_f[idx].ny / model.hparams.patch_size) *
                                                                           (imgs_f[idx].nx / model.hparams.patch_size)),
                                                     static_cast<uint32_t>(imgs_f[idx].nx / model.hparams.patch_size),
                                                     static_cast<uint32_t>(imgs_f[idx].ny / model.hparams.patch_size),
                                                     params.print_patch_tokens, params.l2_normalize, error)) {
                            fprintf(stderr, "%s: failed to write binary embeddings '%s': %s\n", __func__,
                                    out_path.c_str(), error.c_str());
                            ggml_gallocr_free(allocr);
                            free_model();
                            return 1;
                        }
                        fprintf(stderr, "%s: wrote binary embeddings to '%s'\n", __func__, out_path.c_str());
                    } else if (params.classify && output.preds) {
                        // multi-input: label each top-k block with its image path
                        if (params.fnames_inp.size() > 1) {
                            fprintf(stdout, "%s:\n", image_path.c_str());
                        }
                        // Human-readable top-k lines (moved out of dino_predict in the API refactor).
                        for (size_t i = 0; i < output.preds->size(); ++i) {
                            const uint32_t cls_idx = (*output.preds)[i];
                            const float    prob    = (*output.pred_scores)[i];
                            fprintf(stdout, " > %s : %.2f\n", class_label(model, cls_idx).c_str(),
                                    static_cast<double>(prob));
                        }
                        fflush(stdout);
                    }

                    if (!params.embeddings_binary && !params.classify && output.patch_tokens &&
                        !params.image_out.empty()) {
                        const std::string out_path   = params.fnames_inp.size() > 1
                                                           ? pca_out_path(params.image_out, image_path)
                                                           : params.image_out;
                        const int         patch_size = model.hparams.patch_size;
                        const int         out_w      = imgs_f[idx].nx;
                        const int         out_h      = imgs_f[idx].ny;
                        const int         n_patches  = (imgs_f[idx].ny / patch_size) * (imgs_f[idx].nx / patch_size);
                        const int         grid_w     = imgs_f[idx].nx / patch_size;
                        const int         grid_h     = imgs_f[idx].ny / patch_size;

                        pca_project_3d(*output.patch_tokens, n_patches, model.hparams.hidden_size, grid_w, grid_h,
                                       out_w, out_h, out_path);
                        fprintf(stderr, "%s: Saved image to: %s\n", __func__, out_path.c_str());
                    }
                }
            }

            ggml_free(model.ctx);
            ggml_gallocr_free(allocr);
            ggml_backend_buffer_free(model.buffer);
            ggml_backend_free(model.backend);
        } else {
            // Bench path: bench_warmup warmup runs (discarded), then bench_repeats timed runs.
            // Each run processes every input image in chunks of n_batch.
            // Peak RSS is sampled between iterations via getrusage(RUSAGE_SELF).
            // We intentionally skip the PCA visualization here -- --bench is for perf only.
            std::vector<std::vector<ImageF>> chunks;
            for (size_t s = 0; s < imgs_f.size(); s += params.n_batch) {
                const size_t e = std::min(s + (size_t)params.n_batch, imgs_f.size());
                chunks.emplace_back(imgs_f.begin() + (ptrdiff_t)s, imgs_f.begin() + (ptrdiff_t)e);
            }

            std::vector<double> samples;
            samples.reserve(params.bench_repeats);
            size_t peak_rss_kb = 0;
#ifndef _WIN32
            struct rusage ru;
#endif
            for (uint32_t i = 0; i < params.bench_warmup + params.bench_repeats; ++i) {
                int64_t t0 = ggml_time_ms();
                for (const std::vector<ImageF> &chunk : chunks) {
                    if (dino_predict(model, chunk, params, allocr).empty()) {
                        ggml_gallocr_free(allocr);
                        free_model();
                        return 1;
                    }
                }
                ggml_backend_synchronize(model.backend);
                int64_t dt_ms = ggml_time_ms() - t0;
                if (i >= params.bench_warmup) {
                    samples.push_back((double)dt_ms);
                }
#ifdef _WIN32
                PROCESS_MEMORY_COUNTERS pmc;
                if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
                    size_t kb = (size_t)pmc.PeakWorkingSetSize / 1024;
                    if (kb > peak_rss_kb) {
                        peak_rss_kb = kb;
                    }
                }
#else
                if (getrusage(RUSAGE_SELF, &ru) == 0) {
                    size_t kb = (size_t)ru.ru_maxrss;
#if defined(__APPLE__)
                    // macOS: ru_maxrss is bytes; convert to KB.
                    kb = kb / 1024;
#endif
                    if (kb > peak_rss_kb) {
                        peak_rss_kb = kb;
                    }
                }
#endif
                // per-call output vectors are dropped inside the chunk loop
            }

            // Compute summary statistics.
            double sum = 0.0;
            double mn  = samples.empty() ? 0.0 : samples[0];
            double mx  = samples.empty() ? 0.0 : samples[0];
            for (double s : samples) {
                sum += s;
                if (s < mn) {
                    mn = s;
                }
                if (s > mx) {
                    mx = s;
                }
            }
            double mean     = samples.empty() ? 0.0 : sum / (double)samples.size();
            double variance = 0.0;
            for (double s : samples) {
                double d = s - mean;
                variance += d * d;
            }
            double stddev      = samples.size() > 1 ? std::sqrt(variance / (double)(samples.size() - 1)) : 0.0;
            double peak_rss_mb = (double)peak_rss_kb / 1024.0;

            // each timed run covered every input image, so per-image
            // throughput is the mean divided by the image count
            const double n_images       = (double)imgs_f.size();
            const double ms_per_image   = n_images > 0.0 ? mean / n_images : 0.0;
            const double images_per_sec = mean > 0.0 ? n_images * 1000.0 / mean : 0.0;

            const std::string model_label = model_label_from_path(params.model);

            if (params.bench_json) {
                // One JSON object per line on stdout.
                fprintf(stdout, "{\"model\":\"%s\",\"n_threads\":%u,\"n_repeats\":%u,\"n_warmup\":%u,\"samples_ms\":[",
                        model_label.c_str(), params.n_threads, params.bench_repeats, params.bench_warmup);
                for (size_t i = 0; i < samples.size(); ++i) {
                    if (i > 0) {
                        fprintf(stdout, ",");
                    }
                    fprintf(stdout, "%.0f", samples[i]);
                }
                fprintf(stdout,
                        "],\"mean_ms\":%.1f,\"stddev_ms\":%.1f,\"min_ms\":%.0f,\"max_ms\":%.0f,"
                        "\"peak_rss_mb\":%.0f,\"n_images\":%zu,\"batch\":%u,\"ms_per_image\":%.1f,"
                        "\"images_per_sec\":%.1f}\n",
                        mean, stddev, mn, mx, peak_rss_mb, imgs_f.size(), params.n_batch, ms_per_image, images_per_sec);
                fflush(stdout);
            } else {
                fprintf(stderr,
                        "%s: bench(model=%s, n_repeats=%u, n_warmup=%u, n_threads=%u, n_images=%zu, batch=%u) "
                        "mean=%.1f ms stddev=%.1f ms min=%.0f ms max=%.0f ms peak_rss_mb=%.0f "
                        "ms_per_image=%.1f images_per_sec=%.1f\n",
                        __func__, model_label.c_str(), params.bench_repeats, params.bench_warmup, params.n_threads,
                        imgs_f.size(), params.n_batch, mean, stddev, mn, mx, peak_rss_mb, ms_per_image, images_per_sec);
            }

            ggml_free(model.ctx);
            ggml_gallocr_free(allocr);
            ggml_backend_buffer_free(model.buffer);
            ggml_backend_free(model.backend);
        }
    }

    return 0;
}
