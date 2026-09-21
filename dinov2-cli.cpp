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
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
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

// Emit one JSON object on stdout with whatever output fields are set:
// cls (both modes), pooled (feature mode), topk (classify mode),
// patches (feature mode + --print-patch-tokens).
static void print_embeddings_json(const dino_params &params, const dino_model &model, const ImageF &img_f,
                                  const dino_output &output) {
    const int n_patches = (img_f.ny / model.hparams.patch_size) * (img_f.nx / model.hparams.patch_size);
    fprintf(stdout, "{\"model\":\"%s\",\"image\":\"%s\",\"n_patches\":%d,\"hidden\":%u",
            json_escape(model_label_from_path(params.model)).c_str(), json_escape(params.fnames_inp.front()).c_str(),
            n_patches, model.hparams.hidden_size);
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

    if (params.fnames_inp.empty()) {
        fprintf(stderr, "%s: no input images; pass at least one -i FNAME\n", __func__);
        return 1;
    }

    // nothing-to-do guard: no output mode selected
    if (!params.classify && !params.print_embeddings && params.image_out.empty() && params.bench_repeats == 0) {
        fprintf(stderr,
                "%s: nothing to do: no output mode selected; choose one of:\n"
                "  -c, --classify       print top-k classification labels\n"
                "  --print-embeddings   emit embeddings JSON to stdout\n"
                "  -o FNAME, --out      write PCA visualization of patch features to FNAME\n"
                "docs: https://raw.githubusercontent.com/espetro/dinov2.cpp/main/docs/cli.md\n",
                __func__);
        return 1;
    }

    fprintf(stderr, "%s: seed = %d\n", __func__, params.seed);

    // load the image
    Image img = load_image(params.fnames_inp.front());
    if (img.data.empty()) {
        fprintf(stderr, "%s: failed to load image from '%s'\n", __func__, params.fnames_inp.front().c_str());
        return 1;
    }
    fprintf(stderr, "%s: loaded image '%s' (%d x %d)\n", __func__, params.fnames_inp.front().c_str(), img.nx, img.ny);

    // load the model
    if (!dino_model_load({img.nx, img.ny}, params.model, model, params)) {
        fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
        fprintf(stderr,
                "%s: hint: download a model with:\n"
                "  huggingface-cli download dinov2-cpp-core/dinov2-small-gguf --local-dir models\n"
                "docs: https://raw.githubusercontent.com/espetro/dinov2.cpp/main/docs/cli.md\n",
                __func__);
        return 1;
    }

    ImageF img_f;
    if (params.classify) {
        img_f = dino_classify_preprocess(img, model.hparams);
    } else {
        img_f = dino_preprocess(img, model.hparams);
    }

    fprintf(stderr, "%s: preprocessed image (%d x %d)\n", __func__, img_f.nx, img_f.ny);

    // prepare for graph computation, memory allocation and results processing
    {
        ggml_backend_synchronize(model.backend);
        ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));

        if (params.bench_repeats == 0) {
            // Legacy single-shot path: one timed forward pass, then optional output modes.
            int64_t                      t0     = ggml_time_ms();
            std::unique_ptr<dino_output> output = dino_predict(model, img_f, params, allocr);
            ggml_backend_synchronize(model.backend);
            int64_t dt_ms = ggml_time_ms() - t0;
            fprintf(stderr, "%s: graph computation took %lld ms\n", __func__, dt_ms);

            if (!output) {
                return 1;
            }

            if (params.print_embeddings) {
                print_embeddings_json(params, model, img_f, *output);
            } else if (params.classify && output->preds) {
                // Human-readable top-k lines (moved out of dino_predict in the API refactor).
                for (size_t i = 0; i < output->preds->size(); ++i) {
                    const uint32_t idx  = (*output->preds)[i];
                    const float    prob = (*output->pred_scores)[i];
                    fprintf(stdout, " > %s : %.2f\n", class_label(model, idx).c_str(), static_cast<double>(prob));
                }
                fflush(stdout);
            }

            ggml_free(model.ctx);
            ggml_gallocr_free(allocr);
            ggml_backend_buffer_free(model.buffer);
            ggml_backend_free(model.backend);

            if (!params.classify && output->patch_tokens && !params.image_out.empty()) {
                const int patch_size = model.hparams.patch_size;
                const int out_w      = img_f.nx;
                const int out_h      = img_f.ny;
                const int n_patches  = (img_f.ny / patch_size) * (img_f.nx / patch_size);
                const int grid_w     = img_f.nx / patch_size;
                const int grid_h     = img_f.ny / patch_size;

                pca_project_3d(*output->patch_tokens, n_patches, model.hparams.hidden_size, grid_w, grid_h, out_w,
                               out_h, params.image_out);
                fprintf(stderr, "%s: Saved image to: %s\n", __func__, params.image_out.c_str());
            }
        } else {
            // Bench path: bench_warmup warmup runs (discarded), then bench_repeats timed runs.
            // Peak RSS is sampled between iterations via getrusage(RUSAGE_SELF).
            // We intentionally skip the PCA visualization here -- --bench is for perf only.
            std::vector<double> samples;
            samples.reserve(params.bench_repeats);
            size_t peak_rss_kb = 0;
#ifndef _WIN32
            struct rusage ru;
#endif
            for (uint32_t i = 0; i < params.bench_warmup + params.bench_repeats; ++i) {
                int64_t                      t0     = ggml_time_ms();
                std::unique_ptr<dino_output> output = dino_predict(model, img_f, params, allocr);
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
                // Drop `output` to free any per-call buffers before the next iteration.
                output.reset();
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
                        "\"peak_rss_mb\":%.0f}\n",
                        mean, stddev, mn, mx, peak_rss_mb);
                fflush(stdout);
            } else {
                fprintf(stderr,
                        "%s: bench(model=%s, n_repeats=%u, n_warmup=%u, n_threads=%u) mean=%.1f ms "
                        "stddev=%.1f ms min=%.0f ms max=%.0f ms peak_rss_mb=%.0f\n",
                        __func__, model_label.c_str(), params.bench_repeats, params.bench_warmup, params.n_threads,
                        mean, stddev, mn, mx, peak_rss_mb);
            }

            ggml_free(model.ctx);
            ggml_gallocr_free(allocr);
            ggml_backend_buffer_free(model.buffer);
            ggml_backend_free(model.backend);
        }
    }

    return 0;
}
