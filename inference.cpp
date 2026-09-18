#define CRT_SECURE_NO_DEPRECATE // disables "unsafe" warnings on Windows
#include "dinov2.h"
#include "ggml.h"
#include "src/image.h"
#include "ggml-alloc.h"
#include <sys/resource.h>
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

// main function
int main(int argc, char **argv) {
    ggml_time_init();
    dino_params params;
    dino_model  model;

    if (dino_params_parse(argc, argv, params) == false) {
        return 1;
    }

    fprintf(stderr, "%s: seed = %d\n", __func__, params.seed);

    // load the image
    Image img = load_image(params.fname_inp);
    if (img.data.empty()) {
        fprintf(stderr, "%s: failed to load image from '%s'\n", __func__, params.fname_inp.c_str());
        return 1;
    }
    fprintf(stderr, "%s: loaded image '%s' (%d x %d)\n", __func__, params.fname_inp.c_str(), img.nx, img.ny);

    // load the model
    if (!dino_model_load({img.nx, img.ny}, params.model, model, params)) {
        fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
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
            // Legacy single-shot path: one timed forward pass, then PCA visualization.
            int64_t                      t0     = ggml_time_ms();
            std::unique_ptr<dino_output> output = dino_predict(model, img_f, params, allocr);
            ggml_backend_synchronize(model.backend);
            int64_t dt_ms = ggml_time_ms() - t0;
            fprintf(stderr, "%s: graph computation took %lld ms\n", __func__, dt_ms);

            ggml_free(model.ctx);
            ggml_gallocr_free(allocr);
            ggml_backend_buffer_free(model.buffer);
            ggml_backend_free(model.backend);

            if (!params.classify && output->patch_tokens) {
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
            size_t        peak_rss_kb = 0;
            struct rusage ru;
            for (uint32_t i = 0; i < params.bench_warmup + params.bench_repeats; ++i) {
                int64_t                      t0     = ggml_time_ms();
                std::unique_ptr<dino_output> output = dino_predict(model, img_f, params, allocr);
                ggml_backend_synchronize(model.backend);
                int64_t dt_ms = ggml_time_ms() - t0;
                if (i >= params.bench_warmup) {
                    samples.push_back((double) dt_ms);
                }
                if (getrusage(RUSAGE_SELF, &ru) == 0) {
                    size_t kb = (size_t) ru.ru_maxrss;
#if defined(__APPLE__)
                    // macOS: ru_maxrss is bytes; convert to KB.
                    kb = kb / 1024;
#endif
                    if (kb > peak_rss_kb) {
                        peak_rss_kb = kb;
                    }
                }
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
            double mean    = samples.empty() ? 0.0 : sum / (double) samples.size();
            double variance = 0.0;
            for (double s : samples) {
                double d = s - mean;
                variance += d * d;
            }
            double stddev = samples.size() > 1 ? std::sqrt(variance / (double) (samples.size() - 1)) : 0.0;
            double peak_rss_mb = (double) peak_rss_kb / 1024.0;

            // Extract a short model label from the GGUF path: e.g. "models/dinov2-vit-base-patch14/model.f16.gguf"
            // -> "dinov2-vit-base-patch14". Falls back to the full path if nothing matches.
            std::string model_label = params.model;
            {
                std::string needle = "dinov2-vit-";
                std::size_t pos     = model_label.find(needle);
                if (pos != std::string::npos) {
                    std::size_t end = model_label.find('/', pos);
                    if (end == std::string::npos) {
                        end = model_label.find('.', pos);
                    }
                    model_label = model_label.substr(pos, end - pos);
                }
            }

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
                fprintf(stdout, "],\"mean_ms\":%.1f,\"stddev_ms\":%.1f,\"min_ms\":%.0f,\"max_ms\":%.0f,"
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
