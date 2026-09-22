#define CRT_SECURE_NO_DEPRECATE // disables "unsafe" warnings on Windows
#include "src/dinov2-impl.h"
#include "ggml.h"
#include "src/image.h"
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
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

#ifndef DINOV2_VERSION
#define DINOV2_VERSION "dev"
#endif

// CLI-only parameters: everything that is not part of the engine's
// model/ctx/run options (I/O paths, output modes, bench controls).
struct dino_cli_params {
    dino_model_options model_opts;
    dino_ctx_options   ctx_opts;
    dino_run_options   run_opts;
    int32_t            seed               = 42;    // unused: no RNG in inference; kept for CLI compatibility
    bool               print_embeddings   = false; // emit embeddings JSON on stdout (JSONL)
    bool               embeddings_binary  = false; // write preview binary embeddings to -o
    bool               print_patch_tokens = false; // include per-patch embeddings in the output
    std::string        model              = "../model.gguf";
    // input image paths; -i repeats or comma-separates to add more than one.
    // Images are forwarded to the encoder in chunks of ctx_opts.n_batch.
    std::vector<std::string> fnames_inp = {"../assets/tench.jpg"};
    std::string              image_out  = ""; // output of pca visualization (if used; a directory for multi-input)
    // Benchmark controls. bench_repeats=0 disables the bench loop (legacy single-shot path).
    // --bench with no count sets bench_repeats to 5 (the default for one-shot "is it faster").
    uint32_t bench_repeats = 0;
    uint32_t bench_warmup  = 1;
    bool     bench_json    = false;
};

// Write the intentionally unstable version-2 D2EMB preview format.
// grid_w/grid_h are the patch-grid dimensions (zero when patches are absent).
// The vectors are expected to already have the requested normalization applied.
static bool write_embeddings_binary(const std::string &path, const dino_output &output, uint32_t hidden_size,
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

static void print_usage(FILE *out, int argc, char **argv, const dino_cli_params &params) {
    fprintf(out, "usage: %s [options]\n", argv[0]);
    fprintf(out, "\n");
    fprintf(out, "Model:\n");
    fprintf(out, "  -m FNAME, --model     model path (default: %s)\n", params.model.c_str());
    fprintf(out, "  -fa, --flash_attn     enable flash attention, less accurate (default: off)\n");
    fprintf(out, "  -t N, --threads       number of threads to use during computation, 1 or greater (default: %u)\n",
            params.ctx_opts.n_threads);
    fprintf(out, "\n");
    fprintf(out, "Input:\n");
    fprintf(out, "  -i FNAME, --inp       input image file; repeat or comma-separate for several\n");
    fprintf(out, "                        (default: %s)\n",
            params.fnames_inp.empty() ? "" : params.fnames_inp.front().c_str());
    fprintf(out, "  -s N, --seed          accepted for compatibility; has no effect (default: %d)\n", params.seed);
    fprintf(out, "  --batch N             max images per forward pass; inputs run in chunks of N\n");
    fprintf(out, "                        (default: %u, max: %u)\n", params.ctx_opts.n_batch, dino_max_batch);
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
            params.run_opts.topk);
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
                                             char **argv, const dino_cli_params &params) {
    fprintf(stderr, "error: %s has invalid value '%s' (expected %s)\n", option, value, range);
    print_usage(stderr, argc, argv, params);
    exit(1);
}

static bool dino_params_parse(int argc, char **argv, dino_cli_params &params) {
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
                const std::string range = "an integer from 1 through " + std::to_string(dino_max_batch);
                numeric_parse_error(arg.c_str(), value, range.c_str(), argc, argv, params);
            }
            params.ctx_opts.n_batch = static_cast<uint32_t>(parsed);
        } else if (arg == "-o" || arg == "--out") {
            params.image_out = next_value(i);
        } else if (arg == "-t" || arg == "--threads") {
            const char *value = next_value(i);
            int32_t     parsed;
            if (!parse_integer(value, parsed) || parsed <= 0) {
                numeric_parse_error(arg.c_str(), value, "a positive 32-bit integer", argc, argv, params);
            }
            params.ctx_opts.n_threads = static_cast<uint32_t>(parsed);
        } else if (arg == "-k" || arg == "--topk") {
            const char *value = next_value(i);
            int32_t     parsed;
            if (!parse_integer(value, parsed) || parsed <= 0) {
                numeric_parse_error(arg.c_str(), value, "a positive 32-bit integer", argc, argv, params);
            }
            params.run_opts.topk = static_cast<uint32_t>(parsed);
        } else if (arg == "-fa" || arg == "--flash_attn") {
            params.ctx_opts.enable_flash_attn = true;
        } else if (arg == "-c" || arg == "--classify") {
            params.run_opts.classify             = true;
            params.model_opts.require_classifier = true;
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
            params.run_opts.l2_normalize = true;
        } else if (arg == "--preprocess") {
            const std::string value = next_value(i);
            if (value == "bounded") {
                params.ctx_opts.preprocess_mode = dino_preprocess_mode::bounded;
            } else if (value == "hf") {
                params.ctx_opts.preprocess_mode = dino_preprocess_mode::hf;
            } else if (value == "crop518") {
                params.ctx_opts.preprocess_mode = dino_preprocess_mode::crop518;
            } else {
                fprintf(stderr, "error: %s has invalid value '%s' (expected one of: bounded, hf, crop518)\n",
                        arg.c_str(), value.c_str());
                print_usage(stderr, argc, argv, params);
                exit(1);
            }
            preprocess_set = true;
        } else if (arg == "--no-resize") {
            params.ctx_opts.no_resize = true;
        } else if (arg == "--max-tokens") {
            const char *value = next_value(i);
            int32_t     parsed;
            if (!parse_integer(value, parsed) || parsed < 0) {
                numeric_parse_error(arg.c_str(), value, "a non-negative 32-bit integer", argc, argv, params);
            }
            params.ctx_opts.max_tokens = parsed;
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
    if (params.embeddings_binary && params.run_opts.classify) {
        fprintf(stderr, "error: --embeddings-binary cannot be combined with --classify\n");
        print_usage(stderr, argc, argv, params);
        exit(1);
    }
    if (params.embeddings_binary && params.bench_repeats != 0) {
        fprintf(stderr, "error: --embeddings-binary cannot be combined with --bench\n");
        print_usage(stderr, argc, argv, params);
        exit(1);
    }
    if (params.ctx_opts.no_resize && params.ctx_opts.preprocess_mode != dino_preprocess_mode::bounded) {
        fprintf(stderr, "error: --no-resize only applies to --preprocess bounded\n");
        print_usage(stderr, argc, argv, params);
        exit(1);
    }
    if (params.run_opts.classify && !params.image_out.empty()) {
        fprintf(stderr, "error: -o/--out writes feature-mode output (PCA or binary) and cannot be combined with -c\n");
        print_usage(stderr, argc, argv, params);
        exit(1);
    }
    if (params.run_opts.classify && (preprocess_set || params.ctx_opts.no_resize)) {
        fprintf(stderr, "error: --preprocess/--no-resize are feature-mode flags and cannot be combined with -c\n");
        print_usage(stderr, argc, argv, params);
        exit(1);
    }

    return true;
}

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

// Filesystem-safe stem of an input path for per-image output names.
static std::string sanitized_stem(const std::string &input_path) {
    std::string stem = std::filesystem::path(input_path).stem().string();
    for (char &c : stem) {
        if (!std::isalnum(static_cast<unsigned char>(c)) && c != '-' && c != '_') {
            c = '_';
        }
    }
    if (stem.empty()) {
        stem = "input";
    }
    return stem;
}

// Per-image PCA output path for multi-input runs: <out_dir>/<index>-<input-stem>.pca.png.
// The index disambiguates inputs that share a filename stem (a/x.png, b/x.png).
static std::string pca_out_path(const std::string &out_dir, const std::string &input_path, size_t index) {
    return (std::filesystem::path(out_dir) / (std::to_string(index) + "-" + sanitized_stem(input_path) + ".pca.png"))
        .string();
}

static std::string binary_out_path(const std::string &out_path, const std::string &input_path, size_t index,
                                   bool single_input) {
    if (single_input) {
        return out_path;
    }
    return (std::filesystem::path(out_path) / (std::to_string(index) + "-" + sanitized_stem(input_path) + ".d2e"))
        .string();
}

// Emit one JSON object on stdout with whatever output fields are set:
// cls (both modes), pooled (feature mode), topk (classify mode),
// patches (feature mode + --print-patch-tokens). With multiple inputs the
// caller prints one such line per image (JSONL).
static void print_embeddings_json(const dino_cli_params &params, const dino_model &model, const ImageF &img_f,
                                  size_t index, const std::string &image_path, const dino_output &output) {
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
    if (params.run_opts.classify && output.preds) {
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
    dino_cli_params params;
    dino_model      model;

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
    if (!params.run_opts.classify && !params.print_embeddings && !params.embeddings_binary &&
        params.image_out.empty() && params.bench_repeats == 0) {
        fprintf(stderr,
                "%s: nothing to do: no output mode selected; choose one of:\n"
                "  -c, --classify       print top-k classification labels\n"
                "  --print-embeddings   emit embeddings JSON to stdout\n"
                "  -o FNAME, --out      write PCA output or binary embeddings file/directory\n"
                "docs: https://raw.githubusercontent.com/espetro/dinov2.cpp/main/docs/cli.md\n",
                __func__);
        return 1;
    }

    // load every input image
    std::vector<Image> imgs;
    imgs.reserve(params.fnames_inp.size());
    for (const std::string &path : params.fnames_inp) {
        Image img = load_image(path);
        if (img.data.empty()) {
            fprintf(stderr, "%s: failed to load image from '%s'\n", __func__, path.c_str());
            return 1;
        }
        fprintf(stderr, "%s: loaded image '%s' (%d x %d)\n", __func__, path.c_str(), img.nx, img.ny);
        imgs.push_back(std::move(img));
    }

    // load the model
    if (!dino_model_load(params.model, model, params.model_opts)) {
        fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
        fprintf(stderr,
                "%s: hint: download a model with:\n"
                "  hf download dinov2-cpp-core/dinov2-small-gguf --local-dir models\n"
                "docs: https://raw.githubusercontent.com/espetro/dinov2.cpp/main/docs/cli.md\n",
                __func__);
        return 1;
    }
    if (params.run_opts.classify && params.run_opts.topk > model.hparams.num_classes) {
        fprintf(stderr, "%s: --topk (%u) cannot exceed the model's %u classes\n", __func__, params.run_opts.topk,
                model.hparams.num_classes);
        dino_model_unload(model);
        return 1;
    }

    // error paths after this point must release the context and model (the
    // Metal backend aborts at process exit if its buffers were never freed)
    dino_ctx ctx;
    auto     free_model = [&]() {
        dino_ctx_free(ctx);
        dino_model_unload(model);
    };

    // preprocess every input; classify mode always yields 224x224 crops
    std::vector<ImageF> imgs_f;
    imgs_f.reserve(imgs.size());
    const int64_t token_limit = params.ctx_opts.max_tokens >= 0
                                    ? params.ctx_opts.max_tokens
                                    : (int64_t)dino_default_max_tokens(model.hparams.patch_size);
    for (size_t i = 0; i < imgs.size(); ++i) {
        const Image &img = imgs[i];
        // hard cap on patch tokens, applied on the prospective preprocessed
        // size so oversize inputs fail before the (expensive) resize and
        // before any graph is constructed
        if (token_limit > 0 && !params.run_opts.classify) {
            const ImgSize out_size  = dino_feature_output_size(img, model.hparams, params.ctx_opts);
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
        ImageF img_f = params.run_opts.classify ? dino_classify_preprocess(img, model.hparams)
                                                : dino_feature_preprocess(img, model.hparams, params.ctx_opts);
        fprintf(stderr, "%s: preprocessed image '%s' (%d x %d)\n", __func__, params.fnames_inp[i].c_str(), img_f.nx,
                img_f.ny);
        imgs_f.push_back(std::move(img_f));
    }
    std::vector<Image>().swap(imgs);

    // one graph covers a whole chunk, so all images inside an n_batch-sized
    // chunk must share dimensions; different chunks may differ
    for (size_t s = 0; s < imgs_f.size(); s += params.ctx_opts.n_batch) {
        const size_t e = std::min(s + (size_t)params.ctx_opts.n_batch, imgs_f.size());
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
    if (params.fnames_inp.size() > 1 && !params.image_out.empty() && !params.run_opts.classify &&
        params.bench_repeats == 0) {
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
        if (!dino_ctx_init(ctx, model, params.ctx_opts)) {
            fprintf(stderr, "%s: failed to create inference context\n", __func__);
            free_model();
            return 1;
        }

        if (params.bench_repeats == 0) {
            // Single-shot path: run the inputs through dino_predict in chunks
            // of n_batch and emit per-image outputs in input order.
            for (size_t s = 0; s < imgs_f.size(); s += params.ctx_opts.n_batch) {
                const size_t                    e     = std::min(s + (size_t)params.ctx_opts.n_batch, imgs_f.size());
                const std::vector<ImageF>       chunk = {imgs_f.begin() + (ptrdiff_t)s, imgs_f.begin() + (ptrdiff_t)e};
                const int64_t                   t0    = ggml_time_ms();
                const std::vector<dino_output> &outputs = dino_predict(model, ctx, chunk, params.run_opts);
                ggml_backend_sched_synchronize(ctx.sched);
                const int64_t dt_ms = ggml_time_ms() - t0;
                fprintf(stderr, "%s: graph computation took %lld ms\n", __func__, dt_ms);

                if (outputs.empty()) {
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
                                                     params.print_patch_tokens, params.run_opts.l2_normalize, error)) {
                            fprintf(stderr, "%s: failed to write binary embeddings '%s': %s\n", __func__,
                                    out_path.c_str(), error.c_str());
                            free_model();
                            return 1;
                        }
                        fprintf(stderr, "%s: wrote binary embeddings to '%s'\n", __func__, out_path.c_str());
                    } else if (params.run_opts.classify && output.preds) {
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

                    if (!params.embeddings_binary && !params.run_opts.classify && output.patch_tokens &&
                        !params.image_out.empty()) {
                        const std::string out_path   = params.fnames_inp.size() > 1
                                                           ? pca_out_path(params.image_out, image_path, idx)
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

            free_model();
        } else {
            // Bench path: bench_warmup warmup runs (discarded), then bench_repeats timed runs.
            // Each run processes every input image in chunks of n_batch.
            // Peak RSS is sampled between iterations via getrusage(RUSAGE_SELF).
            // We intentionally skip the PCA visualization here -- --bench is for perf only.
            std::vector<std::vector<ImageF>> chunks;
            for (size_t s = 0; s < imgs_f.size(); s += params.ctx_opts.n_batch) {
                const size_t e = std::min(s + (size_t)params.ctx_opts.n_batch, imgs_f.size());
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
                    if (dino_predict(model, ctx, chunk, params.run_opts).empty()) {
                        free_model();
                        return 1;
                    }
                }
                ggml_backend_sched_synchronize(ctx.sched);
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
                        model_label.c_str(), params.ctx_opts.n_threads, params.bench_repeats, params.bench_warmup);
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
                        mean, stddev, mn, mx, peak_rss_mb, imgs_f.size(), params.ctx_opts.n_batch, ms_per_image,
                        images_per_sec);
                fflush(stdout);
            } else {
                fprintf(stderr,
                        "%s: bench(model=%s, n_repeats=%u, n_warmup=%u, n_threads=%u, n_images=%zu, batch=%u) "
                        "mean=%.1f ms stddev=%.1f ms min=%.0f ms max=%.0f ms peak_rss_mb=%.0f "
                        "ms_per_image=%.1f images_per_sec=%.1f\n",
                        __func__, model_label.c_str(), params.bench_repeats, params.bench_warmup,
                        params.ctx_opts.n_threads, imgs_f.size(), params.ctx_opts.n_batch, mean, stddev, mn, mx,
                        peak_rss_mb, ms_per_image, images_per_sec);
            }

            free_model();
        }
    }

    return 0;
}
