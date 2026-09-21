#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#ifdef _WIN32
#include <process.h>
#else
#include <sys/wait.h>
#include <unistd.h>
#endif
#include <vector>

namespace {

static int process_id() {
#ifdef _WIN32
    return _getpid();
#else
    return getpid();
#endif
}

static const std::filesystem::path &test_directory() {
    static const std::filesystem::path directory = [] {
        const auto            now  = std::chrono::steady_clock::now().time_since_epoch().count();
        std::filesystem::path base = std::filesystem::temp_directory_path();
        for (unsigned int attempt = 0; attempt < 100; ++attempt) {
            const auto      candidate = base / ("dinov2-cli-test-" + std::to_string(process_id()) + "-" +
                                           std::to_string(now) + "-" + std::to_string(attempt));
            std::error_code ec;
            if (std::filesystem::create_directory(candidate, ec)) {
                return candidate;
            }
        }
        std::fprintf(stderr, "failed to create unique test directory\n");
        std::abort();
    }();
    return directory;
}

struct TempDirectoryCleanup {
    ~TempDirectoryCleanup() {
        std::error_code ec;
        std::filesystem::remove_all(test_directory(), ec);
    }
};

static std::string shell_quote(const std::string &value) {
    std::string quoted = "'";
    for (const char c : value) {
        if (c == '\'') {
            quoted += "'\\''";
        } else {
            quoted += c;
        }
    }
    quoted += "'";
    return quoted;
}

static int run_cli(const std::string &cli, const std::vector<std::string> &args, std::string &stdout_output,
                   std::string &stderr_output) {
    static unsigned int invocation    = 0;
    const std::string   output_prefix = (test_directory() / ("run-" + std::to_string(invocation++))).string();
    const std::string   stdout_path   = output_prefix + "-stdout.txt";
    const std::string   stderr_path   = output_prefix + "-stderr.txt";
    std::string         command       = shell_quote(cli);
    for (const std::string &arg : args) {
        command += " " + shell_quote(arg);
    }
    command += " >" + shell_quote(stdout_path) + " 2>" + shell_quote(stderr_path);

    const int     status = std::system(command.c_str());
    std::ifstream stdout_file(stdout_path);
    stdout_output.assign(std::istreambuf_iterator<char>(stdout_file), std::istreambuf_iterator<char>());
    std::ifstream stderr_file(stderr_path);
    stderr_output.assign(std::istreambuf_iterator<char>(stderr_file), std::istreambuf_iterator<char>());
    std::remove(stdout_path.c_str());
    std::remove(stderr_path.c_str());

    if (status == -1) {
        return -1;
    }
#ifdef _WIN32
    return status;
#else
    if (!WIFEXITED(status)) {
        return -1;
    }
    return WEXITSTATUS(status);
#endif
}

static bool write_minimal_gguf(const std::string &path, bool add_register_tensor, int64_t register_hidden = 4,
                               int64_t register_count = 1, int64_t trailing_dim = 1, bool add_register_metadata = false,
                               bool add_classifier = false, int64_t metadata_count = 0) {
    gguf_context *gguf = gguf_init_empty();
    if (add_register_metadata) {
        gguf_set_val_u32(gguf, "num_register_tokens",
                         static_cast<uint32_t>(metadata_count > 0 ? metadata_count : register_count));
    }
    gguf_set_val_u32(gguf, "hidden_size", 4);
    gguf_set_val_u32(gguf, "num_hidden_layers", 0);
    gguf_set_val_u32(gguf, "num_attention_heads", 1);
    gguf_set_val_u32(gguf, "patch_size", 14);
    gguf_set_val_u32(gguf, "img_size", 224);
    gguf_set_val_u32(gguf, "ftype", 1);

    ggml_context *tensor_ctx = nullptr;
    if (add_register_tensor || add_classifier) {
        const size_t elements =
            add_register_tensor ? static_cast<size_t>(register_hidden) * register_count * trailing_dim : 0;
        ggml_init_params init_params = {ggml_tensor_overhead() * 4 + sizeof(float) * (elements + 12), nullptr, false};
        tensor_ctx                   = ggml_init(init_params);
        if (add_register_tensor) {
            const int64_t dims[]          = {register_hidden, register_count, trailing_dim, 1};
            ggml_tensor  *register_tokens = ggml_new_tensor(tensor_ctx, GGML_TYPE_F32, 4, dims);
            ggml_set_name(register_tokens, "embeddings.register_tokens");
            for (size_t i = 0; i < elements; ++i) {
                static_cast<float *>(register_tokens->data)[i] = static_cast<float>(i) / 10.0f;
            }
            gguf_add_tensor(gguf, register_tokens);
        }
    }
    if (add_classifier) {
        constexpr uint32_t num_classes = 2;
        gguf_set_val_u32(gguf, "num_classes", num_classes);
        for (uint32_t i = 0; i < num_classes; ++i) {
            gguf_set_val_str(gguf, std::to_string(i).c_str(), std::to_string(i).c_str());
        }
        ggml_tensor *weight = ggml_new_tensor_2d(tensor_ctx, GGML_TYPE_F32, 4, num_classes);
        ggml_set_name(weight, "classifier.weight");
        ggml_tensor *bias = ggml_new_tensor_1d(tensor_ctx, GGML_TYPE_F32, num_classes);
        ggml_set_name(bias, "classifier.bias");
        gguf_add_tensor(gguf, weight);
        gguf_add_tensor(gguf, bias);
    }

    const bool written = gguf_write_to_file(gguf, path.c_str(), false);
    if (tensor_ctx) {
        ggml_free(tensor_ctx);
    }
    gguf_free(gguf);
    return written;
}

} // namespace

int main(int argc, char **argv) {
    TempDirectoryCleanup cleanup;
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s DINOV2_CLI\n", argv[0]);
        return 2;
    }

    const std::string                           cli   = argv[1];
    const std::vector<std::vector<std::string>> cases = {
        {"--seed", "12abc"},
        {"--seed", "+-5"},
        {"--threads", "0"},
        {"--topk", "999999999999999999999"},
        {"--batch", "65"},
        {"--bench-runs", "1.5"},
        {"--bench-warmup", "-1"},
        {"--batch", " 1"},
        {"--seed"},
        {"--embeddings-binary"},
        {"--embeddings-binary", "-c", "-o", (test_directory() / "output").string()},
        {"--embeddings-binary", "--bench", "-o", (test_directory() / "output").string()},
        {"--preprocess", "bogus"},
        {"--max-tokens", "12abc"},
        {"--max-tokens", "-1"},
        {"--no-resize", "--preprocess", "hf"},
        {"-c", "--preprocess", "hf"},
        {"-c", "--no-resize"},
    };
    for (const auto &args : cases) {
        std::string stdout_output;
        std::string stderr_output;
        const int   status = run_cli(cli, args, stdout_output, stderr_output);
        if (status != 1 || !stdout_output.empty() || stderr_output.find("error:") == std::string::npos ||
            stderr_output.find("usage:") == std::string::npos) {
            return 1;
        }
    }

    // A valid backbone-only GGUF must fail classification during loader
    // preflight, before any missing classifier tensor can be accessed.
    const std::string backbone_model = (test_directory() / "backbone.gguf").string();
    if (!write_minimal_gguf(backbone_model, false)) {
        return 1;
    }
    std::string loader_stdout;
    std::string loader_stderr;
    const int   loader_status =
        run_cli(cli, {"-m", backbone_model, "-i", "../assets/tench.jpg", "-c"}, loader_stdout, loader_stderr);
    std::remove(backbone_model.c_str());
    if (loader_status != 1 || !loader_stdout.empty() ||
        loader_stderr.find("num_register_tokens    = 0") == std::string::npos ||
        loader_stderr.find("no non-zero num_classes metadata") == std::string::npos ||
        loader_stderr.find("assert") != std::string::npos) {
        return 1;
    }

    // A register tensor without its count metadata is ambiguous and must not
    // be loaded as a zero-register backbone.
    const std::string inconsistent_model = (test_directory() / "registers-missing-metadata.gguf").string();
    if (!write_minimal_gguf(inconsistent_model, true)) {
        return 1;
    }
    loader_stdout.clear();
    loader_stderr.clear();
    const int inconsistent_status =
        run_cli(cli, {"-m", inconsistent_model, "-i", "../assets/tench.jpg", "-c"}, loader_stdout, loader_stderr);
    std::remove(inconsistent_model.c_str());
    if (inconsistent_status != 1 || !loader_stdout.empty() ||
        loader_stderr.find("missing num_register_tokens metadata") == std::string::npos ||
        loader_stderr.find("assert") != std::string::npos) {
        return 1;
    }

    const auto expect_register_shape_failure = [&](const std::string &name, int64_t hidden, int64_t count,
                                                   int64_t trailing, const char *needle, int64_t metadata_count = 0) {
        const std::string path = (test_directory() / name).string();
        if (!write_minimal_gguf(path, true, hidden, count, trailing, true, false, metadata_count)) {
            return false;
        }
        std::string out;
        std::string err;
        const int   status = run_cli(cli, {"-m", path, "-i", "../assets/tench.jpg", "--print-embeddings"}, out, err);
        std::remove(path.c_str());
        const bool passed = status == 1 && out.empty() && err.find(needle) != std::string::npos;
        return passed;
    };
    if (!expect_register_shape_failure("registers-hidden-mismatch.gguf", 3, 1, 1, "expected [4, 1, 1, 1]") ||
        !expect_register_shape_failure("registers-count-mismatch.gguf", 4, 1, 1, "expected [4, 2, 1, 1]", 2) ||
        !expect_register_shape_failure("registers-trailing-mismatch.gguf", 4, 1, 2, "singleton trailing dimensions")) {
        return 1;
    }

    const std::string topk_model = (test_directory() / "topk.gguf").string();
    if (!write_minimal_gguf(topk_model, false, 4, 1, 1, false, true)) {
        return 1;
    }
    std::string topk_stdout;
    std::string topk_stderr;
    const int   topk_status =
        run_cli(cli, {"-m", topk_model, "-i", "../assets/tench.jpg", "-c", "--topk", "3"}, topk_stdout, topk_stderr);
    std::remove(topk_model.c_str());
    if (topk_status != 1 || !topk_stdout.empty() ||
        topk_stderr.find("cannot exceed the model's 2 classes") == std::string::npos) {
        return 1;
    }

    // --max-tokens 1 must reject any real input after model load, before
    // graph construction, with a clean stderr message and exit 1.
    const std::string cap_model = (test_directory() / "cap.gguf").string();
    if (!write_minimal_gguf(cap_model, false, 4, 1, 1, false, true)) {
        return 1;
    }
    std::string cap_stdout;
    std::string cap_stderr;
    const int   cap_status =
        run_cli(cli, {"-m", cap_model, "-i", "../assets/tench.jpg", "--print-embeddings", "--max-tokens", "1"},
                cap_stdout, cap_stderr);
    std::remove(cap_model.c_str());
    if (cap_status != 1 || !cap_stdout.empty() ||
        cap_stderr.find("patch tokens after preprocessing (limit 1)") == std::string::npos ||
        cap_stderr.find("assert") != std::string::npos) {
        return 1;
    }

    // Record contract: when the real small GGUF is present, a two-input
    // --print-embeddings run must emit one JSONL record per input, in input
    // order, with index/grid/n_patches consistent. Skip silently otherwise.
    const std::string small_model = "../models/dinov2-small/model.gguf";
    if (std::filesystem::exists(small_model)) {
        std::string jsonl_out;
        std::string jsonl_err;
        const int   jsonl_status = run_cli(
            cli, {"-m", small_model, "-i", "../assets/tench.jpg", "-i", "../assets/tench.jpg", "--print-embeddings"},
            jsonl_out, jsonl_err);
        if (jsonl_status != 0) {
            return 1;
        }
        // tench is 612x408; aligned dims 616x420 -> 44x30 grid, 1320 patches
        std::istringstream lines(jsonl_out);
        std::string        line;
        size_t             n = 0;
        while (std::getline(lines, line)) {
            if (line.empty()) {
                continue;
            }
            const std::string index_key = "\"index\":" + std::to_string(n);
            if (line.find(index_key) == std::string::npos || line.find("\"n_patches\":1320") == std::string::npos ||
                line.find("\"grid\":{\"h\":30,\"w\":44}") == std::string::npos ||
                line.find("\"image\":\"../assets/tench.jpg\"") == std::string::npos) {
                return 1;
            }
            ++n;
        }
        if (n != 2) {
            return 1;
        }
    }

    std::string stdout_output;
    std::string stderr_output;
    if (run_cli(cli, {"--help"}, stdout_output, stderr_output) != 0 || !stderr_output.empty() ||
        stdout_output.find("usage:") == std::string::npos) {
        return 1;
    }
    if (run_cli(cli, {"--version"}, stdout_output, stderr_output) != 0 || !stderr_output.empty() ||
        stdout_output.find("dinov2-cli") == std::string::npos) {
        return 1;
    }
    return 0;
}
