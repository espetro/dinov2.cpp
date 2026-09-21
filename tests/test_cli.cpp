#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
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
    const std::string output_prefix = "/tmp/dinov2-cli-test-" + std::to_string(process_id());
    const std::string stdout_path   = output_prefix + "-stdout.txt";
    const std::string stderr_path   = output_prefix + "-stderr.txt";
    std::string       command       = shell_quote(cli);
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

static bool write_minimal_gguf(const std::string &path, bool add_register_tensor) {
    gguf_context *gguf = gguf_init_empty();
    gguf_set_val_u32(gguf, "hidden_size", 4);
    gguf_set_val_u32(gguf, "num_hidden_layers", 0);
    gguf_set_val_u32(gguf, "num_attention_heads", 1);
    gguf_set_val_u32(gguf, "patch_size", 14);
    gguf_set_val_u32(gguf, "img_size", 224);
    gguf_set_val_u32(gguf, "ftype", 1);

    ggml_context *tensor_ctx = nullptr;
    if (add_register_tensor) {
        ggml_init_params init_params = {ggml_tensor_overhead() * 2 + sizeof(float) * 4, nullptr, false};
        tensor_ctx                   = ggml_init(init_params);
        ggml_tensor *register_tokens = ggml_new_tensor_2d(tensor_ctx, GGML_TYPE_F32, 4, 1);
        ggml_set_name(register_tokens, "embeddings.register_tokens");
        static const float values[] = {0.0f, 0.1f, 0.2f, 0.3f};
        std::memcpy(register_tokens->data, values, sizeof(values));
        gguf_add_tensor(gguf, register_tokens);
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
        {"--embeddings-binary", "-c", "-o", "/tmp/dinov2-cli-test-output"},
        {"--embeddings-binary", "--bench", "-o", "/tmp/dinov2-cli-test-output"},
    };
    for (const auto &args : cases) {
        std::string stdout_output;
        std::string stderr_output;
        if (run_cli(cli, args, stdout_output, stderr_output) != 1 || !stdout_output.empty() ||
            stderr_output.find("error:") == std::string::npos || stderr_output.find("usage:") == std::string::npos) {
            return 1;
        }
    }

    // A valid backbone-only GGUF must fail classification during loader
    // preflight, before any missing classifier tensor can be accessed.
    const std::string backbone_model = "/tmp/dinov2-cli-test-backbone.gguf";
    if (!write_minimal_gguf(backbone_model, false)) {
        return 1;
    }
    std::string loader_stdout;
    std::string loader_stderr;
    const int   loader_status = run_cli(cli, {"-m", backbone_model, "-c"}, loader_stdout, loader_stderr);
    std::remove(backbone_model.c_str());
    if (loader_status != 1 || !loader_stdout.empty() ||
        loader_stderr.find("num_register_tokens    = 0") == std::string::npos ||
        loader_stderr.find("no non-zero num_classes metadata") == std::string::npos ||
        loader_stderr.find("assert") != std::string::npos) {
        return 1;
    }

    // A register tensor without its count metadata is ambiguous and must not
    // be loaded as a zero-register backbone.
    const std::string inconsistent_model = "/tmp/dinov2-cli-test-registers.gguf";
    if (!write_minimal_gguf(inconsistent_model, true)) {
        return 1;
    }
    loader_stdout.clear();
    loader_stderr.clear();
    const int inconsistent_status = run_cli(cli, {"-m", inconsistent_model, "-c"}, loader_stdout, loader_stderr);
    std::remove(inconsistent_model.c_str());
    if (inconsistent_status != 1 || !loader_stdout.empty() ||
        loader_stderr.find("missing num_register_tokens metadata") == std::string::npos ||
        loader_stderr.find("assert") != std::string::npos) {
        return 1;
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
