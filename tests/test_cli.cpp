#include <cstdio>
#include <cstdlib>
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

} // namespace

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s DINOV2_CLI\n", argv[0]);
        return 2;
    }

    const std::string                           cli   = argv[1];
    const std::vector<std::vector<std::string>> cases = {
        {"--seed", "12abc"}, {"--seed", "+-5"},       {"--threads", "0"},       {"--topk", "999999999999999999999"},
        {"--batch", "65"},   {"--bench-runs", "1.5"}, {"--bench-warmup", "-1"}, {"--batch", " 1"},
        {"--seed"},
    };
    for (const auto &args : cases) {
        std::string stdout_output;
        std::string stderr_output;
        if (run_cli(cli, args, stdout_output, stderr_output) != 1 || !stdout_output.empty() ||
            stderr_output.find("error:") == std::string::npos || stderr_output.find("usage:") == std::string::npos) {
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
