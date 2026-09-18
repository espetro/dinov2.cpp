# v0.2.0 Release Notes

Pre-built DINOv2 inference binaries, now powered by **ggml v0.24.0**, with **8 pre-converted GGUF weights** on Hugging Face and a fully reproducible CI release matrix.

## Highlights

- **ggml bumped to v0.24.0** ([#2](https://github.com/espetro/dinov2.cpp/pull/2)) — picks up the upstream inference improvements; fixes the transitive `<cmath>` drop so the build is self-contained again.
- **Hugging Face GGUF conversion + publishing pipeline** ([#2](https://github.com/espetro/dinov2.cpp/pull/2), [#3](https://github.com/espetro/dinov2.cpp/pull/3)–[#5](https://github.com/espetro/dinov2.cpp/pull/5)) — every PR that touches conversion scripts runs an end-to-end `safetensors → GGUF → HF upload` job on `ubuntu-latest-large`. All 8 official checkpoints (base/small/large/giant plus the `with-registers` variants) are now published under the [`dinov2-cpp-core`](https://huggingface.co/dinov2-cpp-core) org — no conversion required to run inference.
- **CI release matrix for macOS / Linux / Windows prebuilts** ([#1](https://github.com/espetro/dinov2.cpp/pull/1)) — push a `v*` tag, the `release` workflow builds 4 platform tarballs/zip plus a SHA256SUMS manifest and attaches them to the GitHub Release.
- **Dev harness: doctest + sanitizers + clang-format gate** ([#1](https://github.com/espetro/dinov2.cpp/pull/1)) — `cmake -DBUILD_TESTS=ON` brings in doctest; the `test` job builds with `-fsanitize=address,undefined` on Linux/macOS; a `format-check` job enforces clang-format on the touched file set.
- **OpenCV removed** ([#1](https://github.com/espetro/dinov2.cpp/pull/1)) — zero runtime system dependencies. The binary loads any format ggml can decode, so the heavy image I/O layer is gone.

## What's in the box

- `inference` — CLI for image embedding extraction (`--model`, `--image`, `--format text|json|npy`, `--threads`, etc.).
- `quantize` — GGUF quantization helper (default `Q8_0`).
- Same CLI surface as v0.1.0, rebuilt against ggml v0.24.0.

## Downloads

| Platform | Architecture | Asset |
| --- | --- | --- |
| macOS | arm64 | `dinov2-v0.2.0-bin-macos-arm64.tar.gz` |
| Linux (Ubuntu) | x86_64 | `dinov2-v0.2.0-bin-ubuntu-x64.tar.gz` |
| Linux (Ubuntu) | arm64 | `dinov2-v0.2.0-bin-ubuntu-arm64.tar.gz` |
| Windows | x86_64 | `dinov2-bin-win-cpu-x64.zip` |

Plus `SHA256SUMS.txt` covering all four archives.

Verify with `shasum -a 256 -c SHA256SUMS.txt` (or `sha256sum -c` on Linux).

## Pre-converted GGUF weights

All 8 official DINOv2 checkpoints are pre-converted and hosted on Hugging Face under the [`dinov2-cpp-core`](https://huggingface.co/dinov2-cpp-core) org. No conversion step required — point `--model` at the repo ID and `huggingface-cli` will resolve the GGUF on first run (or download with `huggingface-cli download <repo> --local-dir ./models/<variant>`).

| Variant | HF repo |
| --- | --- |
| ViT-S/14 (base) | [`dinov2-cpp-core/dinov2-small-gguf`](https://huggingface.co/dinov2-cpp-core/dinov2-small-gguf) |
| ViT-B/14 (base) | [`dinov2-cpp-core/dinov2-base-gguf`](https://huggingface.co/dinov2-cpp-core/dinov2-base-gguf) |
| ViT-L/14 (base) | [`dinov2-cpp-core/dinov2-large-gguf`](https://huggingface.co/dinov2-cpp-core/dinov2-large-gguf) |
| ViT-g/14 (base) | [`dinov2-cpp-core/dinov2-giant-gguf`](https://huggingface.co/dinov2-cpp-core/dinov2-giant-gguf) |
| ViT-S/14 (with registers) | [`dinov2-cpp-core/dinov2-with-registers-small-gguf`](https://huggingface.co/dinov2-cpp-core/dinov2-with-registers-small-gguf) |
| ViT-B/14 (with registers) | [`dinov2-cpp-core/dinov2-with-registers-base-gguf`](https://huggingface.co/dinov2-cpp-core/dinov2-with-registers-base-gguf) |
| ViT-L/14 (with registers) | [`dinov2-cpp-core/dinov2-with-registers-large-gguf`](https://huggingface.co/dinov2-cpp-core/dinov2-with-registers-large-gguf) |
| ViT-g/14 (with registers) | [`dinov2-cpp-core/dinov2-with-registers-giant-gguf`](https://huggingface.co/dinov2-cpp-core/dinov2-with-registers-giant-gguf) |

## CI / release hardening (since v0.1.0)

PRs [#2](https://github.com/espetro/dinov2.cpp/pull/2)–[#8](https://github.com/espetro/dinov2.cpp/pull/8) shipped the following release-engineering work on top of the v0.1.0 release pipeline:

- `chore(deps): bump ggml to v0.24.0` ([#2](https://github.com/espetro/dinov2.cpp/pull/2))
- `ci: hf gguf conversion + publishing workflow` ([#2](https://github.com/espetro/dinov2.cpp/pull/2))
- `docs: hf publishing guide` + README rewrite + GGUF weights table ([#2](https://github.com/espetro/dinov2.cpp/pull/2))
- `ci(refactor): port gguf publish to bash script` + `ubuntu-latest-large` runner ([#3](https://github.com/espetro/dinov2.cpp/pull/3))
- `ci(cache): cache .venv-publish + document HF_TOKEN` ([#4](https://github.com/espetro/dinov2.cpp/pull/4))
- `ci(gguf): fit conversion within ubuntu-latest (~14 GB)` via lean deps + cache cleanup ([#5](https://github.com/espetro/dinov2.cpp/pull/5))
- `ci(gguf): hardcode HF_HOME path (runner.home not allowed in env:)` ([#6](https://github.com/espetro/dinov2.cpp/pull/6))
- `perf(gguf): relocate caches to /mnt scratch, hf_xet uploader, dynamic matrix, per-run audit logs` ([#7](https://github.com/espetro/dinov2.cpp/pull/7))
- `fix(ci): emit JSON array from plan job so fromJson(matrix) parses correctly` ([#8](https://github.com/espetro/dinov2.cpp/pull/8))

## Upgrade notes

- The CLI is unchanged from v0.1.0 — existing scripts work as-is.
- ggml v0.24.0 changes mean GGUF files produced against older ggml versions may need to be regenerated against these binaries (the HF repos above are all ggml v0.24.0 compatible).
- Windows binary is built with `BUILD_SHARED_LIBS=OFF` and `CMAKE_INSTALL_RPATH=$ORIGIN`; no DLLs need to be on `PATH`.

## SHA-256

See `SHA256SUMS.txt` on this release page.

## Thanks

- Meta AI for the DINOv2 checkpoints.
- ggml-org for the inference backend.
- Everyone testing the conversion pipeline and reporting issues.
