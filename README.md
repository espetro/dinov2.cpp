<p align="center"><img src="assets/logo/logo-256.png" width="128" alt="dinov2.cpp"></p>

# dinov2.cpp

Run DINOv2 vision models in pure C++ on ggml. No Python, no PyTorch, no system dependencies at runtime.

[![Release](https://img.shields.io/github/v/release/espetro/dinov2.cpp)](https://github.com/espetro/dinov2.cpp/releases)
[![CI](https://github.com/espetro/dinov2.cpp/actions/workflows/build.yml/badge.svg)](https://github.com/espetro/dinov2.cpp/actions/workflows/build.yml)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C.svg)

**See the [GitHub releases page](https://github.com/espetro/dinov2.cpp/releases/latest) for the latest prebuilt binaries.**

**Release post & benchmarks → https://alexlavaee.me/projects/dinov2cpp/**

## Quick start

### Quickstart with agents

Using Claude Code, Cursor or another coding agent? Paste this:

```text
Install https://github.com/espetro/dinov2.cpp via mise (latest), download the small GGUF from the dinov2-cpp-core profile on Hugging Face, and run classification plus `--print-embeddings` on an example image to verify it works.
```

### Manual install

Grab a prebuilt binary, download a weight, run one command. No toolchain needed.

**1. Install a prebuilt binary** from [releases](https://github.com/espetro/dinov2.cpp/releases):

```bash
# Resolve the latest release tag (tarball names embed it)
TAG=$(basename "$(curl -sIL -o /dev/null -w '%{url_effective}' https://github.com/espetro/dinov2.cpp/releases/latest)")

# macOS (Apple Silicon)
curl -LO "https://github.com/espetro/dinov2.cpp/releases/download/$TAG/dinov2-$TAG-bin-macos-arm64.tar.gz"
tar xzf "dinov2-$TAG-bin-macos-arm64.tar.gz"

# Linux x64 / arm64: dinov2-$TAG-bin-ubuntu-x64.tar.gz / dinov2-$TAG-bin-ubuntu-arm64.tar.gz
# Windows (version-stable asset name):
#   https://github.com/espetro/dinov2.cpp/releases/latest/download/dinov2-bin-win-cpu-x64.zip
```

**2. Download a GGUF weight:**

```bash
hf download dinov2-cpp-core/dinov2-small-gguf --local-dir models
```

**3. Run inference** (`-c` for classification; add `-o out.png` for a PCA feature visualization):

```bash
./bin/dinov2-cli -m models/model.gguf -i assets/tench.jpg -c
```

**Embeddings (JSON)** for PyTorch `last_hidden_state[:, 0]` consumers:

```bash
./bin/dinov2-cli -m models/model.gguf -i assets/tench.jpg --print-embeddings
```

Emits `cls`/`pooled` vectors on stdout as one JSON object.

Feature mode bounds the input's shortest edge to 518 px by default
(`--preprocess bounded`) so attention memory stays predictable on large
images; `--preprocess hf` mirrors the HF `AutoImageProcessor` recipe
(256 shortest edge + 224 center crop) for like-for-like comparisons, and
`--preprocess crop518` fixes every input at 518x518 for batching mixed
aspect ratios. `--no-resize` keeps native resolution, and `--max-tokens N`
caps the patch-token count per image (default `4*(518/patch)^2`).

**Preview binary embeddings:** for high-throughput consumers, use
`--embeddings-binary -o output.d2e`. This deliberately unstable preview writes
an explicit 40-byte little-endian header followed by float32 `cls`, `pooled`,
and optional row-major patch vectors. It writes no binary bytes to stdout.
With multiple inputs, `-o` is a directory containing indexed `.d2e` files.
The format may change without compatibility guarantees and is not a standard.

**Batch inference:** repeat `-i` (or comma-separate paths) and set `--batch`;
each input gets one JSON line on stdout, identical to running it alone
(see [docs/cli.md](docs/cli.md#batch-inference)):

```bash
./bin/dinov2-cli -m models/model.gguf -i a.jpg -i b.jpg --batch 2 --print-embeddings
```

That's it. Up to **3x faster than PyTorch on CPU** with up to **4x less memory** (see [docs/benchmarks.md](docs/benchmarks.md)).

## Features

| Feature | Detail |
|:--------|:-------|
| **Zero dependencies** | Image I/O via vendored stb, compute via ggml. Nothing else. |
| **CPU, CUDA, Metal** | Backends via ggml wherever ggml supports them. |
| **f16 GGUF weights** | Plus q4_0 through q8_0 quantization. |
| **PyTorch-parity outputs** | CLS + patch embeddings as JSON, matching the reference implementation. |
| **Cross-platform prebuilts** | macOS arm64, Linux x64/arm64, Windows x64. |

## Pre-converted GGUF weights

Ready-to-download f16 GGUF weights, published by CI to the [`dinov2-cpp-core`](https://huggingface.co/dinov2-cpp-core) Hugging Face profile:

| Model | GGUF download | Size | Recommendation |
|:-----:|:--------------|-----:|:---------------|
| small (no registers) | [`dinov2-cpp-core/dinov2-small-gguf`](https://huggingface.co/dinov2-cpp-core/dinov2-small-gguf) | ~50 MB | Baseline reproduction or task comparison |
| base (no registers) | [`dinov2-cpp-core/dinov2-base-gguf`](https://huggingface.co/dinov2-cpp-core/dinov2-base-gguf) | ~180 MB | Baseline reproduction or task comparison |
| large (no registers) | [`dinov2-cpp-core/dinov2-large-gguf`](https://huggingface.co/dinov2-cpp-core/dinov2-large-gguf) | ~620 MB | Baseline reproduction or task comparison |
| giant (no registers) | [`dinov2-cpp-core/dinov2-giant-gguf`](https://huggingface.co/dinov2-cpp-core/dinov2-giant-gguf) | ~2.2 GB | Baseline reproduction or task comparison |
| small (registers) | [`dinov2-cpp-core/dinov2-with-registers-small-gguf`](https://huggingface.co/dinov2-cpp-core/dinov2-with-registers-small-gguf) | ~50 MB | **Recommended for patch/dense features** |
| base (registers) | [`dinov2-cpp-core/dinov2-with-registers-base-gguf`](https://huggingface.co/dinov2-cpp-core/dinov2-with-registers-base-gguf) | ~180 MB | **Recommended for patch/dense features** |
| large (registers) | [`dinov2-cpp-core/dinov2-with-registers-large-gguf`](https://huggingface.co/dinov2-cpp-core/dinov2-with-registers-large-gguf) | ~620 MB | **Recommended for patch/dense features** |
| giant (registers) | [`dinov2-cpp-core/dinov2-with-registers-giant-gguf`](https://huggingface.co/dinov2-cpp-core/dinov2-with-registers-giant-gguf) | ~2.2 GB | **Recommended for patch/dense features** |

For patch and dense feature workflows, the register-token variants are the recommended starting point. In the settings studied in [Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588), register tokens reduce high-norm patch-token artifacts and produce smoother local feature and attention maps. Use them with `--print-patch-tokens`, PCA, dense features, and object discovery. This is not a universal accuracy claim: the official [DINOv2 results](https://github.com/facebookresearch/dinov2/blob/main/README.md) show classification and retrieval results that depend on the task and model size. Choose the no-register variant for exact baseline reproduction or task-specific classification and retrieval comparisons.

## Documentation

- [CONTRIBUTING.md](CONTRIBUTING.md): pre-converted GGUF weights, build from source, dev harness, tests, PR guidelines
- [docs/cli.md](docs/cli.md): `dinov2-cli` reference: flags, output modes, embeddings JSON schema, workflows
- [docs/build.md](docs/build.md): per-device optimizations, quantization
- [docs/benchmarks.md](docs/benchmarks.md): benchmarks against PyTorch, how to run your own
- [docs/hf-publishing.md](docs/hf-publishing.md): how CI publishes GGUF weights to Hugging Face

## Why this fork

Upstream requires building from source and converting weights yourself. This fork ships what's missing: prebuilt binaries for macOS, Linux and Windows, plus ready-to-download GGUF weights published automatically by CI.

## Credits

Forked from [lavaman131/dinov2.cpp](https://github.com/lavaman131/dinov2.cpp). Built on and heavily inspired by [vit.cpp](https://github.com/staghado/vit.cpp) and [ggml](https://github.com/ggml-org/ggml).

## License

[MIT](LICENSE)
