<p align="center"><img src="assets/logo/logo-256.png" width="128" alt="dinov2.cpp"></p>

# dinov2.cpp

Run DINOv2 vision models in pure C++ on ggml. No Python, no PyTorch, no system dependencies at runtime.

[![Release](https://img.shields.io/github/v/release/espetro/dinov2.cpp)](https://github.com/espetro/dinov2.cpp/releases)
[![CI](https://github.com/espetro/dinov2.cpp/actions/workflows/build.yml/badge.svg)](https://github.com/espetro/dinov2.cpp/actions/workflows/build.yml)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C.svg)

**Release post & benchmarks → https://alexlavaee.me/projects/dinov2cpp/**

## Quick start

### Quickstart with agents

Using Claude Code, Cursor or another coding agent? Paste this:

```text
Install https://github.com/espetro/dinov2.cpp via mise (latest), download the small GGUF from the dinov2-cpp-core profile on Hugging Face, and run classification on an example image to verify it works.
```

### Manual install

Grab a prebuilt binary, download a weight, run one command. No toolchain needed.

**1. Install a prebuilt binary** from [releases](https://github.com/espetro/dinov2.cpp/releases):

```bash
# macOS (Apple Silicon)
curl -LO https://github.com/espetro/dinov2.cpp/releases/download/v0.1.0/dinov2-v0.1.0-bin-macos-arm64.tar.gz
tar xzf dinov2-v0.1.0-bin-macos-arm64.tar.gz

# Linux x64 / arm64: dinov2-v0.1.0-bin-ubuntu-x64.tar.gz / dinov2-v0.1.0-bin-ubuntu-arm64.tar.gz
# Windows: dinov2-bin-win-cpu-x64.zip
```

**2. Download a GGUF weight:**

```bash
huggingface-cli download espetro/dinov2-small-imagenet1k-1-layer-gguf --local-dir models
```

**3. Run inference** (add `-c` for classification, omit for PCA feature visualization):

```bash
./bin/inference -m models/ggml-model.gguf -i assets/tench.jpg -c
```

That's it. Up to **3x faster than PyTorch on CPU** with up to **4x less memory** (see [docs/benchmarks.md](docs/benchmarks.md)).

## Features requires building from source and converting weights yourself. This fork ships what's missing: prebuilt binaries for macOS, Linux and Windows, plus ready-to-download GGUF weights published automatically by CI.

## Why this fork

The upstream project
|---|---|
| **Zero dependencies** | Image I/O via vendored stb, compute via ggml. Nothing else. |
| **CPU, CUDA, Metal** | Backends via ggml wherever ggml supports them. |
| **f16 GGUF weights** | Plus q4_0 through q8_0 quantization. |
| **PyTorch-parity outputs** | Matches the reference implementation. |
| **Cross-platform prebuilts** | macOS arm64, Linux x64/arm64, Windows x64. |

## Pre-converted GGUF weights and building from source

Ready-to-download GGUF weights (published by CI to the [`dinov2-cpp-core`](https://huggingface.co/dinov2-cpp-core) Hugging Face profile) plus full build-from-source instructions live in [CONTRIBUTING.md](CONTRIBUTING.md).

## Documentation

- [CONTRIBUTING.md](CONTRIBUTING.md): pre-converted GGUF weights, build from source, dev harness, tests, PR guidelines
- [docs/build.md](docs/build.md): per-device optimizations, quantization
- [docs/benchmarks.md](docs/benchmarks.md): benchmarks against PyTorch, how to run your own
- [docs/hf-publishing.md](docs/hf-publishing.md): how CI publishes GGUF weights to Hugging Face

## Credits

Built on and heavily inspired by [vit.cpp](https://github.com/staghado/vit.cpp) and [ggml](https://github.com/ggml-org/ggml).

## License

[MIT](LICENSE)
