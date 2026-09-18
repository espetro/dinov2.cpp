<p align="center"><img src="assets/logo/logo-256.png" width="128" alt="dinov2.cpp"></p>

# dinov2.cpp

Run DINOv2 vision models in pure C++ on ggml. No Python, no PyTorch, no system dependencies at runtime.

[![Release](https://img.shields.io/github/v/release/espetro/dinov2.cpp)](https://github.com/espetro/dinov2.cpp/releases)
[![CI](https://github.com/espetro/dinov2.cpp/actions/workflows/build.yml/badge.svg)](https://github.com/espetro/dinov2.cpp/actions/workflows/build.yml)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C.svg)
![Topics](https://img.shields.io/github/topics/espetro/dinov2.cpp?style=flat)

**Latest release: [v0.3.0](https://github.com/espetro/dinov2.cpp/releases/tag/v0.3.0)**

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
huggingface-cli download dinov2-cpp-core/dinov2-small-gguf --local-dir models
```

**3. Run inference** (add `-c` for classification, omit for PCA feature visualization):

```bash
./bin/dinov2-cli -m models/ggml-model.gguf -i assets/tench.jpg -c
```

That's it. Up to **3x faster than PyTorch on CPU** with up to **4x less memory** (see [docs/benchmarks.md](docs/benchmarks.md)).

## Features

| Feature | Detail |
|:--------|:-------|
| **Zero dependencies** | Image I/O via vendored stb, compute via ggml. Nothing else. |
| **CPU, CUDA, Metal** | Backends via ggml wherever ggml supports them. |
| **f16 GGUF weights** | Plus q4_0 through q8_0 quantization. |
| **PyTorch-parity outputs** | Matches the reference implementation. |
| **Cross-platform prebuilts** | macOS arm64, Linux x64/arm64, Windows x64. |

## Pre-converted GGUF weights

Ready-to-download f16 GGUF weights, published by CI to the [`dinov2-cpp-core`](https://huggingface.co/dinov2-cpp-core) Hugging Face profile:

| Model | GGUF download | Size |
|:-----:|:--------------|-----:|
| small (no registers) | [`dinov2-cpp-core/dinov2-small-gguf`](https://huggingface.co/dinov2-cpp-core/dinov2-small-gguf) | ~50 MB |
| base (no registers) | [`dinov2-cpp-core/dinov2-base-gguf`](https://huggingface.co/dinov2-cpp-core/dinov2-base-gguf) | ~180 MB |
| large (no registers) | [`dinov2-cpp-core/dinov2-large-gguf`](https://huggingface.co/dinov2-cpp-core/dinov2-large-gguf) | ~620 MB |
| giant (no registers) | [`dinov2-cpp-core/dinov2-giant-gguf`](https://huggingface.co/dinov2-cpp-core/dinov2-giant-gguf) | ~2.2 GB |
| small (registers) | [`dinov2-cpp-core/dinov2-with-registers-small-gguf`](https://huggingface.co/dinov2-cpp-core/dinov2-with-registers-small-gguf) | ~50 MB |
| base (registers) | [`dinov2-cpp-core/dinov2-with-registers-base-gguf`](https://huggingface.co/dinov2-cpp-core/dinov2-with-registers-base-gguf) | ~180 MB |
| large (registers) | [`dinov2-cpp-core/dinov2-with-registers-large-gguf`](https://huggingface.co/dinov2-cpp-core/dinov2-with-registers-large-gguf) | ~620 MB |
| giant (registers) | [`dinov2-cpp-core/dinov2-with-registers-giant-gguf`](https://huggingface.co/dinov2-cpp-core/dinov2-with-registers-giant-gguf) | ~2.2 GB |

## Documentation

- [CONTRIBUTING.md](CONTRIBUTING.md): pre-converted GGUF weights, build from source, dev harness, tests, PR guidelines
- [docs/build.md](docs/build.md): per-device optimizations, quantization
- [docs/benchmarks.md](docs/benchmarks.md): benchmarks against PyTorch, how to run your own
- [docs/hf-publishing.md](docs/hf-publishing.md): how CI publishes GGUF weights to Hugging Face

## Why this fork

Upstream requires building from source and converting weights yourself. This fork ships what's missing: prebuilt binaries for macOS, Linux and Windows, plus ready-to-download GGUF weights published automatically by CI.

## Credits

Forked from [lavaman131/dinov2.cpp](https://github.com/lavaman131/dinov2.cpp). Built on and heavily inspired by [vit.cpp](https://github.com/staghado/vit.cpp) and [ggml](https://github.com/ggml-org/ggml).

## License

[MIT](LICENSE)

## Topics

Each GitHub repository topic surfaced by `git repo topics` (or clickable
from the repo sidebar) maps to a discoverability surface:

- [dinov2](https://github.com/topics/dinov2) — the Meta vision transformer
  this repo ports.
- [ggml](https://github.com/topics/ggml) — the inference framework; sister
  project `ggerganov/llama.cpp` self-tags the same way.
- [gguf](https://github.com/topics/gguf) — the weight format this repo
  consumes and publishes to Hugging Face.
- [cpp](https://github.com/topics/cpp) — primary language.
- [transformer](https://github.com/topics/transformer) — model family.
- [computer-vision](https://github.com/topics/computer-vision) — project
  domain.
- [inference](https://github.com/topics/inference) — what the binary does.
- [image-classification](https://github.com/topics/image-classification) —
  `-c` flag mode of `dinov2-cli`.
- [self-supervised-learning](https://github.com/topics/self-supervised-learning)
  — the training paradigm behind DINOv2.
- [cmake](https://github.com/topics/cmake) — build system; matches the
  cross-platform release pipeline.
- [huggingface](https://github.com/topics/huggingface) — distribution
  channel for the 8 pre-converted GGUF repos on
  [`dinov2-cpp-core`](https://huggingface.co/dinov2-cpp-core).
