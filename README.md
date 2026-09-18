# dinov2.cpp

Run DINOv2 vision models in pure C++ on ggml. No Python, no PyTorch, no system dependencies at runtime.

[![Release](https://img.shields.io/github/v/release/espetro/dinov2.cpp)](https://github.com/espetro/dinov2.cpp/releases)
[![CI](https://github.com/espetro/dinov2.cpp/actions/workflows/build.yml/badge.svg)](https://github.com/espetro/dinov2.cpp/actions/workflows/build.yml)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C.svg)

**Release post & benchmarks → https://alexlavaee.me/projects/dinov2cpp/**

## Quick start

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

## Why this fork

The upstream project requires building from source and converting weights yourself. This fork ships what's missing: prebuilt binaries for macOS, Linux and Windows, plus ready-to-download GGUF weights published automatically by CI.

## Features

| | |
|---|---|
| **Zero dependencies** | Image I/O via vendored stb, compute via ggml. Nothing else. |
| **CPU, CUDA, Metal** | Backends via ggml wherever ggml supports them. |
| **f16 GGUF weights** | Plus q4_0 through q8_0 quantization. |
| **PyTorch-parity outputs** | Matches the reference implementation. |
| **Cross-platform prebuilts** | macOS arm64, Linux x64/arm64, Windows x64. |

## Pre-converted GGUF weights

f16 GGUF weights are mirrored to Hugging Face under the [`dinov2-cpp`](https://huggingface.co/dinov2-cpp) org as `dinov2-cpp/<variant>-gguf`. Publishing is rolling out now: CI converts and re-publishes each variant monthly (and on demand) from the official checkpoints:

```bash
gh workflow run convert-and-publish-gguf.yml -f variant=all          # all 8 variants
gh workflow run convert-and-publish-gguf.yml -f variant=dinov2-base  # single variant
```

Until a given variant appears on HF, convert it yourself in one command (see table below for source checkpoints):

```bash
python ./scripts/dinov2-to-gguf.py --model_name facebook/dinov2-small-imagenet1k-1-layer
```

See [docs/hf-publishing.md](docs/hf-publishing.md) for the `HF_TOKEN` setup.

```bash
python ./scripts/dinov2-to-gguf.py --model_name facebook/dinov2-small-imagenet1k-1-layer
```

| Model | Source checkpoint | Approx f16 GGUF size |
|:-----:|:------------------|---------------------:|
| small (no registers) | [facebook/dinov2-small-imagenet1k-1-layer](https://huggingface.co/facebook/dinov2-small-imagenet1k-1-layer) | ~50 MB |
| base (no registers) | [facebook/dinov2-base-imagenet1k-1-layer](https://huggingface.co/facebook/dinov2-base-imagenet1k-1-layer) | ~180 MB |
| large (no registers) | [facebook/dinov2-large-imagenet1k-1-layer](https://huggingface.co/facebook/dinov2-large-imagenet1k-1-layer) | ~620 MB |
| giant (no registers) | [facebook/dinov2-giant-imagenet1k-1-layer](https://huggingface.co/facebook/dinov2-giant-imagenet1k-1-layer) | ~2.2 GB |
| small (registers) | [facebook/dinov2-with-registers-small-imagenet1k-1-layer](https://huggingface.co/facebook/dinov2-with-registers-small-imagenet1k-1-layer) | ~50 MB |
| base (registers) | [facebook/dinov2-with-registers-base-imagenet1k-1-layer](https://huggingface.co/facebook/dinov2-with-registers-base-imagenet1k-1-layer) | ~180 MB |
| large (registers) | [facebook/dinov2-with-registers-large-imagenet1k-1-layer](https://huggingface.co/facebook/dinov2-with-registers-large-imagenet1k-1-layer) | ~620 MB |
| giant (registers) | [facebook/dinov2-with-registers-giant-imagenet1k-1-layer](https://huggingface.co/facebook/dinov2-with-registers-giant-imagenet1k-1-layer) | ~2.2 GB |

## Build from source

```bash
git clone --recurse-submodules https://github.com/espetro/dinov2.cpp.git
cd dinov2.cpp
cmake --preset release && cmake --build --preset release
```

Per-device optimizations (AMD hosts, OpenMP), sanitizer presets and benchmark instructions: [docs/build.md](docs/build.md).

## Documentation

- [docs/build.md](docs/build.md): build from source, per-device optimizations, quantization
- [docs/benchmarks.md](docs/benchmarks.md): benchmarks against PyTorch, how to run your own
- [CONTRIBUTING.md](CONTRIBUTING.md): dev harness, tests, PR guidelines

## Credits

Built on and heavily inspired by [vit.cpp](https://github.com/staghado/vit.cpp) and [ggml](https://github.com/ggml-org/ggml).

## License

[MIT](LICENSE)
