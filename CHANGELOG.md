# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Bumped project version in CMakeLists to `0.4.0-dev`. No source changes yet.

## [0.3.0] - 2026-09-18

Rename of the CLI binary, drop of the `quantize` binary, and a minimal-repo
sweep that removes 21 vestigial files. The C++ compute path is unchanged.

### Breaking changes

- **CLI renamed**: `inference` → `dinov2-cli`. The C++ source file moves from
  `inference.cpp` to `dinov2-cli.cpp`. The CMake target, every README and
  `docs/build.md` example, and the `release`/`build` GitHub Actions workflows
  follow. `dinov2.h` public API, `dinov2.cpp` implementation, and the `ggml`
  dependency are unchanged.

  | Before (v0.2.0)                          | After (v0.3.0)                           |
  |:-----------------------------------------|:-----------------------------------------|
  | `cmake --build build --target inference` | `cmake --build build --target dinov2-cli`|
  | `./build/bin/inference -m …`             | `./build/bin/dinov2-cli -m …`            |

- **`quantize` binary removed**. The CLI wrapper `quantize.cpp` and the
  in-tree `dino_model_quantize()` helper are deleted. Quantization is now an
  out-of-tree build step owned by the HF pipeline: the
  `convert-and-publish-gguf` workflow produces all quant variants (f16,
  q4_0, q4_1, q5_0, q5_1, q8_0) and uploads them to
  [`dinov2-cpp-core/<variant>-gguf`](https://huggingface.co/dinov2-cpp-core).
  `dinov2-cli` loads any ggml-supported quant type transparently — pass
  any of those GGUFs as `-m`.

- **`scripts/bench.sh` `--quants` and `--quantize` flags removed**. The
  script now runs f16 only; per-quant benchmarks are produced by
  pre-placing the desired GGUF under `models/<variant>/model.<q>.gguf`
  (which the user pulls from HF).

- **Internal `do_quantize()` helper and its `TEST_CASE` removed**. The
  remaining 9 pure-function tests in `tests/test_dinov2.cpp` (`dino_hparams`
  math, `interpolate_pos_embed` identity / non-square / CLS-preserved /
  output-size, `dino_preprocess` padding / resize / normalization) are
  unchanged.

### Added

- [`ARCHITECTURE.md`](ARCHITECTURE.md): layered model with the dependency
  rule, build-target graph, CLI runtime flow, and the verbatim public API
  surface from `dinov2.h`. Includes four Mermaid diagrams (layer model,
  build graph, CLI flow, GGUF pipeline) that render on GitHub.

### Removed

- **23 files deleted**:
  - `quantize.cpp` — vestigial CLI wrapper.
  - `inference.cpp` — renamed to `dinov2-cli.cpp`.
  - `scripts/benchmark.sh` — deprecation shim (was alias for `bench.sh`).
  - `scripts/benchmark.py` — PyTorch reference bench, replaced by `bench.sh`.
  - `scripts/.env.example`, `requirements.txt` — no consumers after the
    bench rewrites.
  - `src/dinov2_inference/__init__.py`, `src/dinov2_inference/types.py` —
    Python package the `bench.py` import path referenced; both gone after
    the sweep.
  - `src/stb_image_resize2.h` (447 KB, 10,679 LOC) — vendored but never
    wired: `STB_IMAGE_RESIZE2_IMPLEMENTATION` was defined but no `stbir_*`
    call existed. The matching `#define` + `#include` in `src/image.cpp`
    is gone in the same commit.
  - `CMakeSettings.json`, `.clang-tidy` — IDE / linter configs left over
    from the upstream template, never read by the build or CI.
  - 11 unused image assets: `assets/{apple,armadillo,cheetah,coconut,
    giraffe,kiwi,magpie,polars}.jpg*`, `assets/image.png`,
    `assets/pca_visual.jpg`, `assets/readme-assets/OpenCV-table.png`.

### Changed (internal)

- **`.gitignore`** trimmed from 502 lines to 31: only build, python,
  packages, data, OS, env, and local patterns remain.
- **`pyproject.toml`** slimmed to the 6 deps `scripts/dinov2-to-gguf.py`
  uses (`torch`, `transformers`, `gguf`, `huggingface_hub[hf_transfer]`,
  `hf_xet`, `numpy`). `timm`, `torchvision`, `Pillow`, `memory-profiler`,
  and `threadpoolctl` are gone — their imports were only in the deleted
  `benchmark.py`.
- **`scripts/dinov2-to-gguf.py`** inlines the 2-value `GGMLNumpyType`
  enum that used to live in the now-deleted `src/dinov2_inference/types.py`.
  The `PYTHONPATH=src` hack is gone with it.
- **`scripts/publish-gguf.sh`** drops the same `PYTHONPATH` override.
- **`CMakeLists.txt`** drops the `BUILD_QUANTIZE` option block and the
  `quantize` target; renames the `inference` target to `dinov2-cli`.

### Migration

```text
# Before (v0.2.0):
./build/bin/inference -m models/ggml-model.gguf -i assets/tench.jpg -c

# After (v0.3.0):
./build/bin/dinov2-cli -m models/ggml-model.gguf -i assets/tench.jpg -c
```

Build artifacts under `build/` from v0.2.0 must be cleaned first — the
old `inference` and `quantize` targets no longer exist and CMake errors
on missing targets:

```bash
rm -rf build/
cmake -B build -G Ninja -DDINOV2_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

The 4-platform release matrix still ships one binary per platform;
`SHA256SUMS.txt` aggregation is unchanged.

## [0.2.0] - 2026-09-18

Pre-built DINOv2 inference binaries, now powered by **ggml v0.24.0**,
with **8 pre-converted GGUF weights** on Hugging Face and a fully
reproducible CI release matrix.

### Added

- **ggml bumped to v0.24.0** ([#2](https://github.com/espetro/dinov2.cpp/pull/2)).
  Picks up the upstream inference improvements and fixes the
  transitive `<cmath>` drop so the build is self-contained again.
- **Hugging Face GGUF conversion + publishing pipeline**
  ([#2](https://github.com/espetro/dinov2.cpp/pull/2),
  [#3](https://github.com/espetro/dinov2.cpp/pull/3)–
  [#5](https://github.com/espetro/dinov2.cpp/pull/5)). Every PR that
  touches conversion scripts runs an end-to-end
  `safetensors → GGUF → HF upload` job on `ubuntu-latest-large`. All 8
  official checkpoints (base/small/large/giant plus the `with-registers`
  variants) are now published under the
  [`dinov2-cpp-core`](https://huggingface.co/dinov2-cpp-core) org.
- **CI release matrix for macOS / Linux / Windows prebuilts**
  ([#1](https://github.com/espetro/dinov2.cpp/pull/1)). Push a `v*` tag,
  the `release` workflow builds 4 platform tarballs/zip plus a
  `SHA256SUMS` manifest and attaches them to the GitHub Release.
- **Dev harness: doctest + sanitizers + clang-format gate**
  ([#1](https://github.com/espetro/dinov2.cpp/pull/1)).
  `cmake -DBUILD_TESTS=ON` brings in doctest; the `test` job builds with
  `-fsanitize=address,undefined` on Linux/macOS; a `format-check` job
  enforces `clang-format` on the touched file set.
- **CLI flags**: `--model`, `--image`, `--format text|json|npy`,
  `--threads` (carried from v0.1.0, rebuilt against ggml v0.24.0).
- **Bench infrastructure** ([PR B](https://github.com/espetro/dinov2.cpp/pull/...)):
  `--bench`, `--bench-runs N`, `--bench-warmup N`, `--bench-json` flags
  on the CLI; `scripts/bench.sh` (renamed from the old deprecation shim)
  accepts `--models`, `--quants` (later removed in v0.3.0),
  `--repeats`, `--threads`, `--out`, `--aggregate`.

### Removed

- **OpenCV dependency removed** ([#1](https://github.com/espetro/dinov2.cpp/pull/1)).
  Zero runtime system dependencies. The binary loads any format ggml can
  decode, so the heavy image I/O layer is gone.

### Downloads

| Platform           | Architecture | Asset                                       |
|:-------------------|:-------------|:--------------------------------------------|
| macOS              | arm64        | `dinov2-v0.2.0-bin-macos-arm64.tar.gz`      |
| Linux (Ubuntu)     | x86_64       | `dinov2-v0.2.0-bin-ubuntu-x64.tar.gz`       |
| Linux (Ubuntu)     | arm64        | `dinov2-v0.2.0-bin-ubuntu-arm64.tar.gz`     |
| Windows            | x86_64       | `dinov2-bin-win-cpu-x64.zip`                |

Plus `SHA256SUMS.txt` covering all four archives. Verify with
`shasum -a 256 -c SHA256SUMS.txt` (or `sha256sum -c` on Linux).

### Pre-converted GGUF weights

| Variant                     | HF repo                                                                 |
|:----------------------------|:------------------------------------------------------------------------|
| ViT-S/14 (base)             | [`dinov2-cpp-core/dinov2-small-gguf`](https://huggingface.co/dinov2-cpp-core/dinov2-small-gguf)               |
| ViT-B/14 (base)             | [`dinov2-cpp-core/dinov2-base-gguf`](https://huggingface.co/dinov2-cpp-core/dinov2-base-gguf)                 |
| ViT-L/14 (base)             | [`dinov2-cpp-core/dinov2-large-gguf`](https://huggingface.co/dinov2-cpp-core/dinov2-large-gguf)               |
| ViT-g/14 (base)             | [`dinov2-cpp-core/dinov2-giant-gguf`](https://huggingface.co/dinov2-cpp-core/dinov2-giant-gguf)               |
| ViT-S/14 (with registers)   | [`dinov2-cpp-core/dinov2-with-registers-small-gguf`](https://huggingface.co/dinov2-cpp-core/dinov2-with-registers-small-gguf) |
| ViT-B/14 (with registers)   | [`dinov2-cpp-core/dinov2-with-registers-base-gguf`](https://huggingface.co/dinov2-cpp-core/dinov2-with-registers-base-gguf)   |
| ViT-L/14 (with registers)   | [`dinov2-cpp-core/dinov2-with-registers-large-gguf`](https://huggingface.co/dinov2-cpp-core/dinov2-with-registers-large-gguf) |
| ViT-g/14 (with registers)   | [`dinov2-cpp-core/dinov2-with-registers-giant-gguf`](https://huggingface.co/dinov2-cpp-core/dinov2-with-registers-giant-gguf) |

### CI / release hardening (since v0.1.0)

PRs [#2](https://github.com/espetro/dinov2.cpp/pull/2)–
[#8](https://github.com/espetro/dinov2.cpp/pull/8) shipped the
following release-engineering work on top of the v0.1.0 release
pipeline:

- `chore(deps): bump ggml to v0.24.0`
  ([#2](https://github.com/espetro/dinov2.cpp/pull/2))
- `ci: hf gguf conversion + publishing workflow`
  ([#2](https://github.com/espetro/dinov2.cpp/pull/2))
- `docs: hf publishing guide` + README rewrite + GGUF weights table
  ([#2](https://github.com/espetro/dinov2.cpp/pull/2))
- `ci(refactor): port gguf publish to bash script` +
  `ubuntu-latest-large` runner
  ([#3](https://github.com/espetro/dinov2.cpp/pull/3))
- `ci(cache): cache .venv-publish + document HF_TOKEN`
  ([#4](https://github.com/espetro/dinov2.cpp/pull/4))
- `ci(gguf): fit conversion within ubuntu-latest (~14 GB)` via lean
  deps + cache cleanup
  ([#5](https://github.com/espetro/dinov2.cpp/pull/5))
- `ci(gguf): hardcode HF_HOME path (runner.home not allowed in env:)`
  ([#6](https://github.com/espetro/dinov2.cpp/pull/6))
- `perf(gguf): relocate caches to /mnt scratch, hf_xet uploader,
  dynamic matrix, per-run audit logs`
  ([#7](https://github.com/espetro/dinov2.cpp/pull/7))
- `fix(ci): emit JSON array from plan job so fromJson(matrix) parses
  correctly`
  ([#8](https://github.com/espetro/dinov2.cpp/pull/8))

### Migration

- The CLI flag surface is unchanged from v0.1.0 — existing scripts
  work as-is.
- ggml v0.24.0 changes mean GGUF files produced against older ggml
  versions may need to be regenerated against these binaries (the HF
  repos above are all ggml v0.24.0 compatible).
- The Windows binary is built with `BUILD_SHARED_LIBS=OFF` and
  `CMAKE_INSTALL_RPATH=$ORIGIN`; no DLLs need to be on `PATH`.

## [0.1.0] - 2026-08-XX

Initial public release. Pre-built DINOv2 inference binaries for macOS
arm64, Linux x86_64, Linux arm64, and Windows x86_64 with SHA-256
manifest. ggml-backed CPU inference for ViT-S/B/L/g variants (base +
with-registers); classification and PCA feature-extraction modes; image
preprocessing with `stb_image` + `stb_image_write` (no OpenCV
runtime dependency). See the GitHub release page for archives and
checksums.

[Unreleased]: https://github.com/espetro/dinov2.cpp/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/espetro/dinov2.cpp/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/espetro/dinov2.cpp/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/espetro/dinov2.cpp/releases/tag/v0.1.0
