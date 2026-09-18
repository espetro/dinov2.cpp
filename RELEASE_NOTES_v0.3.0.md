# v0.3.0 Release Notes

Rename of the CLI binary, drop of the `quantize` binary, and a minimal-repo
sweep that removes 21 vestigial files. The C++ compute path is unchanged.

## Breaking changes

### `inference` → `dinov2-cli`

The CLI executable is renamed from `inference` to `dinov2-cli`. The C source
file moves from `inference.cpp` to `dinov2-cli.cpp`; everything else (`dinov2.h`
public API, `dinov2.cpp` implementation, `ggml` dependency) is unchanged.

| Before (v0.2.0)                          | After (v0.3.0)                           |
|:-----------------------------------------|:-----------------------------------------|
| `cmake --build build --target inference` | `cmake --build build --target dinov2-cli`|
| `./build/bin/inference -m …`             | `./build/bin/dinov2-cli -m …`            |

The C `main()` function still uses `__func__ = "main"` for stderr messages;
that string refers to the entry point, not the binary name, so it is unchanged.

### `quantize` binary removed

The `quantize` CLI and the in-tree `dino_model_quantize()` function are removed.
Quantization is now an **out-of-tree build step** owned by the HF pipeline:

- The [`convert-and-publish-gguf`](../.github/workflows/convert-and-publish-gguf.yml)
  workflow produces all 8 quant variants (f16, q4_0, q4_1, q5_0, q5_1, q8_0)
  and uploads them to
  [`dinov2-cpp-core/<variant>-gguf`](https://huggingface.co/dinov2-cpp-core).
- `dinov2-cli` loads any ggml-supported quant type transparently — pass any
  of those GGUFs as `-m`.
- `scripts/bench.sh` no longer accepts `--quants`; only f16 is bench input,
  pulled from the HF repo.

The internal `do_quantize(name, tensor)` helper and its TEST_CASE are removed
alongside `dino_model_quantize()`. The other 9 pure-function tests
(`dino_hparams` math, `interpolate_pos_embed` identity / non-square /
CLS-preserved / output-size, `dino_preprocess` padding / resize /
normalization) are unchanged.

## New features

- [`ARCHITECTURE.md`](../ARCHITECTURE.md) — layer model + dependency rule,
  build-target graph, CLI flow, and the verbatim public API surface from
  `dinov2.h`. Mermaid diagrams render in GitHub markdown.

## Internal cleanup

- **23 files deleted**: `quantize.cpp`; `inference.cpp` (renamed);
  `scripts/benchmark.sh` (deprecation shim), `scripts/benchmark.py` (PyTorch
  reference), `scripts/.env.example`, `requirements.txt`;
  `src/dinov2_inference/{__init__.py,types.py}`, `src/dinov2_inference.egg-info/`,
  `src/stb_image_resize2.h`; `CMakeSettings.json`, `.clang-tidy`; and 11 unused
  image assets (`assets/{apple,armadillo,cheetah,coconut,giraffe,kiwi,magpie,polars}.jpg*`,
  `assets/image.png`, `assets/pca_visual.jpg`,
  `assets/readme-assets/OpenCV-table.png`).
- **`.gitignore`** trimmed from 502 lines to 31: only build, python, packages,
  data, OS, env, and local patterns remain.
- **`pyproject.toml`** slimmed to the 6 deps `scripts/dinov2-to-gguf.py` uses:
  `torch`, `transformers`, `gguf`, `huggingface_hub[hf_transfer]`, `hf_xet`,
  `numpy`. `timm`, `torchvision`, `Pillow`, `memory-profiler`, `threadpoolctl`
  are gone (their imports were only in deleted `benchmark.py`).
- **`scripts/dinov2-to-gguf.py`** inlines the 2-value `GGMLNumpyType` enum that
  used to live in `src/dinov2_inference/types.py`; no more `PYTHONPATH=src`
  hack.
- **`CMakeLists.txt`** drops the `BUILD_QUANTIZE` option block and the `quantize`
  target; renames the `inference` target to `dinov2-cli`.

## Upgrade notes

```text
# Before (v0.2.0):
./build/bin/inference -m models/ggml-model.gguf -i assets/tench.jpg -c

# After (v0.3.0):
./build/bin/dinov2-cli -m models/ggml-model.gguf -i assets/tench.jpg -c
```

Build artifacts under `build/` from v0.2.0 must be cleaned first — the old
`inference` and `quantize` targets no longer exist and CMake will error on
missing targets:

```bash
rm -rf build/
cmake -B build -G Ninja -DDINOV2_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

## CI / release artifacts

The 4-platform release matrix still ships one binary per platform;
`SHA256SUMS.txt` aggregation is unchanged.
