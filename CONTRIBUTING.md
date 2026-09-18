# Contributing

Thanks for considering a contribution. Keep PRs focused and small.

## Dev harness

CMake presets (defined in `CMakePresets.json`) configure the standard workflow:

```bash
cmake --preset debug    # or: release, asan, ubsan
cmake --build --preset debug
ctest --preset debug
```

- `debug` / `release`: standard configurations. Both enable `DINOV2_FATAL_WARNINGS` by default, so warnings are errors.
- `asan` / `ubsan`: AddressSanitizer and UndefinedBehaviorSanitizer builds for debugging memory and UB issues.

Build directories are `build-<preset>/`.

## Tests

Unit tests use [doctest](https://github.com/doctest/doctest). Run the suite with:

```bash
cmake --build --preset debug && ctest --preset debug
```

Run the same suite under the `asan` and `ubsan` presets before submitting anything that touches memory handling or the compute path.

## Formatting

CI enforces `clang-format-18`. Before committing:

```bash
clang-format-18 -i <changed files>
```

Format violations fail the lint job.

## Warnings

`DINOV2_FATAL_WARNINGS=ON` (the preset default) promotes warnings to errors. Keep the build clean under this option; do not disable it to mask warnings.

## PR guidelines

- One logical change per PR. Keep refactors separate from features.
- Ensure the full test suite passes under `debug`, `asan` and `ubsan`.
- Update docs (`README.md`, `docs/`) if you change behavior or interfaces.
- Follow conventional commit format (e.g. `feat:`, `fix:`, `docs:`, `ci:`).

## Pre-converted GGUF weights

f16 GGUF weights are mirrored to Hugging Face under the [`dinov2-cpp-core`](https://huggingface.co/dinov2-cpp-core) profile as `dinov2-cpp-core/<variant>-gguf`. Publishing is rolling out now: CI converts and re-publishes each variant monthly (and on demand) from the official checkpoints:

```bash
gh workflow run convert-and-publish-gguf.yml -f variant=all          # all 8 variants
gh workflow run convert-and-publish-gguf.yml -f variant=dinov2-base  # single variant
```

Until a given variant appears on HF, convert it yourself in one command:

```bash
python ./scripts/dinov2-to-gguf.py --model_name facebook/dinov2-small-imagenet1k-1-layer
```

See [docs/hf-publishing.md](docs/hf-publishing.md) for the `HF_TOKEN` setup.

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
