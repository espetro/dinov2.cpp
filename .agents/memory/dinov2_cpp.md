# dinov2.cpp fork — project notes

## Repo identity

- Upstream: `https://github.com/lavaman131/dinov2.cpp` (no releases, no CI)
- This fork: `https://github.com/espetro/dinov2.cpp` (clean working tree, just forked)
- Inspiration credit (README:12): `https://github.com/staghado/vit.cpp` — zero OpenCV, uses stb_image. Direct precedent for dropping OpenCV.

## Storage constraint (user-specified)

- ~3 GB per CI job ceiling. Trigger warning if exceeded.
- Mac storage tight (no external SSD mentioned); OpenCV removal was the biggest storage lever (~3 GB on macOS via brew keg).

## Build system

- CMake ≥3.22.1, C++20. 56-line CMakeLists.txt.
- ggml as `add_subdirectory(ggml)` (submodule, pinned SHA).
- Three executables: `inference`, `realtime`, `quantize` — but we delete `realtime` in the v0.1.0 PR.

## Dependency state

- OpenCV: **dropping entirely** in v0.1.0 PR. Replacement = vendored stb_image + stb_image_write + stb_image_resize + ~80-line power-iter PCA. See `vit.cpp` upstream for the image path.
- ggml: stays pinned at `13bcf9ce50651a8b4238ec6d136f46f2b1b23b6f` for v0.1.0. Post-v0.1.0 bump to tag `v0.23.0` (one minor back from master HEAD `456172ec`).
- gguf (PyPI): pinning `>=0.18.0,<0.20` in pyproject.toml.
- No other vendored deps.

## CI state

- Zero CI today (no `.github/workflows/`).
- v0.1.0 adds: 4-job matrix (ubuntu-x64, ubuntu-arm64, macos-arm64, windows-x64) on `v*` tags. PR smoke on ubuntu-x64 only.
- Caching: ccache (500 MB cap, 7d LRU, save only on master/tags) + apt/brew/vcpkg via `actions/cache@v4`. No `build/` cache. No `uv` cache.
- Submodule URL fix needed: `.gitmodules:1` SSH→HTTPS.

## HF + GGUF pipeline (follow-up, NOT v0.1.0)

- Workflow: `cron: '0 6 1 * *'` monthly + `workflow_dispatch`. 8-variant matrix.
- Repo layout: 8 separate repos under `dinov2-cpp/*-gguf` org namespace.
- Upload: `hf upload` CLI (direct LFS streaming, not `huggingface/upload-folder-to-hub@v1`).
- Disk budget: ~12–13 GB transient per `giant` variant job (tight on the 14 GB runner SSD; use `hf_transfer` + `pytorch-cpu` wheel to shave).
- Secret: `HF_TOKEN` with `Write` scope on target repos.

## Architectural decision: NO ImageBackend port interface

- User explicitly chose "Drop OpenCV entirely, no adapter" over "port/adapter seam".
- Simpler diff, but means anyone wanting webcam/highgui has to fork.
- If we ever add an adapter seam later, the natural insertion point is `src/image.cpp` → split into `src/image_stb.cpp` + `src/image_opencv.cpp` behind `option(DINOV2_USE_OPENCV)`.

## Latent build issues to fix in v0.1.0 PR

- `inference.cpp:8` includes `"ggml/examples/stb_image.h"` — fragile path through submodule. Replaced with vendored `src/stb_image.h`.
- `.gitmodules:1` SSH URL — won't work in CI without deploy key. Switch to HTTPS.
- `ggml/` submodule is **empty** in this checkout — needs `git submodule update --init --recursive` before any local build.

## References (don't re-research)

- llama.cpp CI patterns: `https://github.com/ggml-org/llama.cpp/tree/master/.github/workflows` (esp. `build-cpu.yml`, `build-apple.yml`, `release.yml`)
- llama.cpp ccache save-policy PR: #11661, #23789, #23895 (the ccache race)
- ggml master HEAD: `456172ec` (Sept 2026), tags `v0.15.1` → `v0.24.0`
- ggml pin for v0.1.0: `13bcf9ce50651a8b4238ec6d136f46f2b1b23b6f` (March 2025)
- vit.cpp image preprocessing reference: `https://github.com/staghado/vit.cpp/blob/main/vit.cpp` — port `vit_image_preprocess_bicubic` for the new image.cpp
- OpenCV + Zig incompatibility: `https://github.com/ziglang/zig/issues/13018` (irrelevant now — we dropped OpenCV — but flagging in case someone revisits Zig later)
- mudler/LocalAI brew cache pattern: `https://github.com/mudler/LocalAI/blob/main/.github/workflows/backend_build_darwin.yml`
- HuggingFace publishing reference: `https://huggingface.co/docs/hub/repositories-github-actions`
