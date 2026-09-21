# dinov2.cpp fork — project notes

## 2026-09-21 — embeddings-dropin PR 1 complete (branch feat/embeddings-dropin, 5cd839d..16a5cb6)

- Goal: drop-in replacement for HF PyTorch DINOv2. Swap-score was 2.5/10; PR 1 closes the hard blockers (embeddings output, classify parity, agent-facing CLI/docs).
- `dino_output` now carries cls, pooled `[cls‖mean(patches)]`, patch tokens (feature mode), `preds` (class idx) + `pred_scores`. Registers excluded from patch view in BOTH modes — HF bug #37817 parity; the earlier classify pooling included registers and diverged ~0.0066 prob on with-registers checkpoints.
- Classify preprocessing is 256-shortest-edge + 224-crop per HF `preprocessor_config.json` — NOT the GGUF `img_size` 518 (that's backbone-native; resizing to 518 breaks parity). Gap report's suggested 518 fix was wrong.
- Two latent bugs the gap report missed: `preds[i]` stored probability-as-uint32 (always ~0) not class idx; `forward_head` divided by `n_img_embd²` (1369 at img_size 518) instead of actual patch count — pooled features ~5.3× under-scaled at 224 input.
- CLI: `--print-embeddings` (JSON stdout: model/image/n_patches/hidden/cls/pooled, +patches w/ --print-patch-tokens, +topk w/ -c), `--l2-normalize`, `--version` (CMake `DINOV2_VERSION` define from PROJECT_VERSION, "dev" fallback), `-o` now opt-in for PCA PNG (was default `pca_visual.jpg`, actually PNG bytes), unknown-arg exits 1, argv bounds on all value flags, data on stdout / logs on stderr.
- Docs: `docs/cli.md` is the full agent-facing reference; `--help` links the raw.githubusercontent URL. Model filename is `model.gguf` (HF `dinov2-cpp-core/*-gguf` repos), not `ggml-model.gguf`.
- `scripts/parity_check.py`: HF AutoModel vs CLI JSON. Gates: cls/pooled/flat-patch cosine ≥0.999, top-1 match, |Δprob|<0.05. Per-token-min is diagnostic only (f16 tail ~0.98 expected); strict opt-in via `--patches-token-min-threshold`. Observed: no-reg cls 0.999911 / flat 0.999570; with-reg cls 0.999984 / flat 0.999847.
- CMakePresets.json now has buildPresets — `cmake --build --preset <name>` works (docs always claimed it; it silently couldn't before).
- Deferred minors in `.superpowers/sdd/2026-09-21-embeddings-dropin/progress.md` (json_escape control chars, nan/inf in %.6g, div-by-zero guard, nothing-to-do hint omits --bench).
- PR 2 pending: `--batch N` batching. Known hardcoded-1 sites: dinov2.cpp input tensor (~472), flash reshape (~390), non-flash reshape (~403-404); attention partially batch-aware (`B = cur->ne[3]`).
- Process note: no GitHub Project configured for this repo — user approved proceeding without one; raise later.
- Norma MCP unavailable this session (server dropped) — live_check never ran; don't claim it.

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
