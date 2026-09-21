# Large-image feature preprocessing and wrapper ergonomics design

Status: proposed design only. This document does not claim that any implementation or measurement described below has landed.

Date: 2026-09-22

## 1. Summary

An external beta test of `dinov2-cli` built from `ce9429b` surfaced four defects and one packaging issue:

1. Feature mode has no input-size bound. `dino_preprocess` (`dinov2.cpp:138`) resizes each dimension up to the next multiple of `patch_size`, so a 3440x5601 PNG becomes a 246x401 patch grid (98,646 tokens). Self-attention memory is quadratic in token count (`ggml_mul_mat` KQ product at `dinov2.cpp:467`), which requested a ~222 GB Metal buffer and ended in `Abort trap 6`.
2. Allocation failures abort instead of failing cleanly. `ggml_gallocr_alloc_graph` (`dinov2.cpp:1097`) and `ggml_backend_alloc_ctx_tensors` (`dinov2.cpp:384`) are unchecked.
3. Metric parity vs the HF feature pipeline is a preprocessing-grid mismatch, not numerical error: HF `AutoImageProcessor` crops to 224x224 (256 tokens at patch 14) while the CLI emits the full native grid (1,444 tokens at 518x518 for `assets/tench.jpg`).
4. JSONL records carry no input index and `patches` is flat with no grid dimensions, so consumers cannot reconstruct the spatial grid or join outputs to inputs robustly.
5. `--version` reports `dinov2-cli 0.3.0` while the build is a 0.4.0 beta.

This design bounds feature preprocessing to a shortest-edge 518 resize by default, adds a hard token cap, checks the two allocation call sites, adds an HF-compatible preprocessing mode, extends the record contract, and bumps the project version to 0.4.0.

## 2. Goals

- Feature mode can never request unbounded memory: default resize bounds the shortest edge to 518, and a `--max-tokens` hard cap rejects grids that are still too large, before any graph is constructed.
- Allocation failure is a stderr message plus exit 1, never an abort, for both the model-weight buffer and the per-graph compute buffer.
- `--preprocess hf` reproduces the HF `AutoImageProcessor` feature pipeline exactly (resize shortest edge to 256, center crop 224x224) so parity checks compare like grids.
- Every JSON record is self-describing: input order index, the input path, and the patch-grid dimensions accompany the flat `patches` array.
- `dinov2-cli --version` reports 0.4.0 for pre-release builds; no git tag is created.

## 3. Non-goals

- No change to classifier preprocessing (`dino_classify_preprocess` already does shortest-edge 256 plus 224 center crop).
- No tiling, sliding-window, or chunked attention to support truly unbounded images.
- No stable-version guarantee for the D2EMB preview format; it is explicitly unstable.
- No removal of the `image` JSON field; new keys are additive.
- No implementation in this commit.

## 4. Current behavior, verified against the code

### Feature preprocessing today

`dino_preprocess(const Image &img, const dino_hparams &params)` (`dinov2.cpp:138-158`) calls `preprocess_for_dinov2(img, params.patch_size)` (`src/image.cpp:138-170`), which resizes each dimension **up** to `((v / patch_size) + 1) * patch_size` via `resize_planes` bicubic, scales to [0,1], and applies ImageNet mean/std. The fallback in `dino_preprocess` re-resizes to the same expected multiple if the helper ever disagrees.

Note: the code rounds **up** to the next patch multiple, never down. A 100x50 input at patch 14 yields 112x56 (asserted by `tests/test_dinov2.cpp:338`). The 518 bound below keeps this round-up convention applied to the resized image, for consistency with existing tests.

For the failing case: 3440x5601 at patch 14 gives new_w = (245+1)*14 = 3444, new_h = (400+1)*14 = 5614, a 246x401 = 98,646-token grid.

### Where image size flows

- `dinov2-cli.cpp:184-194` computes `max_img_size` over loaded `Image` dims and passes it to `dino_model_load`.
- `dino_model_load` (`dinov2.cpp:357-365`) uses `img_size` only to size `model.ctx` overhead (`offset` for the interpolated `new_pos_embed` tensor); it does not bound compute.
- `dino_predict` (`dinov2.cpp:1071-1087`) derives `num_patches = (ny/patch) * (nx/patch)` from the preprocessed `ImageF` dims and builds the graph via `build_graph({nx, ny}, ...)`.
- `forward_features` (`dinov2.cpp:536-538`) recomputes `h0`, `w0`, `num_patches` from `img_size`; `attn` allocates the quadratic KQ tensor over `W * H` tokens (`dinov2.cpp:467`).

### Single vs batched input paths

`main` preprocesses every input in a loop (`dinov2-cli.cpp:227-232`): `dino_classify_preprocess` under `-c`, else `dino_preprocess`. Chunks of `params.n_batch` are then checked to share identical post-preprocess dims (`dinov2-cli.cpp:237-251`), because `dino_predict` packs a chunk into one graph (`dinov2.cpp:1069-1079`). Under the 518 bound, images with different aspect ratios still produce different grids, so the existing mixed-size chunk error remains the correct behavior; only the absolute dims change.

### Unchecked allocation sites

- `dinov2.cpp:384`: `model.buffer = ggml_backend_alloc_ctx_tensors(model.ctx, model.backend)` returns `nullptr` on failure; unchecked.
- `dinov2.cpp:1097`: `ggml_gallocr_alloc_graph(allocr, gf)` returns `bool`; unchecked. This is the call that produced the abort: Metal buffer allocation fails inside it and the process continues into `ggml_backend_graph_compute` with an unallocated graph (the observed `Abort trap 6` came from the backend alloc path).
- `dinov2-cli.cpp:270`: `ggml_gallocr_new(...)` can return `nullptr`; unchecked (minor, cheap to guard).
- `dinov2.cpp:1124`: `ggml_backend_graph_compute` is already checked and returns `GGML_STATUS`; keep as is.

### JSON record today

`print_embeddings_json` (`dinov2-cli.cpp:106-142`) emits exactly:

```
{"model":"...","image":"...","n_patches":N,"hidden":H,"cls":[...],"pooled":[...],"topk":[...],"patches":[...]}
```

`model`, `image`, `n_patches`, `hidden` always; `cls` always; `pooled` feature mode only; `topk` classify only; `patches` feature mode + `--print-patch-tokens`. Record order in JSONL already equals input order (the `idx = s + b` loop at `dinov2-cli.cpp:290-296`), but no field carries that index. `n_patches` is computed as `(ny/patch) * (nx/patch)` from the `ImageF` (`dinov2-cli.cpp:108`), with no per-axis breakdown. `image` already holds the path exactly as passed to `-i`.

No C++ test asserts on the exact JSON key set today; `tests/test_cli.cpp` checks stderr/exit codes and GGUF loader failures only. The JSON schema is documented in `docs/cli.md:165-221`, which must be updated in the same change as the emitter.

### HF preprocessing defaults, verified

`https://huggingface.co/facebook/dinov2-small/raw/main/preprocessor_config.json`:

```json
{
  "image_processor_type": "BitImageProcessor",
  "size": {"shortest_edge": 256},
  "crop_size": {"height": 224, "width": 224},
  "do_resize": true, "do_center_crop": true,
  "do_rescale": true, "rescale_factor": 0.00392156862745098,
  "do_normalize": true,
  "image_mean": [0.485, 0.456, 0.406],
  "image_std": [0.229, 0.224, 0.225],
  "resample": 3
}
```

`resample: 3` is PIL `Image.BICUBIC`. The repo's classifier path `dino_classify_preprocess` (`dinov2.cpp:99-136`) already implements exactly this recipe (shortest-edge 256, center crop 224, ImageNet mean/std) and is the parity-verified reference for it.

## 5. Design

### 5.1 Feature preprocessing bound (default)

Add a `--preprocess` mode with values `native518` (new default) and `hf` (section 5.3), plus a `--no-resize` opt-out. Flag naming follows existing conventions (`--batch`, `--bench-runs`, `--print-patch-tokens`): a `--preprocess MODE` selector is preferred over a boolean because a third mode (`native` under `--no-resize`) already exists; `--no-resize` is kept as the explicit escape hatch because it composes orthogonally with `--max-tokens`.

`native518` (default, feature mode only):

1. If `min(img.nx, img.ny) > 518`, compute `scale = 518 / min(nx, ny)`, resize bicubic to `lround(nx*scale) x lround(ny*scale)` (same formula shape as `dino_classify_preprocess` at `dinov2.cpp:103-106`).
2. Apply the existing `dino_preprocess` round-up-to-patch-multiple normalization on the result.
3. Images whose shortest edge is already <= 518 skip step 1 entirely and take the existing code path unchanged. This preserves all current parity artifacts: `assets/tench.jpg` (500x375) already resizes to 518x518 under the round-up rule... 

  Correction to record in implementation: tench.jpg is 500x375, so under the new default its shortest edge (375) is <= 518 and it keeps today's exact output grid of 518x518? No: today's path resizes 500x375 to (500/14+1)*14 x (375/14+1)*14 = 504 x 378 = 36x27 = 972 patches. The recorded parity numbers used that grid. Since step 1 only engages above 518, 500x375 is unchanged and parity artifacts stay valid. Any input whose shortest edge exceeds 518 changes behavior by design.

`--no-resize`: skip step 1; emit the image at its rounded-up native dims (today's exact behavior). Still subject to `--max-tokens`.

`--max-tokens N`: hard cap on `num_patches = (ny/patch) * (nx/patch)` evaluated **after** preprocessing and **before** `dino_predict` builds any graph. Default: `4 * ceil(518 / patch_size)^2` tokens per image (patch 14 gives 4*37*37 = 5,476), which allows roughly a 4:1 aspect extreme at the 518 bound while capping the quadratic attention term near ~120 MB per layer per image at f32. `N = 0` disables the cap. On exceed: `error: image 'X' yields N patches after preprocessing (limit M; use --max-tokens to raise or --no-resize is already implied)` style message on stderr, free the model, exit 1.

Enforcement point: in `main`, inside the preprocess loop at `dinov2-cli.cpp:227-232` (a post-check on `img_f.nx/img_f.ny`), so it covers both `native518` and `--no-resize`, runs before `dino_model_load`'s buffer allocation matters for the image, and precedes every `dino_predict` call. The resize itself is a new preprocessing function (see file plan) rather than a mutation inside `dino_preprocess`, keeping `dino_preprocess` byte-identical for tests and embedders.

Classifier mode (`-c`) is untouched: it always produces 224x224.

Interaction with batching: unchanged. Different aspect ratios still yield different grids after the bound, so mixed-size chunks keep hitting the existing error at `dinov2-cli.cpp:240-249`. Document in `docs/cli.md` that grouping by aspect ratio, not just pixel size, is what matters.

### 5.2 Allocation guard

- Check `ggml_backend_alloc_ctx_tensors` at `dinov2.cpp:384`: on `nullptr`, print `failed to allocate model buffer` plus the backend name to stderr and return false from `dino_model_load`.
- Check `ggml_gallocr_alloc_graph` at `dinov2.cpp:1097`: on false, print the image dims, patch-grid token count, and a hint (`reduce input size; see --max-tokens / --preprocess`), then `ggml_free(ctx_cgraph)` and return `{}` from `dino_predict`. Callers already treat an empty vector as exit-1 (`dinov2-cli.cpp:284-288`, `371-376`).
- Check `ggml_gallocr_new` at `dinov2-cli.cpp:270`: on `nullptr`, free the model and exit 1.

### 5.3 `--preprocess hf`

Feature mode only. Runs the identical recipe as `dino_classify_preprocess` (shortest-edge 256 bicubic, center crop 224x224, ImageNet mean/std): literally reuse that function so the two paths cannot drift. Output is a 16x16 = 256-token grid at patch 14, matching what `AutoImageProcessor` produces for `facebook/dinov2-*`, so `scripts/parity_check.py` compares identical token counts instead of 256-vs-1444 mismatches.

Residual delta expectation: PIL `BICUBIC` applies a support-scaled kernel when downscaling (antialiasing); the repo's `resize_planes` is Catmull-Rom (OpenCV `INTER_CUBIC`) with no downscale prefilter. For images downscaled to the 256 shortest edge, small pixel deltas vs PIL are expected and the existing parity thresholds (`--patches-threshold 0.99`) absorb them. Record this in `docs/parity/README.md`.

### 5.4 JSON/JSONL record contract

Additive keys in `print_embeddings_json`:

- `"index"`: 0-based position of the input in `fnames_inp` (the existing `idx` at `dinov2-cli.cpp:291`). Always present.
- `"input"`: the path exactly as passed to `-i`. Same value as `"image"`; keep both. `image` stays for compatibility, `input` is added because the design brief names it and it reads better next to `index`. (Alternative considered: reuse `image` only. Flagged for the lead; the spec defaults to adding `input` as specified.)
- `"grid": {"h": <ny/patch>, "w": <nx/patch>}`: patch-grid dimensions. Always present.
- `patches` stays flat, documented as row-major `h * w * hidden` (`docs/cli.md:215-218` already describes the scan order; update to reference `grid`).

Contract: one record per input, emitted in input order; `index` is the join key when `image` is ambiguous (duplicate paths, sanitized stems). Update `docs/cli.md` schema table and the stdout/stderr contract section.

D2EMB binary format: the v1 32-byte header (`dinov2.cpp:795-803`, documented at `docs/cli.md:237-246`) has a `patch_count` field but no grid dims, and one reserved u32. Decision: bump the format version to 2 with a 40-byte header adding `grid_w` and `grid_h` u32 fields at offsets 32 and 36. Justification: the format is an explicitly unstable preview with no compatibility guarantees, so a clean version bump beats packing two u16s into the reserved word; readers already must reject unknown versions. `write_embeddings_binary` gains the two dims as parameters. The header test at `tests/test_dinov2.cpp:510-600` must be updated for the new size/version.

### 5.5 Version

Bump `project(... VERSION 0.3.0)` to `0.4.0` in `CMakeLists.txt:3`. `DINOV2_VERSION` propagates through `target_compile_definitions` (`CMakeLists.txt:34`) to `--version` (`dinov2.cpp:1028`). No git tag; pre-release builds now report `dinov2-cli 0.4.0`.

## 6. Tests to add

- `tests/test_dinov2.cpp`: unit tests for the new bounded-preprocess function: a synthetic large image (e.g. 2000x800 at patch 14) yields shortest-edge-bounded dims and a patch count under the default cap; an image at exactly 518 shortest edge is unchanged; a smaller image is unchanged; `--no-resize` equivalent (bound skipped) reproduces `dino_preprocess` dims; the `hf` path on a non-square input yields exactly 224x224 (16x16 grid at patch 14).
- `tests/test_cli.cpp`: subprocess cases: `--max-tokens 1` on `assets/tench.jpg` exits 1 with a stderr message naming the cap; `--preprocess hf` on the minimal-GGUF path reaches model-load failure cleanly (or, with a tiny model fixture, emits `"grid":{"h":16,"w":16}`); `--preprocess bogus` is a parse error; `--max-tokens` rejects non-integer and negative values through `numeric_parse_error`.
- JSON contract: extend `tests/test_cli.cpp` (it already runs the CLI subprocess) or a small Python check in the parity harness to assert each JSONL line contains `index`, `input`, `image`, `grid.h`, `grid.w`, and that `grid.h * grid.w == n_patches`, with `index` matching line order.
- Allocation-failure path: `ggml_gallocr_alloc_graph` failure cannot be triggered deterministically without actually exhausting memory, so no unit test for it; instead, the `--max-tokens` cap test exercises the same exit-1-with-clean-stderr contract before allocation. Note that `allocr` failure inside `dino_predict` is covered by inspection and the empty-vector contract test (`dinov2-cli.cpp:284`).
- Binary header: update the existing `write_embeddings_binary` test for version 2 / 40-byte header and the two new grid fields.

## 7. Docs to update

- `docs/cli.md`: new flags (`--preprocess`, `--no-resize`, `--max-tokens`), the 518 default and its motivation, batching note (aspect ratio grouping), JSON schema table (new keys), D2EMB v2 header table, exit-code table.
- `README.md`: feature-mode default preprocessing sentence and the HF-compatible mode pointer.
- `docs/parity/README.md`: record that default feature preprocessing bounds the shortest edge to 518, that parity runs should use `--preprocess hf` for token-count parity with `AutoImageProcessor`, and the PIL-vs-Catmull-Rom resampling caveat.
- `ARCHITECTURE.md`: the preprocessing description (`src/image.{h,cpp}` row and data-flow diagram) if it states native resolution is preserved.
- `CHANGELOG.md`: entries for the default preprocessing change (behavior change), new flags, record-contract keys, D2EMB v2, and the version bump.

## 8. Task decomposition

Independent tasks; each lists files and its verification gate. Suggested order groups 1-2 first since 3-5 depend on the flag surface.

1. **Version bump.** `CMakeLists.txt`. Verify: `cmake --preset debug && cmake --build build-debug`, `build-debug/bin/dinov2-cli --version` prints `0.4.0`.
2. **Bounded preprocessing + cap + flags.** `dinov2.h` (new params: `preprocess_mode`, `max_tokens`; new function decl), `src/image.{h,cpp}` (bounded resize helper), `dinov2.cpp` (`print_usage`, `dino_params_parse`, cap check), `dinov2-cli.cpp` (preprocess dispatch, cap check before model load or in preprocess loop, error paths through `free_model`). Verify: `ctest --test-dir build-debug`, plus ASAN/UBSAN presets (`build-asan`, `build-ubsan`) since the resize path touches raw buffers.
3. **Allocation guards.** `dinov2.cpp` (three sites in section 5.2), `dinov2-cli.cpp` (gallocr_new check). Verify: ctest; manual review that every new error path frees `ctx_cgraph`/model.
4. **`--preprocess hf`.** `dinov2.cpp` (dispatch to `dino_classify_preprocess`), params plumbing from task 2. Verify: new unit test asserting 224x224 output; parity re-run.
5. **Record contract + D2EMB v2.** `dinov2-cli.cpp` (`print_embeddings_json`, `binary_out_path` call site), `dinov2.{h,cpp}` (`write_embeddings_binary` signature + header), `tests/test_dinov2.cpp`, `tests/test_cli.cpp`. Verify: ctest; `jq` assertion on emitted JSONL.
6. **Docs + changelog.** Files in section 7. Verify: `clang-format-18` clean on touched sources (`git clang-format` or repo script if present); `DINOV2_FATAL_WARNINGS` build clean if the preset exists in `CMakePresets.json`.

Final gates before merge (single reviewer pass over the full diff):

- Debug + ASAN + UBSAN `ctest` green.
- `clang-format` (version 18, matching CI) clean on all touched C++ files.
- `DINOV2_FATAL_WARNINGS=ON` build clean.
- Parity re-run at the final SHA for both small checkpoints (`dinov2-small`, `dinov2-with-registers-small`) with the recorded `docs/parity/README.md` commands, plus a `--preprocess hf` run comparing token counts against HF.
- Real large-image smoke: generate a 3440x5601 PNG (stb write or `scripts/` helper), run feature mode, confirm a bounded grid and clean exit, no abort; run with `--no-resize --max-tokens 0` only if the machine can sustain it, otherwise rely on the cap test.
- Review gates: spec conformance check against section 5, no change to classifier preprocessing, all new error paths free resources and exit 1, docs match emitted keys byte-for-byte.

## 9. Open risks and flagged conflicts

- **Round direction**: the brief described rounding dims *down* to patch multiples; the code rounds *up* (`(v/t + 1) * t`, asserted in tests). This spec keeps round-up for consistency. Changing to round-down would alter existing test expectations and shift every grid by one patch on non-aligned inputs.
- **`input` vs `image`**: the brief asks for an `input` key; the emitter already writes `image` with the same value. Spec adds `input` additively as requested; dropping `image` instead would be a breaking change to documented schema.
- **`dino_model_load` `img_size`**: `max_img_size` is passed before preprocessing exists (`dinov2-cli.cpp:198` vs `:227`), so the loader still sizes ctx overhead from *raw* image dims. Harmless (it is an upper bound for `offset`), but the bounded dims could be used if the bound is computed pre-load; either is acceptable, note in implementation.
- **`--max-tokens` default of 4x the square bound** is a judgment call: it permits ~2.1M-pixel inputs at patch 14 (e.g. 518x2072). The alternative (exactly `(518/patch)^2` = 1369) would reject moderate-aspect images that are safe today under the 518 shortest-edge bound. Flagged for reviewer confirmation.
- **PIL bicubic vs Catmull-Rom**: `--preprocess hf` will not be bit-identical to HF for downscaled inputs; expected residual is within existing parity thresholds but the parity docs must say so.
