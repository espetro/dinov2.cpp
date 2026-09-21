# Large-image feature preprocessing and wrapper ergonomics design

Status: proposed design only. This document does not claim that any implementation or measurement described below has landed.

Date: 2026-09-22

## 1. Summary

An external beta test of `dinov2-cli` built from `ce9429b` surfaced four defects and one packaging issue:

1. Feature mode has no input-size bound. `dino_preprocess` (`dinov2.cpp:138`) resizes each dimension up to the next multiple of `patch_size`, so a 3440x5601 PNG becomes a 246x401 patch grid (98,646 tokens). Self-attention memory is quadratic in token count (`ggml_mul_mat` KQ product at `dinov2.cpp:467`), which requested a ~222 GB Metal buffer and ended in `Abort trap 6`.
2. Allocation failures abort instead of failing cleanly. `ggml_gallocr_alloc_graph` (`dinov2.cpp:1097`) and `ggml_backend_alloc_ctx_tensors` (`dinov2.cpp:384`) are unchecked.
3. Metric parity vs the HF feature pipeline is a preprocessing-grid mismatch, not numerical error: HF `AutoImageProcessor` crops to 224x224 (256 tokens at patch 14) while the CLI emits the full native grid.
4. JSONL records carry no input index and `patches` is flat with no grid dimensions, so consumers cannot reconstruct the spatial grid or join outputs to inputs robustly.
5. `--version` reports `dinov2-cli 0.3.0` while the build is a 0.4.0 beta.

This design bounds feature preprocessing to a shortest-edge 518 resize by default, adds a hard token cap, checks the two allocation call sites, adds HF-compatible and batch-safe cropping preprocessing modes, extends the record contract, and bumps the project version to 0.4.0.

## 2. Goals

- Feature mode can never request unbounded memory: default resize bounds the shortest edge to 518, and a `--max-tokens` hard cap rejects grids that are still too large, before any graph is constructed.
- Allocation failure is a stderr message plus exit 1, never an abort, for both the model-weight buffer and the per-graph compute buffer.
- `--preprocess hf` reproduces the HF `AutoImageProcessor` feature pipeline exactly (resize shortest edge to 256, center crop 224x224) so parity checks compare like grids. `--preprocess crop518` gives a fixed 518x518 (37x37) grid so mixed-dimension batches always share dims.
- Every JSON record is self-describing: input order index, the input path, and the patch-grid dimensions accompany the flat `patches` array.
- `dinov2-cli --version` reports 0.4.0 for pre-release builds; no git tag is created.

## 3. Non-goals

- No change to classifier preprocessing semantics: `dino_classify_preprocess` keeps byte-identical output (it becomes a thin wrapper over the shared resize+center-crop helper).
- No tiling, sliding-window, or chunked attention to support truly unbounded images.
- No stable-version guarantee for the D2EMB preview format; it is explicitly unstable.
- No removal of the `image` JSON field; new keys are additive.
- No implementation in this commit.

## 4. Current behavior, verified against the code

### Feature preprocessing today

`dino_preprocess(const Image &img, const dino_hparams &params)` (`dinov2.cpp:138-158`) calls `preprocess_for_dinov2(img, params.patch_size)` (`src/image.cpp:138-170`), which resizes each dimension to `((v / patch_size) + 1) * patch_size` via `resize_planes` bicubic, scales to [0,1], and applies ImageNet mean/std. The fallback in `dino_preprocess` re-resizes to the same expected multiple if the helper ever disagrees.

This is a strict round-up: a dimension already at a patch multiple still grows by one patch (518 -> 532 -> 38 patches). The new rule is true ceil (`ceil(v / patch) * patch`), so an aligned dimension is unchanged (518 stays 518 -> 37 patches); non-aligned dims produce identical results to today (the existing 100x50 -> 112x56 test is unaffected).

For the failing case: 3440x5601 at patch 14 gives new_w = 3444, new_h = 5614 under either rule, a 246x401 = 98,646-token grid.

### tench.jpg reconciliation

`assets/tench.jpg` is 612x408 (measured with `sips`). Shortest edge 408 <= 518, so the bounded default does not resize it. Under today's strict round-up: 612 -> 616 (44 patches), 408 -> 420 (30 patches), grid 44x30 = 1320 tokens, matching the `n_patches = 1320` observed in the main-branch smoke and in `docs/cli.md`'s schema example. Under true ceil the result is identical (neither dim is a multiple of 14), so the committed parity artifacts under `docs/parity/` remain valid and do not need a re-run for the alignment change alone.

### Where image size flows

- `dinov2-cli.cpp:184-194` computes `max_img_size` over loaded `Image` dims and passes it to `dino_model_load`.
- `dino_model_load` (`dinov2.cpp:357-365`) uses `img_size` only to size `model.ctx` overhead (`offset` for the interpolated `new_pos_embed` tensor); it does not bound compute.
- `dino_predict` (`dinov2.cpp:1071-1087`) derives `num_patches = (ny/patch) * (nx/patch)` from the preprocessed `ImageF` dims and builds the graph via `build_graph({nx, ny}, ...)`.
- `forward_features` (`dinov2.cpp:536-538`) recomputes `h0`, `w0`, `num_patches` from `img_size`; `attn` allocates the quadratic KQ tensor over `W * H` tokens (`dinov2.cpp:467`).

### Single vs batched input paths

`main` preprocesses every input in a loop (`dinov2-cli.cpp:227-232`): `dino_classify_preprocess` under `-c`, else `dino_preprocess`. Chunks of `params.n_batch` are then checked to share identical post-preprocess dims (`dinov2-cli.cpp:237-251`), because `dino_predict` packs a chunk into one graph (`dinov2.cpp:1069-1079`). Under the `bounded` default, images with different aspect ratios still produce different grids, so the existing mixed-size chunk error remains the correct behavior; `crop518` is the batch-safe mode since every input becomes 518x518.

### Unchecked allocation sites

- `dinov2.cpp:384`: `model.buffer = ggml_backend_alloc_ctx_tensors(model.ctx, model.backend)` returns `nullptr` on failure; unchecked.
- `dinov2.cpp:1097`: `ggml_gallocr_alloc_graph(allocr, gf)` returns `bool`; unchecked. This is the call that produced the abort: Metal buffer allocation fails inside it and the process continues into `ggml_backend_graph_compute` with an unallocated graph.
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

### 5.1 Feature preprocessing modes and cap

Add `--preprocess MODE` with three values, a `--no-resize` opt-out, and a `--max-tokens N` cap. `--preprocess` is a selector rather than a boolean because there are three recipes; `--no-resize` is orthogonal and only meaningful with `bounded`, so combining it with `hf` or `crop518` is a usage error (exit 1). Passing `-c` together with `--preprocess` or `--no-resize` is also a usage error: classifier preprocessing is fixed at (256, 224) and silently ignoring the flag would surprise.

All resize+center-crop recipes go through one shared helper `preprocess_resize_crop(img, short_edge, crop)` (new, in `src/image.cpp` or `dinov2.cpp`): shortest-edge resize preserving aspect via `resize_bicubic` with `lround` dims (the existing formula at `dinov2.cpp:103-106`), center crop, float [0,1] + ImageNet mean/std. `dino_classify_preprocess` becomes a call with (256, 224) and must produce byte-identical output to today; `hf` uses (256, 224) as well; `crop518` uses (518, 518) giving a fixed 37x37 grid at patch 14.

A new public function `dino_preprocess_mode(const Image &, const dino_hparams &, const dino_params &)` (name adjustable) dispatches:

- `bounded` (default): if `min(nx, ny) > 518`, bicubic-resize so the shortest edge is 518 preserving aspect (`lround` dims), then apply `dino_preprocess` (true-ceil alignment + normalize). Images with shortest edge <= 518 skip the resize entirely and take today's path.
- `bounded` + `--no-resize`: skip the shortest-edge resize; equivalent to today's exact behavior. Still subject to `--max-tokens`.
- `hf`: `preprocess_resize_crop(img, 256, 224)` -> 16x16 = 256 tokens at patch 14.
- `crop518`: `preprocess_resize_crop(img, 518, 518)` -> 37x37 = 1369 tokens at patch 14; every input shares dims, so mixed-dimension batches always pass the chunk check.

`--max-tokens N`: hard cap on `num_patches = (ny/patch) * (nx/patch)` evaluated after each image is preprocessed in the `main` loop (`dinov2-cli.cpp:227-232`) and before any `dino_predict` call. `N = 0` disables. The default is computed from model hparams at check time (patch size lives in the GGUF): `4 * ceil(518 / patch_size)^2`, i.e. 5,476 tokens for patch 14, covering roughly a 4:1 aspect extreme at the 518 bound. On exceed:

```
error: image 'X' yields N patch tokens after preprocessing (limit M). Use a smaller input, --preprocess crop518, or raise --max-tokens.
```

then free resources through the existing cleanup path and exit 1.

Interaction with batching: under `bounded`, different aspect ratios still yield different grids, so mixed-size chunks keep hitting the existing error at `dinov2-cli.cpp:240-249`; `crop518` removes the mismatch entirely. Document in `docs/cli.md` that grouping by aspect ratio, not just pixel size, is what matters, and that `crop518` is the batch-safe option.

### 5.2 Allocation guard

- Check `ggml_backend_alloc_ctx_tensors` at `dinov2.cpp:384`: on `nullptr`, print `failed to allocate model buffer` to stderr and return false from `dino_model_load`.
- Check `ggml_gallocr_alloc_graph` at `dinov2.cpp:1097`: on false, print the image dims, patch-grid token count, and a hint (`reduce input size; see --max-tokens / --preprocess`), then `ggml_free(ctx_cgraph)` and return `{}` from `dino_predict`. Callers already treat an empty vector as exit-1 (`dinov2-cli.cpp:284-288`, `371-376`).
- Check `ggml_gallocr_new` at `dinov2-cli.cpp:270`: on `nullptr`, free the model and exit 1.

### 5.3 `--preprocess hf`

Runs the identical recipe as `dino_classify_preprocess` through the shared helper with (256, 224), so the classifier and HF-feature paths cannot drift. Output is a 16x16 = 256-token grid at patch 14, matching what `AutoImageProcessor` produces for `facebook/dinov2-*`, so `scripts/parity_check.py` compares identical token counts instead of 256-vs-1320 mismatches.

Residual delta expectation: PIL `BICUBIC` applies a support-scaled kernel when downscaling (antialiasing); the repo's `resize_planes` is Catmull-Rom (OpenCV `INTER_CUBIC`) with no downscale prefilter. For images downscaled to the 256 shortest edge, small pixel deltas vs PIL are expected and the existing parity thresholds (`--patches-threshold 0.99`) absorb them. Record this in `docs/parity/README.md`.

### 5.4 JSON/JSONL record contract

Additive keys in `print_embeddings_json`:

- `"index"`: 0-based position of the input in `fnames_inp` (the existing `idx` at `dinov2-cli.cpp:291`). Always present.
- `"grid": {"h": <ny/patch>, "w": <nx/patch>}`: patch-grid dimensions. Always present.
- `patches` stays flat, documented as row-major `h * w * hidden` (`docs/cli.md:215-218` already describes the scan order; update to reference `grid`).

`image` already carries the path as given and stays; no `input` key is added.

Contract: one record per input, emitted in input order; `index` is the join key when `image` is ambiguous (duplicate paths, sanitized stems). Update `docs/cli.md` schema table and the stdout/stderr contract section.

D2EMB binary format: the v1 32-byte header (`dinov2.cpp:795-803`, documented at `docs/cli.md:237-246`) has a `patch_count` field but no grid dims, and one reserved u32. Decision: bump the format version to 2 with a 40-byte header adding `grid_w` and `grid_h` u32 fields at offsets 32 and 36. Justification: the format is an explicitly unstable preview with no compatibility guarantees, so a clean version bump beats packing two u16s into the reserved word; readers already must reject unknown versions. `write_embeddings_binary` gains the two dims as parameters. The header test at `tests/test_dinov2.cpp:510-600` must be updated for the new size/version.

### 5.5 Version

Bump `project(... VERSION 0.3.0)` to `0.4.0` in `CMakeLists.txt:3`. `DINOV2_VERSION` propagates through `target_compile_definitions` (`CMakeLists.txt:34`) to `--version` (`dinov2.cpp:1028`). No git tag; pre-release builds now report `dinov2-cli 0.4.0`.

## 6. Tests to add

- `tests/test_dinov2.cpp`: unit tests for the bounded path: a 2000x800 image at patch 14 yields shortest-edge-518-aligned dims within the default cap; exactly 518x518 stays 518x518 (37x37) under true ceil; 500x375 is untouched by the bound and aligned by true ceil (504x378); the native path equals today's dims for non-aligned inputs; `hf` on a non-square input yields 224x224; `crop518` on 2000x800 yields 518x518; 112x56 stays 112x56 under true ceil.
- `tests/test_cli.cpp`: subprocess cases: `--max-tokens 1` exits 1 with the cap message; `--preprocess bogus` is a usage error; `--max-tokens 12abc` and `--max-tokens -1` fail strict parsing; `--no-resize --preprocess hf` exits 1; `-c --preprocess hf` exits 1.
- JSON contract (later handoff): assert each JSONL line contains `index`, `image`, `grid.h`, `grid.w`, and `grid.h * grid.w == n_patches`, with `index` matching line order.
- Allocation-failure path: `ggml_gallocr_alloc_graph` failure cannot be triggered deterministically without actually exhausting memory, so no unit test for it; the `--max-tokens` cap test exercises the same exit-1-with-clean-stderr contract before allocation. The empty-vector contract from `dino_predict` remains the guard.
- Binary header (later handoff): update the existing `write_embeddings_binary` test for version 2 / 40-byte header and the two new grid fields.

## 7. Docs to update

- `docs/cli.md`: new flags (`--preprocess`, `--no-resize`, `--max-tokens`), the 518 default and its motivation, batching note (aspect ratio grouping, `crop518` as the batch-safe mode), JSON schema table (new keys), D2EMB v2 header table, exit-code table. The flag table must be updated in the same commit that adds the flags so help and docs do not drift.
- `README.md`: feature-mode default preprocessing sentence and the HF-compatible mode pointer.
- `docs/parity/README.md`: record that default feature preprocessing bounds the shortest edge to 518, that parity runs should use `--preprocess hf` for token-count parity with `AutoImageProcessor`, and the PIL-vs-Catmull-Rom resampling caveat.
- `ARCHITECTURE.md`: the preprocessing description (`src/image.{h,cpp}` row and data-flow diagram) if it states native resolution is preserved.
- `CHANGELOG.md`: entries for the default preprocessing change (behavior change), new flags, record-contract keys, D2EMB v2, and the version bump.

## 8. Task decomposition

Independent tasks; each lists files and its verification gate.

1. **Version bump.** `CMakeLists.txt`. Verify: `cmake --preset debug && cmake --build build-debug`, `build-debug/bin/dinov2-cli --version` prints `0.4.0`.
2. **Bounded preprocessing + cap + flags.** `src/image.{h,cpp}` (true-ceil alignment, bounded resize helper, shared resize+center-crop helper), `dinov2.h` (params `preprocess_mode`, `no_resize`, `max_tokens`; dispatch function decl), `dinov2.cpp` (`print_usage`, `dino_params_parse`, `dino_classify_preprocess` refactor to shared helper, mode dispatch), `dinov2-cli.cpp` (dispatch, cap check, flag-conflict errors). Verify: `ctest --test-dir build-debug`, plus ASAN/UBSAN presets since the resize path touches raw buffers; classifier byte-identity check against a pre-change build.
3. **Allocation guards.** `dinov2.cpp` (three sites in section 5.2), `dinov2-cli.cpp` (gallocr_new check). Verify: ctest; manual review that every new error path frees `ctx_cgraph`/model.
4. **`hf` and `crop518` modes.** Covered by the shared helper in task 2; kept as a separate reviewable commit. Verify: unit tests asserting 224x224 and 518x518 outputs; parity re-run.
5. **Record contract + D2EMB v2.** `dinov2-cli.cpp` (`print_embeddings_json`), `dinov2.{h,cpp}` (`write_embeddings_binary` signature + header), `tests/test_dinov2.cpp`, `tests/test_cli.cpp`, `docs/cli.md`. Verify: ctest; `jq` assertion on emitted JSONL.
6. **Docs + changelog.** Files in section 7. Verify: `clang-format` (version 18) clean on touched sources; `DINOV2_FATAL_WARNINGS` build clean (on by default via `CMakePresets.json`).

Final gates before merge (single reviewer pass over the full diff):

- Debug + ASAN + UBSAN `ctest` green.
- `clang-format` (version 18, matching CI) clean on all touched C++ files.
- `DINOV2_FATAL_WARNINGS=ON` build clean.
- Parity re-run at the final SHA for both small checkpoints (`dinov2-small`, `dinov2-with-registers-small`) with the recorded `docs/parity/README.md` commands, plus a `--preprocess hf` run comparing token counts against HF.
- Real large-image smoke: generate a 3440x5601 PNG, run feature mode, confirm a bounded grid and clean exit, no abort; `--no-resize` should hit the `--max-tokens` error in well under a second; `crop518` and `hf` should report 37x37 and 16x16 grids.
- Review gates: spec conformance check against section 5, classifier preprocessing output byte-identical, all new error paths free resources and exit 1, docs match emitted keys byte-for-byte.

## 9. Open risks and flagged conflicts

- **`dino_model_load` `img_size`**: `max_img_size` is passed before preprocessing runs (`dinov2-cli.cpp:198` vs `:227`), so the loader still sizes ctx overhead from raw image dims. Harmless (it is an upper bound for `offset`), but bounded dims could be used if the bound is computed pre-load; either is acceptable.
- **`--max-tokens` default of 4x the square bound** permits ~2.1M-pixel inputs at patch 14 (e.g. 518x2072). The alternative (exactly `(518/patch)^2` = 1369) would reject moderate-aspect images that are safe under the 518 shortest-edge bound.
- **PIL bicubic vs Catmull-Rom**: `--preprocess hf` will not be bit-identical to HF for downscaled inputs; expected residual is within existing parity thresholds but the parity docs must say so.
- **Parity artifacts**: tench is 612x408; under true ceil its grid is unchanged (44x30 = 1320) and the bounded default does not resize it (408 <= 518), so committed parity artifacts under `docs/parity/` stay valid. Any future artifact recorded on an image whose shortest edge exceeds 518 must state the preprocessing mode used.
