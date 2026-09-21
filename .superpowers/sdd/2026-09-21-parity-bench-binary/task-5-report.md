# Task 5 report: model coverage assessment and scoped backbone support

Date: 2026-09-21
Branch: `feat/batched-inference`

## Status

PASS. Backbone-only DINOv2 feature mode is a small, semantically safe extension of the existing graph. It does not add a new architecture or task head. The loader now tolerates omitted optional register-token metadata and rejects classification requests before any classifier tensor access when the GGUF is backbone-only or incomplete.

DINOv3, depth, segmentation, and other task heads remain out of scope. No weights were downloaded or committed, and no publish workflow was expanded for new repositories.

## Coverage matrix

| Checkpoint family | Converter | Loader feature mode | Loader classification | Publish mapping | Evidence |
|---|---|---|---|---|---|
| `facebook/dinov2-{small,base,large,giant}` | Supported by the existing `AutoModel` path and common tensor conversion | Supported for GGUFs with the DINOv2 backbone tensors; missing `num_register_tokens` defaults to zero | Unsupported by design because no ImageNet head is present; now a clear error | Not mapped | No model download or parity claim in this task |
| `facebook/dinov2-with-registers-{small,base,large,giant}` | Existing register tensor and layer selection path | Supported when register metadata and tensor are present | Unsupported unless a classifier head and labels are present | Not mapped as backbone-only variants | Existing small register classifier parity remains in Task 3 |
| `facebook/dinov2-{size}-imagenet1k-1-layer` | Existing classifier conversion path | Supported | Supported when classifier tensors, `num_classes`, and labels are present | Mapped in the eight-variant publish workflow | Existing small regular parity evidence only |
| `facebook/dinov2-with-registers-{size}-imagenet1k-1-layer` | Existing classifier conversion path | Supported | Supported when classifier tensors, `num_classes`, and labels are present | Mapped in the eight-variant publish workflow | Existing small register parity evidence only |
| DINOv2 depth or segmentation heads | Not implemented | Not applicable | Not applicable | Not mapped | Explicitly deferred |
| DINOv3 | Not implemented | Not applicable | Not applicable | Not mapped | Separate architecture and resource target; explicitly deferred |

## Evidence and behavior inspection

* `scripts/dinov2-to-gguf.py:42-50` selects `AutoModelForImageClassification` only for model names containing `imagenet`; other names use `AutoModel`. The converter writes `num_classes: 0` and no labels for a backbone-only checkpoint, while the shared encoder and embedding tensors remain the same. It currently writes `num_register_tokens: 0` at line 124, but the loader should not require that optional key from independently produced GGUFs.
* `dinov2.cpp:255-272` loads the required backbone hyperparameters and now uses a safe zero default for absent `num_register_tokens`.
* `dinov2.cpp:285-318` validates classification metadata, labels, and `classifier.weight` / `classifier.bias` before graph construction. A backbone-only request with `-c` therefore returns a loader error rather than reaching the `model.tensors.at(...)` calls in `forward_head` at `dinov2.cpp:693-694`.
* Feature mode does not call `forward_head`. `build_graph` at `dinov2.cpp:705-715` only adds the head when `params.classify` is true. `forward_features` always emits CLS and non-register patch tokens, and the feature output path builds pooled vectors from those tokens. This gives backbone-only models the existing CLS, pooled, patch-token, and PCA semantics.
* `dinov2.h:106-112` defines the output contract: CLS in both modes, pooled and patch tokens in feature mode, and classification predictions only in classify mode.
* `scripts/publish-gguf.sh:56-71` maps only the eight ImageNet classifier checkpoint names. `.github/workflows/convert-and-publish-gguf.yml:38-46` repeats that exact eight-variant matrix. `docs/hf-publishing.md:32-43` documents the same eight published repositories. Since the repository does not already intend to publish backbone-only repositories, this task leaves those mappings unchanged.
* `docs/cli.md:121-139` now documents the supported feature-only backbone coverage, the clean classification failure, the zero-register default, the publish limitation, and the explicit DINOv3 and other-head boundaries.

## Changes

* `dinov2.cpp`: added optional GGUF uint32 metadata lookup; defaulted absent `num_register_tokens` to zero; validated classifier metadata, label keys, and tensors before classification loading.
* `tests/test_cli.cpp`: added a no-download malformed-model classification smoke test that asserts exit 1, empty stdout, a loader failure, and no assertion text. Existing parser, help, and version cases remain unchanged.
* `docs/cli.md`: added the coverage and scope decision.
* No converter, publish mapping, or weights were changed because the existing converter and eight-repository workflow do not publish backbone-only models.

## Verification

* `clang-format -i dinov2.cpp tests/test_cli.cpp`: PASS.
* `cmake --build --preset debug -j2`: PASS.
* `cmake --preset debug -DDINOV2_FATAL_WARNINGS=ON && cmake --build --preset debug --clean-first -j2`: PASS.
* `ctest --test-dir build-debug --output-on-failure`: PASS, 3/3 tests. The focused CLI test took 20.87 seconds because each subprocess initializes the CLI backend.
* `git diff --check`: PASS.

## Follow-up recommendations

1. If backbone-only weights become a publishing goal, add explicitly named HF mappings and separate conversion validation plus parity artifacts for each model size. Do not infer base, large, or giant parity from the small checkpoint.
2. Add a real small synthetic backbone-only GGUF fixture to exercise the successful loader path and omitted metadata path in CI. The current repository tests avoid model fixtures and this task did not download a giant or other model.
3. Treat task heads and DINOv3 as separate design and compatibility work rather than extending the current classifier graph.

## Review fix notes

* Replaced the malformed-file CLI smoke test with a deterministic, valid GGUF generated in `tests/test_cli.cpp`. The fixture contains only the required backbone metadata, omits `num_classes` and classifier tensors, and verifies `-c` fails at the classification preflight with no assertion text. The same assertion checks the omitted `num_register_tokens` path reports a zero default.
* Added a second minimal GGUF fixture containing `embeddings.register_tokens` without `num_register_tokens`. The loader now rejects this inconsistency before model tensors are loaded, preventing the tensor from being silently treated as zero registers. It also rejects the inverse mismatch while preserving models with neither metadata nor tensor.
* Re-ran clang-format 18.1.8, the debug build, focused `test_cli`, full ctest, and `git diff --check`.

## Consolidated final review fix wave: 2026-09-21

This fix wave replaces actionable `huggingface-cli download` instructions with `hf download` while preserving repositories, paths, and arguments. Benchmark generation now records the image path `assets/tench.jpg`, `Release` build configuration, Ubuntu platform scope, f16-only quantization, model paths, checksums, commit, and run metadata in generated evidence and workflow provenance. The committed result retains the canonical successful Ubuntu run; earlier failed or parser-debug attempts are explicitly historical in the task-4 report.

The loader now validates positive register-token metadata against `embeddings.register_tokens` dimensions before graph execution: hidden size, register count, and singleton trailing dimensions must match. No-register and valid backbone-only models remain supported. CLI tests cover hidden, count, and trailing-dimension mismatches. New CLI tests use a unique directory under `std::filesystem::temp_directory_path()` and remove it through an RAII cleanup guard. Classification now rejects `--topk` greater than model `num_classes` after model load with exit status 1 and an actionable error; the focused fixture covers this behavior.

Verification performed for this wave: debug build, CTest, CLI-focused fixture tests, benchmark parser regression, actionlint, clang-format 18, and `git diff --check`. No release or merge was run, and no push was performed.
