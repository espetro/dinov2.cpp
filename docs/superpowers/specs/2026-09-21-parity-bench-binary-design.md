# v0.4.0 parity, benchmark, and binary embeddings design

Status: proposed design only. This document does not claim that any implementation or measurement described below has landed.

Date: 2026-09-21

## 1. Summary

Version 0.4.0 should make three related claims auditable without making the CLI or its preview binary format larger than necessary:

1. Parity claims must be backed by committed, reproducible evidence for the regular and register-token small models that were actually run.
2. Benchmark claims must be backed by freshly generated, dated Ubuntu results. Ubuntu is the only supported benchmark platform for this release.
3. Consumers that need higher throughput than JSON can opt into a deliberately simple binary embeddings preview. It is not a stable interchange standard.

The release also tightens numeric CLI parsing. Invalid numeric input must fail before model loading with a useful stderr message and exit code 1. Help and version remain successful, zero-output operations.

## 2. Goals

- Make parity evidence self-contained: CLI version, repository SHA, GGUF path and SHA-256, Hugging Face model id, image inputs, thresholds, exact command, and readable metrics must travel together.
- Run and commit parity evidence for both a regular small checkpoint and a register-token small checkpoint. Evidence names only models that were run.
- Replace the current benchmark seed and placeholder language with dated, platform-labeled Ubuntu measurements produced by the current benchmark command.
- Keep benchmark scope to CPU on Ubuntu, the existing f16 baseline, and the existing forward-pass measurement. Do not add quantization benchmark tables.
- Make every numeric CLI value parse as a complete, range-checked integer, with no partial parses or overflow acceptance.
- Add `--embeddings-binary` as an explicitly preview, intentionally simple, little-endian file output mode for CLS, pooled, and optionally patch embeddings.
- Assess whether backbone-only DINOv2 support is a low-effort follow-up, while deferring task heads and DINOv3 to separate architecture and scope decisions.

## 3. Non-goals

- No quantization-specific benchmark mode, quantization comparison, or new `--quant` benchmark selector.
- No multi-platform benchmark matrix in v0.4.0. macOS, Windows, ARM, GPU, and cross-platform performance claims remain out of scope until adoption justifies the operational cost.
- No promise that the binary preview format is stable, portable across future releases, or optimized for smallest size. Binary format optimization and standardization are explicit non-goals.
- No claim of parity for a model, image, or command that was not actually run and recorded.
- No support for DINOv2 task heads in this change. Classification behavior already present in the CLI is not a new head-porting commitment.
- No DINOv3 support. It requires a separate architecture, weights, preprocessing, and compatibility review.
- No implementation in this design commit. The files below are an implementation map, not a statement that the files have changed.

## 4. Exact file plan

### Files to add

- `docs/superpowers/specs/2026-09-21-parity-bench-binary-design.md`: this design.
- `docs/evidence/parity/2026-09-21-small-regular.md`: committed readable evidence for the regular small model, only after the command passes.
- `docs/evidence/parity/2026-09-21-small-registers.md`: committed readable evidence for the register-token small model, only after the command passes.
- `tests/test_cli.cpp`: subprocess-level parsing and binary-output tests, or the existing project test target if it can cover the same contract without a new file.

### Files to update during implementation

- `dinov2.cpp`: strict parsing and parameter validation.
- `dinov2.h`: parameter and output declarations needed by the binary writer, if the implementation requires them.
- `dinov2-cli.cpp`: mode validation, binary file writing, per-input output naming, and error handling.
- `CMakeLists.txt`: register any new test source or target.
- `docs/cli.md`: document numeric ranges, binary preview semantics, and the stdout/stderr contract.
- `docs/benchmarks.md`: Ubuntu-only method and measured-result interpretation. Remove seed, placeholder, and unsupported platform claims.
- `scripts/parity_check.py`: preserve the current default gates and add optional machine-readable JSON output if it can be done without changing the human default.
- `scripts/bench.sh`: ensure generated output records date, Ubuntu platform label, model identity, command inputs, and actual measurements. Do not add quantization behavior.
- `benchmark_results.txt`: replace the current placeholder content with regenerated Ubuntu results, dated and labeled.

No existing documentation, script, source, or result file is to be edited as part of this design-only commit.

## 5. Numeric CLI parsing contract

The flags covered by this release are `-s/--seed`, `-t/--threads`, `-k/--topk`, `--batch`, `--bench-runs`, and `--bench-warmup`.

Each value must be parsed as a complete base-10 integer. Reject an empty value, a sign or format not explicitly accepted by the existing CLI contract, whitespace-wrapped or partially consumed values, decimal or exponent syntax, trailing characters, overflow, and values outside the semantic range. Parsing must use a checked conversion with an end pointer or equivalent, then check the destination type and the option-specific range before assignment. Do not rely on C or C++ casts from a parsed wide integer.

The implementation must retain the existing valid defaults and document their ranges. The minimum useful ranges are:

| Flag | Required valid range in v0.4.0 |
|:--|:--|
| `--seed` | any representable signed 32-bit integer, unless the RNG API requires a narrower documented range |
| `--threads` | 1 through the implementation's positive thread-count limit |
| `--topk` | 1 through the number of available classifier labels |
| `--batch` | 1 through 64, matching the current documented maximum |
| `--bench-runs` | at least 1 and no greater than the documented practical upper bound |
| `--bench-warmup` | 0 through the documented practical upper bound |

If a build-specific upper bound is needed, the error must print the actual accepted range. Every failure prints one concise `error:` line to stderr, names the flag and supplied value, and exits 1. Model loading and inference must not start after a parse failure. Missing values are also exit 1 and must not be reported as successful help.

`-h/--help` prints help to stdout and exits 0. `--version` prints the version to stdout and exits 0. These two paths must work without a model or image. Unknown options remain errors with exit 1.

Tests must cover non-numeric input, a valid numeric prefix followed by junk, decimal input, signed overflow, zero and negative values where disallowed, upper-bound plus one, missing values, and valid boundary values for every flag. Tests must assert exit code, stderr content, and that help/version behavior is unchanged.

## 6. Binary embeddings preview

### Invocation and mode rules

`--embeddings-binary` selects a preview output mode. It requires `-o PATH`; absence of `-o` is a clean error on stderr with exit 1. The mode is feature embeddings only. It cannot be combined with `-c` or `--bench`; reject conflicting combinations instead of silently ignoring a requested output. `--print-patch-tokens` is accepted with this mode and changes the payload as described below. `--l2-normalize` applies to every emitted vector in the same way as JSON.

For one input image, `-o` is the output file path. For more than one input image, `-o` must name an output directory. Create the directory if needed, then write exactly one file per input. Use a deterministic collision-safe naming rule based on input order and sanitized input stem, for example `<index>-<stem>.d2e`; document the rule and never overwrite two inputs that share a stem. A failure to create or write any output is exit 1.

The mode writes no embedding data to stdout. Stdout remains text-safe for status-free scripting. Diagnostics, including the preview warning, go to stderr. The preview warning must say that the format is intentionally not stable and may change without compatibility guarantees.

### Draft file format, version 1

All integer fields and float32 values are little-endian. There is no padding outside the declared header. A file starts with this 32-byte header:

| Offset | Size | Field |
|:--:|--:|:--|
| 0 | 8 | Magic bytes `D2EMB\\0\\0\\0` |
| 8 | 2 | Format version, unsigned integer, currently `1` |
| 10 | 2 | Header size in bytes, currently `32` |
| 12 | 4 | Hidden dimension `H` |
| 16 | 4 | Pooled dimension, currently `2H` |
| 20 | 4 | Patch count `P`; zero when patches are absent |
| 24 | 4 | Flags |
| 28 | 4 | Reserved, written as zero and ignored by readers |

Flag bit 0 means patch vectors are present. Flag bit 1 means L2 normalization was requested. All other bits are zero in version 1 and readers must reject unknown required bits rather than guessing.

The float32 payload is contiguous and has this documented order:

1. `cls[H]`
2. `pooled[2H]`, where the first `H` values are CLS and the next `H` values are the mean of non-register patch tokens
3. If flag bit 0 is set, `patches[P][H]` in row-major, top-to-bottom and left-to-right patch order, excluding register tokens

With `--l2-normalize`, normalize CLS as one vector, pooled as one vector, and each patch row independently, matching the JSON contract. The header dimensions and patch count describe the payload and must be checked before writing. The writer must use explicit byte serialization, not a C++ struct dump, so compiler padding and host endianness cannot leak into the file.

This draft is a preview contract for the v0.4.0 implementation, not a public standard. A future release may change the magic, header, flags, dimensions, ordering, or extension without a compatibility promise.

## 7. Parity evidence artifacts

The source of truth remains `scripts/parity_check.py`. Its default output stays human-readable. An optional `--json-out PATH` may additionally write a JSON document while retaining the normal table on stdout. JSON output must include the command argv, UTC timestamp, CLI version output, git SHA, absolute or repository-relative GGUF path, SHA-256, Hugging Face model id, image paths, all threshold arguments, per-image metrics, and the final verdict. It must not hide informational metrics or turn an unrun model into a passing row.

Each committed markdown artifact under `docs/evidence/parity/` must include:

- Scope statement naming exactly one regular or register-token small model.
- CLI version and repository git SHA.
- GGUF path as used, SHA-256 checksum, and a note that the checksum identifies the exact file.
- Hugging Face model id used as the reference.
- Image path and any generated image recipe.
- Full parity command, including explicit thresholds and binary path.
- Human-readable metrics for CLS cosine, pooled cosine, flat patch cosine, patch-token mean and informational distribution values, top-1 index and label, and top-1 probability difference.
- Gate interpretation and PASS or FAIL verdict.
- Environment and date sufficient for a reader to reproduce the run.

Regular and register-token evidence must use their matching HF reference and GGUF. Do not combine their rows or infer a result for base, large, giant, DINOv3, or any other checkpoint. The artifact may mention that no evidence exists for those models, but it must not imply coverage.

The currently documented default gates are retained unless an executed experiment and review approve a change: CLS cosine at least 0.999, pooled cosine at least 0.999, flat and mean patch cosine at least 0.99, matching classification top-1, and top-1 probability absolute difference less than 0.05. Patch-token minimum remains informational unless an explicit threshold is supplied. The artifact must record the actual threshold values used, not only the defaults.

## 8. Benchmark provenance and result policy

Regenerate `benchmark_results.txt` from the current CLI and current `scripts/bench.sh`. The release result must be dated, identify Ubuntu and CPU architecture, identify the build and git SHA, identify each exact GGUF path and checksum, name the image, record thread count, batch, warmup count, timed run count, and state that the values are forward-pass wall time. Record mean, standard deviation, minimum, maximum, and peak RSS for each measured model.

The v0.4.0 benchmark set is Ubuntu-only, CPU-only, single image, batch 1, and f16 GGUF. Use the existing 1 warmup and 5 timed runs unless the regenerated artifact explicitly records a reviewed change. Label the platform with the actual runner or host information rather than a guessed generic label. A dated header must say when and where the data was produced.

Remove the current `placeholder`, `seed`, `will be regenerated`, and historical multi-platform headline language from the current-result section. Historical comparisons, if retained, must be clearly separated, dated, and explicitly non-reproducible from this repository. Do not present estimated numbers as measurements. Do not add quantization-specific rows or imply that loading other quantizations constitutes a benchmark result.

`docs/benchmarks.md` must describe how to reproduce the Ubuntu result and how to interpret variance. It must not claim evidence for platforms not run. Benchmark provenance is incomplete if the result cannot be tied to a command, SHA, model checksum, and platform label.

## 9. Model coverage assessment

As part of implementation review, assess backbone-only DINOv2 support as a possible low-effort follow-up. The assessment should identify whether existing model metadata, preprocessing, token outputs, and GGUF loading can support a backbone without introducing a new graph architecture. It should produce a small decision note or issue with effort, missing weights or conversion work, parity needs, and a recommendation.

This assessment is not an acceptance requirement for implementing new task heads. Segmentation, depth, classification heads beyond the current supported path, and other task-specific heads are deferred to separate architecture and scope proposals. DINOv3 is also deferred as a separate architecture and model compatibility effort, regardless of whether backbone-only DINOv2 looks inexpensive.

## 10. Dependencies and model requirements

Runtime changes must use the existing C++20, CMake, ggml, image, and test dependencies. The binary writer must not require a new runtime library. The parity evidence requires the existing Python environment with PyTorch, Transformers, NumPy, and Pillow. JSON output remains optional so that the normal parity workflow does not require a new parser dependency.

Evidence requires two actual small GGUF files, one regular and one with register tokens, and matching Hugging Face reference checkpoints. The exact model ids and checksums belong in the evidence artifacts, not in this design, because this document must not claim runs that have not happened. Tests that need model weights must be explicitly marked integration tests and must fail clearly when fixtures are unavailable. Numeric parser and binary header tests should run without downloading models.

## 11. Review gates

Before implementation is accepted:

1. `--help` and `--version` pass without model files and retain exit 0.
2. Every covered numeric flag rejects all malformed, overflowing, partial, and invalid-range values with stderr and exit 1.
3. Binary mode requires `-o`, handles one and multiple inputs as specified, writes the exact header and payload order, honors normalization, and never writes binary bytes to stdout.
4. Binary tests inspect bytes rather than only round-tripping through the implementation.
5. Parity artifacts contain complete provenance and only report executed models.
6. Benchmark results are regenerated on Ubuntu and contain no placeholder or unsupported-platform claims.
7. The docs match the actual CLI behavior, including preview and non-goal language.
8. Existing tests pass, new tests pass, and `git diff --check` is clean.
9. The diff contains no model downloads, generated binaries, unrelated documentation rewrites, or quantization benchmark changes.

## 12. v0.4.0 readiness criteria

Release v0.4.0 is ready only when all of the following are true:

- The strict numeric parsing contract is implemented and covered by automated tests for every listed flag.
- The preview binary mode is implemented, documented, tested, and visibly labeled as unstable. It is not described as a standard or compatibility promise.
- One passing, fully provenance-linked parity artifact exists for the regular small model and one for the register-token small model. A missing artifact is a release blocker, not permission to generalize from another model.
- `benchmark_results.txt` and `docs/benchmarks.md` contain current, dated Ubuntu-only measurements with reproducibility metadata and no placeholder wording.
- The benchmark data contains no quantization-specific benchmarking.
- Backbone-only DINOv2 coverage has a recorded follow-up assessment, while task heads and DINOv3 remain explicitly deferred.
- All review gates pass, including `git diff --check`, and the release diff does not include work outside this approved scope.

The implementation should land in atomic conventional commits. The design commit itself must contain only this specification and no trailers.
