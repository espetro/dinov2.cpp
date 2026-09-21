# Task 4 report: Ubuntu benchmark evidence

## Status

Blocked. No trustworthy benchmark result or artifact was produced, so the placeholder evidence was not replaced and no benchmark documentation claims were changed.

## Repository and workflow provenance

- Local branch: `feat/batched-inference`
- Local HEAD: `6547bd7bfbe39232cf568215db6262d6178020fa`
- Published `origin/feat/batched-inference`: `12f15a521a259f6273d8662bcf4ef8a465dbc784`
- The local branch is seven commits ahead of its published remote branch. No push was performed.
- The dispatched workflow therefore benchmarked the published remote SHA `12f15a521a259f6273d8662bcf4ef8a465dbc784`, not the current local code. This run cannot be used as evidence for the unpushed local changes.
- Workflow: `.github/workflows/bench.yml`, `ubuntu-latest`, manual dispatch.

## Commands and run

Authentication was available:

```text
gh auth status
  Logged in to github.com account espetro
  Token scopes: gist, project, read:org, repo, workflow
```

The smallest useful run was dispatched first:

```text
gh workflow run bench.yml --ref feat/batched-inference -f variant=small -f repeats=1 -f threads=2
```

Run: [35628865190](https://github.com/espetro/dinov2.cpp/actions/runs/35628865190)

The run was monitored with a bounded `gh run watch 35628865190 --interval 10 --exit-status`. It failed after 51 seconds during `Download f16 GGUF(s)` before benchmark execution or artifact upload.

## Exact blocker

The workflow's download step ran:

```text
huggingface-cli download --repo-type model --local-dir models/dinov2-vit-small-patch14 dinov2-cpp-core/dinov2-small-gguf model.gguf
```

The Ubuntu log shows:

```text
Warning: `huggingface-cli` is deprecated and no longer works. Use `hf` instead.
Hint: `hf` is already installed! Use it directly.
Process completed with exit code 1.
```

The command exited before `Verify staged weights`, `Run benchmarks`, and `Upload benchmark results`. `gh run view 35628865190 --log-failed` contains no downloaded model path, benchmark output, or artifact. No artifact was downloaded because none was uploaded.

## Results and coverage

- Actual benchmark measurements: none.
- Models measured: none; the requested small f16 GGUF could not be staged.
- Register-token variants: unmeasured.
- Precision/backend: no measurement; the workflow is intended for f16 GGUF on the CPU backend.
- Warmup/repeats/threads: the requested run was 1 repeat, 2 threads, with the script default of 1 warmup, but execution never reached `scripts/bench.sh`.
- No local Mac benchmark was substituted for Ubuntu evidence.
- No full or giant sweep was attempted.

## Limitations and follow-up

The workflow must replace the deprecated `huggingface-cli` invocation with the installed `hf download` command, or otherwise pin a working Hugging Face CLI invocation, before another Ubuntu evidence run can proceed. After that fix is published, dispatch at least the small model again and verify the artifact and provenance before replacing placeholder data. The local seven-commit divergence must also be resolved by publishing the code to be measured; this task did not push it.

No models, build outputs, or release artifacts were added to the repository.

## Fix note

The benchmark workflow now uses the supported `hf download` command from the already-installed `huggingface_hub[cli]` package. The repository, local directory, and `model.gguf` arguments are unchanged; no benchmark semantics or inputs were changed.

## Complete rerun result: 2026-09-21

### Status

Blocked. The corrected workflow reached the benchmark binary on Ubuntu, and the binary emitted valid JSON metrics, but `scripts/bench.sh` failed to parse that JSON. No artifact was uploaded. Placeholder benchmark data remains unchanged, and no local Mac timings were substituted.

### Provenance

- Branch pushed: `feat/batched-inference`
- Pushed SHA: `b0359a589187b89139f75b1eca8e16966ba33530`
- Workflow: `.github/workflows/bench.yml`
- Runner: `ubuntu-latest`
- Dispatch command: `gh workflow run bench.yml --ref feat/batched-inference -f variant=small -f repeats=5 -f threads=2`
- Run: [35629370840](https://github.com/espetro/dinov2.cpp/actions/runs/35629370840)
- Run head SHA: `b0359a589187b89139f75b1eca8e16966ba33530`
- Inputs: `variant=small`, `repeats=5`, `threads=2`
- Result: failed after approximately 58 seconds in `Run benchmarks`
- Artifact: none. `Upload benchmark results` was skipped.

### Exact failure log

From `gh run view 35629370840 --log-failed`:

```text
models selected: small
repeats: 5
threads: 2
bench.sh: running dinov2-vit-small-patch14 repeats=5 threads=2...
::error::could not parse bench output for dinov2-vit-small-patch14
{"model":"dinov2-vit-small-patch14","n_threads":2,"n_repeats":5,"n_warmup":1,"samples_ms":[207,214,209,206,207],"mean_ms":208.6,"stddev_ms":3.2,"min_ms":206,"max_ms":214,"peak_rss_mb":104,"n_images":1,"batch":1,"ms_per_image":208.6,"images_per_sec":4.8}
##[error]Process completed with exit code 1.
```

The workflow successfully completed checkout, tool installation, configuration, build, variant resolution, HF download, and staged-weight verification. It failed after the binary produced the JSON shown above, before writing/uploading `benchmark_results.txt`.

### Measurement assessment

The JSON contains an Ubuntu small-model measurement candidate: mean `208.6 ms`, standard deviation `3.2 ms`, minimum `206 ms`, maximum `214 ms`, peak RSS `104 MB`, `5` timed samples, `1` warmup, `2` threads, batch `1`, and `4.8` images/sec. It is not committed as benchmark evidence because the workflow did not complete successfully and did not produce its required artifact. The exact parser blocker must be fixed and the run rerun before evidence is ready for review.

Multi-platform evidence remains deferred. No giant or all-model sweep was attempted.

### Repository outcome

- `benchmark_results.txt`: unchanged placeholder content.
- `docs/benchmarks.md`: unchanged placeholder documentation.
- This report: appended with the complete rerun provenance and failure output.
- Evidence ready for review: **No**, blocked on the `scripts/bench.sh` JSON parsing failure.

### Parser root cause confirmation

The parser is implemented at `scripts/bench.sh:163-169` as a Python f-string using escaped double quotes inside the f-string expression, for example `f"{data[\"mean_ms\"]:.1f}"`. Reproducing that exact expression with the run's JSON values locally produces:

```text
SyntaxError: unexpected character after line continuation character
```

Because stderr from the parser is redirected to `/dev/null`, the workflow surfaces only the generic `could not parse bench output` message and the valid JSON object. This identifies a parser implementation blocker, not a missing model or invalid benchmark output.

## Parser fix: 2026-09-21

Replaced the incompatible f-string expression in `scripts/bench.sh` with `str.format()` using the same five JSON fields and numeric format specifiers. This removes the escaped-quote syntax error without changing benchmark semantics or output format. Added `scripts/test-bench-parser.sh`, which validates the recorded JSON shape and confirms malformed JSON and missing required fields still fail.

## Complete rerun result: 2026-09-21

### Status

Completed. The Ubuntu benchmark workflow succeeded, and the required artifact was downloaded and inspected. Placeholder benchmark evidence was replaced with the artifact's actual Ubuntu data. Multi-platform coverage remains deferred. No quantization-specific benchmark was run.

### Provenance

- Branch: `feat/batched-inference`
- Branch SHA: `50d5dee10da4db7b7820082da7a3ea0eaab5f346`
- Workflow: `.github/workflows/bench.yml`
- Runner: `ubuntu-latest`, platform reported by artifact as `linux-x86_64`
- Dispatch command: `gh workflow run bench.yml --ref feat/batched-inference -f variant=small -f repeats=5 -f threads=2`
- Run: [35629935462](https://github.com/espetro/dinov2.cpp/actions/runs/35629935462)
- Run head SHA: `50d5dee10da4db7b7820082da7a3ea0eaab5f346`
- Artifact: `bench-results-small-35629935462/benchmark_results.txt`
- Artifact generated: `2026-09-21T17:08:06Z`

### Actual measurement

The artifact reported the small f16 GGUF on the CPU backend with one warmup, five timed runs, one 224x224 image, batch 1, and two OpenMP threads:

- Mean: `212.8 ms`
- Standard deviation: `9.1 ms`
- Minimum: `204 ms`
- Maximum: `225 ms`
- Peak RSS: `104 MB`

### Repository outcome

- `benchmark_results.txt`: replaced with the inspected artifact data plus explicit run and commit provenance.
- `docs/benchmarks.md`: updated to report only the actual Ubuntu small-model measurement and deferred multi-platform coverage.
- No models, build outputs, or release artifacts were added to the repository.
