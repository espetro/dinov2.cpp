# Task 4 report: Ubuntu benchmark evidence

## Status

Completed for the feasible Ubuntu small regular model. The committed result is
an actual successful GitHub Actions measurement, not seed data. The benchmark
workflow and script now record provenance and support explicitly named regular
and register-token f16 variants. No models, binaries, build outputs, release
artifacts, or quantization benchmark data were committed.

## Changes

- `.github/workflows/bench.yml` remains Ubuntu-only and CPU-only. Its input
  mapping now supports `small-registers`, `base-registers`,
  `large-registers`, and `giant-registers` in addition to regular variants.
  `all` intentionally remains the four regular sizes to avoid silently
  expanding runner storage and runtime.
- `scripts/bench.sh` now validates the named variants and writes runner, git
  SHA, precision, backend, batch, command, exact GGUF path, and GGUF SHA-256
  alongside the timing columns.
- `benchmark_results.txt` replaces the placeholder table with the successful
  Ubuntu result below.
- `docs/benchmarks.md` labels current evidence Ubuntu-only, documents
  reproduction and limitations, and separates historical comparison data.

## Actual benchmark evidence

- Workflow: `bench.yml`, run [35630688827](https://github.com/espetro/dinov2.cpp/actions/runs/35630688827)
- Workflow result: success
- Generated: `2026-09-21T17:14:58Z`
- Runner: `ubuntu-latest`, reported as `linux-x86_64`
- Run commit: `9d0a6265ae53088e423e4c2417ec8589de181e99`
- Model: regular small, `dinov2-cpp-core/dinov2-small-gguf`
- GGUF path: `models/dinov2-vit-small-patch14/model.gguf`
- GGUF SHA-256: `b7ca009aa416f6be85ea95363f4ef13d3c139999f96ca3fd47226849afb712e`
- Precision/backend: f16 GGUF, CPU
- Image: `assets/tench.jpg`; batch: 1
- Threads: 12; warmup: 1; timed runs: 5
- Command: `scripts/bench.sh --models small --repeats 5 --threads 12 --image assets/tench.jpg --bin build/bin/dinov2-cli --out benchmark_results.txt`
- Measurement: mean `255.0 ms`, stddev `9.1 ms`, min `249 ms`, max `271 ms`, peak RSS `104 MB`

The result and the downloaded workflow provenance artifact were inspected with:

```text
gh run download 35630688827 --name bench-results-small-35630688827 --dir /tmp/bench-run-35630688827
cat /tmp/bench-run-35630688827/benchmark_results.txt
cat /tmp/bench-run-35630688827/benchmark_provenance.txt
```

## Coverage and limitations

Measured coverage is one regular small f16 GGUF on Ubuntu CPU only. Register-token
variants and base, large, and giant sizes remain unmeasured in the committed
result. `all` is not claimed as measured. macOS, Windows, ARM, GPU, and
quantization-specific comparisons remain out of scope.

The successful artifact was produced at run SHA
`9d0a6265ae53088e423e4c2417ec8589de181e99`, while the current local branch is
`6547bd7bfbe39232cf568215db6262d6178020fa`. The local branch was already ahead
of its published remote branch, and no push was performed. Therefore the
artifact is evidence for the benchmark behavior and model command represented
by that successful workflow run, not a claim that the newly edited workflow
mapping itself has executed. A fresh run of the modified workflow requires
publishing the commit, which was intentionally not done for this task.

## Verification

- `bash -n scripts/bench.sh` and `scripts/bench.sh --help`: pass.
- Static workflow token checks: pass.
- `actionlint .github/workflows/bench.yml`: pass.
- `git diff --check`: pass.
