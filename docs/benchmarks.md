# Benchmarks

The table below contains the actual Ubuntu benchmark artifact produced by
[`scripts/bench.sh`](../scripts/bench.sh) in GitHub Actions. Multi-platform
measurements are deferred.

The benchmark ran on `.github/workflows/bench.yml` run
[35630688827](https://github.com/espetro/dinov2.cpp/actions/runs/35630688827)
from `feat/batched-inference` at commit
`9d0a6265ae53088e423e4c2417ec8589de181e99`. The workflow inputs were
`variant=small`, `repeats=5`, and `threads=2`; the uploaded artifact was
`bench-results-small-35630688827/benchmark_results.txt`, accompanied by
`benchmark_provenance.txt` in the same artifact. The staged model was
`models/dinov2-vit-small-patch14/model.gguf` with SHA-256
`b7ca009aa416f6be85ea95363f4ef13d3c139999f96ca3fd47226849afb712e6`. The
artifact recorded `workflow=bench.yml`, `run_id=35630688827`,
`run_attempt=1`, and the same commit SHA.

## Forward pass wall time (CPU backend)

Forward pass only -- the timed block excludes image load, preprocess,
and PCA visualization. Numbers in milliseconds (lower is better).

| Platform        | Arch   | ggml | Model  | mean (ms) | stddev (ms) | min (ms) | max (ms) | Peak RSS |
|-----------------|--------|------|--------|----------:|------------:|---------:|---------:|---------:|
| ubuntu-latest   | x86_64 | cpu  | ViT-S  |     255.0 |         9.1 |      249 |      271 |   104 MB |

This is the only current platform measurement. Other platform rows remain
deferred, and no quantization-specific benchmark was run.

## How to read the tables

Three different "times" are easy to confuse; here's what each column means:

- **mean (ms)** -- forward-pass wall time of the model graph only,
  measured by `dinov2-cli.cpp` between `ggml_time_ms()` calls bracketing
  `dino_predict()`. This is the number to compare across backends and
  against other inference engines.
- **end-to-end wall (not shown)** -- image load + preprocess (image is
  resized and mean-normalized) + forward pass + (optional) PCA + JSON
  serialization. End-to-end is always larger than the forward-pass
  number, typically by 10-30 ms on the small model.
- **peak RSS** -- resident set size peak, sampled via
  `getrusage(RUSAGE_SELF).ru_maxrss` between bench iterations. The model
  weights dominate; RSS grows roughly linearly with model size
  (B ~3x, L ~10x, g ~40x of ViT-S).

`stddev` and `min`/`max` columns in the regenerated markdown table are
produced from `bench_repeats` runs each. A stddev > 5% of the mean
usually means something other than CPU (a noisy shared CI runner, a
cold CPU cache) is in the loop.

## Methodology

- **Image:** `assets/tench.jpg` (a single 224x224 RGB image after
  `dino_preprocess`). Larger images scale forward-pass time roughly
  linearly with the patch grid.
- **Warmup:** 1 discarded run (`--bench-warmup 1`).
- **Timed runs:** 5 (`--bench-runs 5`).
- **Threads:** 2 OpenMP threads, supplied by the workflow input.
- **Batch:** 1 (`--batch` exists for multi-image runs; this table benches
  one image per pass).
- **Backend:** CPU only.
- **Precision:** f16 GGUF.
- **Build:** `cmake --build build --target dinov2-cli` in
  `Release` configuration (no `DEBUG`/`RelWithDebInfo`).

The committed `benchmark_results.txt` is the downloaded artifact from the
Ubuntu workflow run identified above. Additional platform measurements are
deferred.

## Hardware

The evidence currently covers one GitHub-hosted `ubuntu-latest` runner. The
runner's exact hardware may drift without notice; the artifact records the
platform as `linux-x86_64`.

## Caveats

- **Batching exists but the tables run batch=1.** `dino_predict` runs up
  to `--batch` same-size images per forward pass and the CLI chunks
  longer input lists automatically; per-image outputs are identical to
  batch=1. Per-image throughput improves over batch=1 because the
  matmul-heavy layers amortize better across the batch dim. Measure it
  with:

  ```bash
  dinov2-cli -m models/model.gguf -i a.jpg -i b.jpg --batch 2 \
      --bench --bench-runs 5 --bench-json
  ```

  The bench JSON reports `n_images`, `batch`, `ms_per_image` and
  `images_per_sec` alongside the existing fields (`mean_ms` covers one
  full pass over all inputs). See [cli.md](cli.md#batch-inference) for
  the chunking and same-size-per-chunk rules.
- **No KV cache.** DINOv2 is a vision encoder with no autoregressive
  decode step -- there is no KV cache to populate. The "no KV cache"
  caveat in some other engines doesn't apply here.
- **Multi-platform coverage is deferred.** This evidence contains only the
  Ubuntu CPU run described above. Metal and CUDA measurements are not included.
- **CI runner noise.** The `stddev` column is sensitive to load on the shared
  runner; the recorded run measured 9.1 ms of standard deviation.
- **Embeddings flags are single-shot only.** `--print-embeddings` and
  `--print-patch-tokens` exist for output inspection; the bench path
  ignores them and reports timings only.

## Reproduce locally

The committed `benchmark_results.txt` records the Ubuntu CI artifact;
you can reproduce the measurement locally with:

```bash
# Build the dinov2-cli binary first.
cmake --preset release && cmake --build --preset release

# Pre-stage the GGUFs you want to bench under models/<full_name>/model.gguf
# (download from the dinov2-cpp-core/*-gguf repos, or run scripts/publish-gguf.sh).
mkdir -p models/dinov2-vit-small-patch14
cp ~/Downloads/dinov2-vit-small-patch14/model.gguf models/dinov2-vit-small-patch14/

# Single model, f16, 5 timed runs.
scripts/bench.sh --models small --repeats 5

# Full sweep: 4 models, markdown to ./benchmark_results.md.
scripts/bench.sh

# Aggregate per-platform files into one.
scripts/bench.sh --aggregate ./per-platform/*.txt --out benchmark_results.txt
```

For JSON output, pass `--bench-json` to the binary directly:

```bash
./build/bin/dinov2-cli -m models/dinov2-vit-base-patch14/model.gguf \
    -i assets/tench.jpg -t 4 -c \
    --bench --bench-runs 5 --bench-warmup 1 --bench-json
```

## Register and no-register variants

Both variants appear in benchmark tables because they are separate published
checkpoints with different representations and legitimate use cases. Register
tokens are especially relevant to patch and dense feature inspection: the
register-token paper reports fewer high-norm patch-token artifacts and smoother
local feature and attention maps. No-register variants remain necessary for
exact baseline reproduction and fair comparisons with systems or published
results built from those checkpoints.

Runtime rows should stay comparable by model size, precision, backend, input,
and measurement method. Keep register and no-register timings in explicitly
labeled rows or headings rather than combining them. Existing speed and memory
tables are implementation measurements and do not establish downstream
representation quality.

The strongest consistent rationale for registers is dense-feature quality and
local feature-map behavior. Classification and retrieval results in the official
[DINOv2 results](https://github.com/facebookresearch/dinov2/blob/main/README.md)
are mixed by task and model size, so they do not support a universal ranking.
The primary [Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588)
paper provides the motivation and reports feature-map effects and selected
downstream evaluations. Preserve the model variant, task, dataset, evaluation
protocol, and model size when interpreting those results.

## DINOv2 vs PyTorch (historical)

This is the qualitative comparison that motivated the project. Numbers
below were measured on an Intel Core i9-14900HX (24 cores, 32 threads)
with 24 threads, 100-run averages, and are kept here for historical
context only -- they are not reproducible from the current repo.

### With register tokens

Models: `dinov2-with-registers-{size}-imagenet1k-1-layer`

| Model | Max Mem (PyTorch) | Max Mem (dinov2.cpp) | Speed (PyTorch) | Speed (dinov2.cpp) |
|:-----:|:-----------------:|:--------------------:|:---------------:|:------------------:|
| small | ~457 MB           | **~109 MB**          | 297 ms          | **64 ms**          |
| base  | ~720 MB           | **~367 MB**          | 436 ms          | **200 ms**         |
| large | ~1.57 GB          | **~1.2 GB**          | 1331 ms         | **597 ms**         |
| giant | ~4.8 GB           | **~4.4 GB**          | 4472 ms         | **1995 ms**        |

### Without register tokens

Models: `dinov2-{size}-imagenet1k-1-layer`

| Model | Max Mem (PyTorch) | Max Mem (dinov2.cpp) | Speed (PyTorch) | Speed (dinov2.cpp) |
|:-----:|:-----------------:|:--------------------:|:---------------:|:------------------:|
| small | ~455 MB           | **~110 MB**          | 181 ms          | **62 ms**          |
| base  | ~720 MB           | **~367 MB**          | 462 ms          | **197 ms**         |
| large | ~1.55 GB          | **~1.2 GB**          | 1288 ms         | **600 ms**         |
| giant | ~4.8 GB           | **~4.4 GB**          | 4384 ms         | **1969 ms**        |

## Run your own

Bench script entry point: [`scripts/bench.sh`](../scripts/bench.sh).
Build instructions: [build.md](build.md).

## Quantization

Quantization is not produced by this repo. Pull pre-quantized GGUFs
(q4_0, q4_1, q5_0, q5_1, q8_0) directly from
[`dinov2-cpp-core/<variant>-gguf`](https://huggingface.co/dinov2-cpp-core).
The `dinov2-cli` binary loads any ggml-supported quant type transparently
— pass any of those GGUFs as `-m`.