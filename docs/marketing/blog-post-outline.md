# Launch blog post outline

Working title options:

- "dinov2.cpp: DINOv2 embeddings without the Python stack"
- "Running DINOv2 like llama.cpp: one binary, GGUF weights, JSON out"

Positioning one-liner (use verbatim or near-verbatim):

> llama.cpp for vision encoders: DINOv2 on ggml, one binary, GGUF weights,
> embeddings as JSON on stdout.

The analogy holds mechanically (ggml backend, GGUF format, single-binary
CLI, prebuilt releases, HF-hosted weights). Do not overextend it into
performance or maturity claims.

## 1. Hook

Draft:

> DINOv2 is the backbone under a lot of vision pipelines: retrieval,
> dedup, clustering, dense features. Running it today means PyTorch,
> transformers, and a few hundred MB of Python environment before you get
> a single vector out. I wanted `model.gguf` plus one binary that prints
> an embedding. dinov2.cpp is that: the DINOv2 encoder ported to C++ on
> ggml, with prebuilt binaries for macOS, Linux, and Windows, and the
> weights already converted on Hugging Face.

## 2. What it is

- C++20 port of DINOv2 (ViT-S/B/L/g, patch 14) on ggml. Vendored stb for
  image I/O, no other runtime dependencies (`ARCHITECTURE.md`).
- One CLI (`dinov2-cli`) plus a C API (`include/dinov2.h`) for embedding.
- Outputs: ImageNet top-k (`-c`), `cls`/`pooled`/patch embeddings as JSON
  (`--print-embeddings`), PCA feature map PNG (`-o`), bench mode
  (`--bench-json`).
- GGUF weights for all 8 classifier checkpoints on
  `dinov2-cpp-core` HF, f16, published monthly by CI
  (`docs/hf-publishing.md`). Real sizes: 46.9 MB (small), 178.7 MB (base),
  616.5 MB (large), 2.29 GB (giant), measured via `x-linked-size` on
  2026-09-23.
- Install story: download a ~1 MB release archive (mac 1,156,752 B,
  ubuntu x64 945,186 B, ubuntu arm64 895,818 B, win 1,120,200 B per
  `gh release view v0.4.0`), `hf download` a weight, run.

## 3. The demo (centerpiece): local photo dedup

`examples/dedup/dedup.py` is the proof that embeddings-as-JSON is a real
interface, not a demo stub.

- Stdlib-only Python (~30 lines of logic): shells out to
  `dinov2-cli --print-embeddings`, takes `pooled`, pairwise cosine,
  single-linkage clustering, writes an HTML review page
  (`examples/dedup/README.md`).
- Suggested narrative: point it at a real photo dump, show the HTML
  clusters, then show the three lines of code that matter (subprocess,
  cosine, union-find).
- Talking point: threshold 0.99 catches re-encodes and resizes, 0.95
  catches same-scene shots. This is embedding-similarity behavior, tunable,
  not magic.
- Honest limit to name in passing: pairwise comparison is O(n^2) in the
  example script; fine for a few thousand images, real indexes (FAISS,
  sqlite-vec) beyond that.
- Secondary demo if Pages is enabled: the wasm build (~1 MB
  `dinov2-wasm.wasm`, 1,048,008 B locally) runs the encoder in-browser and
  shows CLS cosine plus a patch heatmap (`docs/wasm.md`).

## 4. Does it match the reference? (parity evidence)

- Nightly CI job (`.github/workflows/parity.yml`, cron `17 4 * * *`) runs
  `scripts/parity_check.py` against the HF reference for both small
  checkpoints; a drift fails the job.
- Committed runs (`docs/parity/`): regular small CLS cosine 0.999911,
  pooled 0.999928; registers small CLS 0.999984, pooled 0.999984; patch
  mean cosine 0.9996/0.9998. Gates: cls/pooled >= 0.999, patches >= 0.99.
- Classification matches too: same top-1 label ("tench"), probability diff
  0.000039 on the recorded run.
- Frame it as: the numbers are published, the checker is in the repo, the
  gate runs nightly. Not "trust me".

## 5. Performance

- Committed evidence is narrow on purpose: ViT-S f16, CPU, ubuntu-latest,
  12 threads: 255.0 ms mean, stddev 9.1, 104 MB peak RSS
  (`benchmark_results.txt`, `docs/benchmarks.md`).
- Historical i9-14900HX table exists (small 62 ms vs PyTorch 181 ms) but is
  labeled non-reproducible context. Mention it as history, not a claim.
- wasm estimate is explicitly an estimate: ~4-10 s/image for ViT-S at 518
  px, single-threaded (`docs/wasm.md`).

## 6. Caveats (draft below)

## 7. How it works (optional technical section)

- ggml graph, Catmull-Rom resize, position-embedding interpolation for
  arbitrary grids, `--preprocess bounded|hf|crop518`, `--max-tokens` guard
  against quadratic attention memory on huge inputs.
- Batch inference via repeated `-i` and `--batch`; per-image outputs
  identical to single runs.

## 8. CTA

- Try it: `hf download dinov2-cpp-core/dinov2-small-gguf --local-dir models`
  plus a release binary; three commands total.
- File issues on parity gaps, missing formats, quantization results.
- Star/watch if you want the backbone-only checkpoints and Windows ARM
  binaries when they land.

## Caveats section (suggested draft)

> What this is not, and where the edges are:
>
> - The committed benchmark is one row: ViT-S, f16, CPU, Ubuntu. There is
>   no GPU measurement, no quantization comparison, and no base/large/giant
>   row yet. The 3x-vs-PyTorch table in the docs is an older laptop
>   measurement kept as history, not CI evidence.
> - Parity is recorded for the two small checkpoints on one image and one
>   environment. It is strong evidence, not a proof for every checkpoint,
>   image, and platform; the nightly job exists because drift is possible.
> - Feature-mode preprocessing is not bit-identical to HF's
>   `AutoImageProcessor`: the CLI uses Catmull-Rom, HF uses antialiased
>   bicubic. Under `--preprocess hf` the measured residual on the test
>   image is CLS cosine ~0.996. Use the published GGUFs with the default
>   `bounded` recipe unless you specifically need token-for-token HF
>   comparison.
> - `-fa` (flash attention) is documented as faster but less accurate; a
>   K/V padding bias was fixed in the audit pass, but avoid `-fa` when
>   comparing against HF outputs.
> - Image decode is whatever stb_image supports (JPEG, PNG, BMP, TGA, GIF,
>   PSD, HDR, PNM). No WebP, HEIC, or AVIF.
> - The C API is explicitly unstable for 0.4.x; the CLI JSONL schema is the
>   stable surface (`docs/stability.md`). The `.d2e` binary format is a
>   preview with no compatibility promise.
> - Weights are the ImageNet classifier checkpoints only; backbone-only
>   DINOv2 GGUFs are supported by the code but not published yet.

## Distribution notes

- Cross-post targets: personal blog, Show HN (see show-hn-draft.md), maybe
  r/LocalLLaMA (fits the local-inference audience).
- The parity and benchmarks pages are the credibility backbone; link them
  early, not just at the end.
