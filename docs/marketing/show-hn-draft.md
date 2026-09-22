# Show HN draft

Status: draft. Every number below is verified against a repo artifact or a
live HTTP check (sources inline). Do not add numbers that are not in this
file without re-verifying.

## Headline options

1. `Show HN: dinov2.cpp, DINOv2 vision embeddings from a single C++ binary, no Python`
2. `Show HN: Run DINOv2 locally like llama.cpp: one binary, GGUF weights, JSON embeddings`
3. `Show HN: DINOv2 image embeddings in C++ on ggml, with nightly parity checks against the HF reference`

Option 1 is the safest (self-describing). Option 2 trades on the llama.cpp
analogy, which fits the mechanics (ggml, GGUF, single binary) but borrows a
brand. Option 3 leads with the trust story; longer.

## Body (~150 words)

```text
dinov2.cpp runs Meta's DINOv2 vision encoder in pure C++ on ggml. One
binary, GGUF weights, JSON embeddings on stdout. No Python or PyTorch at
runtime.

Numbers, all committed or reproducible: ViT-S f16 forward pass runs at a
255 ms mean with 104 MB peak RSS on a GitHub ubuntu-latest runner
(benchmark_results.txt, workflow run 35630688827). A nightly CI job gates
embeddings against the Hugging Face reference; the committed runs show CLS
cosine 0.9999+ on both small checkpoints (docs/parity/). Prebuilt CLI
archives for macOS arm64, Linux x64/arm64, and Windows x64 are 0.9 to 1.2
MB each, plus a SHA256SUMS.txt. The encoder also compiles to a ~1 MB wasm
module for in-browser similarity demos.

Weights (47 MB ViT-S up to 2.3 GB giant) live on the dinov2-cpp-core HF
profile. Repo: https://github.com/espetro/dinov2.cpp
```

Word count: ~155. Trim "committed or reproducible" phrasing if needed.

## Links to include

- Repo: https://github.com/espetro/dinov2.cpp
- Parity evidence: https://github.com/espetro/dinov2.cpp/tree/main/docs/parity
- Benchmarks: https://github.com/espetro/dinov2.cpp/blob/main/docs/benchmarks.md
- Demo: wasm demo lives in `wasm/index.html`; needs GitHub Pages enabled
  before it can be linked as a live URL (see README checklist)
- Release post with prior benchmarks: https://alexlavaee.me/projects/dinov2cpp/

## Verified numbers used (with sources)

| Claim | Source |
|:------|:-------|
| 255.0 ms mean, 104 MB peak RSS, ViT-S f16 CPU, 12 threads, batch 1 | `benchmark_results.txt`, `docs/benchmarks.md`, run [35630688827](https://github.com/espetro/dinov2.cpp/actions/runs/35630688827) |
| CLS cosine 0.999911 (regular), 0.999984 (registers) vs HF | `docs/parity/2026-09-21-small-regular.md`, `docs/parity/2026-09-21-small-registers.md`, rerun `2026-09-22-audit-hardening.md` |
| Nightly parity gate | `.github/workflows/parity.yml` (cron `17 4 * * *`) |
| Release asset sizes: mac 1,156,752 B; ubuntu x64 945,186 B; ubuntu arm64 895,818 B; win zip 1,120,200 B | `gh release view v0.4.0`, checked 2026-09-23 |
| wasm module ~1 MB | `docs/wasm.md`; local `build-wasm/bin/dinov2-wasm.wasm` = 1,048,008 B |
| GGUF sizes 46.9 MB to 2.29 GB | `curl -sIL` x-linked-size on the 8 `dinov2-cpp-core/*-gguf` repos, checked 2026-09-23 |

## Explicitly not claimed

- The "up to 3x faster than PyTorch" numbers in `docs/benchmarks.md` are a
  historical i9-14900HX measurement, not the committed CI evidence. Leave
  them out of the post or label them as historical.
- No GPU, no quantization comparison, no base/large/giant benchmark rows:
  all unmeasured per `docs/benchmarks.md`.
- No backbone-only GGUFs published yet (classifier checkpoints only).
