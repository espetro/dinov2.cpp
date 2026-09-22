# Trust and verification

This page collects the evidence behind the project's claims, and the known
limits of that evidence. Everything linked here is in the repo or produced
by CI; you can rerun all of it yourself.

## Parity with the Hugging Face reference

**Nightly gate.** `.github/workflows/parity.yml` runs
`scripts/parity_check.py` every night (cron `17 4 * * *`) on
`ubuntu-latest` for both published small checkpoints, comparing
`dinov2-cli` output against the live HF reference models. Gates: CLS cosine
>= 0.999, pooled cosine >= 0.999, flat and mean patch cosine >= 0.99,
matching top-1 classification, top-1 probability difference < 0.05. Any
thresholded drift fails the job.

**Committed run reports** in [parity/](parity/):

| Checkpoint | CLS cosine | Pooled cosine | Patch mean cosine | Top-1 |
|:-----------|-----------:|--------------:|------------------:|:------|
| dinov2-small (regular) | 0.999911 | 0.999928 | 0.999589 | match, prob diff 0.000039 |
| dinov2-with-registers-small | 0.999984 | 0.999984 | 0.999836 | match, prob diff 0.006114 |

Sources: [2026-09-21 regular](parity/2026-09-21-small-regular.md),
[2026-09-21 registers](parity/2026-09-21-small-registers.md), and the
[2026-09-22 audit-hardening rerun](parity/2026-09-22-audit-hardening.md)
which confirmed the numbers were stable across the flash-attention and
position-embedding fixes.

**Scope of that evidence.** These runs cover two checkpoints, one image,
one environment (the recorded runs were on Apple M1/Metal; the nightly gate
runs on Ubuntu CPU). They are strong, checkable evidence, not a proof that
every checkpoint, image, and platform matches. Reproduction commands are in
[parity/README.md](parity/README.md).

## CI safety nets

- **ASan + UBSan**: the `sanitize` job in
  `.github/workflows/build.yml` builds the whole tree with
  `-fsanitize=address,undefined` and runs `ctest` under it on every PR.
- **JSONL contract**: the `contract` job in `build.yml` pipes real
  `--print-embeddings --print-patch-tokens` output through
  `scripts/check_jsonl_contract.py`, which verifies required keys,
  `index == line order`, and `grid.h * grid.w == n_patches`. The schema is
  documented in [cli.md](cli.md#embeddings-json-schema) and covered by the
  semver policy in [stability.md](stability.md).
- **Strict numeric parsing**: CLI integer flags go through a checked
  `std::from_chars` conversion that rejects empty values, trailing
  characters, and out-of-range inputs (`dinov2-cli.cpp`, `parse_integer`).
  No silent `atoi` truncation.
- **Converter tests**: `tests/test_dinov2_converter.py` runs in the same
  contract job.

## Release integrity

Every release ships a `SHA256SUMS.txt` generated over the binary archives
in `.github/workflows/release.yml` (`sha256sum * > SHA256SUMS.txt`). Verify
a download with `sha256sum -c` against that file.

## What is covered, and what is not

The tier policy ([tiers.md](tiers.md)) and stability contract
([stability.md](stability.md)) define which surfaces are frozen. In short:
the CLI flags, the JSONL schema, `libdinov2`, and the GGUF publishing
layout are tier-1 and versioned; the C API is explicitly unstable for the
0.4.x line; wasm, `dinov2-server`, `examples/`, and the `.d2e` binary
format are tier-2 previews with no compatibility promise.

## Known residual limitations

- **Preprocess kernel delta**: HF `AutoImageProcessor` uses antialiased
  PIL bicubic; the CLI uses Catmull-Rom (`INTER_CUBIC`) without a downscale
  prefilter. Under `--preprocess hf` on `assets/tench.jpg` the measured
  residual is CLS cosine 0.996245, pooled 0.996995
  ([parity/README.md](parity/README.md#preprocessing-note)). The default
  `bounded` recipe is the recommended mode; `hf` exists for token-for-token
  comparisons.
- **Flash attention**: `-fa` is documented as faster but less accurate
  ([cli.md](cli.md)). An audit found and fixed a real bias (unmasked
  zero-padding of the K/V sequence dimension); the parity rerun confirmed
  no regression on the default path. Still, avoid `-fa` when comparing
  against HF outputs.
- **Image formats**: decode is whatever vendored stb_image supports:
  JPEG, PNG, BMP, TGA, GIF, PSD, HDR, PNM. There is no WebP, HEIC, or AVIF
  decode; convert those first.
- **Benchmark scope**: the committed benchmark is one row (ViT-S, f16, CPU,
  Ubuntu, 255.0 ms mean, 104 MB peak RSS; see
  [benchmarks.md](benchmarks.md)). Register variants, other sizes, GPU
  backends, and quantization are unmeasured. The multi-size comparison
  table in that file is labeled historical and is not CI evidence.
- **Published weights**: only the eight `imagenet1k-1-layer` classifier
  checkpoints ship on `dinov2-cpp-core`. Backbone-only checkpoints work in
  feature modes when converted locally but are not published yet
  ([cli.md](cli.md#backbone-only-checkpoints)).
- **wasm**: single-threaded, roughly 4-10 s per ViT-S image, explicitly an
  estimate ([wasm.md](wasm.md)).

## Reporting

Numerical drift, crash reports, and contract violations go through GitHub
issues. Include the GGUF SHA-256, `--version` output, the exact command,
and the input image characteristics; the parity script prints everything
needed.
