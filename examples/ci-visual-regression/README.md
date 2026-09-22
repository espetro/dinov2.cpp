# CI visual regression (example)

A copy-paste GitHub Action that uses release `dinov2-cli` builds as a visual
regression gate: screenshots from your PR are compared against committed
baselines by embedding cosine similarity instead of pixel diffs.

**This is an example, not a gate on this repo.** It is not referenced by any
workflow in `.github/workflows/`; copy `visual-regression.yml` into your own
project and adapt it.

## What it does

1. Resolves the latest dinov2.cpp release tag, downloads the Ubuntu x64
   tarball and `SHA256SUMS.txt`, verifies the archive checksum.
2. Downloads the `dinov2-small` GGUF straight from Hugging Face
   (`/resolve/` URL, no `hf` CLI needed).
3. Embeds every `baseline/*.png|jpg` and `screenshots/*.png|jpg` with
   `--preprocess crop518 --batch 8 --print-embeddings`. `crop518` fixes all
   inputs at 518x518 so grids always match; batching keeps it fast.
4. Compares `pooled` vectors pairwise by basename and fails when any pair
   drops below `THRESHOLD` (default `0.98`).

## Why not pixel diffs

Font rendering, anti-aliasing and GPU rasterization jitter make pixel-exact
screenshot gates flaky. DINOv2 pooled features tolerate subpixel drift while
still firing on real layout and content changes. Tune `THRESHOLD` down
(e.g. `0.95`) if legit UI changes trip the gate.

## Assumed repo layout

```
your-repo/
  baseline/       committed reference screenshots
  screenshots/    written by your test harness on each run
  .github/workflows/visual-regression.yml
```

Filenames must match between the two directories; the comparison joins on
`basename(image)`. New screenshots are reported but do not fail the build.
