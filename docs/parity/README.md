# Parity evidence

These artifacts record actual runs of `scripts/parity_check.py` against the matching Hugging Face reference and GGUF files. They are CPU or Metal and environment specific measurements, not universal guarantees.

## Download the recorded GGUFs

From the repository root, download both public Hugging Face GGUF artifacts into the paths used by the recorded commands:

```sh
hf download dinov2-cpp-core/dinov2-small-gguf model.gguf \
  --local-dir models/dinov2-small

hf download dinov2-cpp-core/dinov2-with-registers-small-gguf model.gguf \
  --local-dir models/dinov2-with-registers-small
```

The model weights are gitignored (`/models` and `*.gguf`) and are not committed.

## Reproduce

From the repository root, use the existing Python environment and a built CLI. The commands below retain the default parity gates explicitly:

```sh
.venv/bin/python scripts/parity_check.py \
  --cli build-debug/bin/dinov2-cli \
  --gguf models/dinov2-small/model.gguf \
  --hf-model facebook/dinov2-small-imagenet1k-1-layer \
  --image assets/tench.jpg \
  --cls-threshold 0.999 --pooled-threshold 0.999 \
  --patches-threshold 0.99 --patches-token-min-threshold 0 \
  --prob-tolerance 0.05

.venv/bin/python scripts/parity_check.py \
  --cli build-debug/bin/dinov2-cli \
  --gguf models/dinov2-with-registers-small/model.gguf \
  --hf-model facebook/dinov2-with-registers-small-imagenet1k-1-layer \
  --image assets/tench.jpg \
  --cls-threshold 0.999 --pooled-threshold 0.999 \
  --patches-threshold 0.99 --patches-token-min-threshold 0 \
  --prob-tolerance 0.05
```

The patch-token minimum is informational here because its threshold is explicitly `0`. A run exits nonzero when a thresholded check fails or a required model, image, or reference dependency cannot be loaded. Model weights are not committed.

## Preprocessing note

Since 0.4.0 the CLI's default feature preprocessing (`--preprocess bounded`)
bounds the shortest edge to 518 and aligns dims by true ceil to patch
multiples. The recorded parity inputs are unaffected: `assets/tench.jpg` is
612x408, its shortest edge is under the bound, and neither dim is patch
aligned, so its grid is unchanged (44x30 = 1320 patches).

HF `AutoImageProcessor` for `facebook/dinov2-*` resizes to shortest edge 256
and center-crops 224x224 (256 tokens at patch 14), which does not match the
CLI's grid. For token-count parity use `--preprocess hf` (same recipe). Even
then a small residual delta is expected when the input is downscaled: HF's
`resample: 3` is PIL bicubic with a support-scaled (antialiased) kernel,
while the CLI uses Catmull-Rom (OpenCV `INTER_CUBIC`) without a downscale
prefilter.

## Runs

* [Regular small classifier, 2026-09-21](2026-09-21-small-regular.md)
* [Register small classifier, 2026-09-21](2026-09-21-small-registers.md)
