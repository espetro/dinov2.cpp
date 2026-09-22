# dinov2-cli reference

`dinov2-cli` runs DINOv2 inference on a GGUF weight: classification,
embeddings as JSON, PCA visualization of patch features, and benchmarking.
All results go to stdout and all logs go to stderr, so stdout is safe to
pipe into `jq` or a JSON parser.

## Install and model download

Use a prebuilt binary from
[releases](https://github.com/espetro/dinov2.cpp/releases), or build from
source:

```bash
cmake --preset release && cmake --build --preset release
# binary lands at ./build-release/bin/dinov2-cli
```

Download a GGUF weight from the
[`dinov2-cpp-core`](https://huggingface.co/dinov2-cpp-core) Hugging Face
profile:

```bash
hf download dinov2-cpp-core/dinov2-small-gguf --local-dir models
# file on disk: models/model.gguf
```

The binary loads any ggml-supported quantization transparently (f16,
q4_0 through q8_0): point `-m` at whichever GGUF you have.

### Choosing a checkpoint

For patch and dense feature work, start with the register-token small model:

```bash
hf download dinov2-cpp-core/dinov2-with-registers-small-gguf --local-dir models
```

The same `dinov2-with-registers-{size}-gguf` naming pattern is available for
`base`, `large`, and `giant`. Use the matching no-register repository when you
need an exact baseline checkpoint or a task-specific classification or retrieval
comparison. The register-token recommendation is based on the feature behavior
reported in [Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588)
and should not be read as a universal accuracy ranking. For classification and
retrieval context, see the official [DINOv2 results](https://github.com/facebookresearch/dinov2/blob/main/README.md).

## Flags

Adapted from `dinov2-cli --help`; run `--help` on your build for the exact
wording. Flags that take a value read it from the next argument.

| Flag | Default | Effect |
|:-----|:--------|:-------|
| `-m FNAME`, `--model` | `../model.gguf` | GGUF model path |
| `-fa`, `--flash_attn` | off | enable flash attention, less accurate |
| `-t N`, `--threads` | `min(4, hardware_concurrency)` | threads used during computation |
| `-i FNAME`, `--inp` | `../assets/tench.jpg` | input image file; repeat or comma-separate for several |
| `-s N`, `--seed` | 42 | accepted for compatibility; has no effect |
| `--batch N` | 1 | max images per forward pass (max 64); inputs run in chunks of N |
| `--preprocess MODE` | `bounded` | feature-mode preprocessing: `bounded` (shortest edge capped at 518), `hf` (shortest edge 256 + center crop 224), `crop518` (shortest edge 518 + center crop 518, fixed grid) |
| `--no-resize` | off | keep native resolution under `--preprocess bounded` (still capped by `--max-tokens`) |
| `--max-tokens N` | `4*(518/patch)^2` | hard cap on patch tokens per image after preprocessing; 0 disables |
| `-c`, `--classify` | off | classify each input image and print top-k labels |
| `-k N`, `--topk` | 5 | number of classes printed with `-c`; must not exceed the model's class count |
| `--print-embeddings` | off | emit one JSON object for one input; one JSON object per line (JSONL) for multiple inputs |
| `--embeddings-binary` | off | write preview binary embeddings to `-o`; no embedding bytes go to stdout |
| `--print-patch-tokens` | off | add per-patch token vectors to the embedding output |
| `--l2-normalize` | off | L2-normalize emitted embedding vectors |
| `-o FNAME`, `--out` | off | write PCA output, or the binary file/directory selected by `--embeddings-binary` |
| `--bench` | off | timed bench loop (5 repeats, 1 warmup) |
| `--bench-runs N` | 5 | timed runs when `--bench` is set |
| `--bench-warmup N` | 1 | warmup runs discarded before timing |
| `--bench-json` | off | bench result as JSON on stdout |
| `-h`, `--help` | | print usage and exit |
| `--version` | | print `dinov2-cli 0.4.0` and exit |

## Output modes

Select at least one output mode per invocation. With none selected the
CLI prints a hint and exits 1.

- **Classification** (`-c`): prints one ` > label : prob` line per top-k
  class on stdout. With multiple inputs each block is headed by a
  `<path>:` line naming its image. Combined with `--print-embeddings`,
  the same results land in the JSON `topk` array and the text lines are
  not printed.
- **Embeddings JSON** (`--print-embeddings`): one JSON object per input
  image on stdout, in input order (JSONL with several `-i`s; a single
  object for one image). The `--bench` loop ignores it and emits its
  own bench output instead. `--print-patch-tokens` adds a `patches`
  field; `--l2-normalize` normalizes the vectors.
- **PCA visualization** (`-o FNAME`): writes a PNG of the patch features
  projected to RGB. Feature mode only; combining it with `-c` is a
  parse-time error, and the `--bench` loop skips it.
  With a single input FNAME is the exact output path. With multiple
  inputs FNAME is a directory (created if missing) and each image
  writes `<index>-<input-stem>.pca.png` inside it, so inputs that share
  a filename stem cannot collide. The writer emits PNG bytes regardless
  of the file extension.
- **Bench** (`--bench*`): warmup runs plus timed runs of the forward
  pass; each run processes every input image in chunks of `--batch`.
  `--bench-json` prints one JSON object on stdout; without it a
  `bench(model=..., ...)` summary line goes to stderr. The bench loop
  skips PCA output and ignores `--print-embeddings`.

Benchmark mode is explicit and takes precedence over ordinary inference
outputs, but it cannot be combined with `--embeddings-binary`; that conflict
is rejected before model loading. Binary mode likewise cannot be combined with
`-c`.

Modes combine. `-c --print-embeddings` produces one object with `cls`
and `topk`. `--print-embeddings -o pca.png` emits the JSON and writes
the PNG in the same run. `--print-patch-tokens` adds patch vectors to JSON
or binary output; on its own it selects no output mode and the run fails the
nothing-to-do guard.

For patch-token inspection, PCA maps, dense features, or object discovery,
prefer a `with-registers` checkpoint. In the settings studied in the
register-token paper, registers reduce high-norm patch-token artifacts and
smooth local feature and attention maps. This is a feature-quality and
visual-behavior recommendation, not a universal classification or retrieval ranking.
Use the matching no-register checkpoint for exact baseline reproduction or a
task-specific comparison.

### Backbone-only checkpoints

Backbone-only DINOv2 checkpoints such as `facebook/dinov2-small`, `base`,
`large`, and `giant` are supported in feature modes when converted to the
repository GGUF layout. They expose CLS, pooled, patch-token, and PCA outputs,
but they do not contain the ImageNet classifier head. Therefore `-c` fails
cleanly with a message that classification requires a classifier GGUF rather
than attempting to access missing tensors. A missing `num_register_tokens`
metadata key is treated as zero registers for feature mode.

The publishing workflow converts the backbone checkpoints alongside the eight
`imagenet1k-1-layer` classifier variants and ships them as
`dinov2-cpp-core/dinov2-backbone-{size}-gguf` (from `facebook/dinov2-{size}`)
and `dinov2-cpp-core/dinov2-backbone-with-registers-{size}-gguf` (from
`facebook/dinov2-with-registers-{size}`), where `{size}` is `small`, `base`,
`large`, or `giant`:

```bash
hf download dinov2-cpp-core/dinov2-backbone-small-gguf --local-dir models
dinov2-cli -m models/model.gguf -i assets/tench.jpg --print-embeddings
```

Use a backbone weight when you want the feature extractor without the
classifier head; for `-c` pick one of the classifier repos above.

DINOv2 task heads other than the existing ImageNet classifier, including depth
and segmentation, are outside this interface. DINOv3 is a separate
architecture and resource target and is also out of scope.

## Feature-mode preprocessing

Feature mode (everything except `-c`) selects one of three recipes with
`--preprocess`:

- `bounded` (default): if the image's shortest edge exceeds 518 px it is
  bicubic-resized to 518 preserving aspect ratio, then each dimension is
  aligned up to a multiple of the patch size. Smaller images keep their
  size, aligned the same way. Use it for PCA visualization and dense
  features: it preserves the whole image without unbounded memory.
- `hf`: the Hugging Face `AutoImageProcessor` recipe for `facebook/dinov2-*`
  checkpoints (shortest edge 256, center crop 224x224, ImageNet mean/std),
  producing a fixed 16x16 = 256-token grid at patch 14. Use it when
  comparing outputs against HF feature pipelines token for token.
- `crop518`: shortest edge 518 plus a 518x518 center crop, producing a
  fixed 37x37 = 1369-token grid at patch 14. Every input shares dims, so
  this is the batch-safe mode for mixed aspect ratios.

`--no-resize` disables the 518 bound under `bounded` (native resolution,
aligned to patch multiples); it is rejected with `hf`/`crop518` and with
`-c`, as is `--preprocess`.

`--max-tokens N` caps the patch-token count per image after preprocessing
(default `4*(518/patch)^2`, e.g. 5476 at patch 14; `0` disables). It is
checked before preprocessing and before any graph is constructed, so
oversize inputs fail fast with exit 1. The cap also guards `--no-resize`
runs: attention memory grows quadratically in token count, and native
megapixel inputs can request hundreds of GB.

## Batch inference

Multiple inputs are given by repeating `-i` and/or comma-separating
paths (the same convention as `scripts/parity_check.py`'s `--image`):

```bash
dinov2-cli -m models/model.gguf -i a.jpg -i b.jpg -i c.jpg --batch 2 --print-embeddings
dinov2-cli -m models/model.gguf -i a.jpg,b.jpg,c.jpg --batch 2 --print-embeddings
```

Images are forwarded to the encoder in sequential chunks of `--batch`
(a short tail chunk is fine; `--batch` larger than the image count just
runs everything at once). Per-image results are identical to running
each image alone: batching changes throughput, not outputs.

**All images inside a chunk must share dimensions** after
preprocessing. Chunks are formed in input order, so a mixed-size list
works only when same-size images land in the same chunk (different
chunks may differ). A chunk that mixes sizes aborts the run with an
error naming the conflicting inputs. In classify mode every input is
center-cropped to 224x224, so dims always match there. Feature mode
bounds the shortest edge to 518 by default (`--preprocess bounded`),
but different aspect ratios still produce different grids; use
`--preprocess crop518` for a fixed 518x518 grid that always matches.
When in doubt, group same-aspect-ratio inputs together, use
`crop518`, or use `-c`.

## Embeddings JSON schema

`--print-embeddings` prints one object per input image, one line each
(a single object for a single input). Real output for
`dinov2-cli -m models/model.gguf -i assets/tench.jpg --print-embeddings`
(arrays truncated):

```json
{
  "model": "models/model.gguf",
  "index": 0,
  "image": "assets/tench.jpg",
  "n_patches": 1320,
  "grid": {"h": 30, "w": 44},
  "hidden": 384,
  "cls": [0.46475, 2.47845, -5.207, -1.29146, ...],
  "pooled": [0.46475, 2.47845, -5.207, -1.29146, ...]
}
```

Fields:

| Field | Type | Present when |
|:------|:-----|:-------------|
| `model` | string | always |
| `index` | int | always |
| `image` | string | always |
| `n_patches` | int | always |
| `grid` | `{"h": int, "w": int}` | always |
| `hidden` | int | always |
| `cls` | float[`hidden`] | always |
| `pooled` | float[2*`hidden`] | feature mode only (absent with `-c`) |
| `topk` | array of objects | with `-c` only |
| `patches` | float[`n_patches`*`hidden`] | feature mode + `--print-patch-tokens` |

- `model`: short label derived from the model path. A path containing a
  `dinov2-vit-*` component (for example
  `models/dinov2-vit-small-patch14/model.gguf`) reports that component;
  anything else reports the path exactly as passed to `-m`.
- `index`: 0-based position of the input in the `-i` list. Records are
  emitted in input order, so `index` equals the JSONL line number; it is
  the join key when `image` is ambiguous (duplicate paths, sanitized
  stems in output filenames).
- `image`: the input path this line describes, exactly as passed to
  `-i`.
- `n_patches`: actual patch count after preprocessing. Not necessarily
  the model's native grid. Under `-c` the input is always 224x224, so
  this is 256 for patch14 models.
- `grid`: patch-grid dimensions `{"h": ny/patch_size, "w":
  nx/patch_size}` of the preprocessed image; `h * w == n_patches`.
- `hidden`: embedding width. 384, 768, 1024, 1536 for the small, base,
  large, giant models.
- `cls`: final-layernorm CLS token, equal to the HF model's
  `last_hidden_state[:, 0]`. This is the standard embedding for
  retrieval and similarity search. Emitted in both classify and feature
  modes.
- `pooled`: `[cls || mean(patch_tokens)]`, `cls` in the first `hidden`
  elements. This is the exact feature the ImageNet1k linear head
  consumes; use it for linear-eval-style downstream tasks.
- `patches`: per-patch token vectors, flat row-major `grid.h * grid.w *
  hidden` floats. Row `p` is patch `p` scanning left-to-right,
  top-to-bottom over the `grid` grid. Register tokens are excluded.
  Absent under `-c` even when the flag is passed.
- `topk`: array of `{"idx": int, "label": string, "prob": float}` sorted
  by descending softmax probability, `k` entries long (`-k`, default 5).
  `label` comes from the GGUF `id2label` map.

### Preview binary embeddings

`--embeddings-binary` is an intentionally simple, unstable preview for
high-throughput consumers. It requires `-o PATH`, cannot be combined with
`-c` or `--bench`, and writes no binary data to stdout. With one input, `-o`
is the exact file path. With multiple inputs, `-o` is created as a directory
and files are named `<zero-based-index>-<sanitized-input-stem>.d2e`, which
prevents collisions between inputs that share a stem. Errors are reported on
stderr and return exit 1.

Version 2 starts with this exact 40-byte little-endian header. Readers should
reject unknown flag bits and must not assume this preview format will remain
compatible:

| Offset | Size | Field |
|:--:|:--:|:--|
| 0 | 8 | Magic `D2EMB\\0\\0\\0` |
| 8 | 2 | Format version `2` |
| 10 | 2 | Header size `40` |
| 12 | 4 | Hidden dimension `H` |
| 16 | 4 | Pooled dimension `2H` |
| 20 | 4 | Patch count `P`, or zero when absent |
| 24 | 4 | Flags: bit 0 patches, bit 1 L2 requested |
| 28 | 4 | Reserved zero |
| 32 | 4 | Patch grid width `W`, or zero when patches are absent |
| 36 | 4 | Patch grid height `H_grid`, or zero when patches are absent |

The float32 payload is contiguous and ordered as `cls[H]`, `pooled[2H]`, then
optional row-major `patches[P][H]`. Pooled is `[cls || mean(non-register
patch tokens)]`. Patch rows exclude register tokens and scan top to bottom,
left to right. L2 normalization applies independently to CLS, pooled, and
each patch row, matching JSON. The writer validates vector lengths before
writing and serializes each integer and float explicitly, rather than dumping
a C++ struct.

This is a preview only. The magic, header, flags, dimensions, ordering, and
extension may change without compatibility guarantees. It is not a standard,
compression format, mmap format, or schema negotiation layer.

### Raw vs `--l2-normalize`

Raw output matches the HF model's float outputs. Pass `--l2-normalize`
when consumers compute cosine similarity: `cls` and `pooled` are each
normalized as a whole vector, and each `patches` row is normalized
independently.

### stdout/stderr contract

All data goes to stdout, all logs go to stderr. Pipe safely:

```bash
dinov2-cli -m models/model.gguf -i assets/tench.jpg --print-embeddings | jq .cls
```

For multi-input JSONL or PCA runs, a later chunk failure can leave stdout or PCA
output partial after earlier chunks were emitted. The process exits 1; consumers
must check the exit status before treating output as complete.

## Workflows

### Classify top-k

```bash
dinov2-cli -m models/model.gguf -i assets/tench.jpg -c -k 3
```

### Batch embeddings (JSONL)

```bash
dinov2-cli -m models/model.gguf -i a.jpg -i b.jpg --batch 2 \
    --print-embeddings > batch.jsonl
```

Each line of `batch.jsonl` is a complete embeddings object whose
`image` field names its input, so `while read` / `jq -c` consumers work
unchanged.

### Embeddings to cosine similarity

```bash
dinov2-cli -m models/model.gguf -i a.jpg -i b.jpg --print-embeddings > both.jsonl
jq -s '.[0].cls' both.jsonl > a.json
jq -s '.[1].cls' both.jsonl > b.json
```

```python
import json, math
a = json.load(open("a.json"))  # already the cls list extracted by jq
b = json.load(open("b.json"))
cos = sum(x * y for x, y in zip(a, b)) / (
    math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))
print(cos)
```

### Patch tokens for downstream tasks

```bash
dinov2-cli -m models/model.gguf -i assets/tench.jpg \
    --print-embeddings --print-patch-tokens > patches.json
```

`patches` is `n_patches` rows of `hidden` floats in row-major order:
reshape to `(n_patches, hidden)` for per-patch features, or to the
`(ny/patch_size, nx/patch_size)` spatial grid for dense prediction.

### PCA visualization

```bash
dinov2-cli -m models/model.gguf -i assets/tench.jpg -o pca.png
dinov2-cli -m models/model.gguf -i a.jpg -i b.jpg -o pca/  # -> pca/0-a.pca.png, pca/1-b.pca.png
```

Feature mode only; `-c` rejects `-o` at parse time and `--bench` skips
it. With multiple inputs `-o` names a directory (created if missing)
that gets one `<index>-<input-stem>.pca.png` per image. Output is PNG
bytes whatever the extension, so use `.png`.

### Bench

```bash
dinov2-cli -m models/model.gguf -i assets/tench.jpg --bench --bench-runs 5 --bench-json
dinov2-cli -m models/model.gguf -i a.jpg -i b.jpg --batch 2 --bench --bench-json
```

With `--bench-json`, stdout gets one JSON object per invocation with
`model`, `n_threads`, `n_repeats`, `n_warmup`, `samples_ms`, `mean_ms`,
`stddev_ms`, `min_ms`, `max_ms`, `peak_rss_mb`, `n_images`, `batch`,
`ms_per_image`, `images_per_sec`. `samples_ms`/`mean_ms` time one full
pass over every input image (each pass chunked at `--batch`), so
`ms_per_image` is `mean_ms / n_images` and `images_per_sec` its
reciprocal in seconds. Without `--bench-json` the same numbers print
to stderr as a `bench(...)` line. Timings cover the forward pass only;
see [benchmarks.md](benchmarks.md) for methodology.

### Quantized variants

```bash
hf download dinov2-cpp-core/dinov2-base-gguf --local-dir models/dinov2-base
dinov2-cli -m models/dinov2-base/model.gguf -i assets/tench.jpg -c
```

Any GGUF works as `-m`: the f16 `model.gguf` files from the
`dinov2-cpp-core/*-gguf` repos, or a q4_0 through q8_0 quantized GGUF,
which loads transparently.

### Threads and flash attention

```bash
dinov2-cli -m models/model.gguf -t 8 -fa -i img.jpg --print-embeddings
```

`-t` sets compute threads; the physical core count is a good default
(more is not always better). `-fa` enables flash attention: faster but
less accurate, so avoid it when comparing against HF outputs.

## Exit codes

| Code | Meaning |
|:-----|:--------|
| 0 | success; also `--help` and `--version` |
| 1 | unknown argument, no input images, image load failure, model load failure, mixed-size batch chunk, `--max-tokens` cap exceeded, no output mode selected, allocation failure, or graph compute failure |

Error messages go to stderr; the model-load failure also prints the
`hf download` hint shown above.

## Troubleshooting

**`failed to load model from '...'`**: `-m` must point at a GGUF file.
Each `dinov2-cpp-core` repo ships its weight as `model.gguf`, so
`--local-dir models` produces `models/model.gguf`. Re-download with:

```bash
hf download dinov2-cpp-core/dinov2-small-gguf --local-dir models
```

**`-c` always runs at 224x224**: classification follows the HF
`preprocessor_config.json` recipe: resize so the shortest edge is 256
(bicubic, aspect preserved), then center-crop 224x224. Feature mode
instead bounds the shortest edge to 518 by default
(`--preprocess bounded`), then aligns each dimension up to a multiple
of the patch size and interpolates position embeddings, so arbitrary
sizes work without unbounded memory. `--preprocess hf` applies the HF
recipe (256 + 224 crop) for feature parity checks, `--preprocess
crop518` fixes every input at 518x518, and `--no-resize` keeps native
resolution subject to `--max-tokens`.

**Preprocessing defaults**: ImageNet mean/std (0.485, 0.456, 0.406 /
0.229, 0.224, 0.225), bicubic interpolation, RGB channel order. Matches
the HF image processor.

**JSON edge cases**: `nan`/`inf` float values would be emitted unquoted,
which is not valid JSON; this only happens on pathological inputs.
String escaping covers `"` and `\` only, which suffices for the bundled
ImageNet labels.

## See also

- [tools/server/README.md](../tools/server/README.md): `dinov2-server`
  exposes the same embeddings over HTTP (tier-2, off by default).
- [examples/dedup/](../examples/dedup/): a stdlib-only script that consumes
  this JSONL output to cluster near-duplicate images.
- [stability.md](stability.md): which parts of this contract are frozen.
