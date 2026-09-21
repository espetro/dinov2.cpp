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
cmake --preset release && cmake --build build-release
# binary lands at ./build-release/bin/dinov2-cli
```

Download a GGUF weight from the
[`dinov2-cpp-core`](https://huggingface.co/dinov2-cpp-core) Hugging Face
profile:

```bash
huggingface-cli download dinov2-cpp-core/dinov2-small-gguf --local-dir models
# file on disk: models/model.gguf
```

The binary loads any ggml-supported quantization transparently (f16,
q4_0 through q8_0): point `-m` at whichever GGUF you have.

## Flags

Verbatim from `dinov2-cli --help` (v0.3.0). Flags that take a value read
it from the next argument.

| Flag | Default | Effect |
|:-----|:--------|:-------|
| `-m FNAME`, `--model` | `../model.gguf` | GGUF model path |
| `-fa`, `--flash_attn` | off | enable flash attention, less accurate |
| `-t N`, `--threads` | 4 | threads used during computation |
| `-i FNAME`, `--inp` | `../assets/tench.jpg` | input image file |
| `-s N`, `--seed` | 42 | RNG seed |
| `-c`, `--classify` | off | classify the image and print top-k labels |
| `-k N`, `--topk` | 5 | number of classes printed with `-c` |
| `--print-embeddings` | off | emit one embeddings JSON object on stdout |
| `--print-patch-tokens` | off | add per-patch token vectors to that JSON |
| `--l2-normalize` | off | L2-normalize emitted embedding vectors |
| `-o FNAME`, `--out` | off | write a PCA visualization of patch features |
| `--bench` | off | timed bench loop (5 repeats, 1 warmup) |
| `--bench-runs N` | 5 | timed runs when `--bench` is set |
| `--bench-warmup N` | 1 | warmup runs discarded before timing |
| `--bench-json` | off | bench result as JSON on stdout |
| `-h`, `--help` | | print usage and exit |
| `--version` | | print `dinov2-cli 0.3.0` and exit |

## Output modes

Select at least one output mode per invocation. With none selected the
CLI prints a hint and exits 1.

- **Classification** (`-c`): prints one ` > label : prob` line per top-k
  class on stdout. Combined with `--print-embeddings`, the same results
  land in the JSON `topk` array and the text lines are not printed.
- **Embeddings JSON** (`--print-embeddings`): one JSON object on stdout
  for a single-shot run; the `--bench` loop ignores it and emits its
  own bench output instead. `--print-patch-tokens` adds a `patches`
  field; `--l2-normalize` normalizes the vectors.
- **PCA visualization** (`-o FNAME`): writes a PNG of the patch features
  projected to RGB. Feature mode only; ignored under `-c` and `--bench`.
  The writer emits PNG bytes regardless of the file extension.
- **Bench** (`--bench*`): warmup runs plus timed runs of the forward
  pass. `--bench-json` prints one JSON object on stdout; without it a
  `bench(model=..., ...)` summary line goes to stderr. The bench loop
  skips PCA output and ignores `--print-embeddings`.

Modes combine. `-c --print-embeddings` produces one object with `cls`
and `topk`. `--print-embeddings -o pca.png` emits the JSON and writes
the PNG in the same run. `--print-patch-tokens` only affects the
embeddings JSON: on its own it selects no output mode and the run fails
the nothing-to-do guard.

## Embeddings JSON schema

`--print-embeddings` prints a single object. Real output for
`dinov2-cli -m models/model.gguf -i assets/tench.jpg --print-embeddings`
(arrays truncated):

```json
{
  "model": "models/model.gguf",
  "image": "assets/tench.jpg",
  "n_patches": 1320,
  "hidden": 384,
  "cls": [0.46475, 2.47845, -5.207, -1.29146, ...],
  "pooled": [0.46475, 2.47845, -5.207, -1.29146, ...]
}
```

Fields:

| Field | Type | Present when |
|:------|:-----|:-------------|
| `model` | string | always |
| `image` | string | always |
| `n_patches` | int | always |
| `hidden` | int | always |
| `cls` | float[`hidden`] | always |
| `pooled` | float[2*`hidden`] | feature mode only (absent with `-c`) |
| `topk` | array of objects | with `-c` only |
| `patches` | float[`n_patches`*`hidden`] | feature mode + `--print-patch-tokens` |

- `model`: short label derived from the model path. A path containing a
  `dinov2-vit-*` component (for example
  `models/dinov2-vit-small-patch14/model.gguf`) reports that component;
  anything else reports the path exactly as passed to `-m`.
- `image`: input path exactly as passed to `-i`.
- `n_patches`: actual patch count after the input is resized up to a
  multiple of the patch size. Not necessarily the model's native grid.
  Under `-c` the input is always 224x224, so this is 256 for patch14
  models.
- `hidden`: embedding width. 384, 768, 1024, 1536 for the small, base,
  large, giant models.
- `cls`: final-layernorm CLS token, equal to the HF model's
  `last_hidden_state[:, 0]`. This is the standard embedding for
  retrieval and similarity search. Emitted in both classify and feature
  modes.
- `pooled`: `[cls || mean(patch_tokens)]`, `cls` in the first `hidden`
  elements. This is the exact feature the ImageNet1k linear head
  consumes; use it for linear-eval-style downstream tasks.
- `patches`: per-patch token vectors in row-major order, patch index to
  `hidden` floats. Row `p` is patch `p` scanning left-to-right,
  top-to-bottom over the `(ny/patch_size, nx/patch_size)` grid. Register
  tokens are excluded. Absent under `-c` even when the flag is passed.
- `topk`: array of `{"idx": int, "label": string, "prob": float}` sorted
  by descending softmax probability, `k` entries long (`-k`, default 5).
  `label` comes from the GGUF `id2label` map.

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

## Workflows

### Classify top-k

```bash
dinov2-cli -m models/model.gguf -i assets/tench.jpg -c -k 3
```

### Embeddings to cosine similarity

```bash
dinov2-cli -m models/model.gguf -i a.jpg --print-embeddings > a.json
dinov2-cli -m models/model.gguf -i b.jpg --print-embeddings > b.json
```

```python
import json, math
a = json.load(open("a.json"))["cls"]
b = json.load(open("b.json"))["cls"]
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
```

Feature mode only; the flag is ignored under `-c` and `--bench`. Output
is PNG bytes whatever the extension, so use `.png`.

### Bench

```bash
dinov2-cli -m models/model.gguf -i assets/tench.jpg --bench --bench-runs 5 --bench-json
```

With `--bench-json`, stdout gets one JSON object per invocation with
`model`, `n_threads`, `n_repeats`, `n_warmup`, `samples_ms`, `mean_ms`,
`stddev_ms`, `min_ms`, `max_ms`, `peak_rss_mb`. Without it the same
numbers print to stderr as a `bench(...)` line. Timings cover the
forward pass only; see [benchmarks.md](benchmarks.md) for methodology.

### Quantized variants

```bash
huggingface-cli download dinov2-cpp-core/dinov2-base-gguf --local-dir models/dinov2-base
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
| 1 | unknown argument, image load failure, model load failure, no output mode selected, or graph compute failure |

Error messages go to stderr; the model-load failure also prints the
`huggingface-cli download` hint shown above.

## Troubleshooting

**`failed to load model from '...'`**: `-m` must point at a GGUF file.
Each `dinov2-cpp-core` repo ships its weight as `model.gguf`, so
`--local-dir models` produces `models/model.gguf`. Re-download with:

```bash
huggingface-cli download dinov2-cpp-core/dinov2-small-gguf --local-dir models
```

**`-c` always runs at 224x224**: classification follows the HF
`preprocessor_config.json` recipe: resize so the shortest edge is 256
(bicubic, aspect preserved), then center-crop 224x224. The GGUFs declare
`img_size=518`, which applies to feature mode instead: there the input
is resized up to a multiple of the patch size and position embeddings
are interpolated, so arbitrary sizes work.

**Preprocessing defaults**: ImageNet mean/std (0.485, 0.456, 0.406 /
0.229, 0.224, 0.225), bicubic interpolation, RGB channel order. Matches
the HF image processor.

**JSON edge cases**: `nan`/`inf` float values would be emitted unquoted,
which is not valid JSON; this only happens on pathological inputs.
String escaping covers `"` and `\` only, which suffices for the bundled
ImageNet labels.
