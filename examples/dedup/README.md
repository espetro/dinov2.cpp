# dedup: near-duplicate photo finder

Dedup a folder of images locally, no cloud. `dedup.py` shells out to
`dinov2-cli --print-embeddings`, compares the `pooled` vectors pairwise with
cosine similarity, clusters near-duplicates (single-linkage union-find), and
writes a small HTML page so you can eyeball the groups before deleting
anything.

Tier-2 example: Python stdlib only, no dependencies, not part of the shipped
binary or CI gates.

## Usage

```bash
python3 dedup.py <dinov2-cli> <model.gguf> <image_dir> [-t 0.95] [-o report.html]
```

```bash
# build the CLI and grab a weight first (see README.md quickstart)
python3 examples/dedup/dedup.py \
  ./build-release/bin/dinov2-cli models/model.gguf ~/Photos/dump \
  -t 0.95 -o dupes.html
open dupes.html
```

- `-t/--threshold`: cosine cutoff for "near duplicate". `0.99` catches
  re-encodes and resizes; `0.95` catches same-scene shots; lower starts
  grouping merely similar images. Tune on a small folder first.
- `-o/--out`: where the HTML review page goes (default `dedup-report.html`).

The script uses `pooled` (`[cls || mean(patches)]`) when present and falls
back to `cls`. For large folders the pairwise pass is O(n^2) in Python; a
few thousand images is fine, hundreds of thousands will want a real index
(FAISS, sqlite-vec, ...).
