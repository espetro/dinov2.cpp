#!/usr/bin/env python3
"""Near-duplicate image finder over dinov2-cli embeddings (tier-2 example).

Runs `dinov2-cli --print-embeddings` on a folder, computes pairwise cosine
similarity on the `pooled` vectors, clusters images above a threshold, and
writes a small HTML page for visual review. Stdlib only.

Usage:
    python3 dedup.py <dinov2-cli> <model.gguf> <image_dir> [-t 0.95] [-o report.html]

Example:
    python3 dedup.py ./build/bin/dinov2-cli models/model.gguf ~/Photos -o dupes.html
"""

import argparse
import html
import json
import math
import subprocess
import sys
from pathlib import Path

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tga", ".webp"}


def embed(cli: str, model: str, images: list[Path]) -> list[dict]:
    """One JSONL record per image; dinov2-cli prints them in input order."""
    proc = subprocess.run(
        [cli, "-m", model, "--print-embeddings"] + [arg for p in images for arg in ("-i", str(p))],
        capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit(f"dinov2-cli failed:\n{proc.stderr}")
    records = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
    if len(records) != len(images):
        sys.exit(f"expected {len(images)} records, got {len(records)}")
    return records


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def clusters(sims: list[tuple[float, int, int]], n: int) -> list[list[int]]:
    """Single-linkage clustering: union pairs above threshold."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for _, i, j in sims:
        pi, pj = find(i), find(j)
        if pi != pj:
            parent[pi] = pj
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [g for g in groups.values() if len(g) > 1]


def write_html(path: Path, images: list[Path], groups: list[list[int]],
               sims: list[tuple[float, int, int]]) -> None:
    rows = []
    for gi, group in enumerate(groups):
        rows.append(f"<h2>cluster {gi + 1} ({len(group)} images)</h2><div>")
        for i in group:
            rows.append(
                f'<figure><img src="{images[i].resolve().as_uri()}" loading="lazy">'
                f"<figcaption>{html.escape(images[i].name)}</figcaption></figure>")
        rows.append("</div><ul>")
        for s, i, j in sims:
            if i in group and j in group:
                rows.append(f"<li>{html.escape(images[i].name)} ~ "
                            f"{html.escape(images[j].name)}: {s:.4f}</li>")
        rows.append("</ul>")
    doc = f"""<!doctype html><meta charset="utf-8"><title>dinov2.cpp dedup</title>
<style>body{{font-family:system-ui;margin:2em}}div{{display:flex;flex-wrap:wrap;gap:1em}}
img{{max-width:220px;max-height:220px}}figure{{margin:0}}figcaption{{font-size:.8em}}</style>
<h1>near-duplicate clusters</h1><p>{len(images)} images scanned, {len(groups)} cluster(s).</p>
{"".join(rows)}"""
    path.write_text(doc, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cli", help="path to the dinov2-cli binary")
    ap.add_argument("model", help="path to a GGUF model")
    ap.add_argument("image_dir", type=Path, help="folder of images to scan")
    ap.add_argument("-t", "--threshold", type=float, default=0.95,
                    help="cosine similarity cutoff for duplicates (default: 0.95)")
    ap.add_argument("-o", "--out", type=Path, default=Path("dedup-report.html"),
                    help="HTML report path (default: dedup-report.html)")
    args = ap.parse_args()

    images = sorted(p for p in args.image_dir.iterdir() if p.suffix.lower() in EXTS)
    if len(images) < 2:
        sys.exit(f"need at least 2 images in {args.image_dir}")

    print(f"embedding {len(images)} images...", file=sys.stderr)
    records = embed(args.cli, args.model, images)
    vecs = [r.get("pooled") or r["cls"] for r in records]

    sims = [(cosine(vecs[i], vecs[j]), i, j)
            for i in range(len(images)) for j in range(i + 1, len(images))]
    sims = sorted((s for s in sims if s[0] >= args.threshold), reverse=True)

    groups = clusters(sims, len(images))
    write_html(args.out, images, groups, sims)
    for gi, group in enumerate(groups):
        print(f"cluster {gi + 1}: " + ", ".join(images[i].name for i in group))
    print(f"wrote {args.out} ({len(groups)} cluster(s), threshold {args.threshold})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
