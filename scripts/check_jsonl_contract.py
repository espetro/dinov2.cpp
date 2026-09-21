#!/usr/bin/env python3
"""Validate dinov2-cli --print-embeddings JSONL output against the record contract.

Reads JSONL from stdin (or a file argument). Checks per line:
  - keys model, index, image, n_patches, grid{h,w}, hidden, cls are present
  - index equals the 0-based line order
  - grid.h * grid.w == n_patches
  - patches (if present) has n_patches * hidden floats

Exits 0 on success, 1 with a message on the first violation.
"""

import json
import sys


def check(stream) -> int:
    n = 0
    for lineno, raw in enumerate(stream):
        line = raw.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            print(f"line {lineno}: invalid JSON: {exc}", file=sys.stderr)
            return 1
        for key in ("model", "index", "image", "n_patches", "grid", "hidden", "cls"):
            if key not in rec:
                print(f"line {lineno}: missing key '{key}'", file=sys.stderr)
                return 1
        if rec["index"] != n:
            print(f"line {lineno}: index {rec['index']} != line order {n}", file=sys.stderr)
            return 1
        grid = rec["grid"]
        if grid.get("h", 0) * grid.get("w", 0) != rec["n_patches"]:
            print(
                f"line {lineno}: grid {grid['h']}x{grid['w']} != n_patches {rec['n_patches']}",
                file=sys.stderr,
            )
            return 1
        if "patches" in rec and len(rec["patches"]) != rec["n_patches"] * rec["hidden"]:
            print(
                f"line {lineno}: patches length {len(rec['patches'])} != "
                f"{rec['n_patches']} * {rec['hidden']}",
                file=sys.stderr,
            )
            return 1
        n += 1
    if n == 0:
        print("no records found", file=sys.stderr)
        return 1
    print(f"ok: {n} record(s) conform to the JSONL contract")
    return 0


def main() -> int:
    if len(sys.argv) > 2:
        print(__doc__, file=sys.stderr)
        return 2
    if len(sys.argv) == 2:
        with open(sys.argv[1], encoding="utf-8") as f:
            return check(f)
    return check(sys.stdin)


if __name__ == "__main__":
    sys.exit(main())
