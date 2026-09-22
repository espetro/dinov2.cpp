# Stability

What is covered by compatibility guarantees, and what is not.

## Stable

- The `dinov2-cli` command-line contract: flags, JSONL/JSON output keys and
  order, human-readable top-k output, exit codes, and stderr messages. See
  [cli.md](cli.md).
- The GGUF schema read by the loader: metadata keys and tensor names written
  by `scripts/dinov2-to-gguf.py` and published on the `dinov2-cpp-core` HF
  profile.

## Unstable

- The C API in `include/dinov2.h` (`libdinov2`): new in the 0.4.x line.
  Function signatures, the `dino_status` enum, params struct layout, and
  accessor semantics may change between minor releases. Pin a tag if you
  embed it.
- The D2EMB binary preview written by `--embeddings-binary`: explicitly
  unstable; the header carries a version field but no compatibility promise.
- The internal C++ header `dinov2.h` (repo root): not installed, no
  compatibility guarantees at all. It changes freely between commits.
