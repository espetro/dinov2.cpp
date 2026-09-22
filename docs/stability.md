# Stability contract

What consumers can rely on, and for how long. Surfaces are split into
**tier-1** (the shipped product; breaking changes follow semver and are
documented in release notes) and **tier-2** (opt-in previews and examples;
may change or disappear between any two commits). Tier definitions and the
promotion process live in [tiers.md](tiers.md).

## Tier-1 surfaces

| Surface | What is covered | Policy |
|:--------|:----------------|:-------|
| `dinov2-cli` flags and output | Flag names/semantics, JSONL/JSON output keys and order, human-readable top-k format, exit codes, stderr messages ([cli.md](cli.md)) | Semver: breaking changes bump minor (0.x) or major and are called out in release notes |
| `--print-embeddings` JSONL schema | Record keys, types, field semantics, per-line ordering (`index` == line number) | Semver, same as the CLI. Additive keys are allowed in minor releases; removed/renamed/retyped keys are breaking |
| `libdinov2` build target | The `dinov2` CMake target name, `include/` as its only public interface dir, `BUILD_SHARED_LIBS` support | Semver |
| C API (`include/dinov2.h`) | `dino_*` functions, `dino_status`, params PODs, accessor borrow semantics | **Unstable for the 0.4.x line**: signatures and struct layouts may change between minor releases. Stabilizes at a later minor (announced in release notes). Pin a tag if you embed it |
| GGUF schema | Metadata keys and tensor names the loader reads, as written by `scripts/dinov2-to-gguf.py` | Stable: the loader keeps reading GGUFs produced by released converters |
| GGUF publishing | The `dinov2-cpp-core` HF profile, repo naming (`dinov2-*-gguf`), `model.gguf` filename | Best-effort stable: repos and file names are not renamed or deleted without a deprecation notice |

## Tier-2 surfaces

Nothing below is a compatibility promise. These are off-by-default previews
and example code; pin a commit if you depend on one.

| Surface | Status |
|:--------|:-------|
| wasm target (`wasm/`, `DINOV2_BUILD_WASM`) | Preview. JS API shape, embind names and build flags may change |
| `dinov2-server` (`tools/server/`, `DINOV2_BUILD_SERVER`) | Preview. Endpoints and response shape may change; single-process, no SLA |
| `examples/` (dedup, ci-visual-regression) | Example code. Copy-paste supported, upstream compatibility not promised |
| `--embeddings-binary` (D2EMB) | Explicitly unstable preview: the header carries a version field but no compatibility promise, format may change at any commit |

## Internal surfaces (no guarantees)

- `src/dinov2-impl.h`: the internal C++ header shared by the engine, CLI, C
  wrapper and tests. **Not installed.** Source-compat only within a commit;
  it changes freely. Out-of-tree consumers should not include it.
- `src/image.h`: internal helpers. Same rule: usable in-repo (tier-2 code
  may), not a public contract.
- Anything reachable only by patching the build (sanitizer flags, internal
  ggml knobs) is unsupported.

## Deprecation mechanics

When a tier-1 surface must break:

1. The old behavior keeps working for at least one minor release where
   technically feasible, with a deprecation warning on stderr.
2. The breaking change lands with a `BREAKING` note in the changelog and
   release notes.
3. `DINO_DEPRECATED` marks doomed C API functions once the C API is stable;
   while it is unstable (0.4.x) functions may simply change.

Tier-2 surfaces skip this process entirely; their preview status is the
warning.
