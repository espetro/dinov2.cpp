# Tiers

Every user-visible surface in this repo is either **tier-1** or **tier-2**.
The split exists so optional previews can ship early without putting the
engine, the CLI contract, or CI health at risk. Per-surface policies are in
[stability.md](stability.md); this page defines the tiers themselves.

## Tier-1: the product

- Ships in every release archive and is covered by the compatibility
  contract.
- Built and tested by default: `cmake --preset debug` builds it, `ctest`
  covers it, a red tier-1 job blocks merge.
- Currently: `libdinov2` (with the C API in `include/dinov2.h`),
  `dinov2-cli` and its JSONL schema, the GGUF loader schema, the converter
  (`scripts/dinov2-to-gguf.py`), and GGUF publishing to `dinov2-cpp-core`.

## Tier-2: opt-in previews and examples

Hard requirements, all enforced by construction:

- **Off by default.** A CMake option (`DINOV2_BUILD_WASM`,
  `DINOV2_BUILD_SERVER`) or a directory nothing tier-1 references
  (`examples/`). The default build must be bit-identical with or without the
  tier-2 code present.
- **Isolated.** Tier-2 code may consume the public C API and in-repo
  internal headers, but nothing tier-1 may include, link, or shell out to
  tier-2 code. Dependencies point downward only; nothing flows back.
- **Non-blocking in CI.** Tier-2 jobs live in
  `.github/workflows/extras.yml` with `continue-on-error: true`. Red is a
  drift signal, never a merge gate.
- **No compatibility promise.** APIs, endpoints and output formats may
  change or be deleted between any two commits. Pin a commit to depend on
  one.

Currently tier-2: the wasm target (`wasm/`), `dinov2-server`
(`tools/server/`), everything under `examples/`, and the D2EMB binary
preview (`--embeddings-binary`).

## Promotion (tier-2 -> tier-1)

A surface graduates when all of these hold:

1. A declared owner and a doc page that commits to a contract
   (endpoints/schema/API written down, not just implemented).
2. Real usage evidence: at least one downstream consumer or sustained
   community use, tracked in an issue.
3. It moves into the default build or the release artifacts, and its CI job
   loses `continue-on-error`.
4. A stability policy entry in [stability.md](stability.md) stating what is
   frozen and the deprecation path.

## Demotion (tier-1 -> tier-2)

Rare and deliberate. Requires a deprecation notice in release notes and at
least one minor release where the surface still works behind a warning.
Demotion is for surfaces that are maintained but no longer part of the
contract (e.g. an experimental output mode that never found consumers), not
for code being deleted outright.
