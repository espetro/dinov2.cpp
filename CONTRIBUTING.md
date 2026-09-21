# Contributing

Thanks for considering a contribution. Keep PRs focused and small.

## Dev harness

CMake presets (defined in `CMakePresets.json`) configure the standard workflow:

```bash
cmake --preset debug    # or: release, asan, ubsan
cmake --build --preset debug
ctest --preset debug
```

- `debug` / `release`: standard configurations. Both enable `DINOV2_FATAL_WARNINGS` by default, so warnings are errors.
- `asan` / `ubsan`: AddressSanitizer and UndefinedBehaviorSanitizer builds for debugging memory and UB issues.

Build directories are `build-<preset>/`.

## Tests

Unit tests use [doctest](https://github.com/doctest/doctest). Run the suite with:

```bash
cmake --build --preset debug && ctest --preset debug
```

Run the same suite under the `asan` and `ubsan` presets before submitting anything that touches memory handling or the compute path.

### Parity check vs PyTorch

`scripts/parity_check.py` compares dinov2-cli embeddings and top-1 classification against the HuggingFace PyTorch reference and reports cosine similarity per image:

```bash
.venv/bin/python scripts/parity_check.py --gguf models/model.gguf
```

It needs the `.venv` deps (torch, transformers, pillow) and a downloaded GGUF.

Default gates are cls/pooled cosine >= 0.999 and patch flat + per-token-mean cosine >= 0.99; the per-token minimum is reported as informational because f16 inference diverges from the f32 reference on a small token tail. Pass `--patches-token-min-threshold <float>` for a strict per-token check.

## Formatting

CI enforces `clang-format-18`. Before committing:

```bash
clang-format-18 -i <changed files>
```

Format violations fail the lint job.

## Warnings

`DINOV2_FATAL_WARNINGS=ON` (the preset default) promotes warnings to errors. Keep the build clean under this option; do not disable it to mask warnings.

## PR guidelines

- One logical change per PR. Keep refactors separate from features.
- Ensure the full test suite passes under `debug`, `asan` and `ubsan`.
- Update docs (`README.md`, `docs/`) if you change behavior or interfaces.
- Follow conventional commit format (e.g. `feat:`, `fix:`, `docs:`, `ci:`).

## Converting weights yourself

The README's ready-to-download GGUFs live at `dinov2-cpp-core/<variant>-gguf` on Hugging Face. To convert a variant yourself from the upstream PyTorch checkpoints:

| Model | Source checkpoint (PyTorch) | Recommendation |
|:-----:|:----------------------------|:---------------|
| small (no registers) | [facebook/dinov2-small-imagenet1k-1-layer](https://huggingface.co/facebook/dinov2-small-imagenet1k-1-layer) | Baseline reproduction or task comparison |
| base (no registers) | [facebook/dinov2-base-imagenet1k-1-layer](https://huggingface.co/facebook/dinov2-base-imagenet1k-1-layer) | Baseline reproduction or task comparison |
| large (no registers) | [facebook/dinov2-large-imagenet1k-1-layer](https://huggingface.co/facebook/dinov2-large-imagenet1k-1-layer) | Baseline reproduction or task comparison |
| giant (no registers) | [facebook/dinov2-giant-imagenet1k-1-layer](https://huggingface.co/facebook/dinov2-giant-imagenet1k-1-layer) | Baseline reproduction or task comparison |
| small (registers) | [facebook/dinov2-with-registers-small-imagenet1k-1-layer](https://huggingface.co/facebook/dinov2-with-registers-small-imagenet1k-1-layer) | **Recommended for patch/dense features** |
| base (registers) | [facebook/dinov2-with-registers-base-imagenet1k-1-layer](https://huggingface.co/facebook/dinov2-with-registers-base-imagenet1k-1-layer) | **Recommended for patch/dense features** |
| large (registers) | [facebook/dinov2-with-registers-large-imagenet1k-1-layer](https://huggingface.co/facebook/dinov2-with-registers-large-imagenet1k-1-layer) | **Recommended for patch/dense features** |
| giant (registers) | [facebook/dinov2-with-registers-giant-imagenet1k-1-layer](https://huggingface.co/facebook/dinov2-with-registers-giant-imagenet1k-1-layer) | **Recommended for patch/dense features** |

The converter also accepts the backbone-only names `facebook/dinov2-{small,base,large,giant}` and `facebook/dinov2-with-registers-{small,base,large,giant}`. These produce feature-only GGUFs and do not invent ImageNet classifier heads or labels.

For the bounded rationale, see [Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588) and the official [DINOv2 results](https://github.com/facebookresearch/dinov2/blob/main/README.md). Register-token checkpoints are recommended for patch and dense feature workflows, while no-register checkpoints remain useful for exact baseline reproduction and task-specific comparisons.

```bash
python ./scripts/dinov2-to-gguf.py --model_name facebook/dinov2-small-imagenet1k-1-layer
```

See [docs/hf-publishing.md](docs/hf-publishing.md) for how CI publishes the mirrors (`HF_TOKEN` setup included).

## Build from source

```bash
git clone --recurse-submodules https://github.com/espetro/dinov2.cpp.git
cd dinov2.cpp
cmake --preset release && cmake --build --preset release
```

Per-device optimizations (AMD hosts, OpenMP), sanitizer presets and benchmark instructions: [docs/build.md](docs/build.md).

## Commit subjects and changelog

`CHANGELOG.md` is auto-generated by [git-cliff](https://git-cliff.org) from
the git history at every `v*` tag push (see
[.github/workflows/changelog.yml](.github/workflows/changelog.yml) and
[cliff.toml](cliff.toml)). This project uses
[conventional commits](https://www.conventionalcommits.org/en/v1.0.0/):

```text
feat(<scope>): <subject>
fix(<scope>):  <subject>
refactor:      <subject>
bench:         <subject>
ci:            <subject>
docs:          <subject>
chore:         <subject>
test:          <subject>
```

Rules:

- **Subject lines stay plain.** No em-dashes. They print directly into the
  changelog under `### Changed` / `### Added` / `### Fixed`. AI-style
  flourish prose causes clutter in the final `CHANGELOG.md`.
- **One logical change per commit.** Each commit maps to one bullet in
  the next release's `[Unreleased]` section; multi-concern commits read
  as a wall of text.
- **Scope is encouraged**, not required. The scope in parens surfaces
  as `*(scope)*` before the subject, which makes dense releases
  skimmable. Use the area affected: `quantize`, `cli`, `bench`,
  `plan`, `release`, `dinov2`, etc.
- **`chore(deps):` and `chore(release):` are auto-skipped** by cliff.toml.
  Use those prefixes for bumps that don't need changelog coverage.
- **Breaking changes**: add an exclamation after the typeword
  (`feat!:` / `refactor!:`) or include `BREAKING CHANGE:` in the body
  footer. `git-cliff` reads the `!` flag and prefixes the bullet with
  `[**breaking**]` in the changelog.
- **Local regen**: `git-cliff --config cliff.toml --unreleased
  --tag vNEXT --next-tag vNEXT` prints the next release section. Useful
  when drafting release notes for a tag in flight.

Hand-editing `CHANGELOG.md`: edit only the lines **above**
`<!-- git-cliff: end of header -->`. Everything below that marker is
regenerated on every `v*` push and any manual entry there is overwritten.
The footer's release-comparison link table is also a manual section,
marked by `<!-- git-cliff: end of body -->`; add a new row when a new tag
ships.
