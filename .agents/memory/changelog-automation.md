# changelog-automation.md

## Tooling landed

The repo now ships a bot-driven CHANGELOG.md via `git-cliff 2.14.1`:

- `cliff.toml` — Keep-A-Changelog 1.1.0 commit_parsers. Maps
  conventional-commit types to Keep-A-Changelog subsections: feat ->
  Added, fix -> Fixed, perf/refactor/bench/ci/docs/test/style/chore/revert
  -> Changed, deprecate -> Deprecated, security -> Security, non-conventional
  -> Miscellaneous. Skips `chore(release):`/`chore(pr):`/`chore(pull):`/
  `chore(deps):`. `filter_unconventional = false` so the upstream
  lavaman131 fork history (pre-conventional-commits) does not silently
  drop from the changelog.
- `.github/workflows/changelog.yml` — fires on `push: tags: v*`,
  serialized with `release.yml` via `concurrency:
  group: ci-${{ github.workflow }}-${{ github.ref }}` so the two
  workflows do not race on the same tag. Uses
  `taiki-e/install-action@v2` to install `git-cliff@2`, then
  `git-cliff --unreleased --tag ${{ github.ref_name }} --next-tag
  ${{ github.ref_name }}`. If the regenerated CHANGELOG.md differs
  from committed, opens a PR via `peter-evans/create-pull-request@v6`
  with branch `changelog/${{ github.ref_name }}`.
- The header (`<!-- git-cliff: end of header -->`) and footer
  (`<!-- git-cliff: end of body -->`) markers in cliff.toml protect
  the manually-maintained prose from regeneration. The body between
  the markers is the bot-driven section.
- `CONTRIBUTING.md` documents the editorial contract: one
  conventional commit per bullet, scopes encouraged,
  `feat!:`/`BREAKING CHANGE:` footer for breaking changes, no em-dashes
  in commit subjects (global AGENTS.md rule extended to commits).

## Key dates (from GitHub Releases API, not hand-authored)

- v0.1.0 published 2026-09-18T17:19:28Z (5-platform binaries incl.
  `dinov2-bin-win-cpu-x64.zip` without version prefix; tag is the
  source of truth). Body: `feat: v0.1.0 - drop OpenCV, cross-platform
  CI + releases, dev harness by @espetro`.
- v0.2.0 published 2026-09-18. Renamed Ubuntu binaries to use
  `v0.2.0-` version prefix; Windows binary kept the no-version name
  until the next cleanup.
- v0.3.0 not yet tagged (local tag was created for git-cliff backfill
  verification, then deleted).

## Why this matters

- Hand-authoring CHANGELOG.md leads to the v0.1.0 placeholder bug:
  forgetting the date of an N-month-old tag. `git-cliff` reads the
  annotated-tag's commit timestamp, so the date is always right.
- All four releases (v0.1.0 / v0.2.0 / v0.3.0 / v0.4.0 / etc.)
  ship on the same day from this single session; future maintainers
  can read CHANGELOG.md without re-fetching from the GitHub API.

## Repository topics (set 2026-09-18 via `gh api PUT /repos/.../topics`)

Applied via the GitHub API by `espetro` (actor in audit log):

```
dinov2, ggml, gguf, cpp, transformer, computer-vision,
inference, image-classification, self-supervised-learning,
cmake, huggingface
```

Research-driven rationale: standard GitHub topic taxonomy only, all
popular (104k repos for `cpp` down to 250 for `dinov2`), no spam
topics (`machine-learning`, `llm`, `awesome` rejected), and no
unsupported-shipped-as-artifact topics (`cuda`, `metal` rejected —
ggml supports them but no GPU prebuilt is shipped).

## State at last touch

- HEAD on `main` is 17 commits ahead of `origin/main` (none pushed
  yet): 6 from earlier work + 5 v0.3.0 cleanup + 5 changelog-automation
  (regen + cliff.toml + workflow + CONTRIBUTING + comparison links +
  README topics) + 1 README topic-docs commit.
- Local v0.3.0 tag deleted; tag is reapplied when the actual release
  ships.
- Git-cliff binary at `/Users/josocjoq/.cargo/bin/git-cliff` (installed
  via `cargo binstall`, not committed to project deps).

## Plan: PR C (manual bench.yml workflow)

Still TODO from the original v0.3.0 plan:
- `.github/workflows/bench.yml` with `on: workflow_dispatch`, that
  fetches a model from HF, runs `scripts/bench.sh`, and uploads the
  output as an artifact for review.
- Must reference `dinov2-cli` (post-rename) and have no quantize
  step (post-drop).
- Should run on the same ubuntu-latest-large runners as
  convert-and-publish-gguf.yml for consistency.

## GPU CI

Deferred per global AGENTS.md guidance. Re-enable checklist:
- `ggml/CMakeLists.txt` for CUDA:11.8/12.x Metal toolchain.
- `CMakeLists.txt` `--ggml-cuda` / `--ggml-metal` options.
- Runners with GPU labels (ubuntu-latest-gpu / macos-latest-arm64).
- README line 60 ("CPU, CUDA, Metal") will need clarification about
  which backends ship vs. which are accessible from source.
