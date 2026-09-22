# Marketing drafts

Launch collateral for dinov2.cpp. Drafts only; nothing here is published
yet. Every number in these files was verified against a repo artifact or a
live HTTP check on 2026-09-23; re-verify before publishing if time passes.

## Contents

| File | Purpose |
|:-----|:--------|
| [hf-model-card-template.md](hf-model-card-template.md) | README/model-card template for the 8 `dinov2-cpp-core/*-gguf` HF repos (they ship only `model.gguf` + `.gitattributes` today, verified via the HF API). Includes a measured size lookup table. |
| [show-hn-draft.md](show-hn-draft.md) | Show HN headline options and ~150-word body, restricted to verified numbers, with a source table. |
| [blog-post-outline.md](blog-post-outline.md) | Full launch-post outline: hook, positioning, dedup-demo centerpiece, parity evidence, caveats draft, CTA. |

Related user-facing page (not marketing-internal): [../trust.md](../trust.md).

## Owner checklist (external steps, in rough order)

- [ ] **Upload the HF model cards.** Render `hf-model-card-template.md`
  once per repo (8 repos, lookup table at the bottom of the template) and
  upload as `README.md`. Needs an HF token with write access to
  `dinov2-cpp-core` (the `HF_TOKEN` secret already used by
  `convert-and-publish-gguf`; see `docs/hf-publishing.md`):
  `hf upload dinov2-cpp-core/<repo> README.md README.md --repo-type model`.
  The YAML frontmatter must be the first bytes of the file.
- [ ] **Enable GitHub Pages** so the wasm demo (`wasm/index.html`) gets a
  live URL the Show HN and blog post can link to. Requires copying
  `dinov2-wasm.js` + `dinov2-wasm.wasm` plus a small GGUF next to the page
  per `docs/wasm.md`, or wiring CI to publish them.
- [ ] **Set repo metadata**: GitHub topics (`ggml`, `gguf`, `dinov2`,
  `cpp`, `embeddings`, `vision-transformer`), the one-line description, and
  a social-preview image (Settings -> General -> Social preview;
  `assets/logo/logo-256.png` exists as a starting point).
- [ ] **Post the Show HN** from `show-hn-draft.md` after Pages and the HF
  cards are live, so every link in the post resolves.
- [ ] **Publish the blog post** from `blog-post-outline.md` (personal site
  first, then cross-posts).
- [ ] **Testimonial permission**: if quoting a user or collaborator in any
  of the above, get explicit written permission first.

## Standing rules for this content

- No hype words, no invented numbers. If a number is not in
  `benchmark_results.txt`, `docs/parity/`, a release listing, or a live
  check, it does not go in.
- The historical PyTorch comparison table is labeled historical; never
  present it as current evidence.
- Backbone-only GGUFs are not published; do not imply they exist.
