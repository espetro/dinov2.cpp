# Register token recommendation documentation design

- **Status:** Approved design, implementation pending
- **Date:** 2026-09-21
- **Scope:** User-facing documentation only

## Goals

1. Help users choose between the existing DINOv2 GGUF variants for their task.
2. Make the four `dinov2-cpp-core/dinov2-with-registers-{small,base,large,giant}-gguf` models the visible recommendation for patch and dense feature workflows.
3. Explain the recommendation with bounded, research-backed language: register tokens reduce high-norm patch-token artifacts and smooth local feature and attention maps in the settings studied by the primary paper.
4. Preserve a clear path for users who need exact baseline reproduction, classification comparisons, or retrieval experiments with the no-register checkpoints.
5. Link readers to the primary research and the official Meta DINOv2 results.

## Non-goals

- No code, model files, conversion scripts, download artifacts, or runtime behavior changes.
- No renaming, removal, or replacement of any existing GGUF repository.
- No claim that register tokens are universally better, faster, or more accurate.
- No new benchmark measurements in this documentation change. Existing measurements must not be presented as evidence for downstream quality unless their task, model, and method are stated.
- No change to CLI flags or output schemas.
- No changes to `docs/build.md`, `docs/hf-publishing.md`, or other documentation outside the four user documentation files listed below.

## Exact files and sections

### `README.md`

Update the `Pre-converted GGUF weights` table at the current model table near the `## Pre-converted GGUF weights` heading.

- Add a `Use` or `Recommendation` column.
- Keep all eight current model rows and their existing download links and size information.
- Mark the four `with-registers` rows as recommended for patch and dense feature use. The recommendation should apply to `--print-patch-tokens`, PCA, dense features, and object discovery.
- Mark the no-register rows as useful for exact baseline reproduction and as valid alternatives for task-specific classification or retrieval comparisons, without making them sound unsupported.
- Add a short note directly below the table that distinguishes feature-map guidance from universal task accuracy guidance.
- Add the primary paper and official Meta results links in the note or a nearby references subsection:
  - [Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588)
  - [official DINOv2 results](https://github.com/facebookresearch/dinov2/blob/main/README.md)

Proposed wording shape, to adapt to the repository's existing voice:

> For patch and dense feature workflows, the register-token variants are the recommended starting point. In the settings studied in [Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588), register tokens reduce high-norm patch-token artifacts and produce smoother local feature and attention maps. Use them with `--print-patch-tokens`, PCA, dense features, and object discovery. This is not a universal accuracy claim: the official [DINOv2 results](https://github.com/facebookresearch/dinov2/blob/main/README.md) show task and model-size-dependent classification and retrieval results. Choose the no-register variant when reproducing an exact baseline or comparing against a no-register checkpoint.

The table cells should be shorter than the note, for example `Recommended for patch/dense features` and `Baseline reproduction or task comparison`. Avoid asserting that either variant wins every classification or retrieval task.

### `docs/cli.md`

Add guidance after `## Install and model download`, before `## Flags`, and add a brief cross-reference near the output mode descriptions for `--print-patch-tokens` and PCA.

The model-download guidance should:

- Show the register-token small model as the default download example for feature-oriented use.
- State that the same naming pattern exists for base, large, and giant.
- Explain that no-register repositories remain available when the user needs a known baseline or exact checkpoint match.
- Link to the paper and official results once, rather than repeating citations throughout the page.

The patch/PCA guidance should be close to the existing descriptions of `--print-patch-tokens` and `-o FNAME`. It should recommend register-token variants for local maps, patch-token inspection, PCA, dense features, and object discovery, while explicitly saying that this recommendation concerns feature quality and visual behavior, not universal classification or retrieval superiority.

Do not imply that registers are included in the emitted `patches` field. Retain the existing statement that register tokens are excluded from the patch view, and clarify only the model-selection recommendation around it.

Suggested concise guidance:

> If you are inspecting patch tokens, generating PCA maps, building dense features, or doing object discovery, start with a `with-registers` checkpoint. The register-token paper reports fewer high-norm patch-token artifacts and smoother local feature and attention maps. For exact no-register baseline reproduction, use the matching no-register checkpoint instead. Classification and retrieval outcomes depend on the task and model size, so this recommendation should not be read as a universal accuracy ranking.

### `docs/benchmarks.md`

Add a subsection near the historical `DINOv2 vs PyTorch` comparison, or immediately before it, titled `## Register and no-register variants`.

This subsection must explain why both variants are benchmarked:

- They are separate published checkpoints with different representations and legitimate use cases.
- Register variants are relevant to patch and dense feature inspection because the paper reports reduced high-norm patch artifacts and smoother local feature and attention maps.
- No-register variants are needed for exact baseline reproduction and fair comparisons with systems or published results built from those checkpoints.
- Runtime benchmark rows should remain comparable by model size, precision, backend, input, and measurement method. Do not combine register and no-register timings into one unlabeled result.

Summarize the evidence without inventing numbers:

- The primary paper provides the motivation for register tokens and reports improvements in feature-map behavior and selected downstream evaluations.
- The official Meta DINOv2 results should be used for classification and retrieval context. Describe those results as mixed or task/model-size dependent rather than as a universal win for either variant.
- Existing historical speed and memory tables are implementation measurements. They do not establish downstream representation quality, and their model names and measurement conditions must remain visible when referenced.

If a table or command example is updated later, label the variant explicitly in the model path or table heading. Include the two reference links in this subsection.

### `CONTRIBUTING.md`

Update the `## Converting weights yourself` source checkpoint table.

- Keep all eight source checkpoint rows.
- Add a `Use` or `Recommendation` column with the same labels and distinction used in `README.md`.
- Mark the register source checkpoints as recommended for patch/dense feature workflows.
- Mark the no-register source checkpoints as baseline reproduction or task-comparison options.
- Add one sentence below the table directing contributors to the paper and official results for the bounded rationale.
- Ensure names, ordering, and labels remain consistent with the README table and the existing Hugging Face links.

## Wording principles

- Say `register-token variants` or `with-registers` when discussing model selection. Do not call them simply `better models`.
- Tie the positive claim to the studied representation behavior: `reduce high-norm patch-token artifacts` and `smooth local feature and attention maps`.
- Use `recommended starting point for patch and dense features` rather than `best` or `always preferred`.
- Name the affected workflows explicitly: `--print-patch-tokens`, PCA, dense features, and object discovery.
- State the limitation in the same section as the recommendation: official classification and retrieval results are task and model-size dependent, and no-register variants remain useful for exact baseline reproduction.
- Keep research claims attributable to the paper or official Meta results. Do not infer a speed, memory, or universal accuracy benefit from the existence of register tokens.
- Use plain prose and existing Markdown conventions. Do not use em dashes, placeholder citations, or vague phrases such as `research proves`.

## Evidence and caveats

Primary evidence:

1. Darcet et al., [Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588), for the register-token motivation and reported effects on high-norm patch artifacts and local feature and attention maps.
2. Meta's [official DINOv2 results](https://github.com/facebookresearch/dinov2/blob/main/README.md), for the classification and retrieval context and the need to avoid a universal ranking claim.

Interpretation constraints:

- The recommendation is specifically about patch and dense feature workflows and the observed quality of local maps. It is not a blanket replacement policy.
- Classification and retrieval comparisons must preserve task, dataset, evaluation protocol, and model size. If those details are not available, use `task/model-size dependent` and do not state which variant wins.
- Exact baseline reproduction requires matching the checkpoint family, including register-token configuration. A no-register checkpoint should remain documented and downloadable.
- Existing speed and memory results must not be used to imply representation quality. If benchmark data compares variants, identify the model variant and measurement conditions.
- Register tokens are excluded from the CLI's emitted patch-token view. The documentation must not suggest that selecting a register checkpoint changes the output schema.

## Validation

Before implementation review:

- Confirm all eight README model rows remain present, links resolve to the same repositories, and only the four `with-registers` rows receive the patch/dense recommendation.
- Confirm the CLI page recommends a register variant for each named feature workflow and separately mentions no-register baseline reproduction.
- Confirm the benchmark page explains why both variants are measured and includes bounded classification/retrieval language without adding unsupported numbers.
- Confirm the contributor source table has the same eight model names, ordering, and recommendation labels as the README table.
- Search the four files for `register`, `with-registers`, `no-register`, `classification`, `retrieval`, `patch`, and `dense` to catch inconsistent terminology.
- Search the diff for em dashes, TODOs, placeholders, `always`, `universally`, `best`, and other unqualified superiority language. Review each match in context.
- Verify that the diff contains no source, model, build, or unrelated documentation changes.
- Run Markdown link and formatting checks available in the repository, if any. No code test run is required because this design concerns documentation only.

## Implementation and review steps

1. Make the documentation edits in `README.md`, `docs/cli.md`, `docs/benchmarks.md`, and `CONTRIBUTING.md` only after this spec is approved.
2. Use the same model labels and recommendation vocabulary in the README and contributor tables.
3. Review each new claim against the two cited sources. Remove any claim that cannot be tied to those sources or to an explicitly labeled local benchmark.
4. Perform the validation checklist above, including a whitespace and diff review.
5. Request review from a documentation owner and a model/evaluation reviewer. The documentation reviewer should check clarity and consistency. The model/evaluation reviewer should check the evidence boundaries, task/model-size caveat, and baseline-reproduction guidance.
6. Land the implementation as a separate conventional commit. This design commit must remain documentation-spec-only.
