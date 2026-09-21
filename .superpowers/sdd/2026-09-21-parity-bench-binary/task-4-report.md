# Task 4 report: Ubuntu benchmark evidence

## Status

Blocked. No trustworthy benchmark result or artifact was produced, so the placeholder evidence was not replaced and no benchmark documentation claims were changed.

## Repository and workflow provenance

- Local branch: `feat/batched-inference`
- Local HEAD: `6547bd7bfbe39232cf568215db6262d6178020fa`
- Published `origin/feat/batched-inference`: `12f15a521a259f6273d8662bcf4ef8a465dbc784`
- The local branch is seven commits ahead of its published remote branch. No push was performed.
- The dispatched workflow therefore benchmarked the published remote SHA `12f15a521a259f6273d8662bcf4ef8a465dbc784`, not the current local code. This run cannot be used as evidence for the unpushed local changes.
- Workflow: `.github/workflows/bench.yml`, `ubuntu-latest`, manual dispatch.

## Commands and run

Authentication was available:

```text
gh auth status
  Logged in to github.com account espetro
  Token scopes: gist, project, read:org, repo, workflow
```

The smallest useful run was dispatched first:

```text
gh workflow run bench.yml --ref feat/batched-inference -f variant=small -f repeats=1 -f threads=2
```

Run: [35628865190](https://github.com/espetro/dinov2.cpp/actions/runs/35628865190)

The run was monitored with a bounded `gh run watch 35628865190 --interval 10 --exit-status`. It failed after 51 seconds during `Download f16 GGUF(s)` before benchmark execution or artifact upload.

## Exact blocker

The workflow's download step ran:

```text
huggingface-cli download --repo-type model --local-dir models/dinov2-vit-small-patch14 dinov2-cpp-core/dinov2-small-gguf model.gguf
```

The Ubuntu log shows:

```text
Warning: `huggingface-cli` is deprecated and no longer works. Use `hf` instead.
Hint: `hf` is already installed! Use it directly.
Process completed with exit code 1.
```

The command exited before `Verify staged weights`, `Run benchmarks`, and `Upload benchmark results`. `gh run view 35628865190 --log-failed` contains no downloaded model path, benchmark output, or artifact. No artifact was downloaded because none was uploaded.

## Results and coverage

- Actual benchmark measurements: none.
- Models measured: none; the requested small f16 GGUF could not be staged.
- Register-token variants: unmeasured.
- Precision/backend: no measurement; the workflow is intended for f16 GGUF on the CPU backend.
- Warmup/repeats/threads: the requested run was 1 repeat, 2 threads, with the script default of 1 warmup, but execution never reached `scripts/bench.sh`.
- No local Mac benchmark was substituted for Ubuntu evidence.
- No full or giant sweep was attempted.

## Limitations and follow-up

The workflow must replace the deprecated `huggingface-cli` invocation with the installed `hf download` command, or otherwise pin a working Hugging Face CLI invocation, before another Ubuntu evidence run can proceed. After that fix is published, dispatch at least the small model again and verify the artifact and provenance before replacing placeholder data. The local seven-commit divergence must also be resolved by publishing the code to be measured; this task did not push it.

No models, build outputs, or release artifacts were added to the repository.
