# HF publishing setup

The [`convert-and-publish-gguf`](../.github/workflows/convert-and-publish-gguf.yml)
workflow converts the 16 DINOv2 variants to f16 GGUF and uploads them to the
`dinov2-cpp-core/<variant>-gguf` repos on Hugging Face. It runs monthly (cron `0 6 1 * *`)
and on demand via `workflow_dispatch`.

## One-time setup

1. Create a fine-grained token at https://huggingface.co/settings/tokens:
   - Type: Fine-grained.
   - Permissions: `Write` access to **Models**, scoped to the `dinov2-cpp-core` org
     (the workflow creates the repos with `hf repo create --exist_ok` if missing).
2. Add the token as an Actions secret on the GitHub repo:
   ```bash
   gh secret set HF_TOKEN
   ```
3. Trigger a first run to verify:
   ```bash
   gh workflow run convert-and-publish-gguf.yml -f variant=dinov2-small
   ```

## Notes

- Uploads use `hf upload` with `HF_HUB_ENABLE_HF_TRANSFER=1` (`hf_transfer` is
  already in `pyproject.toml`).
- The `validate` job converts `dinov2-small` and checks the GGUF loads but never
  uploads.
- The `giant` variants need ~12-13 GB transient disk, so those jobs free runner
  disk space first.

## The 16 published repos

| Variant | Source checkpoint | HF repo |
|---|---|---|
| small | `facebook/dinov2-small-imagenet1k-1-layer` | `dinov2-cpp-core/dinov2-small-gguf` |
| base | `facebook/dinov2-base-imagenet1k-1-layer` | `dinov2-cpp-core/dinov2-base-gguf` |
| large | `facebook/dinov2-large-imagenet1k-1-layer` | `dinov2-cpp-core/dinov2-large-gguf` |
| giant | `facebook/dinov2-giant-imagenet1k-1-layer` | `dinov2-cpp-core/dinov2-giant-gguf` |
| small (registers) | `facebook/dinov2-with-registers-small-imagenet1k-1-layer` | `dinov2-cpp-core/dinov2-with-registers-small-gguf` |
| base (registers) | `facebook/dinov2-with-registers-base-imagenet1k-1-layer` | `dinov2-cpp-core/dinov2-with-registers-base-gguf` |
| large (registers) | `facebook/dinov2-with-registers-large-imagenet1k-1-layer` | `dinov2-cpp-core/dinov2-with-registers-large-gguf` |
| giant (registers) | `facebook/dinov2-with-registers-giant-imagenet1k-1-layer` | `dinov2-cpp-core/dinov2-with-registers-giant-gguf` |
| small (backbone) | `facebook/dinov2-small` | `dinov2-cpp-core/dinov2-backbone-small-gguf` |
| base (backbone) | `facebook/dinov2-base` | `dinov2-cpp-core/dinov2-backbone-base-gguf` |
| large (backbone) | `facebook/dinov2-large` | `dinov2-cpp-core/dinov2-backbone-large-gguf` |
| giant (backbone) | `facebook/dinov2-giant` | `dinov2-cpp-core/dinov2-backbone-giant-gguf` |
| small (backbone, registers) | `facebook/dinov2-with-registers-small` | `dinov2-cpp-core/dinov2-backbone-with-registers-small-gguf` |
| base (backbone, registers) | `facebook/dinov2-with-registers-base` | `dinov2-cpp-core/dinov2-backbone-with-registers-base-gguf` |
| large (backbone, registers) | `facebook/dinov2-with-registers-large` | `dinov2-cpp-core/dinov2-backbone-with-registers-large-gguf` |
| giant (backbone, registers) | `facebook/dinov2-with-registers-giant` | `dinov2-cpp-core/dinov2-backbone-with-registers-giant-gguf` |

The `dinov2-backbone-*` variants convert the backbone-only checkpoints, so
their GGUFs support feature modes only (`--print-embeddings`,
`--print-patch-tokens`, `-o` PCA, `--bench`) and reject `-c`. See
[cli.md](cli.md#backbone-only-checkpoints).
