# HF publishing setup

The [`convert-and-publish-gguf`](../.github/workflows/convert-and-publish-gguf.yml)
workflow converts the 8 DINOv2 variants to f16 GGUF and uploads them to the
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

## The 8 published repos

| Variant | HF repo |
|---|---|
| small | `dinov2-cpp-core/dinov2-small-gguf` |
| base | `dinov2-cpp-core/dinov2-base-gguf` |
| large | `dinov2-cpp-core/dinov2-large-gguf` |
| giant | `dinov2-cpp-core/dinov2-giant-gguf` |
| small (registers) | `dinov2-cpp-core/dinov2-with-registers-small-gguf` |
| base (registers) | `dinov2-cpp-core/dinov2-with-registers-base-gguf` |
| large (registers) | `dinov2-cpp-core/dinov2-with-registers-large-gguf` |
| giant (registers) | `dinov2-cpp-core/dinov2-with-registers-giant-gguf` |
