#!/usr/bin/env bash
# Convert a DINOv2 HuggingFace checkpoint to GGUF and optionally publish to the
# `dinov2-cpp-core` HuggingFace organisation.
#
# Usage:
#   bash scripts/publish-gguf.sh <variant-short-name>
#
# Environment:
#   HF_TOKEN        Required unless SKIP_UPLOAD=1. Token used by `hf` CLI for
#                   repo creation and upload.
#   SKIP_UPLOAD     If set to 1, runs the conversion but skips `hf repo create`
#                   and `hf upload`. Used by CI's validate job.
#
# The script creates a project-local venv at .venv-publish/ (reused across runs
# to avoid reinstalling torch/transformers on every matrix leg), installs the
# minimal runtime deps needed by scripts/dinov2-to-gguf.py (torch CPU +
# transformers + numpy + gguf + huggingface_hub[hf_transfer]), converts the
# model, uploads the resulting ggml-model.gguf to
# dinov2-cpp-core/<variant>-gguf, and then aggressively frees the HF checkpoint
# cache to stay within the ~14 GB disk budget on ubuntu-latest runners.

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "usage: $0 <variant-short-name>" >&2
    echo "  e.g. $0 dinov2-small" >&2
    exit 2
fi

VARIANT="$1"

# Map variant short-name -> facebookresearch HF checkpoint id.
case "$VARIANT" in
    dinov2-small)                       HF_ID="facebook/dinov2-small-imagenet1k-1-layer" ;;
    dinov2-base)                        HF_ID="facebook/dinov2-base-imagenet1k-1-layer" ;;
    dinov2-large)                       HF_ID="facebook/dinov2-large-imagenet1k-1-layer" ;;
    dinov2-giant)                       HF_ID="facebook/dinov2-giant-imagenet1k-1-layer" ;;
    dinov2-with-registers-small)        HF_ID="facebook/dinov2-with-registers-small-imagenet1k-1-layer" ;;
    dinov2-with-registers-base)         HF_ID="facebook/dinov2-with-registers-base-imagenet1k-1-layer" ;;
    dinov2-with-registers-large)        HF_ID="facebook/dinov2-with-registers-large-imagenet1k-1-layer" ;;
    dinov2-with-registers-giant)        HF_ID="facebook/dinov2-with-registers-giant-imagenet1k-1-layer" ;;
    *)
        echo "ERROR: unknown variant '$VARIANT'" >&2
        echo "  known variants: dinov2-{small,base,large,giant} dinov2-with-registers-{small,base,large,giant}" >&2
        exit 2
        ;;
esac

REPO_SLUG="dinov2-cpp-core/${VARIANT}-gguf"
GGUF_PATH="./ggml-model.gguf"
VENV_DIR=".venv-publish"

# ---- HF_TOKEN guard (only when we actually need to upload) -----------------
if [ "${SKIP_UPLOAD:-0}" != "1" ] && [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN is required (set it in the environment or use SKIP_UPLOAD=1)" >&2
    exit 1
fi

# ---- venv setup ------------------------------------------------------------
# We intentionally install ONLY the runtime deps the conversion script needs
# (torch CPU, transformers, numpy, gguf, huggingface_hub with hf_transfer) plus
# the project's local `dinov2_inference` package via PYTHONPATH=src. This avoids
# pulling timm / torchvision / Pillow / memory-profiler (the full pyproject deps)
# which inflates the venv by ~1 GB and risks running the runner out of disk.
if [ ! -d "$VENV_DIR" ]; then
    echo "==> creating venv at $VENV_DIR"
    uv venv "$VENV_DIR"
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"

    echo "==> installing lean runtime deps (torch CPU + transformers + gguf)"
    # --no-cache-dir keeps pip from retaining wheels in the venv, shaving ~hundreds
    # of MB off the disk footprint. The CPU-only torch index is the same one used
    # in pyproject.toml's [tool.uv.index].
    uv pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch
    uv pip install --no-cache-dir \
        transformers \
        numpy \
        'huggingface_hub[hf_transfer]' \
        'gguf>=0.18.0,<0.20'

    # safetensors is pulled by transformers but pin it explicitly so the wheel is
    # cached under a stable name for the matrix cache step.
    uv pip install --no-cache-dir safetensors accelerate
else
    echo "==> reusing venv at $VENV_DIR"
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
fi

# Make the project's `dinov2_inference.types.GGMLNumpyType` importable without
# installing the whole project (which would re-introduce timm/torchvision/Pillow).
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

# Make HF transfer explicit inside the script as well — protects against the
# caller forgetting to export it.
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

# Pin HF cache to the standard location explicitly so the cleanup step below is
# unambiguous regardless of caller overrides.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# ---- conversion ------------------------------------------------------------
echo "==> converting $VARIANT ($HF_ID) -> GGUF"
python scripts/dinov2-to-gguf.py --model_name "$HF_ID"

if [ ! -s "$GGUF_PATH" ]; then
    echo "ERROR: $GGUF_PATH missing or empty after conversion" >&2
    exit 1
fi
SIZE_BYTES=$(stat -c%s "$GGUF_PATH" 2>/dev/null || stat -f%z "$GGUF_PATH")
echo "==> wrote $GGUF_PATH ($SIZE_BYTES bytes)"

# ---- verify the GGUF is loadable ------------------------------------------
echo "==> verifying GGUF is loadable"
python - <<'PYEOF'
import gguf
reader = gguf.GGUFReader("./ggml-model.gguf")
tensors = [t.name for t in reader.tensors]
assert len(tensors) > 0, "no tensors in GGUF"
print(f"OK: {len(tensors)} tensors")
PYEOF

# ---- upload ----------------------------------------------------------------
if [ "${SKIP_UPLOAD:-0}" = "1" ]; then
    echo "==> SKIP_UPLOAD=1, not uploading to $REPO_SLUG"
else
    echo "==> ensuring HF repo $REPO_SLUG exists"
    hf repo create "$REPO_SLUG" --type model --exist-ok

    echo "==> uploading $GGUF_PATH -> $REPO_SLUG (path: model.gguf)"
    hf upload "$REPO_SLUG" "$GGUF_PATH" model.gguf
fi

# ---- cleanup ---------------------------------------------------------------
# Remove the GGUF artifact; keep the venv so the next matrix leg can reuse it.
rm -f "$GGUF_PATH"

# Eagerly free the HF hub cache (3+ GB for the giant variant) — once the GGUF
# is uploaded we don't need the source checkpoint anymore, and on a 14 GB
# runner every gigabyte matters.
if [ -n "${HF_HOME:-}" ] && [ -d "$HF_HOME/hub" ]; then
    echo "==> clearing HF hub cache at $HF_HOME/hub"
    rm -rf "${HF_HOME}/hub/models--facebook--"*
fi
# Also wipe any tmp files the conversion script may have leaked in cwd
# (e.g. partial downloads, .safetensors scratch). This is best-effort.
rm -f tmp_* *.tmp
echo "==> done: $VARIANT"
