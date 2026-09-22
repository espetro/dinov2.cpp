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
# transformers + numpy + gguf + huggingface_hub[hf_transfer] + hf_xet), converts
# the model, uploads the resulting ggml-model.gguf to
# dinov2-cpp-core/<variant>-gguf, and then aggressively frees the HF checkpoint
# cache to stay within the ~14 GB disk budget on ubuntu-latest runners.
#
# Every step's stdout/stderr is mirrored to a per-run log file at
# ./.publish-logs/<variant>-<UTC-timestamp>.log so CI failures have an audit
# trail even after the runner is torn down.

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "usage: $0 <variant-short-name>" >&2
    echo "  e.g. $0 dinov2-small" >&2
    exit 2
fi

VARIANT="$1"

# ---- per-run log file (created up front so every step can tee into it) -----
LOG_FILE="./.publish-logs/${VARIANT}-$(date -u +%Y%m%dT%H%M%SZ).log"
mkdir -p "$(dirname "$LOG_FILE")"
START_EPOCH=$(date -u +%s)

# log: write a line to both stdout and the log file (used for human-readable
# step banners; the actual command output is captured by `step` below).
log() { echo "$@"; echo "$@" >> "$LOG_FILE"; }

# step: run a command, mirror its stdout+stderr to the log file, and fail
# loudly if it returns non-zero. Uses a subshell + PIPESTATUS so the tee's
# exit code doesn't mask the underlying command's exit code.
step() {
    log "==> $*"
    local rc
    ( "$@" 2>&1 | tee -a "$LOG_FILE" )
    rc="${PIPESTATUS[0]}"
    [ "$rc" -eq 0 ] || { log "ERROR: step '$*' exited $rc"; exit "$rc"; }
}

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
    dinov2-backbone-small)              HF_ID="facebook/dinov2-small" ;;
    dinov2-backbone-base)               HF_ID="facebook/dinov2-base" ;;
    dinov2-backbone-large)              HF_ID="facebook/dinov2-large" ;;
    dinov2-backbone-giant)              HF_ID="facebook/dinov2-giant" ;;
    dinov2-backbone-with-registers-small)   HF_ID="facebook/dinov2-with-registers-small" ;;
    dinov2-backbone-with-registers-base)    HF_ID="facebook/dinov2-with-registers-base" ;;
    dinov2-backbone-with-registers-large)   HF_ID="facebook/dinov2-with-registers-large" ;;
    dinov2-backbone-with-registers-giant)   HF_ID="facebook/dinov2-with-registers-giant" ;;
    *)
        echo "ERROR: unknown variant '$VARIANT'" >&2
        echo "  known variants: dinov2-{small,base,large,giant} dinov2-with-registers-{small,base,large,giant}" >&2
        echo "                  dinov2-backbone-{small,base,large,giant} dinov2-backbone-with-registers-{small,base,large,giant}" >&2
        exit 2
        ;;
esac

REPO_SLUG="dinov2-cpp-core/${VARIANT}-gguf"
GGUF_PATH="./ggml-model.gguf"
VENV_DIR=".venv-publish"

log "==> publish-gguf.sh starting for variant='$VARIANT' (HF_ID=$HF_ID) at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
log "==> log file: $LOG_FILE"

# ---- HF_TOKEN guard (only when we actually need to upload) -----------------
if [ "${SKIP_UPLOAD:-0}" != "1" ] && [ -z "${HF_TOKEN:-}" ]; then
    log "ERROR: HF_TOKEN is required (set it in the environment or use SKIP_UPLOAD=1)"
    exit 1
fi

# ---- venv setup ------------------------------------------------------------
# We intentionally install ONLY the runtime deps the conversion script needs
# (torch CPU, transformers, numpy, gguf, huggingface_hub with hf_transfer +
# hf_xet). This avoids pulling timm / torchvision / Pillow /
# memory-profiler (the full pyproject deps) which inflates the venv by ~1 GB
# and risks running the runner out of disk.
if [ ! -d "$VENV_DIR" ]; then
    log "==> creating venv at $VENV_DIR"
    step uv venv "$VENV_DIR"
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"

    log "==> installing lean runtime deps (torch CPU + transformers + gguf + hf_xet)"
    # --no-cache-dir keeps pip from retaining wheels in the venv, shaving ~hundreds
    # of MB off the disk footprint. The CPU-only torch index is the same one used
    # in pyproject.toml's [tool.uv.index].
    step uv pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch
    step uv pip install --no-cache-dir \
        transformers \
        numpy \
        'huggingface_hub[hf_transfer]' \
        'hf_xet' \
        'gguf>=0.18.0,<0.20'

    # safetensors is pulled by transformers but pin it explicitly so the wheel is
    # cached under a stable name for the matrix cache step.
    step uv pip install --no-cache-dir safetensors accelerate
else
    log "==> reusing venv at $VENV_DIR"
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
fi

# Make HF transfer explicit inside the script as well — protects against the
# caller forgetting to export it.
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
# Enable hf_xet high-performance mode (Rust dedup + chunked uploader backend).
export HF_XET_HIGH_PERFORMANCE=1

# Pin HF cache to the standard location explicitly so the cleanup step below is
# unambiguous regardless of caller overrides.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# ---- conversion ------------------------------------------------------------
log "==> converting $VARIANT ($HF_ID) -> GGUF"
step python scripts/dinov2-to-gguf.py --model_name "$HF_ID"

if [ ! -s "$GGUF_PATH" ]; then
    log "ERROR: $GGUF_PATH missing or empty after conversion"
    exit 1
fi
SIZE_BYTES=$(stat -c%s "$GGUF_PATH" 2>/dev/null || stat -f%z "$GGUF_PATH")
log "==> wrote $GGUF_PATH ($SIZE_BYTES bytes)"

# ---- verify the GGUF is loadable ------------------------------------------
log "==> verifying GGUF is loadable"
# Heredoc inline — wrap in a subshell + tee so output is captured without
# invoking `step bash -c ...` (which would mangle the heredoc quoting).
( python - <<'PYEOF' 2>&1 | tee -a "$LOG_FILE" )
import gguf
reader = gguf.GGUFReader("./ggml-model.gguf")
tensors = [t.name for t in reader.tensors]
assert len(tensors) > 0, "no tensors in GGUF"
print(f"OK: {len(tensors)} tensors")
PYEOF

# ---- upload ----------------------------------------------------------------
if [ "${SKIP_UPLOAD:-0}" = "1" ]; then
    log "==> SKIP_UPLOAD=1, not uploading to $REPO_SLUG"
else
    log "==> ensuring HF repo $REPO_SLUG exists"
    step hf repo create "$REPO_SLUG" --type model --exist-ok

    log "==> uploading $GGUF_PATH -> $REPO_SLUG (path: model.gguf)"
    step hf upload "$REPO_SLUG" "$GGUF_PATH" model.gguf
fi

# ---- cleanup ---------------------------------------------------------------
# Remove the GGUF artifact; keep the venv so the next matrix leg can reuse it.
rm -f "$GGUF_PATH"

# Eagerly free the HF hub cache (3+ GB for the giant variant) — once the GGUF
# is uploaded we don't need the source checkpoint anymore, and on a 14 GB
# runner every gigabyte matters.
if [ -n "${HF_HOME:-}" ] && [ -d "$HF_HOME/hub" ]; then
    log "==> clearing HF hub cache at $HF_HOME/hub"
    rm -rf "${HF_HOME}/hub/models--facebook--"*
fi
# Also wipe any tmp files the conversion script may have leaked in cwd
# (e.g. partial downloads, .safetensors scratch). This is best-effort.
rm -f tmp_* *.tmp

# ---- summary ----------------------------------------------------------------
END_EPOCH=$(date -u +%s)
ELAPSED=$((END_EPOCH - START_EPOCH))
HF_URL="https://huggingface.co/$REPO_SLUG"
{
    echo ""
    echo "=========================================="
    echo "publish-gguf.sh summary"
    echo "=========================================="
    echo "log file:    $LOG_FILE"
    echo "variant:     $VARIANT"
    echo "hf_id:       $HF_ID"
    echo "repo_slug:   $REPO_SLUG"
    echo "hf_url:      $HF_URL"
    if [ "${SKIP_UPLOAD:-0}" = "1" ]; then
        echo "gguf_size:   not uploaded (SKIP_UPLOAD=1)"
        echo "uploaded:    no"
    else
        echo "gguf_size:   $SIZE_BYTES bytes"
        echo "uploaded:    yes"
    fi
    echo "elapsed_sec: $ELAPSED"
    echo "finished_at: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "=========================================="
} >> "$LOG_FILE"

# Echo the log path to stdout so CI logs surface it on every run.
echo "==> done: $VARIANT"
echo "==> full log: $LOG_FILE"
