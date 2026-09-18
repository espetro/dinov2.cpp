#!/usr/bin/env bash
# scripts/bench.sh - canonical bench entry point for dinov2.cpp.
#
# Replaces the old scripts/benchmark.sh (now a deprecation shim).
# Produces a markdown table at --out (default ./benchmark_results.md) by
# running the inference binary with --bench across a (models x quants)
# matrix.
#
# Usage:
#   scripts/bench.sh                                # default sweep
#   scripts/bench.sh --models small,base --quants f16,q4_0 --repeats 5
#   scripts/bench.sh --aggregate ./per-platform/*.txt --out benchmark_results.txt
#
# Flags:
#   --models <csv>     comma-separated model sizes: small,base,large,giant
#                      (default: small,base,large,giant)
#   --quants <csv>     comma-separated quants: f16,q4_0,q4_1,q5_0,q5_1,q8_0
#                      (default: f16)
#   --repeats <N>      timed runs per cell (default: 5)
#   --threads <N>      -t argument forwarded to inference (default: 12)
#   --warmup <N>       --bench-warmup forwarded to inference (default: 1)
#   --image <PATH>     input image (default: assets/tench.jpg)
#   --bin <PATH>       path to inference binary (default: build/bin/inference)
#   --quantize <PATH>  path to quantize binary (default: build/bin/quantize)
#   --out <PATH>       output markdown table (default: ./benchmark_results.md)
#   --platform <NAME>  platform label for the table header (default: detected)
#   --aggregate <globs> --out <PATH>  aggregate mode: concatenate per-platform
#                                     *-results.txt files with platform headers.
#                                     --models/--quants/--repeats/--threads are
#                                     ignored in this mode.
#
# Exit codes:
#   0  success
#   1  invocation error
#   2  a required model GGUF is missing (recoverable: run scripts/publish-gguf.sh
#      or download from dinov2-cpp-core/*-gguf first)

set -euo pipefail

# ----------------- defaults -----------------
MODELS_DEFAULT="small,base,large,giant"
QUANTS_DEFAULT="f16"
REPEATS_DEFAULT=5
THREADS_DEFAULT=12
WARMUP_DEFAULT=1
IMAGE_DEFAULT="assets/tench.jpg"
BIN_DEFAULT="build/bin/inference"
QUANTIZE_DEFAULT="build/bin/quantize"
OUT_DEFAULT="./benchmark_results.md"
PLATFORM_DEFAULT="$(uname -s | tr '[:upper:]' '[:lower:]')-$(uname -m)"

# Per-quant ggml ftype id (matches scripts/benchmark.sh historical values).
# Written as a function rather than an associative array for bash 3.x
# compatibility (macOS default shell).
quant_id() {
    case "$1" in
        q4_0) echo 2 ;;
        q4_1) echo 3 ;;
        q5_0) echo 6 ;;
        q5_1) echo 7 ;;
        q8_0) echo 8 ;;
        *)    return 1 ;;
    esac
}

# ----------------- argparse -----------------
usage() {
    sed -n '2,32p' "$0" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

MODELS=""
QUANTS=""
REPEATS=""
THREADS=""
WARMUP=""
IMAGE=""
BIN_PATH=""
QUANTIZE_BIN=""
OUT_PATH=""
PLATFORM_LABEL=""
AGGREGATE_INPUTS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage 0 ;;
        --models)        MODELS="${2:?}"; shift 2 ;;
        --quants)        QUANTS="${2:?}"; shift 2 ;;
        --repeats)       REPEATS="${2:?}"; shift 2 ;;
        --threads)       THREADS="${2:?}"; shift 2 ;;
        --warmup)        WARMUP="${2:?}"; shift 2 ;;
        --image)         IMAGE="${2:?}"; shift 2 ;;
        --bin)           BIN_PATH="${2:?}"; shift 2 ;;
        --quantize)      QUANTIZE_BIN="${2:?}"; shift 2 ;;
        --out)           OUT_PATH="${2:?}"; shift 2 ;;
        --platform)      PLATFORM_LABEL="${2:?}"; shift 2 ;;
        --aggregate)     shift; while [[ $# -gt 0 && "$1" != --* ]]; do AGGREGATE_INPUTS+=("$1"); shift; done ;;
        *)               echo "bench.sh: unknown argument: $1" >&2; usage 1 ;;
    esac
done

MODELS="${MODELS:-$MODELS_DEFAULT}"
QUANTS="${QUANTS:-$QUANTS_DEFAULT}"
REPEATS="${REPEATS:-$REPEATS_DEFAULT}"
THREADS="${THREADS:-$THREADS_DEFAULT}"
WARMUP="${WARMUP:-$WARMUP_DEFAULT}"
IMAGE="${IMAGE:-$IMAGE_DEFAULT}"
BIN_PATH="${BIN_PATH:-$BIN_DEFAULT}"
QUANTIZE_BIN="${QUANTIZE_BIN:-$QUANTIZE_DEFAULT}"
OUT_PATH="${OUT_PATH:-$OUT_DEFAULT}"
PLATFORM_LABEL="${PLATFORM_LABEL:-$PLATFORM_DEFAULT}"

# ----------------- aggregate mode -----------------
if [[ ${#AGGREGATE_INPUTS[@]} -gt 0 ]]; then
    : > "$OUT_PATH"
    for f in "${AGGREGATE_INPUTS[@]}"; do
        if [[ ! -f "$f" ]]; then
            echo "::error::aggregate input not found: $f" >&2
            exit 1
        fi
        # Platform label = filename stem before "-results.txt"
        file_stem="${f##*/}"
        file_stem="${file_stem%-results.txt}"
        {
            echo "# ===== platform: ${file_stem} ====="
            cat "$f"
            echo ""
        } >> "$OUT_PATH"
    done
    echo "bench.sh: aggregated ${#AGGREGATE_INPUTS[@]} files -> $OUT_PATH" >&2
    exit 0
fi

# ----------------- main bench mode -----------------
if [[ ! -x "$BIN_PATH" ]]; then
    echo "::error::inference binary not found at $BIN_PATH. Build first (cmake --build build --target inference)." >&2
    exit 1
fi

if [[ ! -f "$IMAGE" ]]; then
    echo "::error::input image not found at $IMAGE." >&2
    exit 1
fi

IFS=',' read -ra MODEL_ARR <<< "$MODELS"
IFS=',' read -ra QUANT_ARR <<< "$QUANTS"

# Header (markdown table file).
{
    echo "# Generated by scripts/bench.sh on $(date -u +%Y-%m-%dT%H:%M:%SZ). Do not edit by hand."
    echo "# Methodology: ${REPEATS} timed runs after ${WARMUP} warmup on a single image (${IMAGE}). See docs/benchmarks.md."
    echo "# Platform: ${PLATFORM_LABEL}"
    echo ""
    # Two header styles depending on whether any non-f16 quants are requested.
    if [[ ${#QUANT_ARR[@]} -gt 1 || "${QUANT_ARR[0]}" != "f16" ]]; then
        echo "| Model | Quant | mean (ms) | stddev (ms) | min (ms) | max (ms) | peak RSS (MB) |"
        echo "|:------|:------|----------:|------------:|---------:|---------:|---------------:|"
    else
        echo "| Model | mean (ms) | stddev (ms) | min (ms) | max (ms) | peak RSS (MB) |"
        echo "|:------|----------:|------------:|---------:|---------:|---------------:|"
    fi
} > "$OUT_PATH"

# Per-model, per-quant bench cells.
for v in "${MODEL_ARR[@]}"; do
    # Full model name: dinov2-vit-{size}-patch14
    full_model="dinov2-vit-${v}-patch14"
    base_gguf="models/${full_model}/model.gguf"

    if [[ ! -f "$base_gguf" ]]; then
        echo "::error::Missing ${base_gguf}. Run scripts/publish-gguf.sh or download from dinov2-cpp-core/*-gguf first." >&2
        exit 2
    fi

    for q in "${QUANT_ARR[@]}"; do
        cell_gguf="$base_gguf"
        if [[ "$q" != "f16" ]]; then
            if ! qid="$(quant_id "$q")"; then
                echo "::error::unknown quant type: $q (supported: f16, q4_0, q4_1, q5_0, q5_1, q8_0)" >&2
                exit 1
            fi
            cell_gguf="models/${full_model}/model.${q}.gguf"
            if [[ ! -f "$cell_gguf" ]]; then
                if [[ ! -x "$QUANTIZE_BIN" ]]; then
                    echo "::error::quantize binary not found at $QUANTIZE_BIN. Build first." >&2
                    exit 1
                fi
                echo "bench.sh: quantizing ${full_model} -> ${q}..." >&2
                (cd "$(dirname "$QUANTIZE_BIN")" && ./quantize "../../${base_gguf}" "../../${cell_gguf}" "$qid" >/dev/null)
            fi
        fi

        echo "bench.sh: running ${full_model} quant=${q} repeats=${REPEATS} threads=${THREADS}..." >&2

        # One JSON line per bench run; then average. We use --bench-json to get
        # structured output, then drop the bench_warmup samples and compute
        # summary stats here (the C++ side already aggregates, but emitting
        # raw samples makes the script robust to format changes).
        bench_json="$(mktemp)"
        trap 'rm -f "$bench_json"' EXIT

        if ! "$BIN_PATH" \
                -m "$cell_gguf" \
                -i "$IMAGE" \
                -t "$THREADS" \
                -c \
                --bench-runs "$REPEATS" \
                --bench-warmup "$WARMUP" \
                --bench-json \
                > "$bench_json" 2>/dev/null; then
            echo "::error::inference failed for ${full_model} (${q})" >&2
            exit 1
        fi

        # Parse the single JSON object line with python (avoid jq dep).
        # Disable pipefail + errexit for this substitution so a python3-not-found
        # error prints clearly instead of aborting with "no such file".
        set +e +o pipefail
        bench_stats="$(python3 -c '
import json, sys
data = json.loads(open("'"$bench_json"'").read().strip())
print(f"{data[\"mean_ms\"]:.1f} {data[\"stddev_ms\"]:.1f} {data[\"min_ms\"]:.0f} {data[\"max_ms\"]:.0f} {data[\"peak_rss_mb\"]:.0f}")
' 2>/dev/null)" || bench_stats=""
        set -e -o pipefail
        if [[ -z "$bench_stats" ]]; then
            echo "::error::could not parse bench output for ${full_model} (${q})" >&2
            cat "$bench_json" >&2
            exit 1
        fi
        read -r mean_ms stddev_ms min_ms max_ms peak_rss_mb <<<"$bench_stats"

        if [[ ${#QUANT_ARR[@]} -gt 1 || "${QUANT_ARR[0]}" != "f16" ]]; then
            printf "| %s | %s | %s | %s | %s | %s | %s |\n" \
                "$v" "$q" "$mean_ms" "$stddev_ms" "$min_ms" "$max_ms" "$peak_rss_mb" \
                >> "$OUT_PATH"
        else
            printf "| %s | %s | %s | %s | %s | %s |\n" \
                "$v" "$mean_ms" "$stddev_ms" "$min_ms" "$max_ms" "$peak_rss_mb" \
                >> "$OUT_PATH"
        fi
    done
done

{
    echo ""
    echo "# Notes:"
    echo "# - \"mean (ms)\" is the forward-pass wall time of the model graph only."
    echo "#   End-to-end wall time (image load + preprocess + classify + PCA) is higher."
    echo "# - \"peak RSS\" is reported by the inference binary via getrusage(RUSAGE_SELF)."
    echo "# - Numbers will be regenerated by the bench.yml manual-trigger workflow"
    echo "#   once PR C lands (see .agents/plans/2026-09-18-post-v0.2.0.md)."
} >> "$OUT_PATH"

echo "bench.sh: wrote $(wc -l < "$OUT_PATH") rows to $OUT_PATH" >&2