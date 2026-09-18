#!/usr/bin/env bash
# DEPRECATED: scripts/benchmark.sh was replaced by scripts/bench.sh (PR B).
# This shim just forwards to bench.sh so existing invocations keep working.
# The old positional interface (`./scripts/benchmark.sh <threads> <quantize_flag>`)
# is no longer supported -- pass --threads / --quants explicitly to bench.sh.
echo "scripts/benchmark.sh is deprecated; use scripts/bench.sh instead." >&2
exec "$(dirname "$0")/bench.sh" "$@"