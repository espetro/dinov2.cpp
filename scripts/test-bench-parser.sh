#!/usr/bin/env bash
# Focused regression test for the JSON parser embedded in bench.sh.
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
test_dir="$(mktemp -d)"
trap 'rm -rf "$test_dir"' EXIT

mkdir -p "$test_dir/models/dinov2-vit-small-patch14"
touch "$test_dir/models/dinov2-vit-small-patch14/model.gguf" "$test_dir/input.jpg"
cat > "$test_dir/fake-cli" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$BENCH_OUTPUT"
EOF
chmod +x "$test_dir/fake-cli"

run_bench() {
    (cd "$test_dir" && BENCH_OUTPUT="$1" "$repo_dir/scripts/bench.sh" \
        --models small --bin "$test_dir/fake-cli" --image "$test_dir/input.jpg" \
        --out "$test_dir/result.md" >/dev/null)
}

run_bench '{"mean_ms":208.6,"stddev_ms":3.2,"min_ms":206,"max_ms":214,"peak_rss_mb":104}'
gguf_sha256="$(sha256sum "$test_dir/models/dinov2-vit-small-patch14/model.gguf" | cut -d ' ' -f 1)"
grep -Fq "| small | models/dinov2-vit-small-patch14/model.gguf | $gguf_sha256 | 208.6 | 3.2 | 206 | 214 | 104 |" "$test_dir/result.md"
grep -Fq "# Image path: $test_dir/input.jpg" "$test_dir/result.md"
grep -Fq '# Build configuration: unknown' "$test_dir/result.md"

if run_bench '{"mean_ms":208.6,"stddev_ms":3.2,"min_ms":206,"max_ms":214}' 2>/dev/null; then
    echo 'parser accepted a missing required field' >&2
    exit 1
fi

if run_bench 'not-json' 2>/dev/null; then
    echo 'parser accepted malformed JSON' >&2
    exit 1
fi

echo 'bench parser validation passed'
