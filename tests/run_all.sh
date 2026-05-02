#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# Run the full ion7-llm test suite.
#
# Files starting with `0` need no model (pure-Lua + module-load tests),
# files starting with `1` need a generation model (ION7_MODEL), files
# starting with `2`+ may need extra resources (embedding model, draft
# model). Discovery is alphabetic so the prefix dictates ordering.
#
# Usage:
#   ION7_MODEL=/path/to/model.gguf bash tests/run_all.sh
#
# Optional environment:
#   ION7_CORE_SRC    Override the ion7-core source root (defaults to
#                    a sibling checkout under ../ion7-core/src/).
#   ION7_EMBED       Embedding model (.gguf) for the embedding suite.
#   ION7_DRAFT       Draft model (.gguf) for speculative decoding tests.
#   ION7_GPU_LAYERS  Override n_gpu_layers (default 0 — pure CPU).
#   ION7_SKIP        Whitespace-separated list of file basenames to skip.
# ──────────────────────────────────────────────────────────────────────────

set -e

PASS=0
FAIL=0
SKIP=0

cd "$(dirname "$0")/.."

run_suite() {
    local file="$1"
    local name
    name="$(basename "$file" .lua)"

    if [ -n "$ION7_SKIP" ] && echo "$ION7_SKIP" | grep -qw "$(basename "$file")"; then
        printf "\n\033[33m══ SKIP %-40s\033[0m\n" "$name (in ION7_SKIP)"
        SKIP=$((SKIP + 1))
        return
    fi

    printf "\n\033[1m══ %-40s \033[0m%s\n" "$name" "$(printf '═%.0s' {1..18})"
    if ION7_MODEL="$ION7_MODEL" \
       ION7_CORE_SRC="$ION7_CORE_SRC" \
       ION7_EMBED="$ION7_EMBED" \
       ION7_DRAFT="$ION7_DRAFT" \
       ION7_GPU_LAYERS="$ION7_GPU_LAYERS" \
       luajit "$file"; then
        PASS=$((PASS + 1))
    else
        FAIL=$((FAIL + 1))
    fi
}

for f in tests/[0-9][0-9]_*.lua; do
    [ -f "$f" ] || continue
    run_suite "$f"
done

# ── Summary ──────────────────────────────────────────────────────────────
printf "\n\033[1m%s\033[0m\n" "$(printf '═%.0s' {1..60})"
printf "  Suites: \033[32m%d passed\033[0m" "$PASS"
[ "$FAIL" -gt 0 ] && printf "  \033[31m%d FAILED\033[0m" "$FAIL"
[ "$SKIP" -gt 0 ] && printf "  \033[33m%d skipped\033[0m" "$SKIP"
printf "\n\033[1m%s\033[0m\n" "$(printf '═%.0s' {1..60})"

[ "$FAIL" -eq 0 ]
