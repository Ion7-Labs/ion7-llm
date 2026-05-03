#!/usr/bin/env bash
#
# publish.sh — publish the current Ion7 rockspec to luarocks.org.
#
# Reads the api key from the LUAROCKS_API_KEY environment variable.
# Never accept it as an argument — that would leave it in the shell
# history and `ps` output.
#
# Steps :
#   1. Locate the rockspec next to this script's repo root.
#   2. Refuse to publish if the rockspec's `source.tag` is not present
#      on `origin` (you forgot `git push --tags`, presumably).
#   3. `luarocks pack` to make sure the rockspec actually fetches.
#   4. `luarocks upload` (--api-key kept out of the visible command).
#
# Usage :
#   LUAROCKS_API_KEY=$(cat ~/.config/ion7/luarocks-key) bash bin/publish.sh
#   LUAROCKS_API_KEY=xxx bash bin/publish.sh --force        # overwrite an existing version on luarocks
#   LUAROCKS_API_KEY=xxx bash bin/publish.sh --pack-only    # produce the .src.rock without uploading
#
# Flags :
#   --force        passes --force to `luarocks upload` (re-publish
#                  the same version-revision)
#   --pack-only    stop after `luarocks pack` ; useful for verifying
#                  the rockspec without touching the public manifest

set -euo pipefail

if [ "${LUAROCKS_API_KEY:-}" = "" ]; then
    cat <<EOF >&2
[publish] LUAROCKS_API_KEY is not set.

Set it from a file you keep outside the repo, e.g.

    LUAROCKS_API_KEY=\$(cat ~/.config/ion7/luarocks-key) bash bin/publish.sh

EOF
    exit 1
fi

DO_FORCE=0
DO_PACK_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --force)     DO_FORCE=1 ;;
        --pack-only) DO_PACK_ONLY=1 ;;
        *) echo "[publish] unknown flag: $arg" >&2 ; exit 1 ;;
    esac
done

cd "$(git rev-parse --show-toplevel)"

# ── Find the rockspec ─────────────────────────────────────────────────────
RS=$(ls ion7-*.rockspec 2>/dev/null | head -1 || true)
if [ -z "$RS" ]; then
    echo "[publish] no ion7-*.rockspec in $(pwd)" >&2
    exit 1
fi

# ── Confirm the source tag is on origin ───────────────────────────────────
TAG=$(grep -E 'tag\s*=\s*"v[^"]+"' "$RS" | head -1 | sed -E 's/.*"(v[^"]+)".*/\1/')
if [ -z "$TAG" ]; then
    echo "[publish] could not parse source.tag from $RS" >&2
    exit 1
fi
if ! git ls-remote --tags origin "$TAG" 2>/dev/null | grep -q "$TAG\$"; then
    echo "[publish] tag $TAG is not on origin." >&2
    echo "         git push origin $TAG  before publishing." >&2
    exit 1
fi

# ── Pack first (catches a missing tag, broken submodule init, etc.) ───────
echo "[publish] packing $RS"
luarocks pack "$RS"

if [ "$DO_PACK_ONLY" = "1" ]; then
    PACKED=$(ls -1t ion7-*.src.rock 2>/dev/null | head -1 || true)
    echo "[publish] pack-only — produced $PACKED"
    exit 0
fi

# ── Upload ────────────────────────────────────────────────────────────────
EXTRA=""
[ "$DO_FORCE" = "1" ] && EXTRA="--force"

echo "[publish] uploading $RS to luarocks.org"
# `luarocks upload` reads the api key from $LUAROCKS_API_KEY when
# --api-key is not passed, but spelling it out keeps the intent clear
# without echoing the secret.
luarocks upload --api-key="$LUAROCKS_API_KEY" $EXTRA "$RS"

echo "[publish] done. https://luarocks.org/modules/$(echo "$RS" | sed -E 's/-[0-9].*//')/$(echo "$RS" | sed -E 's/^[^-]+-([0-9]+\.[0-9]+\.[0-9]+(beta[0-9]+)?-[0-9]+).*/\1/')"
