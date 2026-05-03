#!/usr/bin/env bash
#
# bump-version.sh — sync the version across the rockspec, source files,
# and (optionally) git tag of an Ion7 repo.
#
# Touches :
#   - <package>-<version>-<rev>.rockspec   (renamed in place)
#       version = "<lr_ver>-<rev>"
#       source.tag = "v<git_ver>"
#   - src/**/init.lua                      _VERSION / VERSION constants
#   - lints the resulting rockspec via `luarocks lint`
#
# Convention :
#   git tag         vX.Y.Z[-<pre>]      (e.g. v0.1.0-beta2, v0.2.0)
#   rockspec ver    X.Y.Z[<pre>]-<rev>  (luarocks dislikes dashes inside
#                                        version, so beta2 stays glued)
#   source const    "X.Y.Z[-<pre>]"     (matches the git tag, dashes kept)
#
# Usage :
#   bash bin/bump-version.sh <git-version>           # in-place edits only
#   bash bin/bump-version.sh <git-version> --tag     # + commit + git tag
#   bash bin/bump-version.sh <git-version> --tag --push
#                                                    # + push branch + tag
#
# Examples :
#   bash bin/bump-version.sh 0.1.0-beta2
#   bash bin/bump-version.sh 0.2.0 --tag --push
#
# Re-run idempotent : each step replaces or no-ops cleanly when the
# target value is already in place.

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: $0 <git-version> [--tag] [--push] [--rev N]" >&2
    exit 1
fi

GIT_VER="$1" ; shift

DO_TAG=0
DO_PUSH=0
REV=1
while [ $# -gt 0 ]; do
    case "$1" in
        --tag)  DO_TAG=1 ;;
        --push) DO_PUSH=1 ;;
        --rev)  shift ; REV="$1" ;;
        *) echo "[bump-version] unknown flag: $1" >&2 ; exit 1 ;;
    esac
    shift
done

# Strip "v" prefix if the caller typed one ("v0.2.0" → "0.2.0").
GIT_VER="${GIT_VER#v}"

# luarocks rockspec version : strip dashes inside the X.Y.Z[-pre] part.
# 0.2.0-beta1 → 0.2.0beta1   (rockspec version = "0.2.0beta1-1")
LR_VER="${GIT_VER//-/}"

cd "$(git rev-parse --show-toplevel)"

# ── Detect the package name from the rockspec ─────────────────────────────
RS_GLOB=$(ls ion7-*.rockspec 2>/dev/null | head -1 || true)
if [ -z "$RS_GLOB" ]; then
    echo "[bump-version] no ion7-*.rockspec found in $(pwd)" >&2
    exit 1
fi
PKG=$(echo "$RS_GLOB" | sed -E 's/-[0-9].*//')
NEW_RS="${PKG}-${LR_VER}-${REV}.rockspec"

# ── Rename the rockspec file (no-op if already correct) ───────────────────
if [ "$RS_GLOB" != "$NEW_RS" ]; then
    if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        git mv "$RS_GLOB" "$NEW_RS" 2>/dev/null || mv "$RS_GLOB" "$NEW_RS"
    else
        mv "$RS_GLOB" "$NEW_RS"
    fi
    echo "[bump-version] renamed $RS_GLOB → $NEW_RS"
fi

# ── Rewrite version + tag inside the rockspec ─────────────────────────────
sed -i.bak -E "s/^version = \".*\"/version = \"${LR_VER}-${REV}\"/" "$NEW_RS"
sed -i.bak -E "s/(tag\s*=\s*)\"v[^\"]*\"/\\1\"v${GIT_VER}\"/" "$NEW_RS"
rm -f "$NEW_RS.bak"

# ── Sync _VERSION / VERSION constants in src ──────────────────────────────
# Match either `_VERSION = "X"` or `<mod>.VERSION = "X"` patterns.
while IFS= read -r f; do
    sed -i.bak -E \
        -e "s/(_VERSION[[:space:]]*=[[:space:]]*)\"[^\"]*\"/\\1\"${GIT_VER}\"/" \
        -e "s/((^|\.)VERSION[[:space:]]*=[[:space:]]*)\"[^\"]*\"/\\1\"${GIT_VER}\"/" \
        "$f"
    rm -f "$f.bak"
done < <(grep -rlE '_VERSION[[:space:]]*=|\.VERSION[[:space:]]*=' src 2>/dev/null || true)

# ── Lint the rockspec ─────────────────────────────────────────────────────
echo "[bump-version] lint $NEW_RS"
luarocks lint "$NEW_RS"

echo "[bump-version] $PKG → ${GIT_VER}  (rockspec ${LR_VER}-${REV})"

# ── Optional commit + tag + push ──────────────────────────────────────────
if [ $DO_TAG -eq 1 ]; then
    git add -A
    if git diff --cached --quiet; then
        echo "[bump-version] no changes to commit (working tree clean)"
    else
        git commit -m "chore: bump version to v${GIT_VER}"
    fi

    if git rev-parse "v${GIT_VER}" >/dev/null 2>&1; then
        echo "[bump-version] tag v${GIT_VER} already exists — skipping"
    else
        git tag -a "v${GIT_VER}" -m "${PKG} v${GIT_VER}"
        echo "[bump-version] tagged v${GIT_VER}"
    fi

    if [ $DO_PUSH -eq 1 ]; then
        BRANCH=$(git rev-parse --abbrev-ref HEAD)
        git push origin "$BRANCH"
        git push origin "v${GIT_VER}"
        echo "[bump-version] pushed $BRANCH and v${GIT_VER}"
    fi
fi
