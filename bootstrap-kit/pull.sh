#!/usr/bin/env bash
# pull.sh — pull the agent's outputs back from the box. Run from your laptop:
#
#     ./pull.sh <HOST>        # HOST = an ~/.ssh/config alias or user@ip
#
# Claude on the box writes bootstrap.sh (the setup it built) plus LOG.md and
# BLOCKERS.md (its records) as it works; this copies them back over the local
# copies so the repo tracks the latest. The inverse of prep.sh shipping out.
set -euo pipefail

# ~/bootstrap-kit on the box (matches prep.sh)
REMOTE_DIR="bootstrap-kit"            

HOST="${1:-}"
[ -n "$HOST" ] || { echo "usage: ./pull.sh <HOST>   (ssh alias or user@ip)" >&2; exit 1; }
KIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
say() { echo "[pull] $*" >&2; }

# Agent-written outputs to pull back; missing ones (e.g. no blockers) are skipped.
FILES=(bootstrap.sh LOG.md BLOCKERS.md)

say "pulling ${FILES[*]} from $HOST:~/$REMOTE_DIR -> $KIT_DIR"
for f in "${FILES[@]}"; do
  if scp -q "$HOST:$REMOTE_DIR/$f" "$KIT_DIR/$f"; then
    say "  + $f"
  else
    say "  ! $f not found on box, skipped"
  fi
done
say "done. review with: git -C \"$KIT_DIR\" diff"
