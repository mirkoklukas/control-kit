#!/usr/bin/env bash
# prep.sh — laptop-side base setup. Run ONCE per box from your laptop:
#
#     ./prep.sh <HOST>        # HOST = an ~/.ssh/config alias or user@ip
#
# Readies the box's base (the Docker "FROM" + auth), then ships the kit:
#   - primes git HTTPS credentials on the box from your local `gh` token
#   - replicates your git identity (user.name / user.email)
#   - copies the declared secret files to ~/.secrets/ on the box (by file, never inlined)
#   - installs uv (the on-box bootstrap uses it to sync the real project)
#   - ships this kit folder to ~/bootstrap-kit on the box
#
# This is the ONLY place secrets are handled. It hardcodes its own git host + secrets list.
# Claude on the box may edit this file; pull it back and re-run to apply the fix.
set -euo pipefail

# --- config (edit for your setup) --------------------------------------------
GIT_HOST="github.com"
REMOTE_DIR="bootstrap-kit"            # lands at ~/bootstrap-kit on the box
# Secret files (paths on the LAPTOP) to copy to ~/.secrets/ on the box. Empty = none.
SECRETS=(
  # "$HOME/.secrets/hf.env"
)

# --- args --------------------------------------------------------------------
HOST="${1:-}"
[ -n "$HOST" ] || { echo "usage: ./prep.sh <HOST>   (ssh alias or user@ip)" >&2; exit 1; }
KIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
say() { echo "[prep] $*" >&2; }

# --- laptop preconditions ----------------------------------------------------
command -v gh  >/dev/null || { echo "need: gh (GitHub CLI), authenticated" >&2; exit 1; }
command -v ssh >/dev/null || { echo "need: ssh" >&2; exit 1; }
GH_TOKEN="$(gh auth token 2>/dev/null)" || { echo "run: gh auth login" >&2; exit 1; }
GIT_NAME="$(git config --global user.name  || true)"
GIT_EMAIL="$(git config --global user.email || true)"

# --- git credentials + identity (token piped via stdin, never in argv) -------
say "priming git HTTPS credentials + identity on $HOST"
printf 'https://x-access-token:%s@%s\n' "$GH_TOKEN" "$GIT_HOST" \
  | ssh "$HOST" "umask 077; cat > ~/.git-credentials; \
      git config --global credential.helper store; \
      git config --global user.name  '$GIT_NAME'; \
      git config --global user.email '$GIT_EMAIL'"

# --- secrets (by file, never inlined) ----------------------------------------
if [ "${#SECRETS[@]}" -gt 0 ]; then
  say "copying ${#SECRETS[@]} secret file(s) to ~/.secrets/"
  ssh "$HOST" "umask 077; mkdir -p ~/.secrets"
  for f in "${SECRETS[@]}"; do
    [ -f "$f" ] || { echo "missing secret: $f" >&2; exit 1; }
    scp -q "$f" "$HOST:.secrets/$(basename "$f")"
  done
fi

# --- uv + glow (markdown viewer) + ship the kit -------------------------------------------------------
say "installing uv on $HOST (if absent)"
ssh "$HOST" 'command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh'

say "installing glow on $HOST"
ssh "$HOST" 'command -v glow >/dev/null 2>&1 || { GLOW_VER=2.0.0; mkdir -p ~/.local/bin; \
  curl -fsSL "https://github.com/charmbracelet/glow/releases/download/v${GLOW_VER}/glow_${GLOW_VER}_Linux_x86_64.tar.gz" \
    | tar -xz -C ~/.local/bin --strip-components=1 --wildcards "*/glow"; }'

say "shipping kit -> ~/$REMOTE_DIR"
ssh "$HOST" "mkdir -p ~/$REMOTE_DIR"
if command -v rsync >/dev/null; then
  rsync -az --delete \
    --exclude '.git' --exclude '__pycache__' --exclude '.venv' --exclude 'prep.sh' \
    "$KIT_DIR/" "$HOST:$REMOTE_DIR/"
else
  scp -q -r "$KIT_DIR/." "$HOST:$REMOTE_DIR/"
fi

say "done. next: ssh $HOST, then run install and run Claude:"
say "  curl -fsSL https://claude.ai/install.sh | bash"
say "  cd ~/ && claude --permission-mode bypassPermissions"
say "tell claude: 'Read ~/bootstrap-kit/BOOTSTRAP.md and complete the bootstrap.'"
# say "  cd ~/ && claude --permission-mode bypassPermissions 'Read ~/bootstrap-kit/BOOTSTRAP.md and complete the bootstrap.'"

