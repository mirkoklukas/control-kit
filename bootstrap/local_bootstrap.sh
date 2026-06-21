#!/usr/bin/env bash
#
# local_bootstrap.sh — LAPTOP side of the controlkit cloud bootstrap.
#
# Provisions a fresh GPU instance so that bootstrap.sh can run there with no
# further input, then copies bootstrap.sh over. It does NOT run bootstrap.sh —
# you ssh in and run it yourself (keeps the box-side recipe self-contained and
# Docker-portable).
#
# What it sets up on the box:
#   - git credentials (credential.helper store + ~/.git-credentials, from local gh)
#   - git identity    (user.name / user.email, mirrored from THIS laptop)
#   - optionally Claude (one of two auth branches)
#   - a copy of bootstrap.sh at ~/bootstrap.sh
#
# Everything laptop-specific lives here (your gh auth, SSH key, git identity,
# secrets). Nothing personal is hardcoded: the git identity is read from this
# machine's own `git config`.
#
# Prerequisites (on this laptop):
#   - gh authenticated        : gh auth status
#   - git identity set        : git config user.name / user.email
#   - SSH access to the box    : ssh "$HOST" true   (see ../docs/cloud-setup.md)
#   - for CLAUDE_AUTH=key     : Anthropic key at ~/.secrets/anthropic.api
#
# Usage:
#   ./local_bootstrap.sh HOST                      # HOST = hostname or user@host (whatever `ssh HOST` accepts)
#   CLAUDE_AUTH=login ./local_bootstrap.sh HOST    # + install Claude, you log in manually (subscription)
#   CLAUDE_AUTH=key   ./local_bootstrap.sh HOST    # + install Claude, key from ~/.secrets/anthropic.api
#
# Then finish on the box:
#   ssh HOST
#   bash ~/bootstrap.sh
#
# Re-running is safe.
#
set -euo pipefail

# ---- config ----------------------------------------------------------------
HOST="${1:?usage: ./local_bootstrap.sh HOST   [CLAUDE_AUTH=key|login]}"
CLAUDE_AUTH="${CLAUDE_AUTH:-none}"                     # none | key | login
KEY_FILE="${KEY_FILE:-$HOME/.secrets/anthropic.api}"  # laptop path to the Anthropic key (CLAUDE_AUTH=key)
HERE="$(cd "$(dirname "$0")" && pwd)"                  # this script's dir, to find bootstrap.sh
# ---------------------------------------------------------------------------

echo "== 1. check laptop prerequisites =="
command -v gh >/dev/null || { echo "gh not installed on this laptop"; exit 1; }
gh auth status >/dev/null 2>&1 || { echo "gh not authenticated — run: gh auth login"; exit 1; }
NAME="$(git config --get user.name  || true)"
EMAIL="$(git config --get user.email || true)"
{ [ -n "$NAME" ] && [ -n "$EMAIL" ]; } \
  || { echo "set your local git identity first: git config --global user.name/user.email"; exit 1; }
ssh -o BatchMode=yes -o ConnectTimeout=10 "$HOST" true \
  || { echo "cannot ssh to '$HOST' (check the hostname and your SSH key)"; exit 1; }

echo "== 2. configure git on the box (credentials + identity) =="
# Identity: mirror THIS laptop's git config — nothing hardcoded.
ssh "$HOST" "git config --global user.name \"$NAME\"; git config --global user.email \"$EMAIL\""
# Credentials: persistent store, primed from local gh. The token is streamed over
# stdin (read into a var on the box), never printed, never an argv, never committed.
ssh "$HOST" 'git config --global credential.helper store'
gh auth token | ssh "$HOST" \
  'umask 077; read -r TOK; printf "https://x-access-token:%s@github.com\n" "$TOK" > ~/.git-credentials; chmod 600 ~/.git-credentials'

echo "== 3. Claude on the box (CLAUDE_AUTH=$CLAUDE_AUTH) =="
case "$CLAUDE_AUTH" in
  none)
    echo "   skipped — set CLAUDE_AUTH=login or CLAUDE_AUTH=key to install Claude on the box"
    ;;

  login)
    # --- Branch A: manual login (uses your Claude subscription) --------------
    # Install Claude; you finish auth interactively. The login flow works over a
    # plain SSH session: it prints a URL you open on your laptop and a code you
    # paste back. Nothing secret leaves your laptop here.
    ssh "$HOST" 'command -v claude >/dev/null || curl -fsSL https://claude.ai/install.sh | bash'
    echo "   Claude installed. Finish the login yourself:"
    echo "       ssh $HOST"
    echo "       claude        # it prints a URL — open it on your laptop, paste the code back"
    ;;

  key)
    # --- Branch B: API key from ~/.secrets/anthropic.api --------------------
    # Copy the key file to the same path on the box and export it for login
    # shells, so Claude runs non-interactively. API usage is billed pay-as-you-go.
    test -s "$KEY_FILE" || { echo "   no key at $KEY_FILE on this laptop (CLAUDE_AUTH=key needs it)"; exit 1; }
    ssh "$HOST" 'umask 077; mkdir -p ~/.secrets; cat > ~/.secrets/anthropic.api' < "$KEY_FILE"
    ssh "$HOST" 'grep -q ANTHROPIC_API_KEY ~/.bashrc \
                || echo "export ANTHROPIC_API_KEY=\$(cat ~/.secrets/anthropic.api)" >> ~/.bashrc'
    ssh "$HOST" 'command -v claude >/dev/null || curl -fsSL https://claude.ai/install.sh | bash'
    echo "   Claude installed; key copied to ~/.secrets/anthropic.api and exported in ~/.bashrc on the box."
    ;;

  *)
    echo "   unknown CLAUDE_AUTH='$CLAUDE_AUTH' (use: none | login | key)"; exit 1
    ;;
esac

echo "== 4. copy bootstrap.sh to the box =="
scp "$HERE/bootstrap.sh" "$HOST:~/bootstrap.sh"

cat <<EOF

== done — box provisioned, bootstrap.sh is at ~/bootstrap.sh ==
Finish on the box:
    ssh $HOST
    bash ~/bootstrap.sh
EOF
