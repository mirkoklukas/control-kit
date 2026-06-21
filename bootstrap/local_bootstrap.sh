#!/usr/bin/env bash
#
# local_bootstrap.sh — LAPTOP side of the controlkit cloud bootstrap ("control plane").
#
# What it does, in order:
#   1. check laptop prerequisites (gh authenticated, box reachable over SSH)
#   2. copy your GitHub credential to the box (minted from local gh, never typed)
#   3. optionally install Claude on the box, via one of two auth branches
#   4. run the box-side bootstrap.sh over SSH
#
# It deliberately holds NO GPU/Python/apt logic — that all lives in bootstrap.sh
# so that file stays box-local and Docker-portable. This wrapper only does the
# things that need YOUR laptop: your gh auth, your SSH key, your secrets.
#
# Prerequisites (on this laptop):
#   - gh authenticated              : gh auth status
#   - SSH access to the instance    : ssh "$BOX" true   (key set up; see ../docs/cloud-setup.md)
#   - for CLAUDE_AUTH=key           : Anthropic key stored at ~/.secrets/anthropic.api
#
# Usage:
#   ./local_bootstrap.sh user@host                     # bootstrap only (no Claude on the box)
#   CLAUDE_AUTH=login ./local_bootstrap.sh user@host   # + Claude, you log in manually (uses your subscription)
#   CLAUDE_AUTH=key   ./local_bootstrap.sh user@host   # + Claude, API key from ~/.secrets/anthropic.api
#
# Re-running is safe: every remote step is guarded / idempotent.
#
set -euo pipefail

# ---- config ----------------------------------------------------------------
BOX="${1:?usage: ./local_bootstrap.sh user@host   [CLAUDE_AUTH=key|login]}"
CLAUDE_AUTH="${CLAUDE_AUTH:-none}"                     # none | key | login
KEY_FILE="${KEY_FILE:-$HOME/.secrets/anthropic.api}"  # laptop path to the Anthropic key (CLAUDE_AUTH=key)
HERE="$(cd "$(dirname "$0")" && pwd)"                  # this script's dir, so we can find bootstrap.sh
# ---------------------------------------------------------------------------

echo "== 1. check laptop prerequisites =="
command -v gh >/dev/null || { echo "gh not installed on this laptop"; exit 1; }
gh auth status >/dev/null 2>&1 || { echo "gh not authenticated — run: gh auth login"; exit 1; }
ssh -o BatchMode=yes -o ConnectTimeout=10 "$BOX" true \
  || { echo "cannot ssh to '$BOX' (check the address and your SSH key)"; exit 1; }

echo "== 2. copy GitHub credential to the box =="
# gh mints the token locally; we stream it straight into a root-only file on the
# box. It is never printed, never committed, never passed as an argv. The box-side
# bootstrap.sh turns ~/.gh_token into a persistent git login.
gh auth token | ssh "$BOX" 'umask 077; cat > ~/.gh_token'

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
    ssh "$BOX" 'command -v claude >/dev/null || curl -fsSL https://claude.ai/install.sh | bash'
    echo "   Claude installed. Finish the login yourself:"
    echo "       ssh $BOX"
    echo "       claude        # it prints a URL — open it on your laptop, paste the code back"
    ;;

  key)
    # --- Branch B: API key from ~/.secrets/anthropic.api --------------------
    # Copy the key file to the same path on the box and export it for login
    # shells, so Claude runs non-interactively. API usage is billed pay-as-you-go.
    test -s "$KEY_FILE" || { echo "   no key at $KEY_FILE on this laptop (CLAUDE_AUTH=key needs it)"; exit 1; }
    ssh "$BOX" 'umask 077; mkdir -p ~/.secrets; cat > ~/.secrets/anthropic.api' < "$KEY_FILE"
    ssh "$BOX" 'grep -q ANTHROPIC_API_KEY ~/.bashrc \
                || echo "export ANTHROPIC_API_KEY=\$(cat ~/.secrets/anthropic.api)" >> ~/.bashrc'
    ssh "$BOX" 'command -v claude >/dev/null || curl -fsSL https://claude.ai/install.sh | bash'
    echo "   Claude installed; key copied to ~/.secrets/anthropic.api and exported in ~/.bashrc on the box."
    ;;

  *)
    echo "   unknown CLAUDE_AUTH='$CLAUDE_AUTH' (use: none | login | key)"; exit 1
    ;;
esac

echo "== 4. run the box-side bootstrap over SSH =="
# Stream this repo's bootstrap.sh to the box and run it. After it clones the repo,
# the box carries its own copy for later re-runs:
#     ssh "$BOX" 'cd control-kit && bash bootstrap.sh'
ssh "$BOX" 'bash -s' < "$HERE/bootstrap.sh"

echo "== done. connect with:  ssh $BOX =="
