#!/usr/bin/env bash
# local_bootstrap.sh — run on your LAPTOP. The "control plane": it prepares a fresh
# GPU box (copies credentials, optionally installs Claude) and then triggers the
# box-side bootstrap.sh over SSH. No GPU/Python logic lives here.
#
# Prereqs on the laptop:
#   - gh authenticated   (gh auth status)
#   - SSH access to the box (key set up; see docs/cloud-setup.md)
#
# Usage:
#   ./local_bootstrap.sh user@host
#   # also install + API-key Claude on the box (for developing on the box later):
#   INSTALL_CLAUDE=1 ANTHROPIC_API_KEY=sk-ant-... ./local_bootstrap.sh user@host
set -euo pipefail

BOX="${1:?usage: ./local_bootstrap.sh user@host}"

echo "== 1. check local prereqs =="
command -v gh >/dev/null || { echo "gh not installed locally"; exit 1; }
gh auth status >/dev/null 2>&1 || { echo "gh not authenticated — run: gh auth login"; exit 1; }
ssh -o BatchMode=yes -o ConnectTimeout=10 "$BOX" true \
  || { echo "cannot ssh to $BOX (check the address / your SSH key)"; exit 1; }

echo "== 2. hand the box your git credential =="
# gh mints the token locally; it lands on the box owner-only, never printed here.
gh auth token | ssh "$BOX" 'umask 077; cat > ~/.gh_token'

echo "== 3. (optional) Claude on the box =="
if [ "${INSTALL_CLAUDE:-0}" = "1" ]; then
  # install Claude Code on the box
  ssh "$BOX" 'command -v claude >/dev/null || curl -fsSL https://claude.ai/install.sh | bash'
  if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
    # auth option A: API key (headless, non-interactive, pay-as-you-go)
    printf '%s\n' "$ANTHROPIC_API_KEY" | ssh "$BOX" 'umask 077; cat > ~/.anthropic_key'
    ssh "$BOX" 'grep -q ANTHROPIC_API_KEY ~/.bashrc || echo "export ANTHROPIC_API_KEY=\$(cat ~/.anthropic_key)" >> ~/.bashrc'
    echo "   Claude installed + API key set (exported in ~/.bashrc on the box)."
  else
    # auth option B: interactive login (uses your Claude subscription)
    echo "   Claude installed, no API key given -> log in interactively:"
    echo "       ssh $BOX"
    echo "       claude     # follow the prompt: open the URL on your laptop, paste the code back"
  fi
else
  echo "   skipped (set INSTALL_CLAUDE=1 to install; add ANTHROPIC_API_KEY=... for key auth, or omit it to log in)"
fi

echo "== 4. run the box-side bootstrap over SSH =="
# Pipe THIS repo's bootstrap.sh to the box and run it. The cloned repo will then
# carry its own copy for later re-runs:  ssh "$BOX" 'cd control-kit && bash bootstrap.sh'
ssh "$BOX" 'bash -s' < "$(dirname "$0")/bootstrap.sh"

echo "== done. connect with:  ssh $BOX =="
