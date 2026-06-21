#!/usr/bin/env bash
# bootstrap.sh — stand up controlkit on a fresh NVIDIA GPU box (Lambda Stack, Ubuntu 22.04).
#
# Runs ON THE BOX. Re-runnable, non-interactive. This is the "data plane": all the
# system + app setup, no SSH, no laptop-only logic — so it ports straight into a
# Dockerfile RUN later.
#
# Assumes:
#   - NVIDIA driver already present (nvidia-smi works). This script does NOT install
#     a driver (needs a reboot on a VM; comes from the host in Docker).
#   - ~/.gh_token exists (your laptop dropped it there; see local_bootstrap.sh).
#   - passwordless sudo (true on Lambda Stack's `ubuntu` user).
#
# Usage:   bash bootstrap.sh
# Or from the laptop:   ssh "$BOX" 'bash -s' < bootstrap.sh
#
# This is v0 — verify it against a real box on the first run and correct it in
# place (see docs/cloud-bootstrap.md).
set -euo pipefail

# ---- config (edit for your setup; no secrets here) -------------------------
GIT_USER_NAME="Mirko Klukas"
GIT_USER_EMAIL="mirko.klukas@gmail.com"
REPO_URL="${REPO_URL:-https://github.com/mirkoklukas/control-kit.git}"
WORKDIR="${WORKDIR:-$HOME}"          # or an attached volume, e.g. /mnt/data
APP_DIR="${APP_DIR:-control-kit}"    # repo dir to cd into
# ---------------------------------------------------------------------------

echo "== 0. verify box (assumptions) =="
command -v nvidia-smi >/dev/null && nvidia-smi \
  || { echo "no NVIDIA driver (nvidia-smi) — not installed by this script; see docs/cloud-bootstrap.md Target section"; exit 1; }
lsb_release -a || true
uname -m

echo "== 1. system packages =="
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -y
sudo apt-get install -y git curl
# uncomment if you need offscreen MuJoCo rendering:
# sudo apt-get install -y libegl1 libgles2

echo "== 2. uv =="
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version

echo "== 3. persistent git login + clone =="
test -s "$HOME/.gh_token" || { echo "missing ~/.gh_token (run the laptop handoff first)"; exit 1; }
git config --global user.name  "$GIT_USER_NAME"
git config --global user.email "$GIT_USER_EMAIL"
git config --global credential.helper store
printf 'https://x-access-token:%s@github.com\n' "$(cat "$HOME/.gh_token")" > "$HOME/.git-credentials"
chmod 600 "$HOME/.git-credentials"
cd "$WORKDIR"
dir=$(basename "$REPO_URL" .git)
[ -d "$dir" ] || git clone "$REPO_URL"
cd "$APP_DIR"

echo "== 4-5. python deps + GPU JAX =="
# `gpu` extra = jax[cuda12], Linux-only, inherits the jax version from `mjx`.
uv sync --extra mjx --extra gpu

echo "== 6. verify GPU =="
uv run python -c "import jax; assert jax.default_backend()=='gpu', jax.devices(); print('GPU OK', jax.devices())"

echo "== 7. run examples =="
uv run python examples/00_minimal.py
uv run python examples/01_mpc_cartpole.py

echo "== done =="
