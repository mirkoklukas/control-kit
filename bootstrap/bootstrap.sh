#!/usr/bin/env bash
#
# bootstrap.sh — BOX side of the controlkit cloud bootstrap ("data plane").
#
# Runs ON THE GPU INSTANCE. All system + app setup lives here: no SSH, no laptop
# logic, no secrets baked in — so this file ports straight into a Dockerfile RUN
# later. local_bootstrap.sh (laptop side) is what gets credentials here and calls
# this; you can also run it by hand once you're on the box.
#
# Steps (these mirror bootstrap/cloud-bootstrap.md):
#   0.   verify the box matches our assumptions (NVIDIA driver present)
#   1.   system packages (git, curl)
#   2.   uv (Python / venv manager)
#   3.   persistent git login (from ~/.gh_token) + clone
#   4-5. python deps + GPU JAX   (uv sync --extra mjx --extra gpu)
#   6.   verify JAX actually sees the GPU
#   7.   run the examples
#
# Assumes:
#   - NVIDIA driver already present (nvidia-smi works). NOT installed here: a
#     driver needs a reboot on a VM and comes from the host inside Docker.
#   - ~/.gh_token exists (local_bootstrap.sh dropped it, or you scp'd it).
#   - passwordless sudo (true for Lambda Stack's `ubuntu` user).
#
# Usage (on the box):    bash bootstrap.sh
# From the laptop:       ssh "$BOX" 'bash -s' < bootstrap.sh
#
# Re-runnable: the clone is guarded and `uv sync` is idempotent. This is v0 —
# verify it against a real box on the first run and correct it in place (see
# bootstrap/cloud-bootstrap.md).
#
set -euo pipefail

# ---- config (edit for your setup; NO secrets here) -------------------------
GIT_USER_NAME="Mirko Klukas"
GIT_USER_EMAIL="mirko.klukas@gmail.com"
REPO_URL="${REPO_URL:-https://github.com/mirkoklukas/control-kit.git}"
WORKDIR="${WORKDIR:-$HOME}"          # where to clone (e.g. an attached volume: /mnt/data)
APP_DIR="${APP_DIR:-control-kit}"    # repo dir name to cd into after cloning
# ---------------------------------------------------------------------------

echo "== 0. verify box (assumptions) =="
# The one hard requirement we do NOT install: the NVIDIA driver. Fail loudly so
# you notice on the first run rather than silently landing on CPU later.
command -v nvidia-smi >/dev/null && nvidia-smi \
  || { echo "no NVIDIA driver (nvidia-smi) — not installed by this script; see bootstrap/cloud-bootstrap.md (Target)"; exit 1; }
lsb_release -a || true
uname -m

echo "== 1. system packages =="
export DEBIAN_FRONTEND=noninteractive    # never block on an apt prompt (also matters in Docker builds)
sudo apt-get update -y
sudo apt-get install -y git curl
# uncomment if you need OFFSCREEN MuJoCo rendering (no window); see cloud-bootstrap.md step 8:
# sudo apt-get install -y libegl1 libgles2

echo "== 2. uv =="
# uv manages the Python toolchain + venv and fetches a suitable Python itself.
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version

echo "== 3. persistent git login + clone =="
# Turn the one-shot ~/.gh_token into a credential store that survives reboots, so
# clone/pull/commit/push all work for the life of the box (and for any repo on
# this account). The token is read from disk, never embedded in this script.
test -s "$HOME/.gh_token" || { echo "missing ~/.gh_token — run the laptop handoff first (bootstrap/local_bootstrap.sh)"; exit 1; }
git config --global user.name  "$GIT_USER_NAME"
git config --global user.email "$GIT_USER_EMAIL"
git config --global credential.helper store
printf 'https://x-access-token:%s@github.com\n' "$(cat "$HOME/.gh_token")" > "$HOME/.git-credentials"
chmod 600 "$HOME/.git-credentials"
cd "$WORKDIR"
[ -d "$APP_DIR" ] || git clone "$REPO_URL" "$APP_DIR"
cd "$APP_DIR"

echo "== 4-5. python deps + GPU JAX =="
# `mjx` -> mujoco-mjx + (CPU) JAX;  `gpu` -> jax[cuda12] (Linux-only, inherits the
# same jax version, adds the CUDA PJRT plugin + bundled NVIDIA libraries).
uv sync --extra mjx --extra gpu

echo "== 6. verify GPU =="
# Hard-fail if JAX fell back to CPU — that means the GPU stack isn't wired up.
uv run python -c "import jax; assert jax.default_backend()=='gpu', jax.devices(); print('GPU OK', jax.devices())"

echo "== 7. run examples =="
uv run python examples/00_minimal.py
uv run python examples/01_mpc_cartpole.py

echo "== done =="
