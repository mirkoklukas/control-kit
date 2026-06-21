#!/usr/bin/env bash
#
# bootstrap.sh — BOX side of the controlkit cloud bootstrap ("data plane").
#
# Runs ON THE GPU INSTANCE. All system + app setup, no SSH, no secrets, no
# personal config — so it ports straight into a Dockerfile RUN later.
#
# It ASSUMES the box was already provisioned by bootstrap/local_bootstrap.sh
# (run from your laptop):
#   - git credentials configured  (credential.helper store + ~/.git-credentials)
#   - git identity configured     (user.name / user.email)
# so it can clone/pull/push with no input of its own. It knows nothing about who
# you are; it only needs the repo URL.
#
# Steps:
#   0.   verify the box (NVIDIA driver present, git provisioned)
#   1.   system packages (git, curl)
#   2.   uv (Python / venv manager)
#   3.   clone the repo
#   4-5. python deps + GPU JAX   (uv sync --extra mjx --extra gpu)
#   6.   verify JAX actually sees the GPU
#   7.   run the examples
#
# Also assumes:
#   - NVIDIA driver present (nvidia-smi). NOT installed here: a driver needs a
#     reboot on a VM and comes from the host inside Docker.
#   - passwordless sudo (true for Lambda Stack's `ubuntu` user).
#
# Usage (on the box):    bash ~/bootstrap.sh
#               or:      bash bootstrap/bootstrap.sh   (from a clone)
#
# Re-runnable: the clone is guarded and `uv sync` is idempotent. This is v0 —
# verify it against a real box on the first run and correct it in place (see
# bootstrap/cloud-bootstrap.md).
#
set -euo pipefail

# ---- config (overridable via env; no secrets, no personal identity) --------
REPO_URL="${REPO_URL:-https://github.com/mirkoklukas/control-kit.git}"
WORKDIR="${WORKDIR:-$HOME}"          # where to clone (e.g. an attached volume: /mnt/data)
APP_DIR="${APP_DIR:-control-kit}"    # repo dir name to cd into after cloning
# ---------------------------------------------------------------------------

echo "== 0. verify box (assumptions) =="
# The one hard requirement we do NOT install: the NVIDIA driver.
command -v nvidia-smi >/dev/null && nvidia-smi \
  || { echo "no NVIDIA driver (nvidia-smi) — not installed by this script; see bootstrap/cloud-bootstrap.md (Target)"; exit 1; }
# Git must already be provisioned by local_bootstrap.sh (file test, no git binary needed).
test -f "$HOME/.git-credentials" \
  || { echo "git credentials missing — run bootstrap/local_bootstrap.sh from your laptop first"; exit 1; }
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

echo "== 3. clone =="
# Credentials + identity were set up by local_bootstrap.sh, so this just works.
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
