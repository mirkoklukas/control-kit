#!/usr/bin/env bash
# bootstrap.sh — provisions a fresh instance to make every goals.yaml target pass.
#
# Built by Claude during the discovery run; this is the deterministic replay.
# On a fresh box: clear the BLOCKERS.md prerequisites, then run `bash bootstrap.sh`.
# No Claude needed.
#
# Conventions (keep these as you add sections):
#   - idempotent: safe to run repeatedly
#   - assumes BLOCKERS.md prerequisites are already satisfied; fails loudly if not
set -euo pipefail

# ---------------------------------------------------------------------------
# Preconditions (from ENVIRONMENT.md) — fail loudly if missing.
# ---------------------------------------------------------------------------
command -v uv   >/dev/null 2>&1 || { echo "FATAL: uv not on PATH"; exit 1; }
command -v git  >/dev/null 2>&1 || { echo "FATAL: git not on PATH"; exit 1; }
command -v nvcc >/dev/null 2>&1 || { echo "FATAL: nvcc (CUDA toolkit) not on PATH"; exit 1; }

# Install everything under $HOME (local disk), never the NFS mount.
cd "$HOME"

# ---------------------------------------------------------------------------
# Task: run-kit — install https://github.com/mirkoklukas/run-kit
#   pass: (cd ~/run-kit && uv run --extra dev pytest)
# ---------------------------------------------------------------------------
[ -d "$HOME/run-kit" ] || git clone https://github.com/mirkoklukas/run-kit.git "$HOME/run-kit"
# Build the dev env (pytest + deps) on local disk. Idempotent.
( cd "$HOME/run-kit" && uv sync --extra dev )

# ---------------------------------------------------------------------------
# Task: control-kit — install https://github.com/mirkoklukas/control-kit
#   Must share the same parent folder as run-kit ($HOME); it consumes run-kit
#   as a local editable checkout via [tool.uv.sources] runkit = ../run-kit.
#   pass: (cd ~/control-kit && uv run --extra ppo python -m lab.hexapod_ppo.v0.smoke)
#
#   The smoke is MJX-bound and needs the `ppo` extra (brax + mujoco-mjx + jax).
#   `uv run --extra ppo` resolves a CPU jaxlib (the GPU-only `jax[cuda12]` lives
#   in the separate `gpu` extra, which the pass command does not request), so the
#   smoke runs on CPU — slow (several minutes incl. one-time JIT compile) but
#   exits 0. Pre-sync here to front-load the heavy download.
# ---------------------------------------------------------------------------
[ -d "$HOME/control-kit" ] || git clone https://github.com/mirkoklukas/control-kit.git "$HOME/control-kit"
( cd "$HOME/control-kit" && uv sync --extra ppo )

echo "[bootstrap] done — run the goals.yaml 'pass' commands to verify."
