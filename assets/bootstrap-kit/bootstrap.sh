#!/usr/bin/env bash
# bootstrap.sh — provisions a fresh instance to make every goals.yaml target pass.
#
# Built by Claude during the discovery run; this is the deterministic replay.
# On a fresh box: clear the BLOCKERS.md prerequisites, then run `bash bootstrap.sh`.
# No Claude needed.
#
# Conventions (keep these as you add sections):
#   - idempotent: safe to run repeatedly
#   - pinned refs: exact tag/SHA, never `main`
#   - assumes BLOCKERS.md prerequisites are already satisfied; fails loudly if not

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$ROOT/repos"

# Pin everything here so the script and goals.yaml stay in sync.
ROS_DISTRO=jazzy   # must match this box's Ubuntu

# helper: idempotent clone at a pinned ref
clone_at() {  # clone_at <url> <ref> <dir>
  local url="$1" ref="$2" dir="$3"
  [ -d "$dir/.git" ] || git clone "$url" "$dir"
  git -C "$dir" fetch --all --tags --quiet
  git -C "$dir" checkout --quiet "$ref"
}

# ─────────────────────────────────────────────────────────────────────────────
# target: foo   (https://github.com/org/foo @ v1.2.0)   pass: pytest tests/unit
# ─────────────────────────────────────────────────────────────────────────────
# TODO(claude): fill in discovered install steps, e.g.
#   clone_at https://github.com/org/foo v1.2.0 "$ROOT/repos/foo"
#   ( cd "$ROOT/repos/foo" && pip install -e ".[test]" )

# ─────────────────────────────────────────────────────────────────────────────
# target: bar   (https://github.com/org/bar @ a1b2c3d)  pass: its test suite
# ─────────────────────────────────────────────────────────────────────────────
# TODO(claude): fill in discovered install steps

# ─────────────────────────────────────────────────────────────────────────────
# target: ros2  (distro $ROS_DISTRO)   pass: demo_nodes_cpp talker publishes
# ─────────────────────────────────────────────────────────────────────────────
# TODO(claude): fill in discovered install steps; remember tests that need ROS
#   must `source /opt/ros/$ROS_DISTRO/setup.bash` first.

echo "bootstrap.sh: all sections completed."
