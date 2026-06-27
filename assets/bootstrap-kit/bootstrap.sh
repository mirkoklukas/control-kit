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
