# LOG.md

Dead ends and rationale — the reasoning bootstrap.sh can't show.

---

## run-kit
- Plain `uv`-managed package. `uv run --extra dev pytest` auto-creates `.venv`
  and installs; 8 tests pass in ~0.3s. Nothing tricky.

## control-kit
- Depends on run-kit as a **local editable** checkout via
  `[tool.uv.sources] runkit = { path = "../run-kit", editable = true }`. So
  run-kit must be cloned as a sibling under the same parent (`$HOME`) — which
  matches goals.yaml's note. Clone order: run-kit first, then control-kit.
- The smoke (`lab.hexapod_ppo.v0.smoke`) is MJX-bound and needs the `ppo` extra
  (brax + mujoco-mjx + jax, ~3GB of CUDA wheels via torch/jax deps).
- GPU note: `jax[cuda12]` lives in a SEPARATE `gpu` extra. The pass command only
  requests `--extra ppo`, so jax resolves to a **CPU** jaxlib and prints
  "CUDA-enabled jaxlib is not installed. Falling back to cpu." This is expected —
  the oracle exercises the CPU path. It still exits 0, just slow: ~7 min wall
  (dominated by a one-time JIT compile + the tiny CPU training run). Adding the
  `gpu` extra in bootstrap.sh would be pointless because `uv run --extra ppo`
  reconciles the env back to ppo-only and would strip the cuda jaxlib anyway.
- Did NOT install `glow` (a README "system requirement") — it's only for the
  terminal markdown renderer, not needed by either pass command.
