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
- The smoke (`lab.hexapod_ppo.v0.smoke`) is MJX-bound (brax + mujoco-mjx + jax,
  ~3GB of CUDA wheels via torch/jax deps).
- EXTRA CHANGED: goals.yaml now pins the pass to `--extra vm`, not `--extra ppo`
  (goals.yaml was edited after the first discovery run). `vm = controlkit[gpu,ppo]`
  — the GPU superset: ppo PLUS the `gpu` extra's `jax[cuda12]`. bootstrap.sh now
  does `uv sync --extra vm` to match. Under `vm` the smoke runs on the **GPU**
  (nvcc / CUDA toolkit precondition), prints "SMOKE PASS" and `exit=0` in ~5 min
  wall (one-time JIT compile + the tiny training run). The old `ppo` note (CPU
  fallback, "CUDA-enabled jaxlib is not installed") no longer applies. Syncing
  `ppo` instead of `vm` would just make the `vm` pass command re-resolve and
  download the cuda jaxlib at pass time, so we sync `vm` to front-load it.
- Did NOT install `glow` (a README "system requirement") — it's only for the
  terminal markdown renderer, not needed by either pass command.
