# hexapod_ppo / v1

Train the radial hexapod to **walk straight in +x** with PPO (Brax on MJX).

## Run

Everything needs the `vm` extra (`jax[cuda12]`+ `brax` + the `mjx` stack). Real runs go on a GPU
box; launch the baseline preset (see Presets) with the recommended env prefix:

```bash
TF_CPP_MIN_LOG_LEVEL=2 JAX_COMPILATION_CACHE_DIR=~/.cache/jax \
  uv run --extra vm python -m lab.hexapod_ppo.v1.run exp:config.yaml --tag=baseline
```

The env prefix is recommended but optional (drop it and it still runs):
- `TF_CPP_MIN_LOG_LEVEL=2` — hide XLA autotuning warnings.
- `JAX_COMPILATION_CACHE_DIR=~/.cache/jax` — cache the compiled + autotuned
  executable. The **first** run pays a several-minute MJX training-step compile
  (XLA + GPU autotuning, *not* a hang — see Notes); this lets later same-config
  runs skip it.
- `JAX_LOG_COMPILES=1` — log each XLA compile: a heartbeat during the silent compile,
  reassurance it's compiling and not hung.
  - Note this *adds* output rather than hiding it, so it works against a quiet log.
    Drop it if you just want fewer lines; keep it for long runs where the heartbeat helps.

MJX only flies on CUDA/TPU. CPU works but is slow, and the Apple GPU can't run MJX
at all (see `docs/gotchas.md`), so real runs happen on a GPU box. Output lands in
an immutable runkit run dir under `runs/` (gitignored): `runs/<date>/<time>_hexapod_ppo_<tag>/policy.pkl`.

