# Gotchas / known issues

Verified footguns and environment quirks, so we don't rediscover them. Each
entry: symptom, cause, what to do.

## MJX cannot run on the Apple (Metal) GPU

**Verified 2026-06-20 on an Apple M3 Max, macOS 14.**

**Symptom.** On this Mac, `jax.default_backend()` is `cpu` and the only device
is `CpuDevice`, so MJX runs on CPU and is slow (~120 s for the `03b` cartpole
run). Natural question: can we use the Apple GPU via `jax-metal`?

**Finding — the GPU works for JAX, but not for MJX.** We tested it in an isolated
env (details below):

- `jax-metal` **does** load and work. With the jax it targets it registered the
  M3 Max as a `METAL` device and ran a 2000×2000 matmul on the GPU.
- `mjx.step` **fails to compile** on Metal:
  ```
  error: failed to legalize operation 'mhlo.cholesky'
    qh, _ = jax.scipy.linalg.cho_factor(d.qM)   # mjx/_src/smooth.py:304
  ```
  MJX factorizes the joint-space inertia matrix `M` every step
  (`smooth.factor_m`) with a Cholesky decomposition, and **Metal's XLA backend
  can't lower `mhlo.cholesky`**. It's on the critical path of the forward
  dynamics every step, so there is no avoiding it.

**Why it won't improve soon.** `jax-metal` is stuck at `0.1.1` (targets
`jax==0.4.34`, no updates). Our project uses jax `0.10.2`, which `jax-metal`
almost certainly won't even load against (PJRT ABI drift). Until Apple's Metal
backend implements `cholesky` (and likely other LAPACK-style ops MJX needs), MJX
on the Mac GPU is a dead end.

**What to do.**
- Locally: **CPU is the only path for MJX.** Fine for correctness / small models;
  just expect the slowness.
- For acceleration: run the *same* (device-agnostic) code on **NVIDIA CUDA or a
  TPU** (cloud box / Colab) with `jax[cuda12]` + `mujoco-mjx`. Batched MPPI
  (`vmap` over N candidates) is exactly the workload that flies there.

**How it was tested (to re-verify later):** isolated `uv` venv, pinned
`jax==0.4.34 jaxlib==0.4.34 jax-metal==0.1.1 mujoco-mjx==3.2.5`; the matmul
succeeded on `METAL`, `mjx.step` raised the `mhlo.cholesky` legalization error
above.

## The MuJoCo viewer needs `mjpython` on macOS

`mujoco.viewer.launch_passive` must run under **`mjpython`**, not plain
`python` (the GUI needs the macOS main thread). So:

```bash
uv run mjpython examples/03b_mpc_cartpole.py --render   # not `python`
```

`mjpython` ships with the `mujoco` package and is on the venv path.

## CPU MJX + per-step MPPI renders in slow motion

In `examples/03b_mpc_cartpole.py --render` the viewer advances at ~2 Hz and looks
like slow motion. Not a bug: there's no real-time pacing, so the frame rate is
`1 / (loop-body time)`, dominated by the per-step `policy()` call (N×T `mjx.step`
on CPU, ~0.5 s). Levers: fewer `SAMPLES`/`HORIZON`, a real GPU, or decouple
rendering from planning (plan every k steps). See `docs/mpc.md`.

## `mjd_transitionFD` does not support RK4

The finite-difference transition Jacobian needs an Euler/implicit integrator.
The LQR example (`examples/02`) temporarily switches `model.opt.integrator` to
`mjINT_EULER` for the linearization and restores it afterward. See
`docs/mujoco-concepts.md` §3.
