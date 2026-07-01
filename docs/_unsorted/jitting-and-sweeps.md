# JIT, baking, and sweeps (brax + MJX)

Short version of a long thread. Applies to `lab/hexapod_ppo/v1`.

## The one rule

`jax.jit` traces a function **once per input signature** (shapes + dtypes) and caches
the compiled program. During that trace:

- **Traced arguments** (arrays flowing in as function args) stay live — their values
  can change every call without recompiling.
- **Everything else** — anything read from `self`, closures, or globals — is
  **concrete at trace time and baked in as a constant**. It's frozen into the
  compiled program.

Consequence: if you mutate `self.cfg.action_scale` on an existing env and call the
same jitted fn again, **it does not recompile and keeps the old value** (silent
staleness). The value only changes if you build a *new* env (new bound method → fresh
compile).

So: **live at runtime ⟺ it came in through a traced argument.** For a brax env, the
traced args of `step` are `state` and `action`. Anything you want to vary must ride in
the `state` pytree (in practice `state.info`), as arrays.

## Why `self.cfg.X` bakes but `state.info[...]` doesn't

brax jits **bound methods** (`jax.jit(env.reset)`), so `self` is already filled in — it
is *not* a traced argument, just a captured constant. Hence `self.cfg.X` bakes. `state`
is a registered pytree of arrays, so its leaves (including `info`) are traced → live.
That's why `reward_weights` are carried in `state.info` (seeded from cfg at reset), not
read off `self.cfg`.

You can't pass the whole env/`Cfg` as a traced arg: a plain object isn't a pytree of
arrays (strings, tuples, PPO hyperparams aren't valid JAX leaves), and most fields
aren't read in `step` anyway.

## What recompiles

- **New env instance / new cfg values baked in step-reset** → fresh compile.
- **Structural change**: a new `info` key, a changed shape/dtype, or a Python int used
  as a `scan` length (e.g. `decimation`) → recompile. Structure is part of the signature.
- **Values that are traced** (in `state.info`) → *no* recompile when they change.

## Persistent compilation cache (cross-process)

`run.py` enables JAX's on-disk cache (`jax_compilation_cache_dir`, shared dir under
`~/.cache/controlkit/jax`, override with `CONTROLKIT_JAX_CACHE`). It saves compiled
executables keyed by **HLO + shapes + jax/jaxlib version + device**. So a **same-config
rerun skips the multi-minute training-step compile** (loads from disk instead).

The cache key includes baked constants. Two implications for sweeps:

- Sweeping a value that's **baked** (e.g. `kp`, `kv`, `action_scale` read off cfg,
  network sizes, `num_envs`) → different HLO → **cache miss**, full recompile per point.
- Sweeping a value that's **traced** (in `state.info`, like `reward_weights`) → identical
  HLO → **cache hit** across the sweep (only the first point compiles).

So the persistent cache and "put it in `state.info`" work together: tracing a swept
scalar is what lets the disk cache be reused across sweep points.

## Curriculum

`ppo.train` gives **no hook** to change something mid-training from outside. A curriculum
has to be self-contained inside `step`/`reset`: carry the variable in `state.info`, update
it there by a rule (e.g. widen the command range when tracking is good), sample from it at
reset. Tracing the variable is the prerequisite; the update rule is separate.

## Actuator gains (`kp`/`kv`)

Set once per run via cfg; applied to the `MjModel` in `env.__init__` before
`mjx.put_model` (`gainprm[0]=kp`, `biasprm[1]=-kp`, `biasprm[2]=-kv`). They live in the
**model**, which is closed over → **baked**, so each distinct `kp`/`kv` recompiles and
misses the persistent cache. Fine for a few runs. Cheap gain sweeps / per-env
randomization would instead go through brax's `DomainRandomizationVmapWrapper` (gains as a
randomized per-env model field) — deferred.
