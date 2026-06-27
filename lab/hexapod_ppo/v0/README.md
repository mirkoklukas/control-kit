# hexapod_ppo / v0

Train the radial hexapod to **walk straight in +x** with PPO (Brax on MJX).

The reward, ids, command, and kinematics all live in the stable core
(`controlkit.hexapod_reward`, see `docs/hexapod-rewards.md`). This experiment only
glues them to training: `env.py` wraps the MJX dynamics + `compute_reward_2` into
a Brax `Env`, and `__main__.py` runs `brax.training.agents.ppo` as a tracked
runkit experiment, saving the policy into the run dir.

## Run

Everything needs the `ppo` extra (`brax` + the mjx stack):

```bash
# full training (GPU box) -- cloud-shaped defaults in config.py
uv run --extra ppo python -m lab.hexapod_ppo.v0 --tag=baseline

# override any Cfg field as key=value; runkit meta-flags are --flags
uv run --extra ppo python -m lab.hexapod_ppo.v0 vx=0.4 num_envs=2048 --tag=fast
```

MJX only flies on CUDA/TPU. CPU works but is slow, and the Apple GPU can't run MJX
at all (see `docs/gotchas.md`), so real runs happen on a GPU box. Output lands in
an immutable runkit run dir under `runs/` (gitignored): `runs/<date>/<time>_hexapod_ppo_<tag>/policy.pkl`.

## Presets (`configs/`)

Named run variants live in `configs/*.yaml`. runkit resolves them with the `exp:`
prefix (relative to this folder, so cwd doesn't matter) and layers them over the
`config.py` defaults; CLI `key=value` still wins on top, so presets compose with
ad-hoc overrides:

```bash
uv run --extra ppo python -m lab.hexapod_ppo.v0 exp:configs/baseline.yaml --tag=baseline
uv run --extra ppo python -m lab.hexapod_ppo.v0 exp:configs/baseline.yaml vx=0.4 --tag=fast   # override on top
```

- **`baseline.yaml`** — the main A100-40GB run (50M steps), `num_evals=40` so progress
  prints land every ~2 training steps instead of the sparse default.
- **`throughput.yaml`** — baseline with `num_envs=8192` to push A100 utilization (same
  sample budget, fewer sequential unrolls); an A/B against `baseline`.
- **`smoke.yaml`** — the wiring check below.

Unlike CLI flags, YAML *can* set the tuple fields (`policy_hidden` / `value_hidden`):
write them as lists.

## Smoke test (local, proves wiring, won't learn)

Tiny settings so it compiles and runs a couple of PPO iterations end-to-end:

```bash
uv run --extra ppo python -m lab.hexapod_ppo.v0 exp:configs/smoke.yaml --tag=smoke
```

You should see two `eval_reward` lines print and a `policy.pkl` get saved. It runs
on CPU in a minute or two; the reward won't meaningfully improve at this scale.

## Knobs

All in `config.py` (`Cfg`): the task (`vx`), control (`decimation`, `action_scale`,
`episode_length`), termination thresholds, and the PPO hyperparameters. Defaults
are GPU/cloud-shaped. Override scalars on the CLI as `key=value`.

Note: **tuple fields can't be overridden on the CLI** (runkit mangles them), so set
`policy_hidden` / `value_hidden` in `config.py`, not as flags.

### How the PPO knobs fit together

One **training step** = collect a batch of rollouts, then do several epochs of
minibatch SGD on it. The data sizing (all from `brax…ppo.train`):

- **`unroll_length`** — length of one rollout segment (kept intact so GAE runs along
  its time axis).
- **`batch_size`** — size of one **minibatch** = one gradient step, counted in
  *segments* (so a grad step sees `batch_size · unroll_length` transitions).
- **`num_minibatches`** — minibatches per epoch = grad steps per pass over the batch.
  The collected batch is `batch_size · num_minibatches` segments.
- **`num_updates_per_batch`** — epochs (passes) over that same batch; each transition
  is reused this many times.
- **`num_envs`** — *only* the parallelism width of collection: brax runs `num_envs`
  envs for `unroll_length`, repeated `batch_size · num_minibatches / num_envs` times.
  It does **not** change how much data a step trains on, just how parallel the
  collection is (the GPU-utilization knob). Constraint: `batch_size · num_minibatches`
  must be divisible by `num_envs`.

So, per training step:

```
env_step_per_training_step = batch_size · unroll_length · num_minibatches · action_repeat   (note: no num_envs)
```

- **`num_timesteps`** — total env transitions **collected** over the whole run (the
  sample budget, summed over all envs). Rounded *up* to a whole number of epochs, so
  actual ≥ requested. The optimizer touches each transition `num_updates_per_batch`
  times; eval rollouts are extra and not counted here.
- **`num_evals`** — number of eval points: `progress_fn` (and `policy_params_fn`) fire
  once at step 0 plus after each of `num_evals − 1` epochs, roughly evenly spaced in
  env steps. Raise it for more frequent logging (each eval costs a rollout).

## Layout

```
config.py    Cfg -- all knobs (task, control, termination, PPO)
configs/     named run presets (baseline / throughput / smoke), loaded as exp:configs/<name>.yaml
env.py       HexapodEnv (Brax Env): action->ctrl, MJX step x decimation,
             compute_reward_2, 66-dim obs, foot_state/last_action via State.info
__main__.py  @experiment("hexapod_ppo"): build env -> brax PPO -> save policy.pkl
_compat.py   shim: re-adds jax.device_put_replicated (removed in jax 0.10) for brax 0.14
```

## Notes / gotchas

- **`_compat.py` is required.** brax 0.14 calls `jax.device_put_replicated`, which
  jax 0.10 removed; the shim re-adds it. `__main__.py` imports it before brax.
- **Env reset** seeds `FootState.last_contact` from the current stance, so feet
  don't register a spurious touchdown on step 0.
- **Command is fixed** (`Command.straight(vx)`), closed over in the env. To train a
  command-conditioned policy later, sample the command per episode and thread it
  through `State.info` (and handle Brax's AutoReset not resetting `info`).
- **Quieting startup noise:** the mjx Warp-probe prints are silenced by `warp-lang`
  (in the `mjx` extra); XLA GPU autotuning warnings can be hidden with
  `TF_CPP_MIN_LOG_LEVEL=2`.
