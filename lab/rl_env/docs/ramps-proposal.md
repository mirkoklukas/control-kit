# Proposal: ramps (curricula) for config values

Status: proposal, 2026-09-27. Nothing implemented. The design is open; this records
where the discussion landed and what is still undecided.

## Problem

Some reward weights should change during training: a cost that is too strong early
makes the cheapest behaviour "don't move" (seen with `w_support` 10/20: duty 0.92,
vx ~0.01). Today there are two ad-hoc mechanisms, both coded into `env.py` and
`test_policy.py`:

- `k_c` -- Hwangbo et al.'s curriculum factor, `k_c <- k_c ** 0.997` per PPO iteration,
  multiplying a group of terms (the `REG` list in `env.py`).
- `k_support` -- a linear ramp for one term (0 until `support_start` steps, then to 1
  over `support_ramp`).

Adding a ramp for another value means more code in both files. And both have the same
delivery bug (below).

### Found on the way: the ramps never reached the envs

The training callback pushes `k_c` / `k_support` with `venv.set_attr(...)`. SB3's
`set_attr` does a plain `setattr` on the *wrapped* env, so the value lands on the
`Monitor` wrapper as a stray attribute; `WalkEnv` keeps its own. Consequences:

- every training run so far used `k_c = kc0 = 0.4` throughout (the `k_c` in the
  progress lines is the callback's counter, not what the envs used);
- with the current code, `k_support` stays 0 in the envs: the support cost is off.

`venv.env_method("set_wrapper_attr", name, value)` does reach `WalkEnv` (gymnasium's
`set_wrapper_attr` walks through the wrappers). Measured: pushing a whole `WalkEnvCfg`
to 12 subprocesses takes 0.17 ms, ~0.006% of a PPO iteration (2.7 s).

## Prior art (from memory, not re-checked against current versions)

- Hwangbo et al. 2019: one global factor on all costs, geometric growth. Grouping fixed
  in code.
- Isaac Lab (manager-based envs): a curriculum manager with named terms; e.g.
  `modify_reward_weight(term_name, weight, num_steps)` sets a reward term's weight
  after N steps. Registration by name, logic in configured functions.
- legged_gym: performance-driven curricula on the *task* (terrain level, command
  range), not on reward weights.
- optax / SB3: schedules as functions of the step for optimizer settings.

## Proposed design

### 1. The only state that enters the env: a curriculum step

The reward changes over training, and an env cannot know where training stands unless
something tells it. The options were: push the *result* (`k_c`, ramped weights), push
the *input* (a step), or let the env count its own steps.

**Proposal: the env receives a curriculum step (an int). Making sure it does is the
user's job** -- for SB3, the training callback pushes it once per iteration:

```python
venv.env_method("set_wrapper_attr", "curriculum_step", model.num_timesteps)
```

Clock: training env steps (`num_timesteps`, summed over envs). Same for all envs,
independent of `n_envs`, and what checkpoints already record (`ckpt.info["steps"]`), so
a resumed run evaluates its ramps at the same point. (A self-counting env would need
`n_envs` and lockstep stepping; it fits JAX envs better, where the step lives in the
env state.)

### 2. The config holds the ramps and evaluates them

Ramps are data in the config, next to the values they scale, plus one pure method:

```python
@dataclass
class Ramp:
    values: tuple = ()        # config field names it scales, e.g. ("w_support",)
    shape: str = "linear"     # "linear" | "geometric" | ...
    start: int = 0            # curriculum steps before it starts
    length: int = 1           # linear: steps from 0 to 1
    x0: float = 0.0           # geometric: start value, x <- x ** rate per ...
    rate: float = 1.0

@dataclass
class WalkEnvCfg:
    w_support: float = 5.0    # full strength
    ...
    curriculum: Ramp = ...    # today's k_c group
    support: Ramp = ...       # today's k_support

    def at(self, step: int) -> "WalkEnvCfg":
        """This config with every registered field scaled by its ramp's value at `step`.
        Returns a new config; `self` is not modified."""
```

- **Registration by config field name** (`"w_support"`), not reward-term name: any
  numeric field can ramp, not only weights (later e.g. the command speed or
  `reset_noise`).
- **Multiply**: `value_at(step) = value * ramp(step)`, ramp in [0, 1]. The field keeps
  its meaning ("full strength").
- **Pure**: `at(step)` returns a new config (`dataclasses.replace`). The full-strength
  config is what runkit records and what `eval` uses.

### 3. The env asks its config

```python
self.base_cfg = cfg                                  # full strength, never modified
self.cfg = cfg.at(self.curriculum_step)              # what step() reads
```

Re-evaluated when the step changes (once per iteration is enough). The env knows only
"there is a step; ask the config". `REG`, `k_c` and `k_support` disappear from
`env.py`.

**Cached values**: `WalkEnv` derives some constants from its config in `__init__`
(`self.cmd`, `q_safe`, tip / flatness cosines, `max_steps`). If a ramped field is one of
those, they must be recomputed when `self.cfg` changes -- one "derive constants" step,
run in `__init__` and after every re-evaluation. Or: allow ramps only on fields read
live, checked at startup.

### 4. Around it

- **Logging**: each ramp's value per iteration in the metrics, `ramp/<name>`, and in
  the progress line.
- **eval**: full strength by default (`cfg`); optionally "as trained" (`cfg.at(steps of
  the checkpoint)`), to compare an eval's return with the training curve.
- **Training callbacks**: `Progress` currently does metrics, progress, the curriculum
  and checkpointing. With this design the curriculum part shrinks to one push of the
  step; checkpointing could become its own small callback.
- **JAX / MJX later**: `at(step)` over numbers is jit-able, with the step as an array in
  the env state.

## Open questions

- **Where the ramps live**: in `WalkEnvCfg` (self-contained: the env config says how it
  changes over training) or in the training config `PolicyCfg` (the env config stays
  "the reward at full strength", training says how to get there). The `at(step)`
  design leans to the former.
- **Named ramp fields vs a list** of ramps: named fields are CLI-friendly
  (`env.support.start=2e6`); a list is general but awkward to override (and runkit's
  handling of a tuple of dataclasses would need checking).
- **A value in two ramps**: multiply, or reject at startup? Unknown field names
  (typos) should be rejected at startup either way.
- **Shapes**: linear and geometric cover today; cosine / piecewise if needed. Is
  multiplying enough, or do some values need interpolation from a start value (e.g.
  the command speed)?
- **Performance-triggered ramps** ("start once vx > 0.8 cmd", legged_gym style): more
  robust than fixed step counts, but stateful -- state in the checkpoint, and the
  progress line should say why it switched.
- **Does this belong in runkit?** The mechanism is generic (a config value as a function
  of a training step). If it proves itself here, it could move to runkit's
  `proposals.md`.

## Independent of this proposal

The delivery bug should be fixed regardless: replace `set_attr(...)` with
`env_method("set_wrapper_attr", ...)` in `test_policy.py`, so the current `k_c` and
`k_support` actually reach the envs.
