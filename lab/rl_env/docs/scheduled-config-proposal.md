# Proposal: `ScheduledConfig` -- config values that follow a schedule

Status: proposal, rewritten 2026-09-28 (notation: top-level `_schedule`; names `ScheduledConfig` / `_schedule` chosen the same day). Nothing implemented. Records where the
discussion landed and what is still open.

## Problem

Some values should change during training -- mostly reward weights: a cost that is
too strong early makes "don't move" the cheapest behaviour (seen with `w_support`
10 / 20: duty 0.92, vx ~0.01). Today there are two ad-hoc mechanisms, coded into
`env.py` and `test_policy.py`:

- `k_c` -- Hwangbo et al.'s curriculum factor, `k_c <- k_c ** 0.997` per PPO
  iteration, multiplying a fixed group of terms (the `REG` list in `env.py`);
- `k_support` -- a linear ramp for one term (0 until `support_start` steps, then to 1
  over `support_ramp`).

Each new ramp means more code in both files, and nothing about them is visible in the
config beyond a few scalars.

(Found on the way, fixed 2026-09-27: the callback pushed `k_c` / `k_support` with
`venv.set_attr`, which only sets the attribute on the `Monitor` wrapper, so no earlier
run ever saw them change. `env_method("set_wrapper_attr", ...)` reaches the env; pushing
a whole `WalkEnvCfg` to 12 subprocesses takes 0.17 ms, ~0.006% of a PPO iteration.)

## Prior art (read from the sources, 2026-09-28)

**legged_gym** (Isaac Gym, `legged_robot.py`):
- Reward weights never change: scaled by `dt` once in `_prepare_reward_function`.
- Two curricula on the *task*, performance-driven, run in `reset_idx`:
  - terrain (per env): level up if the robot walked > half the terrain length, down
    if it covered < half of what its command asked for;
  - commands (once per episode length): widen the velocity range by +-0.5 m/s when
    the mean tracking reward > 80% of its maximum.
- Clock: `common_step_counter`, counted inside the env.

**Isaac Lab** (`envs/mdp/curriculums.py`, `managers/curriculum_manager.py`):
- A curriculum manager in the env, built from the env config's `curriculum` section:
  named terms `CurrTerm(func=..., params={...})`.
- `_reset_idx` calls `curriculum_manager.compute(env_ids)`: changes happen when envs
  reset, never mid-episode.
- Clock: `env.common_step_counter`, +1 per `env.step()` of the whole batch (so policy
  steps per env). The env owns it: one process, all envs in lockstep.
- `modify_reward_weight(term_name, weight, num_steps)`: a step change -- once the
  counter passes `num_steps`, set the term's weight (in place, in the reward
  manager's term config).
- `modify_env_param(address, modify_fn, modify_params)`: the general form. `address`
  is a dotted path to any env attribute; `modify_fn(env, env_ids, old_value, **params)`
  returns the new value or `NO_CHANGE`; the term reads, calls, writes back in place.
  `modify_term_cfg` is the same with short addresses (`"rewards.<term>.weight"`).
- Terms return a state value (e.g. the current weight) for logging.

**Others**: Hwangbo et al. 2019 (one global factor on all costs, geometric growth);
optax / SB3 (schedules as functions of the step, for optimizer settings).

## Proposed design

Three pieces, each plain and separately testable: the config (values), a
`_schedule` section in the same config (how values change), and a
`ScheduledConfig` that applies one to the other.

### 1. The config values stay plain data

`WalkEnvCfg` (and every other section) keeps its values at full strength -- "the
reward and task as they are meant to be". No methods, no knowledge of curricula. With
the schedules switched off, this is the config; it is what `eval` uses by default.

### 2. `_schedule`: a top-level section that says how config keys are overridden

One `_schedule` section at the top of the experiment config. The leading underscore
marks it as not a value but meta: it configures how other keys are overridden over
training. It **mirrors the config's structure**: under the path of each controlled
entry sits that entry's schedule, including the operation. Chosen notation
(2026-09-28):

```yaml
env:                                       # plain values, full strength
  w_torque: 0.005
  w_support: 5.0
  cmd_vx: 0.15
model:
  kp: 10.0
_schedule:
  enabled: true                            # all schedules on / off
  env:
    w_support:
      enabled: true                        # this one on / off
      op: scale                            # value * s(step)
      kind: linear                         # 0 until start, then to 1 over length
      start: 1000000
      length: 2000000
    w_torque: {op: scale, kind: geometric, x0: 0.4, rate: 0.997}
    cmd_vx:   {op: lerp, from: 0.05, kind: linear, start: 0, length: 3000000}
  model:                                   # any section, e.g. later:
    kp: {op: lerp, from: 5.0, kind: linear, start: 0, length: 2000000}
```

- **At the top**: one place to see -- and switch off -- everything a run schedules,
  across all sections.
- **Nested keys, not dotted strings**: runkit splits CLI keys on dots, so
  `"env.w_support"` as a key could not be addressed from the command line. Nested, a
  schedule's path is its value's path with `_schedule.` in front:
  - `env.w_support=4` -- the full-strength value (its schedule stays);
  - `_schedule.env.w_support.start=2e6` -- the schedule;
  - `_schedule.env.w_support.enabled=false` -- this schedule off;
  - `_schedule.enabled=false` -- all off.
- **Leaf vs branch**: a mapping that contains `op` / `kind` is a schedule; anything
  else is a path step into the config.
- **`enabled`** at two levels (section, key), both default `true`; a schedule is active
  only if both are on. An inactive entry keeps its full-strength value.
- **Operations**: `scale` (`value * s`, for weights: "fraction of full strength"),
  `lerp` (`from + (value - from) * s`, e.g. a command speed from 0.05 to 0.15), later
  `set` if needed. Op-specific parameters (`from`) sit in the entry.
- **Schedules** (`kind`): `linear` (`start`, `length`), `geometric` (Hwangbo's k_c:
  `x0`, `rate`, per what -- see open questions), `step` (`at`; Isaac Lab's
  `modify_reward_weight`). s(step) in [0, 1].
- **Groups repeat** (the ten k_c-scaled weights each carry the same schedule). In YAML
  files, anchors handle it (`&shaping` / `*shaping`); on the CLI they are changed one
  by one.
- **The key path is the name**: for validation ("`env.w_suport` is not a config
  field") and for logging (`schedule/env.w_support` -> its s(step)).
- **Only the top level for now.** `_schedule` sections inside sub-configs (keys
  relative to that level, collected by `ScheduledConfig`, an entry scheduled twice an
  error) are possible, if a need shows up.
- **runkit**: `_schedule` is a plain nested-dict field; `ScheduledConfig` parses the
  entries into schedule objects itself, so runkit only has to pass the dict through
  and merge CLI overrides into it (to verify).

### 3. `ScheduledConfig`: (config, step) -> adjusted config

```python
class ScheduledConfig:
    def __init__(self, cfg): ...
        """Read cfg._schedule, parse the schedules, validate: every key exists in
        cfg and ends at a number. Fails here, not mid-training."""
    def __call__(self, step) -> object:
        """A new config: every active entry changed per its schedule at `step`
        (dataclasses.replace along the path; the config it was built from is not
        modified; `_schedule` is carried along unchanged)."""
    def values(self, step) -> dict: ...
        """s(step) per key path (`env.w_support` -> 0.3), for logging."""
    def sub(self, path) -> "ScheduledConfig": ...
        """The same schedule restricted to a sub-config (e.g. "env"), with keys
        relative to it -- for code that only holds that sub-config."""
```

```python
sc = ScheduledConfig(cfg)
new_cfg = sc(step)
```

"Step" is just a number `ScheduledConfig` does not interpret: normally training
steps, but a caller could pass a curriculum level raised by performance logic -- the
stateful part stays outside, `ScheduledConfig` stays pure.

- **Pure**: same (config, step), same result. No in-place mutation (unlike Isaac Lab),
  so the full-strength config survives, and a resumed run evaluates to the same values
  at the same step.
- **Generic**: it addresses fields by path, so it works with any dataclass config -- it
  could later live in `controlkit` or runkit, without touching `WalkEnvCfg`.
- **Named for what it is**: a config whose values follow a schedule --
  `ScheduledConfig(cfg)(step)`, neutral about the use (curricula, annealing, warm-ups).
  "Curriculum" stays the name for the higher-level idea: a later `EnvCurriculum` that
  acts on an *environment* -- modifier functions `(env) -> None` that change anything,
  Isaac Lab style (terrain levels, command ranges, per-env state).
  `ScheduledConfig` would be one of its modifiers there.

### How the env uses it

The env only holds `WalkEnvCfg`; the schedules for its fields sit at the top. The env
gets its part: `sc.sub("env")`.

```python
# WalkEnv
self.base_cfg = cfg                         # full strength, never modified
self.schedule = schedule                    # ScheduledConfig over WalkEnvCfg (sc.sub("env"))
self.schedule_step = 0                      # delivered from outside (see below)

def reset(...):
    self.cfg = self.schedule(self.schedule_step)       # always from the base
    ...                                                 # then derive cached constants
```

- **At reset**, as in Isaac Lab: a new setting applies from the next episode on, so an
  episode's return never mixes two reward definitions. (Today's push once per PPO
  iteration can change weights mid-episode.)
- **Always from the base**: applying a schedule to an already adjusted config would
  compound the factors.
- **Cached constants**: `WalkEnv` derives some values from its config (`self.cmd`,
  `q_safe`, tip / flatness cosines, `max_steps`). They are recomputed after every
  schedule update -- one "derive constants" step, also used by `__init__`.
- `REG`, `k_c` and `k_support` disappear from `env.py`.
- Alternative: the training loop evaluates the whole config (`sc(step)`) and pushes
  `new_cfg.env` into the envs; the env then knows nothing about curricula, but changes
  land mid-episode unless the env defers them to its next reset.

### The step: the only state that enters the env

An env cannot know where training stands unless told. **The env receives a schedule
step; making sure it does is the user's job.** For SB3 the training callback pushes it
once per iteration:

```python
venv.env_method("set_wrapper_attr", "schedule_step", model.num_timesteps)
```

Clock: training env steps (`num_timesteps`, summed over envs) -- the unit of `steps`
in the config and of `ckpt.info["steps"]`, independent of `n_envs`. (Isaac Lab counts
per-env steps inside the env; that works there because one process steps all envs.
In a JAX / MJX port the step would live in the env state, Isaac-style.)

### Around it

- **Logging**: `sc.values(step)`, recorded as `schedule/<key path>` in the metrics
  and shown in the progress line.
- **eval**: the base config by default (full strength); optionally "as trained":
  `ScheduledConfig(cfg)(checkpoint_steps)`, to compare with the training curve.
- **Training callbacks**: the curriculum (`k_c`) part of `Progress` shrinks to pushing the
  step. Checkpointing could become its own small callback.

## Open questions

- **Data ramps vs modifier functions.** Ramps as data (shapes) are recorded in
  `config.yaml` and overridable from the CLI, but limited to the shapes we implement.
  Isaac Lab's terms are arbitrary functions -- flexible, but a function is not config
  data. A middle way: shapes as data, plus an escape hatch for a custom function
  referenced by name.
- **Operations beyond `scale` and `lerp`**: `set` (the schedule gives the value
  directly)? Tuples or ranges (e.g. a command range widening, legged_gym style)?
- **Task curricula and performance triggers.** legged_gym's main levers are
  performance-driven *task* curricula (command range, terrain level), not weight
  schedules. That fits our floor -> incline -> wall plan. A performance trigger needs
  state (which `EnvCurriculum` would keep, and the checkpoint store) and a way for
  the env to see performance (e.g. the tracking reward at reset). Per-env curricula
  (like terrain levels) would need a curriculum that works per env, as Isaac Lab's
  `env_ids` does. Both belong to the later `EnvCurriculum`, not `ScheduledConfig`.
- **Future: a declared clock unit.** `ScheduledConfig` is unit-free: it maps a number
  through the schedules, and the schedule parameters are in the unit of whatever clock
  the caller feeds. The unit could become part of the `_schedule` section, so it is
  recorded:

  ```yaml
  _schedule:
    clock: steps       # steps (training env steps, default) | progress (fraction of `steps`, 0..1)
  ```

  `progress` makes schedules independent of run length ("start at 10%, done at 30%"),
  like SB3's `progress_remaining` for learning rates. But learning milestones are mostly
  absolute ("walks after ~0.5M steps"), extending / resuming a run would stretch every
  schedule, runs of different length stop being comparable at the same step, and
  performance levels or open-ended runs have no [0, 1]. Hence `steps` as the default.
- **Geometric per what**: Hwangbo's k_c grows per PPO iteration; in training steps that
  is `x0 ** (rate ** (step / steps_per_iteration))` -- `per` as a parameter.
- **Where `ScheduledConfig` lives**: `lab/rl_env` first; `controlkit` or runkit once it is
  generic and proven.
