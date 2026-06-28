"""Config for hexapod_ppo v0.

A leaf module so both run.py (the runner) and env.py (the Brax ``Env``) can
import ``Cfg`` without a circular import (env wraps Cfg; run imports env).
"""
import dataclasses
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class Cfg:
    """Config for hexapod_ppo v0 -- all the knobs in one dataclass.

    runkit builds this from CLI key=value tokens (introspecting the ``cfg: Cfg``
    annotation on ``run``), so any field is overridable, e.g.::

        uv run --extra ppo python -m lab.hexapod_ppo.v0.run num_envs=2048 vx=0.4 --tag=fast

    The PPO defaults are GPU/cloud-shaped; for a quick local smoke pass tiny values
    (see smoke.py). Sim dt = 0.004 s.
    """
    # --- task: command a fixed forward speed in +x (base frame) ---
    vx: float = 0.3              # commanded forward speed (m/s)

    # --- control / sim ---
    decimation: int = 10         # sim steps per control step (0.04 s control dt -> ~25 Hz)
    action_scale: float = 0.5    # ctrl = clip(action_scale * action, joint range) [rad]
    episode_length: int = 100    # max control steps before the episode truncates/resets
    reset_joint_noise: float = 0.05   # uniform rad noise on initial joint angles

    # --- reward (the rest default in controlkit RewardWeights) ---
    air_time_target: float = 0.4
    contact_force_thresh: float = 1.0

    # --- termination ---
    z_nominal: float = 0.20      # standing trunk height (m)
    z_min_frac: float = 0.5      # done if trunk z < z_min_frac * z_nominal
    up_min: float = 0.5          # done if base up-axis . world up < up_min (tilt > 60 deg)

    # --- PPO (brax) ---
    seed: int = 0
    num_timesteps: int = 50_000_000  # total env steps to train for (the budget, summed over envs)
    num_envs: int = 4096             # parallel envs stepped each iter (the rollout batch width)
    num_eval_envs: int = 128         # parallel envs used at each eval (separate from training)
    num_evals: int = 10
    batch_size: int = 1024
    num_minibatches: int = 32
    num_updates_per_batch: int = 4
    unroll_length: int = 20       # steps each env rolls out per PPO iteration (rollout segment)
    learning_rate: float = 3e-4
    entropy_cost: float = 1e-2
    discounting: float = 0.97
    gae_lambda: float = 0.95
    clipping_epsilon: float = 0.2
    reward_scaling: float = 1.0
    normalize_observations: bool = True
    policy_hidden: tuple = (128, 128, 128)
    value_hidden: tuple = (256, 256)

    # --- sample episodes (saved each eval: 1 deterministic + N stochastic, for replay/plots) ---
    n_sample_episodes: int = 4   # stochastic episodes per eval (deterministic one always saved too)

    def save(self, path) -> None:
        """Write every field to a YAML file at `path` (complete + self-contained).

        Tuples are written as lists so the YAML round-trips cleanly. The output
        is independent of these defaults -- loading it later reproduces this Cfg
        regardless of how config.py changes.
        """
        data = {
            f.name: (list(v) if isinstance(v := getattr(self, f.name), tuple) else v)
            for f in dataclasses.fields(self)
        }
        Path(path).write_text(yaml.safe_dump(data, sort_keys=False))

    @classmethod
    def load(cls, path) -> "Cfg":
        """Build a Cfg from a YAML file at `path` (plain path, no scheme prefixes).

        Unknown keys are rejected; tuple-typed fields accept YAML lists. Any field
        absent from the file falls back to its default here.
        """
        data = yaml.safe_load(Path(path).read_text()) or {}
        types = {f.name: f.type for f in dataclasses.fields(cls)}
        unknown = set(data) - set(types)
        if unknown:
            raise ValueError(f"unknown Cfg field(s) in {path}: {sorted(unknown)}")
        for k, v in data.items():
            if types[k] in (tuple, "tuple") and isinstance(v, list):
                data[k] = tuple(v)
        return cls(**data)
