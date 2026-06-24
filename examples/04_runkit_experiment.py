"""Mock experiment. Structure-only — training is faked.

Run:
    uv run python examples/04_runkit_experiment.py seed=7 num_timesteps=500000 --tag=baseline
    # or, equivalently:
    uv run runkit run examples/04_runkit_experiment.py seed=7 --tag=baseline

The @experiment decorator creates the run dir, captures provenance, etc.
This file describes only WHAT to run (the Config + body); never WHERE it
lands or HOW it's tracked.
"""
import math
import pickle
from dataclasses import dataclass

from runkit import experiment, RunContext, main


# --- PARAMETERS: the experiment definition (cfg) ---
@dataclass
class HexapodConfig:
    seed: int = 1
    num_timesteps: int = 1_000_000
    feet_air_time: float = 1.0
    lin_vel_weight: float = 1.0


# --- THE RUN: receives a prepared ctx, just trains into ctx.out ---
@experiment(name="hexapod")
def run(cfg: HexapodConfig, ctx: RunContext):
    ckpt = ctx.out / "checkpoints"
    reward = 0.0
    for step in range(0, cfg.num_timesteps, 250_000):
        reward = cfg.lin_vel_weight * (1 - math.exp(-step / 4e5)) \
            + 0.1 * cfg.feet_air_time
        (ckpt / f"step_{step}.pkl").write_bytes(
            pickle.dumps({"step": step, "reward": reward}))
    (ctx.out / "final_policy.pkl").write_bytes(
        pickle.dumps({"cfg": vars(cfg), "reward": reward}))
    return {"final_reward": reward}


if __name__ == "__main__":
    main(run)
