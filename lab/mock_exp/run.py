"""Mock experiment — runkit @experiment flow, no real training.

The smallest example of an experiment that plugs into runkit: define a Cfg and
a `run(cfg, ctx)`, and the @experiment decorator creates the run dir, hands the
body a RunContext, freezes the config, and dumps a non-None return value. The
"training" is faked (a toy reward curve), so this is structure-only and runs
anywhere (no jax / MuJoCo).

Run:
    uv run python -m lab.mock_exp.run seed=7 num_steps=500000 --tag=baseline
    # equivalently, via the runkit console script:
    uv run runkit run lab/mock_exp/run.py seed=7 --tag=baseline

Everything written under ctx.out lands in the run dir below ./runs (gitignored).
Useful flags: --dry-run (resolve + print cfg, no run dir), --tag=LABEL,
--runs-dir=DIR, --out=DIR.
"""
import math
import pickle
from dataclasses import dataclass

from runkit import RunContext, experiment, main


@dataclass
class Cfg:
    seed: int = 1
    num_steps: int = 1_000_000
    lr: float = 3e-4
    reward_scale: float = 1.0


@experiment(name="mock")
def run(cfg: Cfg, ctx: RunContext):
    """Fake a training run: write toy checkpoints + a final result into ctx.out."""
    ckpt = ctx.out / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)
    reward = 0.0
    for step in range(0, cfg.num_steps, 250_000):
        reward = cfg.reward_scale * (1 - math.exp(-step / 4e5))
        (ckpt / f"step_{step}.pkl").write_bytes(
            pickle.dumps({"step": step, "reward": reward}))
    (ctx.out / "results" / "final.pkl").write_bytes(
        pickle.dumps({"cfg": vars(cfg), "reward": reward}))
    print(f"[mock] done: seed={cfg.seed}  reward={reward:.4f}  ->  {ctx.out}")
    return {"final_reward": reward}


if __name__ == "__main__":
    main(run)
