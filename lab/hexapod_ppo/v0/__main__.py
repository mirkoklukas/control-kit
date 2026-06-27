"""hexapod_ppo v0 -- train the radial hexapod to walk straight in +x (Brax PPO / MJX).

A runkit experiment: ``@experiment`` creates the run dir and hands the body a
RunContext; we build the MJX ``HexapodEnv`` and hand it to
``brax.training.agents.ppo``. The trained params land in ``ctx.out``.

MJX only flies on CUDA/TPU (CPU works but is slow; the Apple GPU can't run MJX --
see docs/gotchas.md), so the real run happens on a GPU box. The cloud-shaped
defaults live in config.py; override per run on the CLI:

    uv run --extra ppo python -m lab.hexapod_ppo.v0 --tag=baseline
    uv run --extra ppo python -m lab.hexapod_ppo.v0 vx=0.4 num_envs=2048 --tag=slow

Quick local smoke (tiny, just proves the pipeline wires up; won't learn):

    uv run --extra ppo python -m lab.hexapod_ppo.v0 \
        num_envs=8 num_timesteps=4096 batch_size=8 num_minibatches=2 \
        unroll_length=10 episode_length=100 num_evals=1
"""
import functools
import pickle
from pathlib import Path

import mujoco

from . import _compat  # noqa: F401  -- patches jax.device_put_replicated for brax 0.14
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo

from runkit import RunContext, experiment, main

from .config import Cfg
from .env import HexapodEnv

ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / "models" / "hexapod.xml"


@experiment(name="hexapod_ppo")
def run(cfg: Cfg, ctx: RunContext):
    """Train PPO on the walk-straight hexapod env; save params into ctx.out."""
    mj_model = mujoco.MjModel.from_xml_path(str(MODEL))
    env = HexapodEnv(mj_model, cfg)

    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=tuple(cfg.policy_hidden),
        value_hidden_layer_sizes=tuple(cfg.value_hidden),
    )

    def progress(step, metrics):
        r = metrics.get("eval/episode_reward", float("nan"))
        std = metrics.get("eval/episode_reward_std", float("nan"))
        print(f"[hexapod_ppo] step={int(step):>11}  eval_reward={float(r):8.3f} +/- {float(std):.3f}")

    make_inference, params, _ = ppo.train(
        environment=env,
        num_timesteps=cfg.num_timesteps,
        num_envs=cfg.num_envs,
        num_eval_envs=cfg.num_eval_envs,
        num_evals=cfg.num_evals,
        episode_length=cfg.episode_length,
        batch_size=cfg.batch_size,
        num_minibatches=cfg.num_minibatches,
        num_updates_per_batch=cfg.num_updates_per_batch,
        unroll_length=cfg.unroll_length,
        learning_rate=cfg.learning_rate,
        entropy_cost=cfg.entropy_cost,
        discounting=cfg.discounting,
        gae_lambda=cfg.gae_lambda,
        clipping_epsilon=cfg.clipping_epsilon,
        reward_scaling=cfg.reward_scaling,
        normalize_observations=cfg.normalize_observations,
        network_factory=network_factory,
        seed=cfg.seed,
        progress_fn=progress,
    )

    # Persist params (normalizer + policy) for later eval/replay.
    (ctx.out / "policy.pkl").write_bytes(pickle.dumps(params))
    print(f"[hexapod_ppo] saved policy -> {ctx.out / 'policy.pkl'}")
    return {"config": vars(cfg)}


if __name__ == "__main__":
    main(run)
