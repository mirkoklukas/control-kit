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
import math
import pickle
import time
from datetime import datetime
from pathlib import Path

import jax
import mujoco
import numpy as np

from . import _compat  # noqa: F401  -- patches jax.device_put_replicated for brax 0.14
from brax.training.agents.ppo import networks as ppo_networks # type: ignore
from brax.training.agents.ppo import train as ppo # type: ignore

from runkit import RunContext, experiment, main

from .config import Cfg
from .env import HexapodEnv

ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / "models" / "hexapod.xml"


def _fmt_dur(seconds: float) -> str:
    """Compact h/m/s duration, e.g. '1h02m', '3m45s', '12s'."""
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{sec:02d}s"
    return f"{sec}s"


def _make_episode_sampler(env, make_policy, n_steps, seed):
    """Build a jitted rollout: ``params -> (qpos, qvel)`` for an ``n_steps`` episode.

    Runs the *deterministic* policy from a fixed reset (same initial state every
    call), so the saved episodes are directly comparable across evals. No
    auto-reset: if the robot falls, the trajectory simply records it.
    """
    key = jax.random.PRNGKey(seed)

    @jax.jit
    def sample(params):
        policy = make_policy(params, deterministic=True)
        state = env.reset(key)

        def body(carry, _):
            st, k = carry
            k, ak = jax.random.split(k)
            action, _ = policy(st.obs, ak)
            st = env.step(st, action)
            ps = st.pipeline_state
            return (st, k), (ps.qpos, ps.qvel)

        _, (qpos, qvel) = jax.lax.scan(body, (state, key), None, length=n_steps)
        return qpos, qvel

    return sample


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

    # Total env steps brax will actually run: it rounds the budget up to whole
    # epochs (mirrors ppo.train's own accounting), so steps-left hits 0 at the end.
    # action_repeat defaults to 1; these are control steps (one per env.step).
    steps_per_train_step = cfg.batch_size * cfg.unroll_length * cfg.num_minibatches
    epochs = max(cfg.num_evals - 1, 1)
    total_steps = (
        epochs
        * math.ceil(cfg.num_timesteps / (epochs * steps_per_train_step))
        * steps_per_train_step
    )

    clock = {}  # closure state: t0 (start) + last (time, step) for the interval rate

    def progress(step, metrics):
        now = time.monotonic()
        step = int(step)
        r = float(metrics.get("eval/episode_reward", float("nan")))
        std = float(metrics.get("eval/episode_reward_std", float("nan")))

        clock.setdefault("t0", now)
        elapsed = now - clock["t0"]
        rate = None  # env steps/s since the previous eval
        if "t" in clock and now > clock["t"]:
            rate = (step - clock["step"]) / (now - clock["t"])
        clock["t"], clock["step"] = now, step

        pct = 100.0 * step / total_steps
        left = max(total_steps - step, 0)
        rate_s = f"{rate / 1e3:6.1f}k/s" if rate else "      -- "
        eta = _fmt_dur(left / rate) if rate else "--"
        print(
            f"[hexapod_ppo] {datetime.now():%H:%M:%S} +{_fmt_dur(elapsed):>6} "
            f"step={step:>11,}/{total_steps:,} ({pct:4.1f}%) {rate_s} eta {eta:>6} "
            f"reward={r:7.3f} +/- {std:.3f}"
        )

    # Every eval, roll the current policy out for ~5 s and save a replayable
    # state trajectory to results/ (same .npz layout the MPC `play` viewer reads).
    # Built lazily on the first call, when brax hands us `make_policy`.
    n_sample_steps = max(1, round(5.0 / env.control_dt))
    _sampler = []

    def save_sample_episode(step, make_policy, params):
        if not _sampler:
            _sampler.append(
                _make_episode_sampler(env, make_policy, n_sample_steps, cfg.seed))
        qpos, qvel = _sampler[0](params)
        results = ctx.out / "results"
        results.mkdir(parents=True, exist_ok=True)
        np.savez(
            results / f"sample_episode_{int(step)}.npz",
            qpos=np.asarray(qpos), qvel=np.asarray(qvel),
            timestep=env.control_dt, model=str(MODEL.relative_to(ROOT)),
        )
        print(f"[hexapod_ppo] sample episode @ step {int(step)} -> "
              f"results/sample_episode_{int(step)}.npz")

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
        policy_params_fn=save_sample_episode,
    )

    # Persist params (normalizer + policy) for later eval/replay.
    (ctx.out / "policy.pkl").write_bytes(pickle.dumps(params))
    print(f"[hexapod_ppo] saved policy -> {ctx.out / 'policy.pkl'}")
    return {"config": vars(cfg)}


if __name__ == "__main__":
    main(run)
