"""hexapod_ppo v1 -- train the radial hexapod to walk straight in +x (Brax PPO / MJX).

A runkit experiment: ``@experiment`` creates the run dir and hands the body a
RunContext; we build the MJX ``HexapodEnv`` and hand it to
``brax.training.agents.ppo``. The trained params land in ``ctx.out``.

MJX only flies on CUDA/TPU (CPU works but is slow; the Apple GPU can't run MJX --
see docs/gotchas.md), so the real run happens on a GPU box. The cloud-shaped
defaults live in config.py; override per run on the CLI:

    uv run --extra ppo python -m lab.hexapod_ppo.v1.run --tag=baseline
    uv run --extra ppo python -m lab.hexapod_ppo.v1.run vx=0.4 num_envs=2048 --tag=slow

Quick local smoke (tiny, just proves the pipeline wires up; won't learn):

    uv run --extra ppo python -m lab.hexapod_ppo.v1.run \
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
import jax.numpy as jnp
import mujoco
import numpy as np
from rich.console import Console

from . import _compat  # noqa: F401  -- patches jax.device_put_replicated for brax 0.14
from brax.training.agents.ppo import networks as ppo_networks # type: ignore
from brax.training.agents.ppo import train as ppo # type: ignore

from runkit import RunContext, experiment, main

from controlkit import ui
from controlkit.viz import plot_episodes

from .config import Cfg
from .env import HexapodEnv

ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / "models" / "hexapod.xml"

console = Console()  # rich console for readable progress output


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


def _make_episode_sampler(env, make_policy, n_steps, seed, n_stochastic):
    """Build a jitted sampler: ``params -> (det, sto)`` rollouts of ``n_steps``.

    Returns ``(sample, term_names)``. ``sample`` gives one **deterministic** episode
    from a fixed reset (same initial state every call, so it is comparable across
    evals) plus ``n_stochastic`` episodes from the *sampling* policy with independent
    reset + action noise (so they show the exploration spread). All run without
    auto-reset: a fall is just recorded. Each rollout yields
    ``(qpos, qvel, done, reward, terms)`` stacked along the time axis (``terms`` is
    ``[n, len(term_names)]``, the weighted per-term contributions that sum to
    ``reward``, in ``term_names`` order); the stochastic batch is ``vmap``-ed, so its
    leaves carry a leading ``[K]`` axis.

    ``term_names`` is the per-term label order, derived from the env's reward metrics
    keys (``reward/*``) and aligned with the ``terms`` columns. Pulled via
    ``eval_shape`` (abstract, no device run -- avoids the one-time MJX step compile).
    """
    base = jax.random.PRNGKey(seed)

    probe = jax.eval_shape(
        env.step,
        jax.eval_shape(env.reset, base),
        jax.ShapeDtypeStruct((env.action_size,), jnp.float32),
    )
    term_names = tuple(
        p[-1].key[len("reward/"):]
        for p, _ in jax.tree_util.tree_leaves_with_path(probe.metrics)
    )

    def rollout(policy, reset_key, act_key):
        state = env.reset(reset_key)

        def body(carry, _):
            st, k = carry
            k, ak = jax.random.split(k)
            action, _ = policy(st.obs, ak)
            st = env.step(st, action)
            ps = st.pipeline_state
            terms = jnp.stack(jax.tree_util.tree_leaves(st.metrics))  # sorted-key order
            return (st, k), (ps.qpos, ps.qvel, st.done, st.reward, terms)

        _, out = jax.lax.scan(body, (state, act_key), None, length=n_steps)
        return out

    @jax.jit
    def sample(params):
        det_policy = make_policy(params, deterministic=True)
        sto_policy = make_policy(params, deterministic=False)
        det = rollout(det_policy, base, base)                  # fixed reset
        keys = jax.random.split(jax.random.fold_in(base, 1), n_stochastic)

        def one(k):
            rk, ak = jax.random.split(k)
            return rollout(sto_policy, rk, ak)

        sto = jax.vmap(one)(keys)                              # leaves: [K, n, ...]
        return det, sto

    return sample, term_names


@experiment(name="hexapod_ppo_v1")
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
        rate_s = f"{rate / 1e3:.1f}k/s" if rate else "--"
        eta = _fmt_dur(max(total_steps - step, 0) / rate) if rate else "--"
        rcolor = "green" if r >= 0 else "red"

        console.print(
            f"[dim]{datetime.now():%H:%M:%S}[/] "
            f"[bold cyan]{pct:5.1f}%[/] "
            f"step [bold]{step:,}[/][dim]/{total_steps:,}[/]  "
            f"[dim]{rate_s} · eta {eta} · +{_fmt_dur(elapsed)}[/]  "
            f"reward [bold {rcolor}]{r:7.3f}[/] [dim]± {std:.3f}[/]"
        )
        # Per-term reward breakdown (episode sums over eval envs); these add up to
        # the total above. Keys come straight from the env's reward/* metrics.
        prefix = "eval/episode_reward/"
        terms = {k[len(prefix):]: round(float(v), 3)
                 for k, v in metrics.items() if k.startswith(prefix)}
        ui.print_tree(terms, label="terms")

    # Every eval, roll the current policy out for ~5 s and save replayable state
    # trajectories to results/ (same .npz layout the MPC `play` viewer reads) plus
    # a state plot. Files are indexed by eval counter (0,1,2,...), then sample idx:
    # idx 0 is the deterministic episode, 1..K the stochastic ones. The sampler is
    # built lazily on the first call, when brax hands us `make_policy`.
    n_sample_steps = max(1, round(5.0 / env.control_dt))
    z_min = cfg.z_min_frac * env.stand_height
    _sampler = []      # lazily-built sampler (closure over make_policy)
    _eval = [0]        # progress-update counter -> filename index

    def save_sample_episode(step, make_policy, params):
        if not _sampler:
            _sampler.append(_make_episode_sampler(
                env, make_policy, n_sample_steps, cfg.seed, cfg.n_sample_episodes))
        sample, term_names = _sampler[0]
        det, sto = sample(params)
        e = _eval[0]
        _eval[0] += 1

        # Each rollout is (qpos, qvel, done, reward, terms); idx 0 = deterministic,
        # 1..K = the stochastic batch (peel off its leading [K] axis). `terms` is
        # [n, len(TERM_NAMES)], the weighted per-term contributions summing to reward.
        det = [np.asarray(a) for a in det]
        sto = [np.asarray(a) for a in sto]
        episodes = [(*[a for a in det], "deterministic")]
        for i in range(sto[0].shape[0]):
            episodes.append((*[a[i] for a in sto], "stochastic"))

        results = ctx.out / "results"
        results.mkdir(parents=True, exist_ok=True)
        for idx, (qpos, qvel, done, reward, terms, kind) in enumerate(episodes):
            np.savez(
                results / f"sample_episode_{e:03d}_{idx}.npz",
                qpos=qpos, qvel=qvel, done=done,
                reward=reward, reward_terms=terms, term_names=np.asarray(term_names),
                timestep=env.control_dt, model=str(MODEL.relative_to(ROOT)),
                step=int(step), kind=kind, cmd_vx=cfg.vx, z_min=z_min,
            )
        plot = results / f"sample_episodes_{e:03d}.png"
        plot_episodes(
            [(q, v, k) for q, v, _d, _r, _t, k in episodes], out=plot,
            control_dt=env.control_dt, cmd_vx=cfg.vx, z_min=z_min, step=int(step))

        console.print(
            f"  [dim]eval {e:03d} @ step {int(step):,}: saved {len(episodes)} episodes "
            f"(1 det + {len(episodes) - 1} stochastic) + {plot.name}[/]"
        )
        if int(step) == 0:
            console.print(
                "  [yellow]compiling training step (one-time, minutes) -- "
                "next progress line after the first epoch.[/]"
            )

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
    console.print(f"[green]saved policy[/] -> {ctx.out / 'policy.pkl'}")
    return {"config": vars(cfg)}


if __name__ == "__main__":
    main(run)
