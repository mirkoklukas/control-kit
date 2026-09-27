"""Policy test: learn to walk forward on the floor with PPO (stable-baselines3, CPU).

A runkit ``Experiment`` with three roles -- ``run`` (train), ``eval`` (one
deterministic episode of the checkpoint) and ``viz`` (summary of a run):

    uv run --extra mjx --extra sb3 python -m lab.rl_env.test_policy steps=10e6 --tag=gait
    uv run --extra mjx --extra sb3 python -m lab.rl_env.test_policy eval        # latest ok run
    uv run --extra mjx --extra sb3 python -m lab.rl_env.test_policy eval 2cc9   # run by id prefix
    uv run --extra mjx --extra sb3 python -m lab.rl_env.test_policy viz         # latest run
    uv run --extra mjx --extra sb3 python -m lab.rl_env.test_policy bench       # env steps/s vs n_envs
    uv run --extra mjx ctk play runs/test_policy/latest/out/eval/rollout.npz

Or through runkit, which takes the extras (mjx, sb3) and the runs root from
``lab/rl_env/experiment.toml`` and relaunches under ``uv run`` itself (from the
repo root; ``bench`` stays ``python -m``):

    runkit run  lab.rl_env.test_policy steps=10e6 --tag=gait
    runkit eval lab.rl_env.test_policy
    runkit viz  lab.rl_env.test_policy
    cd "$(runkit latest lab.rl_env.test_policy)"                           # the latest run dir

Run dirs: ``{root}/test_policy/{date}_{time}_{hex8}[_{tag}]/`` (root from
``--root``, else ``experiment.toml``, else ``./runs``). runkit's records at the top
level (``status.yaml`` carries ``progress`` / ``total`` and the latest checkpoint),
plus what runkit writes when asked, plus this experiment's own files under ``out/``:

    checkpoints/current/                  every 20 PPO iterations and at the end, replacing
      model.zip, vecnormalize.pkl         the previous one (runkit's ctx.checkpoint;
                                          without a name: numbered ones); checkpoint.yaml
                                          info: it, steps, ep_return, vx, k_c
    metrics/run.jsonl                     one row per PPO iteration (ctx.record)
    metrics/eval.jsonl                    one row per eval (ctx.record("eval", ...))
    out/progress.png                      training curve, from metrics/run.jsonl
    out/eval/eval.yaml                    latest eval summary
    out/eval/rollout.npz, rollout.xml     its replay

Config: :class:`PolicyCfg` = the robot model (``model.*``, :class:`.config.ModelCfg`
with ``pad_cells`` defaulting to 1), the env's task / reward / curriculum (``env.*``,
:class:`.config.WalkEnvCfg`), ``steps`` / ``n_envs``, and ``seed`` (a fresh random
seed per run, recorded in ``config.yaml``; ``seed=<n>`` repeats a run). E.g. ``steps=5e6
env.w_air=15 model.kp=12``. ``run`` takes ``key=value`` overrides and
``--tag`` / ``--root``; ``eval`` and ``viz`` use the run's own ``config.yaml``.
``bench`` is not a runkit role (no run dir); it is dispatched here, before runkit.

Environment, observation, action and reward: :mod:`.env` (:class:`WalkEnv`).

Progress: one line per PPO iteration in the terminal (steps, fps, episode return
and length, forward speed, k_c, and the per-term reward breakdown); the same rows go
to ``metrics/run.jsonl``, plotted to ``out/progress.png`` at every checkpoint.
"""

import dataclasses
import sys
import time
from pathlib import Path

import numpy as np

from runkit import Experiment, RunContext, load_metrics, random_seed

from .config import ModelCfg, WalkEnvCfg
from .env import TERMS, WalkEnv
from .model import build

REPO = Path(__file__).resolve().parents[2]
N_STEPS = 1024      # PPO rollout length per env and iteration


@dataclasses.dataclass
class PolicyCfg:
    """Config of the policy test: model + env + training knobs."""
    # one pad cell: magnets are off on the floor, extra cells only cost contacts
    model: ModelCfg = dataclasses.field(default_factory=lambda: ModelCfg(pad_cells=1))
    env: WalkEnvCfg = dataclasses.field(default_factory=WalkEnvCfg)
    steps: int = 10_000_000         # PPO env steps
    n_envs: int = 12                # parallel envs (SubprocVecEnv); 12 of 16 cores
    seed: int = random_seed()       # fresh per run, recorded; PPO + env j gets seed + j


# ---------------------------------------------------------------------- training
def make_env(cfg: PolicyCfg, rank: int):
    """A factory for env number ``rank`` (seeded ``cfg.seed + rank``), for SubprocVecEnv."""
    def _f():
        from stable_baselines3.common.monitor import Monitor
        return Monitor(WalkEnv(cfg.model, cfg.env, cfg.seed + rank))
    return _f


def _plot(rows: list[dict], png_path: Path) -> None:
    """The training curve from the per-iteration metric rows (``metrics/run.jsonl``)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if len(rows) < 2:
        return
    col = lambda k: np.array([float(r[k]) for r in rows])
    # x: PPO's num_timesteps -- policy actions collected so far, summed over all
    # parallel envs (one step = one control step, 1 / policy_hz s of sim time)
    x = col("steps") / 1e6
    xlabel = "training steps, millions (all envs)"
    fig, ax = plt.subplots(3, 2, figsize=(11, 10))
    ax[0, 0].plot(x, col("ep_return"))
    ax[0, 0].set(title="episode return", ylabel="summed reward per episode")
    ax[0, 1].plot(x, col("ep_len"))
    ax[0, 1].set(title="episode length", ylabel="policy steps per episode")
    ax[1, 0].plot(x, col("vx"), label="measured"); ax[1, 0].axhline(float(rows[0]["cmd_vx"]), ls="--", c="k", lw=1, label="commanded")
    ax[1, 0].set(title="forward speed", ylabel="body x velocity (m/s)"); ax[1, 0].legend()
    for k in TERMS:
        # "reward/<term>"; plain "<term>" in runs recorded before the rename; terms a
        # run did not have (e.g. drag in older runs) are skipped
        key = f"reward/{k}" if f"reward/{k}" in rows[0] else k
        if key in rows[0]:
            ax[1, 1].plot(x, col(key), label=k, lw=1)
    ax[1, 1].set(title="reward terms", ylabel="reward per policy step (mean)"); ax[1, 1].legend(fontsize=7, ncol=2)
    if "duty" in rows[0]:
        ax[2, 0].plot(x, col("duty"))
        ax[2, 0].set(title="duty factor", ylabel="fraction of time a foot is planted")
        ax[2, 1].plot(x, col("air_s"), label="measured"); ax[2, 1].axhline(float(rows[0]["air_target"]), ls="--", c="k", lw=1, label="target")
        ax[2, 1].set(title="air time per foot step", ylabel="seconds in the air"); ax[2, 1].legend()
    for a in ax.flat:
        if a.has_data():
            a.set_xlabel(xlabel)
        a.grid(alpha=0.3)
    fig.text(0.5, 0.005, "x axis: environment steps PPO has trained on, summed over all "
             "parallel envs; one step = one policy action (1 / policy_hz s of simulated time)",
             ha="center", fontsize=8, color="0.35")
    fig.tight_layout(rect=(0, 0.02, 1, 1)); fig.savefig(png_path, dpi=100); plt.close(fig)


exp = Experiment("test_policy")

@exp.run
def train(cfg: PolicyCfg, ctx: RunContext) -> dict:
    """PPO on :class:`WalkEnv`. Checkpoints, progress and metrics through runkit
    (``ctx.checkpoint`` / ``ctx.progress`` / ``ctx.record``), the curve into
    ``out/``. Returns the last progress row (the run's summary)."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
    import torch
    
    run_dir = ctx.out
    # PPO runs whole iterations of n_envs x n_steps, so it overshoots `steps` up to
    # the next multiple; that multiple is the honest total
    per_iter = cfg.n_envs * N_STEPS
    ctx.progress(total=-(-cfg.steps // per_iter) * per_iter)   # status.yaml: 0 of total

    n_envs = cfg.n_envs
    venv = VecNormalize(SubprocVecEnv([make_env(cfg, s) for s in range(n_envs)]),
                        norm_obs=True, norm_reward=True, clip_obs=10.0)
    model = PPO(
        "MlpPolicy", venv, n_steps=N_STEPS, batch_size=4096, n_epochs=5, learning_rate=3e-4,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, max_grad_norm=1.0,
        policy_kwargs=dict(net_arch=dict(pi=[256, 128], vf=[256, 128]),
                           activation_fn=torch.nn.Tanh, log_std_init=-1.0),
        verbose=0, device="cpu", seed=cfg.seed)

    class Progress(BaseCallback):
        """Per-iteration terminal line + metric row + progress; curriculum update;
        checkpoints."""

        def _on_training_start(self):
            self.t0, self.k_c, self.it = time.time(), cfg.env.kc0, 0
            self.rows = []                     # all metric rows, for the curve
            self._reset_acc()
            print(f"{n_envs} envs x {model.n_steps} "
                  f"steps/iter | policy dt {venv.get_attr('dt')[0]:.3f}s", flush=True)
            print(f"{'it':>4} {'steps':>9} {'fps':>6} {'return':>7} {'len':>5} {'vx':>6} "
                  f"{'duty':>5} {'air_s':>5} {'k_c':>5} {'k_sup':>5}  "
                  + " ".join(f"{k[:6]:>6}" for k in TERMS), flush=True)

        def _k_support(self):
            """Support ramp: 0 until support_start training steps, then linear to 1 over
            support_ramp steps."""
            e = cfg.env
            return float(np.clip((self.num_timesteps - e.support_start) / max(e.support_ramp, 1),
                                 0.0, 1.0))

        def _reset_acc(self):
            self.acc = {k: 0.0 for k in TERMS}
            self.vx, self.n, self.planted, self.air = 0.0, 0, 0.0, []

        def _on_step(self):
            for info in self.locals["infos"]:
                for k, v in info["terms"].items():
                    self.acc[k] += v
                self.vx += info["vx"]
                self.planted += float(np.mean(info["planted"]))
                self.air += info["air_td"]
                self.n += 1
            return True

        def _on_rollout_end(self):
            self.it += 1
            ep = [e["r"] for e in model.ep_info_buffer]
            el = [e["l"] for e in model.ep_info_buffer]
            row = {"it": self.it, "steps": self.num_timesteps,
                   "fps": int(self.num_timesteps / (time.time() - self.t0)),
                   "ep_return": float(np.mean(ep)) if ep else 0.0,
                   "ep_len": float(np.mean(el)) if el else 0.0,
                   "vx": self.vx / max(self.n, 1), "duty": self.planted / max(self.n, 1),
                   "air_s": float(np.mean(self.air)) if self.air else 0.0,
                   "k_c": self.k_c, "k_support": self._k_support(),
                   "cmd_vx": cfg.env.cmd_vx, "air_target": cfg.env.air_target}
            # reward terms under "reward/<term>" (mean per policy step, times dt)
            row.update({f"reward/{k}": v / max(self.n, 1) for k, v in self.acc.items()})
            ctx.record(**row)                  # metrics/run.jsonl
            ctx.progress(self.num_timesteps)   # status.yaml
            self.rows.append(row)
            self.last = row
            print(f"{row['it']:4d} {row['steps']:9d} {row['fps']:6d} {row['ep_return']:7.2f} "
                  f"{row['ep_len']:5.0f} {row['vx']:6.3f} {row['duty']:5.2f} {row['air_s']:5.2f} "
                  f"{row['k_c']:5.2f} {row['k_support']:5.2f}  "
                  + " ".join(f"{row[f'reward/{k}']:6.3f}" for k in TERMS), flush=True)
            self._reset_acc()
            self.k_c = self.k_c ** cfg.env.kc_rate             # curriculum
            self.training_env.set_attr("k_c", self.k_c)
            self.training_env.set_attr("k_support", self._k_support())
            # CHECKPOINT SCHEDULE (1/2): every 20 PPO iterations (20 x n_envs x N_STEPS
            # = ~245k env steps with 12 envs); (2/2) at the end, in _on_training_end
            if self.it % 20 == 0:
                self._checkpoint()

        def _checkpoint(self):
            """Save the model + normalization stats as runkit checkpoint "current", and
            redraw the curve. When: see CHECKPOINT SCHEDULE -- every 20 PPO iterations
            (_on_rollout_end) and once at the end (_on_training_end)."""
            # one checkpoint, replaced at every save (disk space); complete only once
            # the block exits, the previous one is removed only then. ctx.checkpoint()
            # without a name gives numbered ones: checkpoints/000001/, 000002/, ...
            with ctx.checkpoint("current") as ckpt:
                model.save(ckpt.dir / "model.zip")
                venv.save(str(ckpt.dir / "vecnormalize.pkl"))
                last = getattr(self, "last", {})
                ckpt.info.update(it=self.it, steps=self.num_timesteps, k_c=self.k_c,
                                 **{k: last[k] for k in ("ep_return", "vx") if k in last})
            _plot(self.rows, run_dir / "progress.png")

        def _on_training_end(self):
            self._checkpoint()          # CHECKPOINT SCHEDULE (2/2): the final one

    progress = Progress()
    model.learn(total_timesteps=cfg.steps, callback=progress)
    venv.close()
    print(f"done: {_rel(ctx.dir)} (checkpoints/, metrics/run.jsonl, out/progress.png)")
    last = getattr(progress, "last", {})
    return {k: last[k] for k in ("steps", "ep_return", "ep_len", "vx", "duty", "air_s") if k in last}






def _rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(REPO))
    except ValueError:
        return str(p)


@exp.eval
def evaluate(cfg: PolicyCfg, ctx: RunContext) -> dict:
    """One deterministic episode of the run's latest checkpoint, reset with the run's seed.

    Writes ``out/eval/eval.yaml`` (summary) and ``out/eval/rollout.npz`` + ``.xml``
    (replay), and records the summary to ``metrics/eval.jsonl``.
    """
    import yaml
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    ckpts = ctx.checkpoints()                        # complete ones, oldest first
    if not ckpts:
        raise SystemExit(f"{_rel(ctx.dir)}: no checkpoint yet")
    ckpt = ckpts[-1]
    eval_dir = ctx.out / "eval"
    eval_dir.mkdir(exist_ok=True)

    venv = VecNormalize.load(str(ckpt.dir / "vecnormalize.pkl"),
                             DummyVecEnv([lambda: WalkEnv(cfg.model, cfg.env, cfg.seed)]))
    venv.training, venv.norm_reward = False, False
    model = PPO.load(ckpt.dir / "model.zip", device="cpu")
    env = venv.venv.envs[0]
    env.k_c, env.k_support = 1.0, 1.0           # full costs
    obs = venv.reset()
    log = {k: [] for k in ("qpos", "qvel", "ctrl", "force")}
    ret, vx = 0.0, []
    for _ in range(env.max_steps):
        act, _ = model.predict(obs, deterministic=True)
        obs, r, done, info = venv.step(act)
        d = env.data
        for k, v in (("qpos", d.qpos), ("qvel", d.qvel), ("ctrl", d.ctrl)):
            log[k].append(np.copy(v))
        log["force"].append(info[0]["force"])
        ret += float(venv.get_original_reward()[0]); vx.append(info[0]["vx"])
        if done[0]:
            break
    T = len(log["qpos"])
    F = np.asarray(log["force"])
    planted = F > cfg.env.contact_force_min
    summary = {
        "checkpoint": ckpt.name, "checkpoint_steps": ckpt.info.get("steps"),
        "steps": T, "seconds": round(T * env.dt, 3), "full_length": T >= env.max_steps,
        "return": round(ret, 3), "cmd_vx": cfg.env.cmd_vx, "vx": round(float(np.mean(vx)), 4),
        "distance": round(float(log["qpos"][-1][0] - log["qpos"][0][0]), 4),
        "duty": [round(float(x), 3) for x in planted.mean(0)],
        "below_min_support": round(float((planted.sum(1) < cfg.env.min_support).mean()), 3),
    }
    print(f"episode: {T} steps ({summary['seconds']} s), return {ret:.2f}, mean vx "
          f"{summary['vx']:.3f} (cmd {cfg.env.cmd_vx}), distance {summary['distance']:.2f} m, "
          f"{'full length' if summary['full_length'] else 'terminated'}")
    print(f"gait: duty per foot {summary['duty']}, <{cfg.env.min_support} feet planted "
          f"{100 * summary['below_min_support']:.0f}% of steps")

    print(f"checkpoint: {ckpt.name} ({ckpt.info.get('steps')} training steps)")
    ctx.record("eval", **summary)                    # metrics/eval.jsonl
    (eval_dir / "eval.yaml").write_text(yaml.safe_dump(summary, sort_keys=False))
    xml = eval_dir / "rollout.xml"
    spec, _ = build(cfg.model, write=False)
    xml.write_text(spec.to_xml())
    np.savez(eval_dir / "rollout.npz", **{k: np.asarray(v) for k, v in log.items()},
             timestep=env.dt, model=_rel(xml))
    print(f"saved {_rel(eval_dir)}/eval.yaml, rollout.npz (+ .xml)")
    return summary


@exp.viz
def show(cfg: PolicyCfg, ctx: RunContext) -> None:
    """How did this run go: status, last progress row, eval summary, what to open."""
    import yaml
    d, out = ctx.dir, ctx.out
    status = yaml.safe_load((d / "status.yaml").read_text()) if (d / "status.yaml").exists() else {}
    print(f"status   {status.get('status')}  started {status.get('started')}  "
          f"duration {status.get('duration_s')} s  seed {cfg.seed}")
    if status.get("total"):
        print(f"steps    {status.get('progress') or 0:,} of {status['total']:,}  "
              f"latest checkpoint {status.get('checkpoint')}")
    rows = load_metrics(d, "run")
    if rows:
        r = rows[-1]
        print(f"progress it {r['it']}, {r['steps'] / 1e6:.2f}M steps: return "
              f"{r['ep_return']:.2f}, vx {r['vx']:.3f} (cmd {r['cmd_vx']}), "
              f"duty {r['duty']:.2f}, air {r['air_s']:.2f} s, k_c {r['k_c']:.2f}")
        _plot(rows, out / "progress.png")          # needs >= 2 rows
        if (out / "progress.png").exists():
            print(f"curve    {_rel(out / 'progress.png')}")
    else:
        print("progress none yet")
    ev = out / "eval" / "eval.yaml"
    if ev.exists():
        e = yaml.safe_load(ev.read_text())
        print(f"eval     vx {e['vx']} (cmd {e['cmd_vx']}), distance {e['distance']} m, "
              f"duty {e['duty']}, return {e['return']}")
        print(f"replay   uv run --extra mjx ctk play {_rel(out / 'eval' / 'rollout.npz')}")
    else:
        print(f"eval     none yet -- python -m lab.rl_env.test_policy eval {ctx.id.split('_')[-1]}")


def bench(cfg: PolicyCfg, secs: float = 5.0) -> None:
    """Random-action env steps/s for several n_envs (SubprocVecEnv)."""
    from stable_baselines3.common.vec_env import SubprocVecEnv
    for n in (1, 4, 8, 12, 14, 16):
        venv = SubprocVecEnv([make_env(cfg, s) for s in range(n)])
        venv.reset()
        acts = np.zeros((n, 12), np.float32)
        k, t0 = 0, time.time()
        while time.time() - t0 < secs:
            venv.step(acts + np.random.uniform(-0.3, 0.3, acts.shape).astype(np.float32))
            k += 1
        print(f"n_envs {n:2d}: {k * n / (time.time() - t0):7.0f} env steps/s", flush=True)
        venv.close()


def entry(argv: list[str]) -> None:
    """``bench [key=value ...]`` here; everything else (run / eval / viz / root /
    latest) goes to runkit."""
    if argv and argv[0] == "bench":
        from runkit.config import build_cfg, parse_overrides
        bench(build_cfg(PolicyCfg, parse_overrides(argv[1:])))
    else:
        exp.main(argv)


if __name__ == "__main__":
    entry(sys.argv[1:])
