"""Policy test: learn to walk forward on the floor with PPO (stable-baselines3, CPU).

A runkit ``Experiment`` with two roles -- ``run`` (train) and ``eval`` (one
deterministic episode of a checkpoint):

    uv run --extra mjx --extra sb3 python -m lab.rl.test_policy steps=10e6 --tag=gait
    uv run --extra mjx --extra sb3 python -m lab.rl.test_policy --branch 2cc9 steps=15e6  # continue
    uv run --extra mjx --extra sb3 python -m lab.rl.test_policy eval        # latest checkpoint
    uv run --extra mjx --extra sb3 python -m lab.rl.test_policy eval 2cc9   # run by id prefix
    uv run --extra mjx ctk play runs/test_policy/latest/checkpoints/current/eval/rollout.npz

Or through runkit, which takes the extras (mjx, sb3) and the runs root from
``lab/rl/experiment.toml`` and relaunches under ``uv run`` itself (from the
repo root):

    runkit run  lab.rl.test_policy steps=10e6 --tag=gait
    runkit run  lab.rl.test_policy --branch 2cc9 steps=15e6 env.w_support=10
    runkit eval lab.rl.test_policy
    runkit eval runs/test_policy/latest/checkpoints/current              # no module needed
    cd "$(runkit latest lab.rl.test_policy)"                           # the latest run dir

Branching (``--branch RUN[:CHECKPOINT]``): a new run that continues from a
checkpoint -- the weights, the optimizer, the normalization stats and the step
count, so the reward schedule goes on where it was. Its config is the parent's
with the command line on top; ``steps`` is the total to reach, counted from the
start of the parent (``steps=15e6`` continues a 10M run by 5M).

Run dirs: ``{root}/test_policy/{date}_{time}_{hex8}[_{tag}]/`` (root from
``--root``, else ``experiment.toml``, else ``./runs``). runkit's records at the top
level (``status.yaml`` carries ``progress`` / ``total`` and the latest checkpoint),
plus what runkit writes when asked, plus this experiment's own files under ``out/``:

    checkpoints/current/                  every 20 PPO iterations and at the end, replacing
      checkpoint.yaml                     the previous one (runkit's ctx.checkpoint; without
                                          a name: numbered ones); info: it, steps, ep_return, vx
      state/model.zip, vecnormalize.pkl   what a branch continues from
      state/config.yaml                   the config training ran with at that step (the
                                          schedule evaluated there)
      eval/eval.yaml                      an eval of this checkpoint: summary, ...
      eval/eval.jsonl                     ... the same as a row per eval ...
      eval/rollout.npz, rollout.xml       ... its replay, and gait.txt; replaced with the
                                          checkpoint
    metrics/run.jsonl                     one row per PPO iteration: the values to watch
                                          (it, steps, fps, ep_return, ep_len, vx, duty, air_s)
    metrics/reward.jsonl                  the reward terms, one row per PPO iteration
    metrics/schedule.jsonl                the scheduled env values pushed to the envs, one
                                          row at the start and one per PPO iteration
    out/progress.png                      training curve, from metrics/run.jsonl
    (older runs: model.zip at the checkpoint's top, the eval in out/eval/ and
    metrics/eval.jsonl)

Config: :class:`PolicyCfg` = the robot model (``mjmodel.*``, :class:`.config.MjModelCfg`
with ``pad_cells`` defaulting to 1), the env's task / reward / curriculum (``env.*``,
:class:`.config.WalkEnvCfg`), ``steps`` / ``n_envs``, and ``seed`` (a fresh random
seed per run, recorded in ``config.yaml``; ``seed=<n>`` repeats a run). E.g. ``steps=5e6
env.w_air=15 mjmodel.kp=12``. ``run`` takes ``key=value`` overrides and
``--tag`` / ``--root`` / ``--branch``; ``eval`` uses the run's own ``config.yaml``.

Environment, observation, action and reward: :mod:`.env` (:class:`WalkEnv`).

Progress: no prints; runkit's ``ctx.progress`` / ``ctx.record`` only. Per PPO iteration,
steps, fps, episode return and length, forward speed, duty and air time go to
``metrics/run.jsonl`` and the reward terms (named by the env's ``info["terms"]``) to ``metrics/reward.jsonl``,
plotted to ``out/progress.png`` at every checkpoint. The scheduled values pushed to the
envs go to ``metrics/schedule.jsonl``.

Reward schedule: ``PolicyCfg._schedule`` (see :mod:`.scheduled_config`). Once per PPO
iteration the progress callback evaluates it at the current step and pushes the
resulting env config into every env; envs start with the schedule at step 0 (a branch:
at its checkpoint's step). eval uses the full-strength ``env`` config.
"""

import dataclasses
import functools
import sys
import time
from pathlib import Path

import numpy as np

from runkit import (Checkpoint, Experiment, RunContext, load_config, random_seed,
                    record, save_config)

from .config import MjModelCfg, WalkEnvCfg
from . import gait
from .env import WalkEnv
from .mjmodel import build
from .scheduled_config import ScheduledConfig, geometric, linear, scale, schedule

REPO = Path(__file__).resolve().parents[2]
N_STEPS = [512, 1024][1];      # PPO rollout length per env and iteration
BATCH_SIZE = [3072, 4096][1]; # We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.


# The default reward schedule (``PolicyCfg._schedule``), in training env steps.
# Shaping costs: Hwangbo et al.'s curriculum factor, x0 ** (rate ** i) per PPO
# iteration i (per = one iteration = 12 envs x 1024 steps).
SHAPING = scale(geometric(x0=0.4, rate=0.997, per=12 * N_STEPS))
# Support cost: off until the policy walks, then ramped in slowly (too strong too early
# and not stepping at all is the cheapest option).
SUPPORT = scale(linear(start=500_000, length=2_000_000))


@dataclasses.dataclass
class PolicyCfg:
    """Config of the policy test: model + env + training knobs + the reward schedule."""
    # one pad cell: magnets are off on the floor, extra cells only cost contacts
    mjmodel: MjModelCfg = dataclasses.field(default_factory=lambda: MjModelCfg(pad_cells=1))
    env: WalkEnvCfg = dataclasses.field(default_factory=WalkEnvCfg)
    steps: int = 10_000_000         # PPO env steps
    n_envs: int = 12                # parallel envs (SubprocVecEnv); 12 of 16 cores
    seed: int = random_seed()       # fresh per run, recorded; PPO + env j gets seed + j
    # How `env` values change over training (scheduled_config.py): mirrors the config,
    # e.g. `_schedule.env.w_support.start=2e6`, `_schedule.env.w_support.enabled=false`,
    # `_schedule.enabled=false` (full strength throughout).
    _schedule: dict = schedule({
        "env.w_torque": SHAPING,
        "env.w_action_rate": SHAPING,
        "env.w_joint_speed": SHAPING,
        "env.w_orient": SHAPING,
        "env.w_height": SHAPING,
        "env.w_clear": SHAPING,
        "env.w_drag": SHAPING,
        "env.w_slip": SHAPING,
        "env.w_ankle": SHAPING,
        "env.w_air": SHAPING,
        "env.w_support": SUPPORT,
    })
    # `_schedule.env.w_support.start=2e6` changes just that leaf: runkit merges an
    # override of a dict field into its default


# ---------------------------------------------------------------------- training
def make_env(cfg: PolicyCfg, rank: int, step: int = 0):
    """A factory for env number ``rank`` (seeded ``cfg.seed + rank``), for SubprocVecEnv.
    Its reward config is the schedule at ``step`` (0: the start of training)."""
    def _f():
        from stable_baselines3.common.monitor import Monitor
        return Monitor(WalkEnv(cfg.mjmodel, ScheduledConfig(cfg)(step).env, cfg.seed + rank))
    return _f


def _plot(rows: list[dict], png_path: Path, *, reward_rows: list[dict] = (),
          cmd_vx: float | None = None, air_target: float | None = None) -> None:
    """The training curve from the per-iteration metric rows (``metrics/run.jsonl``),
    the reward terms from ``metrics/reward.jsonl`` (``reward_rows``), and the commanded
    speed / air-time target as reference lines. Older runs kept the terms (``reward/<k>``
    or plain ``<k>``) and the references (``cmd_vx``, ``air_target``) in their run rows."""
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
    cmd_vx = rows[0].get("cmd_vx") if cmd_vx is None else cmd_vx
    air_target = rows[0].get("air_target") if air_target is None else air_target
    ax[1, 0].plot(x, col("vx"), label="measured")
    if cmd_vx is not None:
        ax[1, 0].axhline(float(cmd_vx), ls="--", c="k", lw=1, label="commanded")
    ax[1, 0].set(title="forward speed", ylabel="body x velocity (m/s)"); ax[1, 0].legend()
    if reward_rows:                              # the reward stream: every key but these
        xr = np.array([float(r["steps"]) for r in reward_rows]) / 1e6
        for k in reward_rows[0]:
            if k not in ("it", "steps") and not k.startswith("_"):
                ax[1, 1].plot(xr, [float(r[k]) for r in reward_rows], label=k, lw=1)
    else:                                        # older runs: "reward/<k>" in the run rows
        for key in rows[0]:
            if key.startswith("reward/"):
                ax[1, 1].plot(x, col(key), label=key.removeprefix("reward/"), lw=1)
    ax[1, 1].set(title="reward terms", ylabel="reward per policy step (mean)"); ax[1, 1].legend(fontsize=7, ncol=2)
    if "duty" in rows[0]:
        ax[2, 0].plot(x, col("duty"))
        ax[2, 0].set(title="duty factor", ylabel="fraction of time a foot is planted")
        ax[2, 1].plot(x, col("air_s"), label="measured")
        if air_target is not None:
            ax[2, 1].axhline(float(air_target), ls="--", c="k", lw=1, label="target")
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
def train(cfg: PolicyCfg, ctx: RunContext, branch: Checkpoint | None = None) -> dict:
    """PPO on :class:`WalkEnv`. Checkpoints, progress and metrics through runkit
    (``ctx.checkpoint`` / ``ctx.progress`` / ``ctx.record``), the curve into
    ``out/``. Returns the last progress row (the run's summary).

    ``branch``: the checkpoint to continue from (``--branch``), else None. Its
    ``state/`` has the model (weights and optimizer) and the normalization stats,
    its ``info`` the iteration and step count -- the schedule continues there.
    ``cfg.steps`` stays the total, counted from the parent's start."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
    import torch

    run_dir = ctx.out
    start = int(branch.info["steps"]) if branch else 0          # env steps so far
    start_it = int(branch.info["it"]) if branch else 0          # PPO iterations so far
    if start >= cfg.steps:
        raise SystemExit(f"the checkpoint is at {start:,} steps, steps={cfg.steps:,}: "
                         f"nothing to train -- pass a larger total, e.g. steps={2 * start}")
    # PPO runs whole iterations of n_envs x n_steps, so it overshoots `steps` up to
    # the next multiple; that multiple is the honest total
    per_iter = cfg.n_envs * N_STEPS
    ctx.progress(start, total=start + -(-(cfg.steps - start) // per_iter) * per_iter)

    n_envs = cfg.n_envs
    envs = SubprocVecEnv([make_env(cfg, s, step=start) for s in range(n_envs)])
    if branch:
        # the parent's running obs / reward statistics, and its model: weights,
        # Adam moments, num_timesteps; the hyperparameters are the parent's too
        venv = VecNormalize.load(str(branch.state / "vecnormalize.pkl"), envs)
        venv.training, venv.norm_reward = True, True
        model = PPO.load(branch.state / "model.zip", env=venv, device="cpu")
        model.set_random_seed(cfg.seed)
    else:
        venv = VecNormalize(envs, norm_obs=True, norm_reward=True, clip_obs=10.0)
        model = PPO(
            "MlpPolicy", venv, n_steps=N_STEPS, batch_size=BATCH_SIZE, n_epochs=5, learning_rate=3e-4,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.005, max_grad_norm=1.0,
            policy_kwargs=dict(net_arch=dict(pi=[256, 128], vf=[256, 128]),
                               activation_fn=torch.nn.Tanh, log_std_init=-1.0),
            verbose=0, device="cpu", seed=cfg.seed)

    schedule = ScheduledConfig(cfg)            # validated here: a bad key fails before training

    class Progress(BaseCallback):
        """Per-iteration terminal line + metric row + progress; reward schedule update;
        checkpoints."""

        def _on_training_start(self):
            self.t0, self.it = time.time(), start_it
            self.rows = []                     # this run's rows, for the curve
            self.reward_rows = []              # this run's reward rows, for the curve
            self._reset_acc()
            self._push(schedule(start))        # the envs were built with it; recorded

        def _push(self, new_cfg):
            """Push ``new_cfg.env`` into every env and record its scheduled values in
            their own stream (the config over training, not a training metric)."""
            # set_wrapper_attr, not set_attr: SB3's set_attr only sets the attribute on
            # the outermost wrapper (Monitor), so WalkEnv would never see it
            self.training_env.env_method("set_wrapper_attr", "cfg", new_cfg.env)
            ctx.record("schedule", it=self.it, steps=self.num_timesteps,
                       **{".".join(e.path): functools.reduce(getattr, e.path, new_cfg)
                          for e in schedule.entries})

        def _reset_acc(self):
            self.acc = {}                      # term -> sum; names from the env's info
            self.vx, self.n, self.planted, self.air = 0.0, 0, 0.0, []
            self.supported = 0                 # steps with >= min_support feet planted

        def _on_step(self):
            for info in self.locals["infos"]:
                for k, v in info["terms"].items():
                    self.acc[k] = self.acc.get(k, 0.0) + v
                self.vx += info["vx"]
                self.planted += float(np.mean(info["planted"]))
                self.supported += int(np.sum(info["planted"]) >= cfg.env.min_support)
                self.air += info["air_td"]
                self.n += 1
            return True

        def _on_rollout_end(self):
            self.it += 1
            ep = [e["r"] for e in model.ep_info_buffer]
            el = [e["l"] for e in model.ep_info_buffer]
            row = {
                "it": self.it, "steps": self.num_timesteps,
                "fps": int((self.num_timesteps - start) / (time.time() - self.t0)),
                "ep_return": float(np.mean(ep)) if ep else 0.0,
                "ep_len": float(np.mean(el)) if el else 0.0,
                "vx": self.vx / max(self.n, 1), 
                "duty": self.planted / max(self.n, 1),
                # fraction of steps with >= min_support feet planted (1 for a crawl)
                "support": self.supported / max(self.n, 1),
                "air_s": float(np.mean(self.air)) if self.air else 0.0,
            }
            # run: the few values to watch training by; the reward terms (mean per
            # policy step, times dt) go to a stream of their own
            terms = {k: v / max(self.n, 1) for k, v in self.acc.items()}
            ctx.record(**row)                  # metrics/run.jsonl
            ctx.record("reward", it=self.it, steps=self.num_timesteps, **terms)
            self.reward_rows.append({"steps": self.num_timesteps, **terms})
            ctx.progress(self.num_timesteps)   # status.yaml
            self.rows.append(row)
            self.last = row
            self._reset_acc()
            # reward schedule: the env config for the next rollout
            self._push(schedule(self.num_timesteps))
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
                # state/: what a branch continues from (see train's `branch`)
                model.save(ckpt.state / "model.zip")
                venv.save(str(ckpt.state / "vecnormalize.pkl"))
                # the config training ran with at this step (the schedule evaluated
                # there): for reading, not needed to continue -- a branch
                # re-evaluates the schedule from `steps`
                save_config(schedule(self.num_timesteps), ckpt.state / "config.yaml")
                last = getattr(self, "last", {})
                ckpt.info.update(it=self.it, steps=self.num_timesteps,
                                 **{k: last[k] for k in ("ep_return", "vx") if k in last})
            _plot(self.rows, run_dir / "progress.png", reward_rows=self.reward_rows,
                  cmd_vx=cfg.env.cmd_vx, air_target=cfg.env.air_target)

        def _on_training_end(self):
            self._checkpoint()          # CHECKPOINT SCHEDULE (2/2): the final one

    progress = Progress()
    # a branch: num_timesteps goes on from the checkpoint's (reset_num_timesteps=False
    # adds total_timesteps to it), so the schedule and the step axis continue
    model.learn(total_timesteps=cfg.steps - start, callback=progress,
                reset_num_timesteps=not branch)
    venv.close()
    last = getattr(progress, "last", {})
    return {k: last[k] for k in ("steps", "ep_return", "ep_len", "vx", "duty", "air_s") if k in last}






def _rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(REPO))
    except ValueError:
        return str(p)


@exp.eval
def evaluate(ckpt: Checkpoint) -> dict:
    """Run one deterministic episode of a trained policy and report how it walks.

    The policy is the one saved in ``ckpt`` (by default the latest checkpoint of the
    latest run). The env is reset with the run's seed and uses the run's full-strength
    ``env`` config from its ``config.yaml`` -- not the scheduled values training used at
    the checkpoint's step -- so every eval scores against the same target reward.

    Steps:
        1. Load the policy and the observation normalization from the checkpoint.
        2. Roll out one episode (until termination or ``episode_s``), logging the
           simulation state and the per-foot contact state at every policy step.
        3. Summarize: return, speed, distance, duty per foot, support violations.
        4. Print the summary and a gait diagram (which feet are planted, when).
        5. Save everything into the checkpoint's ``eval/`` folder.

    Files written to ``ckpt.eval`` (emptied when the checkpoint is saved again):
        eval.yaml: the summary of this eval.
        eval.jsonl: the summary appended as one row per eval.
        gait.txt: the gait diagram, strips and the full per-step table.
        rollout.npz, rollout.xml: the logged episode and its model, for replay with
            ``ctk play``.

    Args:
        ckpt: the checkpoint to evaluate (runkit picks it from the command line).

    Returns:
        The summary dict (also in ``eval.yaml``).
    """
    import yaml
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    # --- 1. load the policy and its observation normalization -----------------------
    # The config is the run's own (full-strength weights), not the checkpoint's
    # scheduled one.
    cfg = load_config(PolicyCfg, ckpt.run / "config.yaml")
    eval_dir = ckpt.eval

    # VecNormalize holds the running mean / std of the observations seen in training;
    # the policy only understands normalized observations. Freeze the statistics
    # (training=False) and keep rewards raw (norm_reward=False).
    venv = VecNormalize.load(str(ckpt.state / "vecnormalize.pkl"),
                             DummyVecEnv([lambda: WalkEnv(cfg.mjmodel, cfg.env, cfg.seed)]))
    venv.training, venv.norm_reward = False, False
    model = PPO.load(ckpt.state / "model.zip", device="cpu")

    # --- 2. roll out one episode ------------------------------------------------------
    # Step the WalkEnv itself and use the VecNormalize only to normalize observations.
    # Stepping the vec env would reset the env automatically at the end of the episode,
    # so the last logged frame would be the reset state instead of the final one.
    env = venv.venv.envs[0]
    obs, _ = env.reset()

    # Per policy step: the simulation state (qpos, qvel, ctrl, for the replay) and the
    # per-foot contact state from the env's info (normal force, planted, position).
    log = {k: [] for k in ("qpos", "qvel", "ctrl", "force", "planted", "foot_pos")}
    ret, vx = 0.0, []
    for _ in range(env.max_steps):
        act, _ = model.predict(venv.normalize_obs(obs), deterministic=True)
        obs, r, terminated, truncated, info = env.step(act)

        d = env.data
        for k, v in (("qpos", d.qpos), ("qvel", d.qvel), ("ctrl", d.ctrl)):
            log[k].append(np.copy(v))
        for k in ("force", "planted", "foot_pos"):
            log[k].append(info[k])

        # the env's raw reward (not normalized) and the forward speed in the body frame
        ret += float(r)
        vx.append(info["vx"])
        if terminated or truncated:
            break

    # --- 3. summarize -----------------------------------------------------------------
    # `planted` is the env's definition: loaded (normal force above contact_force_min)
    # and the pad flat (tilt within pad_tilt_max_deg). Shape (T, 4).
    T = len(log["qpos"])
    planted = np.asarray(log["planted"])
    summary = {
        # which policy, and how long the episode lasted
        "checkpoint": ckpt.name,
        "checkpoint_steps": ckpt.info.get("steps"),
        "steps": T,
        "seconds": round(T * env.dt, 3),
        "full_length": T >= env.max_steps,
        # task: return, commanded vs. measured speed, distance along world x
        "return": round(ret, 3),
        "cmd_vx": cfg.env.cmd_vx,
        "vx": round(float(np.mean(vx)), 4),
        "distance": round(float(log["qpos"][-1][0] - log["qpos"][0][0]), 4),
        # gait: fraction of time each foot is planted, and fraction of steps with
        # fewer than min_support feet planted (0 for a clean crawl)
        "duty": [round(float(x), 3) for x in planted.mean(0)],
        "below_min_support": round(float((planted.sum(1) < cfg.env.min_support).mean()), 3),
    }

    # --- 4. print the summary and the gait diagram ------------------------------------
    print(f"episode: {T} steps ({summary['seconds']} s), return {ret:.2f}, mean vx "
          f"{summary['vx']:.3f} (cmd {cfg.env.cmd_vx}), distance {summary['distance']:.2f} m, "
          f"{'full length' if summary['full_length'] else 'terminated'}")
    print(f"gait: duty per foot {summary['duty']}, <{cfg.env.min_support} feet planted "
          f"{100 * summary['below_min_support']:.0f}% of steps")

    # Two layouts of the same planted array (gait.py): strips (one line per foot,
    # time along the line) for the whole episode, and a table (one row per step) --
    # the first 2 s in the terminal, all of it in gait.txt.
    strip = gait.strips(planted, env.dt, min_support=cfg.env.min_support)
    # table = gait.rows(planted, env.dt, seconds=2.0, min_support=cfg.env.min_support)
    # table_full = gait.rows(planted, env.dt, min_support=cfg.env.min_support)
    print(f"\n{strip}\n")
    print(f"checkpoint: {ckpt.name} ({ckpt.info.get('steps')} training steps)")

    # --- 5. save into the checkpoint's eval/ ------------------------------------------
    (eval_dir / "gait.txt").write_text(f"{strip}\n\n{strip}\n")
    record(eval_dir / "eval.jsonl", **summary)
    (eval_dir / "eval.yaml").write_text(yaml.safe_dump(summary, sort_keys=False))

    # The replay for `ctk play`: the logged episode (npz) with the time per frame and
    # the path of the model it ran on, written next to it as MJCF (xml).
    xml = eval_dir / "rollout.xml"
    spec, _ = build(cfg.mjmodel, write=False)
    xml.write_text(spec.to_xml())
    np.savez(eval_dir / "rollout.npz", **{k: np.asarray(v) for k, v in log.items()},
             timestep=env.dt, model=_rel(xml))
    print(f"saved {_rel(eval_dir)}/eval.yaml, rollout.npz (+ .xml)")
    return summary


if __name__ == "__main__":
    exp.main(sys.argv[1:])
