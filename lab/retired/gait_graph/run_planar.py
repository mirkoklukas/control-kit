"""gait_graph planar rollout -- plan a flat-ground gait by walking the graph.

A runkit experiment: from a start ``(stance, body)`` we repeatedly ``branch``
(sample witness bodies + candidate stances, score by canonical-body advance,
gate by stability), sample a next stance ``∝ softmax(beta·score)``, step onto it,
and record the ``(stance, posture)`` sequence. The postures form a replayable
qpos trajectory for ``models/weld0.xml`` (quasi-static keyframes, no dynamics).

    uv run --extra mjx python -m lab.gait_graph.run_planar --tag=planar
    uv run --extra mjx python -m lab.gait_graph.run_planar T=30 task_vxy='[0.15,0.0]' S=800

Outputs land in ctx.out/results: rollout.npz (qpos/feet/scores) + rollout.png
(body path + support polygons).
"""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jaxlie import SE3

from runkit import RunContext, experiment, main

from lab.gait_graph.branch import branch
from lab.gait_graph.kinematics import PLANTED
from lab.gait_graph.stance import Posture, Stance, complement, infer_posture
from lab.gait_graph.task import Task

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models" / "weld0.xml"
ALL_IDS = jnp.arange(6)


@dataclass
class Cfg:
    """Knobs for the planar gait-graph rollout (runkit builds this from CLI)."""
    seed: int = 0

    # --- branch sampling widths ---
    B: int = 400            # witness body samples per branch
    S: int = 400            # candidate stance samples per branch
    tau: float = 0.05       # stability gate: min(stb_old, stb_new) must exceed this

    # --- task: desired per-step body xy advance (the canonical-body step) ---
    task_vxy: tuple = (0.1, 0.0)
    sigma: float = 0.3      # task-score bandwidth

    # --- rollout ---
    T: int = 20             # gait steps (transitions)
    select_beta: float = 4.0    # temperature on scores for stance selection (∝ softmax(beta·score))
    start_ids: tuple = (0, 2, 4)   # legs planted at the start (tripod)
    body_z: float = 0.241185       # start body height (home)


def rollout(cfg: Cfg):
    """Walk the graph for ``cfg.T`` steps; return aligned (stances, postures, scores).

    ``stances``/``postures`` have one entry per node (start + each step); ``scores``
    has one per transition (the chosen stance's canonical-body advance).
    """
    key = jax.random.PRNGKey(cfg.seed)
    task = Task(step=jnp.asarray(cfg.task_vxy, float), sigma=jnp.asarray(cfg.sigma, float))

    ids0 = jnp.asarray(cfg.start_ids)
    cur_stance = Stance(ids0, PLANTED[ids0])
    last_stance = Stance(complement(ids0), PLANTED[complement(ids0)])   # swing legs at home
    body = SE3.from_translation(jnp.array([0.0, 0.0, cfg.body_z]))
    _, theta0 = infer_posture(body, Stance(ALL_IDS, PLANTED))   # start posture (all-6 IK at home)
    posture = Posture(body, theta0, PLANTED)                    # feet = home layout (all 6)

    stances = [cur_stance]
    postures = [posture]
    scores = []

    for _ in range(cfg.T):
        key, key_branch, key_sel = jax.random.split(key, 3)
        next_ids = complement(cur_stance.foot_ids)
        ns, post, sc = branch(key_branch, last_stance, cur_stance, posture, next_ids, task,
                              B=cfg.B, S=cfg.S, tau=cfg.tau)
        if sc.shape[0] == 0:
            break                                              # dead end: no feasible successor
        probs = jax.nn.softmax(cfg.select_beta * sc)
        i = int(jax.random.choice(key_sel, sc.shape[0], p=probs))
        # advance: previous stance <- where we stood; current <- the chosen next.
        last_stance, cur_stance = cur_stance, ns[i]
        posture = post[i]
        stances.append(cur_stance)
        postures.append(posture)
        scores.append(float(sc[i]))

    return stances, postures, scores


def _qpos(posture: Posture):
    """weld0.xml qpos (25,) for a posture: [pos(3), quat wxyz(4), theta(18)]."""
    wxyz_xyz = np.asarray(posture.body.wxyz_xyz)               # [qw,qx,qy,qz, px,py,pz]
    quat, pos = wxyz_xyz[:4], wxyz_xyz[4:]
    theta = np.asarray(posture.theta).reshape(-1)             # leg-major [coxa,femur,tibia]*6
    return np.concatenate([pos, quat, theta])


def _plot(stances, body_xy, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    n = len(stances)
    for t, st in enumerate(stances):
        f = np.asarray(st.foot_positions)[:, :2]
        c = plt.cm.viridis(t / max(n - 1, 1))
        ax.plot([*f[:, 0], f[0, 0]], [*f[:, 1], f[0, 1]], "-", color=c, alpha=0.5, lw=1)
        ax.scatter(f[:, 0], f[:, 1], color=c, s=15, zorder=2)
    ax.plot(body_xy[:, 0], body_xy[:, 1], "-o", color="k", ms=3, lw=1.5, zorder=3, label="body")
    ax.legend(loc="best")
    ax.set(title="planar gait rollout (body path + supports)", xlabel="x", ylabel="y")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


@experiment(name="gait_graph_planar")
def run(cfg: Cfg, ctx: RunContext):
    """Roll out the planar gait graph; save the qpos trajectory + a path plot."""
    stances, postures, scores = rollout(cfg)

    qpos = np.stack([_qpos(p) for p in postures])             # (F, 25)
    feet = np.stack([np.asarray(s.foot_positions) for s in stances])   # (F, n, 3)
    foot_ids = np.stack([np.asarray(s.foot_ids) for s in stances])     # (F, n)
    body_xy = qpos[:, :2]
    forward_x = float(body_xy[-1, 0] - body_xy[0, 0])

    results = ctx.out / "results"
    results.mkdir(parents=True, exist_ok=True)
    np.savez(
        results / "rollout.npz",
        qpos=qpos, qvel=np.zeros((qpos.shape[0], 24)),        # quasi-static: qvel = 0
        feet=feet, foot_ids=foot_ids, scores=np.asarray(scores),
        timestep=0.3, model=str(MODEL.relative_to(ROOT)),     # 0.3s/keyframe for replay pacing
    )
    _plot(stances, body_xy, results / "rollout.png")

    print(f"[gait_graph] steps={len(scores)}/{cfg.T}  forward_x={forward_x:+.3f}m  -> {ctx.out}")
    return {"steps": len(scores), "forward_x": forward_x}


if __name__ == "__main__":
    main(run)
