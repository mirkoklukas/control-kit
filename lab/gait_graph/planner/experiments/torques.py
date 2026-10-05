"""Actuator torques of sampled stances: minimum-norm vs. least-torque foot forces.

For random body poses on flat ground, sample a stance with `gait_graph`'s :class:`Kit`,
then compute the foot forces and actuator torques (:mod:`..statics`) for the four-foot
stance and for each tripod (one leg lifted), two ways:

- min-norm: zero internal forces (what :func:`controlkit.forces.stance_forces` gives);
- least torque: the internal forces chosen to minimize ``|tau|^2``.

    uv run runkit run lab.gait_graph.planner.experiments.torques
    uv run runkit run lab.gait_graph.planner.experiments.torques n_stances=64 gravity_tilt_deg=90

Prints, per support (all four feet, or leg i lifted): the peak |tau| and |tau| (L2) of
both, how much least torque saves, and the largest equilibrium residual (should be ~0).
Saves one row per (stance, support) to ``out/torques.jsonl``.

No limits yet (adhesion, friction, torque bounds): "holds" is the next step.
"""
import dataclasses
import json
import math
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

import controlkit.forces as forces
from controlkit.se3 import SE3
from runkit import Experiment, RunContext

from ...config import Cfg
from ...robot import make_robot
from ...stance import Kit
from ... import terrain
from ..statics import least_torque, min_norm, residual, statics


@dataclasses.dataclass
class TorquesCfg:
    """The `gait_graph` robot and terrain (``graph.*``), and the sampling."""
    graph: Cfg = dataclasses.field(default_factory=Cfg)
    n_stances: int = 32             # stances to sample
    seed: int = 0
    body_xy: float = 0.15           # body position ~ U[-body_xy, body_xy] in x and y (m)
    body_dz: float = 0.05           # body height ~ body_height + U[-body_dz, body_dz] (m)
    body_yaw_deg: float = 30.0      # body yaw ~ U[-.., ..] (deg)
    gravity_tilt_deg: float = 0.0   # gravity tilted from -z by this (0: floor, 90: wall)
    gravity_azimuth_deg: float = 0.0


def gravity(tilt_deg: float, azimuth_deg: float) -> np.ndarray:
    """(3,) gravity vector, tilted from -z towards azimuth (about z, from +x)."""
    th, ph = math.radians(tilt_deg), math.radians(azimuth_deg)
    return 9.81 * np.array([math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph),
                            -math.cos(th)])


def body_pose(x, y, z, yaw) -> SE3:
    """A level body pose at (x, y, z) with the given yaw (rad)."""
    q = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]                 # wxyz
    return SE3(jnp.asarray([*q, x, y, z], float))


def sample_stances(cfg: TorquesCfg, kit: Kit, rng: np.random.Generator):
    """Up to ``n_stances`` valid stances at random body poses.

    Returns:
        A list of postures (:class:`controlkit.kinematics.Posture`), one per valid stance.
    """
    key, postures, tries = jax.random.PRNGKey(cfg.seed), [], 0
    while len(postures) < cfg.n_stances and tries < 10 * cfg.n_stances:
        tries += 1
        body = body_pose(*rng.uniform(-cfg.body_xy, cfg.body_xy, 2),
                         cfg.graph.body_height + rng.uniform(-cfg.body_dz, cfg.body_dz),
                         math.radians(rng.uniform(-cfg.body_yaw_deg, cfg.body_yaw_deg)))
        key, k = jax.random.split(key)
        ok, posture, _ = kit.sample_stance(k, kit.candidates(body), body)
        if bool(ok):
            postures.append(posture)
    return postures


exp = Experiment("planner_torques")


@exp.run
def run(cfg: TorquesCfg, ctx: RunContext) -> dict:
    """Sample stances, compute min-norm and least-torque torques per support, print.

    Returns:
        Summary per support: mean peak |tau| for both, mean saving, max residual.
    """
    g_cfg = cfg.graph
    L = g_cfg.num_legs
    robot = make_robot(g_cfg)
    scene = terrain.make_scene(g_cfg.boxes)
    footholds = terrain.sample_footholds(jax.random.PRNGKey(cfg.seed), scene,
                                         g_cfg.num_footholds, edge_margin=g_cfg.edge_margin)
    kit = Kit(robot, scene, g_cfg, footholds)
    mx, foot_ids = forces.model_from_robot(           # the mass model gait_graph scores with
        robot, body_density=g_cfg.body_density, leg_density=g_cfg.leg_density,
        foot_density=g_cfg.foot_density, link_radius=g_cfg.link_radius,
        foot_radius=g_cfg.foot_radius, body_half_height=g_cfg.body_half_height)
    mass = float(mx.body_subtreemass[1])
    g = gravity(cfg.gravity_tilt_deg, cfg.gravity_azimuth_deg)

    t0 = time.perf_counter()
    postures = sample_stances(cfg, kit, np.random.default_rng(cfg.seed))
    print(f"torques: {len(postures)} stances sampled ({time.perf_counter() - t0:.0f} s); "
          f"robot {mass:.2f} kg, {robot.leg.num_joints} joints per leg; gravity "
          f"tilt {cfg.gravity_tilt_deg:.0f} deg, azimuth {cfg.gravity_azimuth_deg:.0f} deg")

    supports = {"all four": np.ones(L, bool)}
    supports |= {f"leg {i} lifted": np.arange(L) != i for i in range(L)}

    @jax.jit
    def solve(qpos, planted):
        st = statics(mx, foot_ids, qpos, g)
        F_mn, tau_mn = min_norm(st, planted)
        F_lt, tau_lt = least_torque(st, planted)
        fz_planted = jnp.where(planted, F_lt[:, 2], jnp.inf)     # planted feet only
        return dict(peak_mn=jnp.abs(tau_mn).max(), l2_mn=jnp.linalg.norm(tau_mn),
                    peak_lt=jnp.abs(tau_lt).max(), l2_lt=jnp.linalg.norm(tau_lt),
                    res_mn=residual(st, planted, F_mn), res_lt=residual(st, planted, F_lt),
                    fz_min=fz_planted.min())

    rows = []
    with open(ctx.out / "torques.jsonl", "w") as f:
        for k, posture in enumerate(postures):
            qpos = forces.to_qpos(posture)
            for name, planted in supports.items():
                r = {key: float(v) for key, v in solve(qpos, jnp.asarray(planted)).items()}
                r |= {"stance": k, "support": name}
                rows.append(r)
                f.write(json.dumps(r) + "\n")
            ctx.progress(k + 1, total=len(postures))

    print(f"\n  {'support':<13} | {'peak |tau| (N m)':^25} | {'|tau| L2 (N m)':^25} "
          f"| {'max residual':>12} | {'feet pull':>9}")
    print(f"  {'':<13} | {'min-norm':>11} {'least':>13} | {'min-norm':>11} {'least':>13} |"
          f"{'':>14}|")
    summary = {}
    for name in supports:
        rs = [r for r in rows if r["support"] == name]
        col = lambda key: np.array([r[key] for r in rs])
        save = 100 * (1 - col("l2_lt") / col("l2_mn"))
        print(f"  {name:<13} | {col('peak_mn').mean():11.2f} {col('peak_lt').mean():13.2f} "
              f"| {col('l2_mn').mean():11.2f} {col('l2_lt').mean():9.2f} "
              f"({save.mean():3.0f}%) | {max(col('res_mn').max(), col('res_lt').max()):12.1e} "
              f"| {100 * (col('fz_min') < 0).mean():8.0f}%")
        summary[name] = {"peak_min_norm": round(col("peak_mn").mean(), 3),
                         "peak_least": round(col("peak_lt").mean(), 3),
                         "l2_saving_pct": round(save.mean(), 1),
                         "residual_max": float(max(col("res_mn").max(), col("res_lt").max())),
                         "pull_pct": round(100 * (col("fz_min") < 0).mean(), 1)}
    print("  (means over the stances; % = mean L2 saving of least torque; residual: "
          "equilibrium |A F - b|, ~0;\n   feet pull: stances where a planted foot needs a "
          "negative vertical force (least torque) -- would need adhesion on flat ground)")
    print(f"  saved {ctx.out}/torques.jsonl")
    return summary


if __name__ == "__main__":
    exp.main(sys.argv[1:])
