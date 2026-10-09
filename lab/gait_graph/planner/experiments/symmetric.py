"""Symmetric full stances on the wall: how foot radius and body gap shape forces and torques.

The four feet on the diagonals (radially out along each leg's mount) at a distance
``r`` from the body centre in the surface plane, the body at a gap ``h`` from the surface,
joint angles from the leg IK (knee highest), pads laid flat. A symmetric stance on a
vertical wall is the same stance on the floor with gravity rotated into the surface plane:
here gravity points along the body's -x (down the wall, the body facing up).

Per (r, h), all four feet planted:

- ``s``, ``dist``, ``dist_mean``: the hold margin and the disturbance margins (/ weight);
- ``N``, ``T``: the least-torque foot forces within the limits (unique;
  :func:`..statics.least_torque_qp`): the largest push / pull (+ / -) and tangential force
  over the feet (N). Not the "most central" forces of :func:`..statics.foot_forces_lp`:
  with four feet that LP has many optimal force distributions, and the solver's pick
  hides the peel (constant N);
- ``hip yaw``, ``hip pitch``, ``knee``: the largest |torque| per joint type, for those
  forces (N m), and ``sum tau^2``;
- ``ankle``: the largest ankle angle needed to lay a pad flat (deg); > ``ankle_range_deg``:
  the pad cannot lie flat (that stance is not valid).

    uv run runkit run lab.gait_graph.planner.experiments.symmetric
    uv run runkit run lab.gait_graph.planner.experiments.symmetric gravity=floor

Prints one table (rows r, h); returns the rows.
"""
import dataclasses
import sys
from dataclasses import replace

import jax.numpy as jnp
import numpy as np
from runkit import Experiment, RunContext

from ..climb_model.config import MjModelCfg
from ..climb_model.mjmodel import make_robot
from ..climb_model.poses import pose_qpos
from ..climb_statics import ClimbModel
from ..statics import disturbance_margin, hold_margin, least_torque_qp


@dataclasses.dataclass
class SymmetricCfg:
    """The robot, the grid and the hold parameters."""
    mjmodel: MjModelCfg = dataclasses.field(default_factory=MjModelCfg)
    radii: tuple = (0.20, 0.24, 0.28, 0.32, 0.34)   # foot distance from the body centre (m)
    gaps: tuple = (0.16, 0.13, 0.10, 0.08)          # body (mount plane) to the surface (m)
    gravity: str = "wall"           # wall: g along the body's -x in the surface plane; floor
    adhesion: float = 40.0          # A per foot (N)
    mu: float = 0.5                 # friction coefficient


exp = Experiment("planner_symmetric")


@exp.run
def run(cfg: SymmetricCfg, ctx: RunContext) -> list:
    """Print the table over (r, h).

    Returns:
        One dict per (r, h) with the columns of the table.
    """
    cm = ClimbModel(cfg.mjmodel)
    robot = make_robot(cfg.mjmodel)
    g = jnp.array([-9.81, 0.0, 0.0] if cfg.gravity == "wall" else [0.0, 0.0, -9.81])
    normals = jnp.tile(jnp.array([0.0, 0.0, 1.0]), (4, 1))
    planted = (0, 1, 2, 3)
    tau_max = float(cfg.mjmodel.forcerange[1])
    w = cm.mass * 9.81
    lim = cfg.mjmodel.ankle_range_deg
    print(f"planner_symmetric: {cfg.gravity}, symmetric full stances; weight {w:.1f} N, A = "
          f"{cfg.adhesion:.0f} N, mu = {cfg.mu}, tau_max = {tau_max:.0f} N m, ankle limit {lim:.0f} deg")
    print(f"  {'r (m)':>5} | {'h (m)':>5} | {'ankle':>5} | {'s*':>5} | {'dist':>5} | {'mean':>5} "
          f"| {'N max':>6} | {'N min':>6} | {'T max':>5} | {'hip yaw':>7} | {'hip pitch':>9} "
          f"| {'knee':>5} | {'sum tau^2':>9}")
    rows = []
    for r in cfg.radii:
        for h in cfg.gaps:
            mj = replace(cfg.mjmodel, stand_foot_radius=r)
            try:
                q, _, ankles = pose_qpos(cm.model, robot, mj, h)
            except RuntimeError:
                print(f"  {r:5.2f} | {h:5.2f} | unreachable")
                continue
            ankle = float(np.degrees(np.abs(ankles)).max())
            st = cm.statics(jnp.asarray(q), g)
            s = float(hold_margin(st, planted, normals, cfg.adhesion, cfg.mu, tau_max)[0])
            dist, lam, _ = disturbance_margin(st, planted, normals, cfg.adhesion, cfg.mu, tau_max)
            dist = float(dist) / w if s >= 1 else 0.0
            mean = float(np.mean(lam)) / w if s >= 1 else 0.0
            F, tau, _ = least_torque_qp(st, planted, normals, cfg.adhesion, cfg.mu, tau_max)
            F = np.asarray(F)
            N, T = F[:, 2], np.linalg.norm(F[:, :2], axis=1)
            tau = np.abs(np.asarray(tau)).reshape(4, 3)
            row = dict(r=r, h=h, ankle=ankle, s=s, dist=dist, mean=mean, N_max=float(N.max()),
                       N_min=float(N.min()), T_max=float(T.max()), hip_yaw=float(tau[:, 0].max()),
                       hip_pitch=float(tau[:, 1].max()), knee=float(tau[:, 2].max()),
                       tau2=float((tau ** 2).sum()))
            rows.append(row)
            flag = "*" if ankle > lim else " "
            print(f"  {r:5.2f} | {h:5.2f} | {ankle:4.0f}{flag} | {s:5.2f} | {dist:5.2f} | {mean:5.2f} "
                  f"| {row['N_max']:+6.1f} | {row['N_min']:+6.1f} | {row['T_max']:5.1f} "
                  f"| {row['hip_yaw']:7.2f} | {row['hip_pitch']:9.2f} | {row['knee']:5.2f} "
                  f"| {row['tau2']:9.2f}")
        print("  " + "-" * 104)
    print(f"  * ankle beyond +/-{lim:.0f} deg: the pad cannot lie flat (not a valid stance); "
          f"N, T, torques: least-torque forces within the limits; N: + push, - pull")
    return rows


if __name__ == "__main__":
    exp.main(sys.argv[1:])
