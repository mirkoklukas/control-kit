"""Checks of :mod:`.statics` on sampled stances of the `gait_graph` robot.

    uv run --extra mjx python -m lab.gait_graph.planner.test_statics
    uv run --extra mjx python -m lab.gait_graph.planner.test_statics 100     # stances

1. :func:`min_norm` equals :func:`controlkit.forces.stance_forces`.
2. :func:`least_torque` satisfies equilibrium and never has more torque (L2) than
   min-norm.
3. :func:`hold_margin` (qpax) matches the same LP solved by HiGHS (scipy), for four cases:
   floor with four feet / one leg lifted (no magnets), wall and ceiling with one leg
   lifted (adhesion).
4. On the floor without magnets, a tripod holds at all (s* > 0) exactly when the centre
   of mass lies above its triangle.
5. :func:`point_mass_statics` (centre of mass + feet only) gives the same margin as the
   full model without torque constraints (tau_max=None), on the wall.

Prints one line per check and exits non-zero if any fails.
"""
import sys

import jax
import jax.numpy as jnp
import numpy as np
from mujoco import mjx
from scipy.optimize import linprog

import controlkit.forces as forces

from .. import terrain
from ..robot import make_robot
from ..stance import Kit
from .experiments.torques import TorquesCfg, sample_stances
from .statics import (hold_lp, hold_margin, least_torque, min_norm, point_mass_statics,
                      residual, statics)

ADHESION, MU, TAU_MAX = 40.0, 0.5, 30.0
CASES = {  # name: (gravity, planted feet, adhesion)
    "floor, 4 feet": ((0.0, 0.0, -9.81), (0, 1, 2, 3), 0.0),
    "floor, leg 0 lifted": ((0.0, 0.0, -9.81), (1, 2, 3), 0.0),
    "wall, leg 0 lifted": ((9.81, 0.0, 0.0), (1, 2, 3), ADHESION),
    "ceiling, leg 0 lifted": ((0.0, 0.0, 9.81), (1, 2, 3), ADHESION),
}
TOL_S = 2e-3            # |s*(qpax) - s*(HiGHS)|
TOL_VIOLATION = 1e-2    # qpax solution's constraint violation (N, N m)


def setup(n_stances):
    """The robot's mjx model, its foot ids and ``n_stances`` sampled postures (qpos)."""
    tc = TorquesCfg(n_stances=n_stances)
    cfg = tc.graph
    robot = make_robot(cfg)
    scene = terrain.make_scene(cfg.boxes)
    footholds = terrain.sample_footholds(jax.random.PRNGKey(0), scene, cfg.num_footholds,
                                         edge_margin=cfg.edge_margin)
    kit = Kit(robot, scene, cfg, footholds)
    mx, foot_ids = forces.model_from_robot(
        robot, body_density=cfg.body_density, leg_density=cfg.leg_density,
        foot_density=cfg.foot_density, link_radius=cfg.link_radius,
        foot_radius=cfg.foot_radius, body_half_height=cfg.body_half_height)
    postures = sample_stances(tc, kit, np.random.default_rng(0))
    return mx, foot_ids, jnp.stack([forces.to_qpos(p) for p in postures])


def check(name, ok, detail):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}")
    return ok


def main(argv):
    n = int(argv[0]) if argv else 50
    mx, foot_ids, qpos = setup(n)
    print(f"test_statics: {qpos.shape[0]} stances")
    results = []
    g_floor = jnp.array([0.0, 0.0, -9.81])
    up = jnp.array([0.0, 0.0, 1.0])

    # 1 + 2: min-norm vs. forces.py, least torque
    @jax.jit
    def forces_both(q, planted):
        st = statics(mx, foot_ids, q, g_floor)
        F_mn, tau_mn = min_norm(st, planted)
        F_lt, tau_lt = least_torque(st, planted)
        F_ref, tau_ref = forces.stance_forces(mx, foot_ids, q, planted)
        return (jnp.abs(F_mn - F_ref).max() / jnp.abs(F_ref).max(),
                jnp.abs(tau_mn - tau_ref.reshape(-1)).max() / jnp.abs(tau_ref).max(),
                residual(st, planted, F_lt), jnp.linalg.norm(tau_lt) - jnp.linalg.norm(tau_mn))
    dF, dtau, res, more = (np.array(v) for v in zip(*[
        forces_both(q, jnp.array([True, True, True, False])) for q in qpos]))
    results.append(check("min_norm == forces.stance_forces (relative, float32)",
                         max(dF.max(), dtau.max()) < 1e-2,
                         f"max |dF| / max|F| {dF.max():.1e}, |dtau| / max|tau| {dtau.max():.1e}"))
    results.append(check("least_torque: equilibrium, |tau| <= min-norm",
                         res.max() < 1e-2 and more.max() < 1e-4,
                         f"max residual {res.max():.1e}, max |tau|_lt - |tau|_mn {more.max():.1e}"))

    # 3: hold_margin (qpax) vs. HiGHS
    s_floor_tripod = None
    for name, (g, planted, adhesion) in CASES.items():
        normals = jnp.tile(up, (len(planted), 1))
        run = jax.jit(jax.vmap(lambda q: hold_margin(
            statics(mx, foot_ids, q, jnp.asarray(g)), planted, normals, adhesion, MU, TAU_MAX)))
        s, _, viol = (np.asarray(v) for v in run(qpos))
        ref = []
        for q in qpos:
            c, A_eq, b_eq, G, h = (np.asarray(a, float) for a in hold_lp(
                statics(mx, foot_ids, q, jnp.asarray(g)), planted, normals, adhesion, MU,
                TAU_MAX))
            r = linprog(c, A_ub=G, b_ub=h, A_eq=A_eq, b_eq=b_eq, bounds=(None, None),
                        method="highs")
            ref.append(-r.fun if r.status == 0 else np.nan)
        err = np.abs(s - np.array(ref))
        ok = bool(np.all(np.isfinite(s)) and err.max() < TOL_S and viol.max() < TOL_VIOLATION)
        results.append(check(f"hold_margin == HiGHS ({name})", ok,
                             f"max |ds*| {err.max():.1e}, max violation {viol.max():.1e}, "
                             f"holds {np.mean(s >= 1):.0%}"))
        if name == "floor, leg 0 lifted":
            s_floor_tripod = s

    # 4: floor tripod, no magnets: holds at all <=> centre of mass above the triangle
    fwd = jax.jit(lambda q: mjx.forward(mx, mjx.make_data(mx).replace(qpos=q)))
    inside = []
    for q in qpos:
        d = fwd(q)
        c = np.asarray(d.subtree_com[1][:2])
        tri = np.asarray(d.xpos[foot_ids])[[1, 2, 3], :2]
        cross = [np.cross(np.append(tri[(k + 1) % 3] - tri[k], 0),
                          np.append(c - tri[k], 0))[2] for k in range(3)]
        inside.append(all(x > 0 for x in cross) or all(x < 0 for x in cross))
    agree = np.mean((s_floor_tripod > 1e-3) == np.array(inside))
    results.append(check("floor tripod: s* > 0 <=> centre of mass above the triangle",
                         agree == 1.0, f"agree {agree:.1%}, inside {np.mean(inside):.0%}"))

    # 5: point mass vs. full model, torques out of play (wall, leg 0 lifted)
    g_wall, planted = jnp.array([9.81, 0.0, 0.0]), (1, 2, 3)
    normals = jnp.tile(up, (3, 1))
    mass = float(mx.body_subtreemass[1])

    @jax.jit
    def both(q):
        d = mjx.forward(mx, mjx.make_data(mx).replace(qpos=q))
        s_full = hold_margin(statics(mx, foot_ids, q, g_wall), planted, normals, ADHESION,
                             MU, None)[0]
        pm = point_mass_statics(d.subtree_com[1], d.xpos[foot_ids], mass, g_wall)
        s_pm = hold_margin(pm, planted, normals, ADHESION, MU, None)[0]
        return s_full, s_pm
    s_full, s_pm = (np.array(v) for v in zip(*[both(q) for q in qpos]))
    err = np.abs(s_full - s_pm).max()
    results.append(check("point mass == full model without torque limits (wall)", err < TOL_S,
                         f"max |ds*| {err:.1e}, holds {np.mean(s_pm >= 1):.0%}"))

    print(f"{sum(results)}/{len(results)} passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
