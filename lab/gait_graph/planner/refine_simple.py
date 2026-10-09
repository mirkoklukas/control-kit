"""Refine a sequence of transfers with a simplified model (TOWR-like, quasi-static):
:func:`refine_sequence_simple` (``docs/trajopt-nlp.md``).

The robot as one rigid body (all mass ``M`` at a fixed point of the body frame, the stand
pose's centre of mass, legs included), massless legs, no joints:

- **reach**: each foot inside a box in its shoulder frame (rotating with the body), fitted
  from the real leg (:func:`reach_box`: IK-reachable within the joint limits, the ankle
  within its cone for a level surface);
- **forces** as variables (the supporting feet at each body); equilibrium as equalities,
  ``sum F + M g = 0`` and ``sum (p_j - c) x F_j = 0``; the round-cone margins (adhesion
  included) >= ``margin``;
- **pushes** (12, as the fast push check) carried by the minimum-norm response
  ``F_e = F + G^+ w`` (``G``: the feet's grasp map about the centre of mass): purely
  geometric, no extra variables; their cone margins >= ``margin`` too.

Variables per transfer (33): B_plant, B_lift (offsets from the warm start, 6 + 6), the
planting foot (3), the forces of T at B_plant and of T' at B_lift (9 + 9). Objective:
``sum |F|^2`` + ``reg`` x |offsets|^2. Joint angles afterwards by the closed-form IK
(``Kit.move_body``), then checked with the full model.
"""
from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
from mujoco import mjx
from scipy.optimize import minimize

from controlkit.kinematics.types import Foothold, Posture
from controlkit.se3 import SE3

from .refine import RefineCfg, _cone_margins, _push_dirs
from .stance import move_body
from .step_fast import FastPlanner


def reach_box(kit, nominal_theta, normal=(0.0, 0.0, 1.0), n=4000, valid=0.98, seed=0):
    """An axis-aligned box in the shoulder frame around the foot at ``nominal_theta``, grown
    (the three axes in turns, 5 mm at a time) while at least ``valid`` of its points are
    reachable (some IK branch within
    the joint limits with the tibia within the ankle cone of ``normal``, given in the
    shoulder frame; level body: the world normal).

    Returns:
        ``(lo, hi)``: (3,) corners of the box, shoulder frame (m).
    """
    leg = kit.robot.leg
    lim = jnp.asarray(leg.limits)
    nrm = jnp.asarray(normal, float)
    cos_a = jnp.cos(jnp.deg2rad(kit.cfg.ankle_limit_deg))
    center = leg.foot(jnp.asarray(nominal_theta))

    def ok(p):
        reach, ths = leg.ik_from_foot(p)
        inl = jnp.all((ths >= lim[:, 0]) & (ths <= lim[:, 1]), -1)
        ank = jax.vmap(leg.contact_vector)(ths) @ nrm >= cos_a
        return jnp.any(reach & inl & ank)

    ok_b = jax.jit(jax.vmap(ok))
    u = jax.random.uniform(jax.random.PRNGKey(seed), (n, 3), minval=-1.0, maxval=1.0)
    frac = lambda h: float(jnp.mean(ok_b(center + u * h)))
    half = jnp.full(3, 0.005)
    grown = True
    while grown:                               # grow the axes in turns, 5 mm at a time
        grown = False
        for axis in range(3):
            h = half.at[axis].add(0.005)
            if frac(h) >= valid:
                half, grown = h, True
    return np.asarray(center - half), np.asarray(center + half)


def refine_sequence_simple(pl: FastPlanner, cfg: RefineCfg, start_stance, warms, legs,
                           box) -> dict:
    """Refine a sequence of transfers jointly with the simplified model (module docstring).

    Args:
        pl: the fast planner pieces (kit, climb model, the check settings in ``pl.cfg``).
        cfg: the NLP's knobs (``margin``, ``reg``, the trust region, ``maxiter``, ``ftol``).
        start_stance: (L,) foothold indices before the first transfer.
        warms: the sampled transfers, in order.
        legs: per transfer ``(i_k, i'_k)``.
        box: ``(lo, hi)`` the reach box in the shoulder frame (:func:`reach_box`).

    Returns:
        dict as :func:`.refine.refine_sequence`: ``planted`` / ``lift`` (postures with joint
        angles from the IK; ``ik_ok`` per transfer), ``feet``, ``success``, ``message``,
        ``nit``, ``eq_viol``, ``ineq_viol``, ``cost0``, ``cost``, ``t``.
    """
    kit, cm, fc = pl.kit, pl.cm, pl.cfg
    robot, fh = kit.robot, kit.footholds
    L, K = robot.num_legs, len(warms)
    r = kit.cfg.foot_radius
    M, g = cm.mass, jnp.array([0.0, 0.0, -9.81])
    dirs = _push_dirs(M * 9.81, fc.push_min, fc.twist_min)               # (12, 6)
    lo, hi = jnp.asarray(box[0]), jnp.asarray(box[1])
    # the centre of mass in the body frame (the stand pose; legs included)
    d = mjx.forward(cm.mx, mjx.make_data(cm.mx).replace(qpos=jnp.asarray(cm.model.key_qpos[
        [cm.model.key(k).name for k in range(cm.model.nkey)].index("stand")])))
    base = SE3(jnp.concatenate([d.xquat[1], d.xpos[1]]))
    com_b = base.inverse().apply(d.subtree_com[1])
    pos_start = np.asarray(fh.position[jnp.asarray(start_stance)])
    nrm = jnp.stack([fh.normal[jnp.asarray(w.stance)] for w in warms])  # (K, L, 3)
    f0 = jnp.stack([fh.position[jnp.asarray(w.stance)[i]] for w, (i, _) in zip(warms, legs)])
    Bp0 = jnp.stack([w.plant.body.wxyz_xyz for w in warms])
    Bl0 = jnp.stack([w.lift.body.wxyz_xyz for w in warms])
    I = jnp.array([i for i, _ in legs])
    T_idx = jnp.array([[j for j in range(L) if j != i] for i, _ in legs])
    Tn_idx = jnp.array([[j for j in range(L) if j != n] for _, n in legs])
    nv = 33

    def stances(f):
        cur, out = jnp.asarray(pos_start), []
        for k, (i, _) in enumerate(legs):
            cur = cur.at[i].set(f[k])
            out.append(cur)
        return jnp.stack(out)

    def grasp(P, c):
        """(6, 9) G: forces at the 3 feet P (3, 3) to (force, moment about c)."""
        cols = []
        for a in range(3):
            rr = P[a] - c
            cross = jnp.array([[0.0, -rr[2], rr[1]], [rr[2], 0.0, -rr[0]], [-rr[1], rr[0], 0.0]])
            cols.append(jnp.concatenate([jnp.eye(3), cross]))
        return jnp.concatenate(cols, 1)

    def body_rows(B, F, P_all, n_all, idx):
        """Equilibrium (6), reach rows (>= 0, 4 x 6), cone margins (>= 0, 13 x 3)."""
        c = B.apply(com_b)
        P = P_all[idx] + r * n_all[idx]                                 # the pivots
        Gm = grasp(P, c)
        w_grav = jnp.concatenate([-M * g, jnp.zeros(3)])               # what the feet carry
        eq = Gm @ F.reshape(-1) - w_grav
        sh = robot.shoulders(B)
        loc = jax.vmap(lambda s, p: s.inverse().apply(p))(sh, P_all + r * n_all)   # (L, 3)
        reach = jnp.concatenate([(loc - lo).reshape(-1), (hi - loc).reshape(-1)])
        Gp = Gm.T @ jnp.linalg.inv(Gm @ Gm.T)                           # (9, 6) min-norm response
        Fe = F.reshape(1, -1) + dirs @ Gp.T                            # (12, 9)
        n_idx = n_all[idx]
        marg = jnp.concatenate([_cone_margins(F.reshape(3, 3), n_idx, fc.adhesion, fc.mu)[None],
                                jax.vmap(lambda f: _cone_margins(f.reshape(3, 3), n_idx,
                                                                 fc.adhesion, fc.mu))(Fe)])
        return eq, jnp.concatenate([reach, marg.reshape(-1) - cfg.margin])

    def one(zk, Sk, nk, bp0, bl0, fk0, i, ti, tni):
        Bp = SE3(bp0) @ SE3.exp(zk[0:6])
        Bl = SE3(bl0) @ SE3.exp(zk[6:12])
        f, Fp, Fl = zk[12:15], zk[15:24], zk[24:33]
        ep, ip = body_rows(Bp, Fp, Sk, nk, ti)
        el, il = body_rows(Bl, Fl, Sk, nk, tni)
        eq = jnp.concatenate([ep, el, jnp.atleast_1d((f - fk0) @ nk[i])])
        cost = (jnp.sum(Fp ** 2) + jnp.sum(Fl ** 2)
                + cfg.reg * (jnp.sum(zk[:12] ** 2) + jnp.sum((f - fk0) ** 2)))
        return eq, jnp.concatenate([ip, il]), cost

    def parts(z):
        Z = z.reshape(K, nv)
        S = stances(Z[:, 12:15])
        return jax.vmap(one)(Z, S, nrm, Bp0, Bl0, f0, I, T_idx, Tn_idx)

    objective = lambda z: jnp.sum(parts(z)[2])
    eqs = lambda z: parts(z)[0].reshape(-1)
    ineqs = lambda z: parts(z)[1].reshape(-1)
    fns = {k: jax.jit(f) for k, f in (("obj", objective), ("grad", jax.grad(objective)),
                                      ("eq", eqs), ("jeq", jax.jacfwd(eqs)),
                                      ("in", ineqs), ("jin", jax.jacfwd(ineqs)))}
    np64 = lambda f: (lambda z: np.asarray(f(jnp.asarray(z, jnp.float32)), np.float64))

    # warm start: the forces of the minimum-norm split at the warm bodies
    S0 = stances(f0)

    def warm_forces(B, Sk, nk, idx):
        c = B.apply(com_b)
        P = Sk[idx] + r * nk[idx]
        Gm = grasp(P, c)
        return (Gm.T @ jnp.linalg.solve(Gm @ Gm.T, jnp.concatenate([-M * g, jnp.zeros(3)])))

    z0 = np.concatenate([np.concatenate([
        np.zeros(12), np.asarray(f0[k]),
        np.asarray(warm_forces(SE3(Bp0[k]), S0[k], nrm[k], T_idx[k])),
        np.asarray(warm_forces(SE3(Bl0[k]), S0[k], nrm[k], Tn_idx[k]))]) for k in range(K)])
    tx, tr, tf = cfg.trust_xyz, cfg.trust_rot, cfg.trust_foot
    bounds = []
    for k in range(K):
        bounds += ([(-tx, tx)] * 3 + [(-tr, tr)] * 3) * 2
        bounds += [(float(f0[k, a]) - tf, float(f0[k, a]) + tf) for a in range(3)]
        bounds += [(None, None)] * 18
    t0 = time.perf_counter()
    for f_ in fns.values():
        jax.block_until_ready(f_(jnp.asarray(z0, jnp.float32)))
    t_compile = time.perf_counter() - t0
    t0 = time.perf_counter()
    cost0 = float(np64(fns["obj"])(z0))
    res = minimize(np64(fns["obj"]), z0, jac=np64(fns["grad"]), method="SLSQP", bounds=bounds,
                   constraints=[dict(type="eq", fun=np64(fns["eq"]), jac=np64(fns["jeq"])),
                                dict(type="ineq", fun=np64(fns["in"]), jac=np64(fns["jin"]))],
                   options=dict(maxiter=cfg.maxiter, ftol=cfg.ftol))
    dt = time.perf_counter() - t0
    Z = np.asarray(res.x).reshape(K, nv)

    # joint angles by the IK: the bodies with the refined feet held
    pos = pos_start.copy()
    planted, lift, ik_ok = [], [], []
    for k, ((i, nxt), w) in enumerate(zip(legs, warms)):
        pos[i] = Z[k, 12:15]
        S = jnp.asarray(w.stance)
        pool = Foothold(fh.position.at[S].set(jnp.asarray(pos)), fh.normal)
        Bp = SE3(Bp0[k]) @ SE3.exp(jnp.asarray(Z[k, 0:6], jnp.float32))
        Bl = SE3(Bl0[k]) @ SE3.exp(jnp.asarray(Z[k, 6:12], jnp.float32))
        okp, pp, _ = move_body(robot, kit.scene, kit.cfg, pool, w.planted, S, jnp.ones(L, bool), Bp)
        okl, pq, _ = move_body(robot, kit.scene, kit.cfg, pool, w.lift, S, jnp.ones(L, bool), Bl)
        planted.append(pp)
        lift.append(pq)
        ik_ok.append(bool(okp) and bool(okl))
    return dict(planted=planted, lift=lift, feet=Z[:, 12:15], ik_ok=ik_ok,
                success=bool(res.success), message=str(res.message), nit=int(res.nit),
                eq_viol=float(np.abs(np64(fns["eq"])(res.x)).max()),
                ineq_viol=float(np.maximum(0.0, -np64(fns["in"])(res.x)).max()),
                cost0=cost0, cost=float(res.fun), t=dt,
                prof={"compile (once)": [1, t_compile], "solve": [int(res.nit), dt]})
