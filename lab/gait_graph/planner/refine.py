"""Refine one transfer with a smooth NLP: :func:`refine_transfer` (``docs/trajopt-nlp.md``,
"Planning / refining NLP over transfers").

A transfer ``(T, S, T')`` (:mod:`.transfer`): the planting leg i lands from B_plant (closing
S), the body moves on the four feet of S to B_lift, the lifting leg i' lifts. Starting from
a sampled transfer (the warm start), optimize with SLSQP (Sequential Least Squares
Programming, ``scipy.optimize.minimize``), gradients and constraint Jacobians from JAX:

- **variables** (39): B_plant and B_lift as tangent offsets from the warm start
  (``B = B_warm exp(delta)``, 6 + 6), the planting foot (3, a free 3D point), the 12 joint
  angles at B_plant and at B_lift (12 + 12);
- **equalities**: forward kinematics, all four feet of S reached at both bodies (each foot
  centre ``foothold + r n``); the planting foot on the surface (the warm foothold's plane);
- **inequalities**: the ankle cone (tibia within ``ankle_limit_deg`` of the normal), and
  stability as smooth rows: the least-torque split (:func:`.statics.least_torque_kkt`) of
  gravity and of the 12 pushes within adhesion, the friction pyramids and tau_max
  (:func:`.statics.limit_rows`), with a margin ``margin`` (N, N m), for T at B_plant
  (leg i at its planted angles) and T' at B_lift;
- **bounds**: joint limits; a trust region around the warm start (``trust_xyz``,
  ``trust_rot``, ``trust_foot``), since collisions are not constraints (v0);
- **objective**: sum tau^2 (undisturbed) at B_plant (T) + at B_lift (T'), plus
  ``reg`` x |offsets|^2;
- optionally a **goal** B': B_lift fixed to it (planning from B to B' with one transfer).

Not in v0: clearance (collisions are only checked afterwards), the node path B -> B_plant,
the in-air leg's posture.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from controlkit.kinematics.types import Posture
from controlkit.se3 import SE3

from .scoring import G
from .stance import foot_center
from .statics import least_torque_kkt, limit_rows
from .step_fast import FastPlanner
from .transfer import TransferResult


@dataclass
class RefineCfg:
    """The NLP's knobs."""
    margin: float = 0.0             # stability rows must hold with this slack (N, N m)
    reg: float = 1e-3               # weight on |offsets|^2 (m^2, rad^2)
    trust_xyz: float = 0.03         # bound on the body offsets' translation (m)
    trust_rot: float = 0.15         # ... and rotation (rad)
    trust_foot: float = 0.03        # bound on the planting foot's move (m)
    maxiter: int = 200              # SLSQP iterations
    ftol: float = 1e-8              # SLSQP tolerance


def _push_dirs(weight, push_min, twist_min):
    f = jnp.concatenate([jnp.eye(3), jnp.zeros((3, 3))], 1) * (push_min * weight)
    m = jnp.concatenate([jnp.zeros((3, 3)), jnp.eye(3)], 1) * (twist_min * weight)
    return jnp.concatenate([f, -f, m, -m])                             # (12, 6)


def refine_transfer(pl: FastPlanner, cfg: RefineCfg, warm: TransferResult, i: int,
                    next_leg: int, goal: SE3 = None) -> dict:
    """Refine one transfer.

    Args:
        pl: the fast planner pieces (kit, climb model, the check settings in ``pl.cfg``).
        cfg: the NLP's knobs.
        warm: the sampled transfer (:func:`.transfer.plan_transfer`).
        i: the planting leg.
        next_leg: the lifting leg.
        goal: the end body B' (SE3): B_lift is fixed to it (its offset variables bounded to
            0; the warm start's joint angles at B_lift then start infeasible). None: B_lift
            free (a variable around the warm start).

    Returns:
        dict: ``plant`` / ``planted`` / ``lift`` (refined postures), ``stance`` (S),
        ``success``, ``message``, ``nit`` (SLSQP), ``eq_viol``, ``ineq_viol`` (largest
        violations), ``cost0``, ``cost`` (objective before / after), ``t`` (s).
    """
    kit, cm, fc = pl.kit, pl.cm, pl.cfg
    robot, leg, fh = kit.robot, kit.robot.leg, kit.footholds
    L = robot.num_legs
    S = jnp.asarray(warm.stance)
    T_idx = tuple(j for j in range(L) if j != i)
    Tn_idx = tuple(j for j in range(L) if j != next_leg)
    pos0 = fh.position[S]                                              # (L, 3)
    nrm = fh.normal[S]                                                 # (L, 3)
    f0 = pos0[i]
    Bp0, Bl0 = warm.plant.body, (warm.lift.body if goal is None else goal)
    r = kit.cfg.foot_radius
    cos_a = float(np.cos(np.radians(kit.cfg.ankle_limit_deg)))
    dirs = _push_dirs(cm.mass * 9.81, fc.push_min, fc.twist_min)
    tau_max = float(cm.mj.forcerange[1])

    def unpack(z):
        Bp = Bp0 @ SE3.exp(z[0:6])
        Bl = Bl0 @ SE3.exp(z[6:12])
        return Bp, Bl, z[12:15], z[15:27].reshape(L, 3), z[27:39].reshape(L, 3)

    def centers(f):
        return pos0.at[i].set(f) + r * nrm

    def fk(B, th):
        sh = robot.shoulders(B)
        return jax.vmap(lambda s, t: s.apply(leg.foot(t)))(sh, th)    # (L, 3)

    def tibia(B, th):
        sh = robot.shoulders(B)
        return jax.vmap(lambda s, t: s.rotation().apply(leg.contact_vector(t)))(sh, th)

    def statics_rows(B, th, idx):
        st = cm.statics(cm.qpos(Posture(B, th), nrm), G)
        F, tau, H, M = least_torque_kkt(st, idx)
        Fe = F[None] + (dirs @ H.T).reshape(len(dirs), len(idx), 3)
        te = st.bias_joint[None] - Fe.reshape(len(dirs), -1) @ M.T
        n_idx = nrm[jnp.asarray(idx)]
        rows = jnp.concatenate([limit_rows(F, tau, n_idx, fc.adhesion, fc.mu, tau_max)[None],
                                jax.vmap(lambda f, t: limit_rows(f, t, n_idx, fc.adhesion,
                                                                 fc.mu, tau_max))(Fe, te)])
        return rows.reshape(-1), jnp.sum(tau ** 2)

    def objective(z):
        Bp, Bl, f, tp, tl = unpack(z)
        _, ep = statics_rows(Bp, tp, T_idx)
        _, el = statics_rows(Bl, tl, Tn_idx)
        return ep + el + cfg.reg * (jnp.sum(z[:12] ** 2) + jnp.sum((f - f0) ** 2))

    def eq(z):
        Bp, Bl, f, tp, tl = unpack(z)
        c = centers(f)
        return jnp.concatenate([(fk(Bp, tp) - c).reshape(-1), (fk(Bl, tl) - c).reshape(-1),
                                jnp.atleast_1d((f - f0) @ nrm[i])])

    def ineq(z):                                                       # >= 0
        Bp, Bl, f, tp, tl = unpack(z)
        ank = jnp.concatenate([jnp.sum(tibia(Bp, tp) * nrm, 1), jnp.sum(tibia(Bl, tl) * nrm, 1)]) - cos_a
        rp, _ = statics_rows(Bp, tp, T_idx)
        rl, _ = statics_rows(Bl, tl, Tn_idx)
        return jnp.concatenate([ank, -rp - cfg.margin, -rl - cfg.margin])

    f_obj, g_obj = jax.jit(objective), jax.jit(jax.grad(objective))
    f_eq, j_eq = jax.jit(eq), jax.jit(jax.jacfwd(eq))
    f_in, j_in = jax.jit(ineq), jax.jit(jax.jacfwd(ineq))
    np64 = lambda f: (lambda z: np.asarray(f(jnp.asarray(z, jnp.float32)), np.float64))

    z0 = np.concatenate([np.zeros(12), np.asarray(f0), np.asarray(warm.planted.thetas).reshape(-1),
                         np.asarray(warm.lift.thetas).reshape(-1)])
    lim = np.asarray(leg.limits)                                       # (3, 2)
    th_b = [tuple(lim[k % 3]) for k in range(2 * 3 * L)]
    tx, tr, tf = cfg.trust_xyz, cfg.trust_rot, cfg.trust_foot
    body_b = [(-tx, tx)] * 3 + [(-tr, tr)] * 3
    lift_b = body_b if goal is None else [(0.0, 0.0)] * 6              # B_lift = B'
    bounds = body_b + lift_b + [(float(f0[k]) - tf, float(f0[k]) + tf) for k in range(3)] + th_b
    t0 = time.perf_counter()
    cost0 = float(np64(f_obj)(z0))
    res = minimize(np64(f_obj), z0, jac=np64(g_obj), method="SLSQP", bounds=bounds,
                   constraints=[dict(type="eq", fun=np64(f_eq), jac=np64(j_eq)),
                                dict(type="ineq", fun=np64(f_in), jac=np64(j_in))],
                   options=dict(maxiter=cfg.maxiter, ftol=cfg.ftol))
    dt = time.perf_counter() - t0
    z = jnp.asarray(res.x, jnp.float32)
    Bp, Bl, f, tp, tl = unpack(z)
    planted = Posture(Bp, tp)
    ok_p, plant = kit.lift_leg(planted, S, i, fc.lift_height)          # leg i just above f
    return dict(plant=plant, planted=planted, lift=Posture(Bl, tl), stance=S, foot=np.asarray(f),
                success=bool(res.success), message=str(res.message), nit=int(res.nit),
                eq_viol=float(np.abs(np64(f_eq)(res.x)).max()),
                ineq_viol=float(np.maximum(0.0, -np64(f_in)(res.x)).max()),
                cost0=cost0, cost=float(res.fun), t=dt)


# --------------------------------------------------------------------------- #
# A sequence of transfers, jointly                                            #
# --------------------------------------------------------------------------- #
def _cone_margins(F, normals, adhesion, mu, eps=1e-3):
    """Round-cone margins (N) of forces F (p, 3), smoothed at zero tangential force:
    ``(mu a - sqrt(t^2 + eps^2)) / sqrt(1 + mu^2)``, ``a`` the adhesion-shifted normal part."""
    Fs = F + adhesion * normals
    a = jnp.sum(Fs * normals, -1)
    t = jnp.sqrt(jnp.sum((Fs - a[:, None] * normals) ** 2, -1) + eps ** 2)
    return (mu * a - t) / np.sqrt(1.0 + mu ** 2)


def refine_sequence(pl: FastPlanner, cfg: RefineCfg, start_stance, warms, legs) -> dict:
    """Refine a sequence of transfers jointly (``docs/trajopt-nlp.md``).

    Per transfer k (planting leg i_k, lifting leg i'_k) the variables of
    :func:`refine_transfer` (B_plant, B_lift as offsets from the warm start, the planting
    foot, the joint angles at both bodies: 39). Coupled through the footholds: transfer k's
    stance holds the start footholds (fixed) and the earlier transfers' planting feet
    (variables). Stability as smooth rows: per body (T_k at B_plant, T'_k at B_lift), per
    supporting foot and per case (gravity + the 12 pushes) the round-cone margin of the
    least-torque split >= ``margin``, and the undisturbed servo torques within tau_max.
    Objective: sum tau^2 over all bodies + ``reg`` x |offsets|^2. The start and the paths
    between transfers are not optimized.

    Args:
        pl: the fast planner pieces.
        cfg: the NLP's knobs.
        start_stance: (L,) foothold indices before the first transfer.
        warms: the sampled transfers (:class:`.transfer.TransferResult`), in order.
        legs: per transfer ``(i_k, i'_k)``.

    Returns:
        dict: ``planted`` / ``lift`` (refined postures per transfer), ``feet`` (K, 3)
        planting feet, ``success``, ``message``, ``nit``, ``eq_viol``, ``ineq_viol``,
        ``cost0``, ``cost``, ``t`` (s).
    """
    kit, cm, fc = pl.kit, pl.cm, pl.cfg
    robot, leg, fh = kit.robot, kit.robot.leg, kit.footholds
    L, K = robot.num_legs, len(warms)
    r = kit.cfg.foot_radius
    cos_a = float(np.cos(np.radians(kit.cfg.ankle_limit_deg)))
    dirs = _push_dirs(cm.mass * 9.81, fc.push_min, fc.twist_min)
    tau_max = float(cm.mj.forcerange[1])
    pos_start = np.asarray(fh.position[jnp.asarray(start_stance)])     # (L, 3)
    nrm = jnp.stack([fh.normal[jnp.asarray(w.stance)] for w in warms])  # (K, L, 3)
    f0 = jnp.stack([fh.position[jnp.asarray(w.stance)[i]] for w, (i, _) in zip(warms, legs)])
    Bp0 = jnp.stack([w.plant.body.wxyz_xyz for w in warms])           # (K, 7)
    Bl0 = jnp.stack([w.lift.body.wxyz_xyz for w in warms])
    I = jnp.array([i for i, _ in legs])
    T_idx = jnp.array([[j for j in range(L) if j != i] for i, _ in legs])        # (K, 3)
    Tn_idx = jnp.array([[j for j in range(L) if j != n] for _, n in legs])
    nv = 39

    def stances(f):
        """(K, L, 3) foothold positions of each transfer's stance S_k."""
        cur, out = jnp.asarray(pos_start), []
        for k, (i, _) in enumerate(legs):
            cur = cur.at[i].set(f[k])
            out.append(cur)
        return jnp.stack(out)

    def fk(B, th):
        sh = robot.shoulders(B)
        return jax.vmap(lambda s, t: s.apply(leg.foot(t)))(sh, th)

    def tibia(B, th):
        sh = robot.shoulders(B)
        return jax.vmap(lambda s, t: s.rotation().apply(leg.contact_vector(t)))(sh, th)

    def stability(B, th, idx, n):
        st = cm.statics(cm.qpos(Posture(B, th), n), G)
        F, tau, H, M = least_torque_kkt(st, idx)
        Fe = F[None] + (dirs @ H.T).reshape(len(dirs), 3, 3)
        n_idx = n[idx]
        marg = jnp.concatenate([_cone_margins(F, n_idx, fc.adhesion, fc.mu)[None],
                                jax.vmap(lambda f: _cone_margins(f, n_idx, fc.adhesion, fc.mu))(Fe)])
        rows = jnp.concatenate([marg.reshape(-1) - cfg.margin, tau_max - tau, tau_max + tau])
        return rows, jnp.sum(tau ** 2)                               # rows >= 0

    def one(zk, Sk, nk, bp0, bl0, fk0, i, ti, tni):
        Bp = SE3(bp0) @ SE3.exp(zk[0:6])
        Bl = SE3(bl0) @ SE3.exp(zk[6:12])
        f, tp, tl = zk[12:15], zk[15:27].reshape(L, 3), zk[27:39].reshape(L, 3)
        c = Sk + r * nk
        eq = jnp.concatenate([(fk(Bp, tp) - c).reshape(-1), (fk(Bl, tl) - c).reshape(-1),
                              jnp.atleast_1d((f - fk0) @ nk[i])])
        ank = jnp.concatenate([jnp.sum(tibia(Bp, tp) * nk, 1), jnp.sum(tibia(Bl, tl) * nk, 1)]) - cos_a
        rp, ep = stability(Bp, tp, ti, nk)
        rl, el = stability(Bl, tl, tni, nk)
        cost = ep + el + cfg.reg * (jnp.sum(zk[:12] ** 2) + jnp.sum((f - fk0) ** 2))
        return eq, jnp.concatenate([ank, rp, rl]), cost

    def parts(z):
        Z = z.reshape(K, nv)
        S = stances(Z[:, 12:15])
        return jax.vmap(one)(Z, S, nrm, Bp0, Bl0, f0, I, T_idx, Tn_idx)

    objective = lambda z: jnp.sum(parts(z)[2])
    eqs = lambda z: parts(z)[0].reshape(-1)
    ineqs = lambda z: parts(z)[1].reshape(-1)
    f_obj, g_obj = jax.jit(objective), jax.jit(jax.grad(objective))
    f_eq, j_eq = jax.jit(eqs), jax.jit(jax.jacfwd(eqs))
    f_in, j_in = jax.jit(ineqs), jax.jit(jax.jacfwd(ineqs))
    prof = {}                                      # name -> [calls, seconds]

    def np64(f, name=None):
        def g(z):
            t = time.perf_counter()
            out = np.asarray(jax.block_until_ready(f(jnp.asarray(z, jnp.float32))), np.float64)
            if name:
                c = prof.setdefault(name, [0, 0.0])
                c[0] += 1
                c[1] += time.perf_counter() - t
            return out
        return g

    z0 = np.concatenate([np.concatenate([np.zeros(12), np.asarray(f0[k]),
                                         np.asarray(w.planted.thetas).reshape(-1),
                                         np.asarray(w.lift.thetas).reshape(-1)])
                         for k, w in enumerate(warms)])
    lim = np.asarray(leg.limits)
    tx, tr, tf = cfg.trust_xyz, cfg.trust_rot, cfg.trust_foot
    bounds = []
    for k in range(K):
        bounds += ([(-tx, tx)] * 3 + [(-tr, tr)] * 3) * 2
        bounds += [(float(f0[k, a]) - tf, float(f0[k, a]) + tf) for a in range(3)]
        bounds += [tuple(lim[a % 3]) for a in range(2 * 3 * L)]
    t0 = time.perf_counter()
    for f_, z_ in ((f_obj, z0), (g_obj, z0), (f_eq, z0), (j_eq, z0), (f_in, z0), (j_in, z0)):
        jax.block_until_ready(f_(jnp.asarray(z_, jnp.float32)))     # compile first
    t_compile = time.perf_counter() - t0
    t0 = time.perf_counter()
    cost0 = float(np64(f_obj)(z0))
    res = minimize(np64(f_obj, "objective"), z0, jac=np64(g_obj, "gradient"), method="SLSQP",
                   bounds=bounds,
                   constraints=[dict(type="eq", fun=np64(f_eq, "equalities"),
                                     jac=np64(j_eq, "equality Jacobian")),
                                dict(type="ineq", fun=np64(f_in, "inequalities"),
                                     jac=np64(j_in, "inequality Jacobian"))],
                   options=dict(maxiter=cfg.maxiter, ftol=cfg.ftol))
    dt = time.perf_counter() - t0
    prof["SLSQP itself (the rest)"] = [res.nit, dt - sum(v[1] for v in prof.values())]
    prof["compile (once)"] = [1, t_compile]
    Z = jnp.asarray(res.x, jnp.float32).reshape(K, nv)
    planted = [Posture(SE3(Bp0[k]) @ SE3.exp(Z[k, 0:6]), Z[k, 15:27].reshape(L, 3)) for k in range(K)]
    lift = [Posture(SE3(Bl0[k]) @ SE3.exp(Z[k, 6:12]), Z[k, 27:39].reshape(L, 3)) for k in range(K)]
    return dict(planted=planted, lift=lift, feet=np.asarray(Z[:, 12:15]), success=bool(res.success),
                message=str(res.message), nit=int(res.nit),
                eq_viol=float(np.abs(np64(f_eq)(res.x)).max()),
                ineq_viol=float(np.maximum(0.0, -np64(f_in)(res.x)).max()),
                cost0=cost0, cost=float(res.fun), t=dt, prof=prof)
