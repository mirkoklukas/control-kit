"""One transfer as a TOWR-style NLP: feet as free 3D points, reach as a signed distance,
foot forces as variables (``docs/trajopt-nlp.md``, "TOWR-style transfer").

Transfer ``(T, S, T')`` from a fixed body B: the planting leg i lands on a new foothold f
(closing S), the lifting leg i' lifts (opening T'). Four bodies:

    B (fixed)  supports T,  -- leg i lands at B_plant --  B_plant supports T, reaches S
    B_lift     supports T', reaches S  -- leg i' lifts --  B'  supports T', reaches T'

Variables: B_plant, B_lift, B' (position + roll / pitch / yaw, 6 each), the foothold f (3),
and the planted feet's forces at each of the four bodies (3 feet x 3 each). No joint angles.

- **reach**: per body and foot it must reach, ``sdf(foot in the shoulder frame) >= margin``
  (:class:`ReachSDF`, Catmull-Rom; margin >= h).
- **equilibrium** per body: ``G(c, feet) F = b_b`` (:func:`.statics.point_mass_statics`),
  with the centre of mass c approximated without joint angles (:func:`com_approx`).
- **limits** per supporting foot: adhesion and the friction pyramid (:func:`.statics.limit_rows`).
- **foothold on the surface**: ``f_z = 0`` (the floor).
- **body clearance**: the body box's corners above the floor.
- **margins**: each supporting foot's normal force >= ``min_normal`` x its share of the
  weight (else the optimum puts the whole weight on one foot: a knife edge); f at least
  ``foot_sep`` from the other footholds.
- **objective**: maximize ``x(B') - x(B)``, plus small regularizers (body steps, forces).

The centre of mass needs every foot, also the legs in the air: leg i at B at its old
foothold (just lifted); leg i' at B' at ``air_foot`` in its shoulder frame (it rides along).

Not in v0: joint limits beyond the reach grid's, the ankle cone, leg collisions (all need
joint angles: checked afterwards by IK), pushes, the body's paths between the poses.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import linprog, minimize

from controlkit.se3 import SE3

from .statics import PYRAMID_FACES, limit_rows, min_norm_slacks, point_mass_statics


@dataclass
class TowrCfg:
    """The NLP's knobs."""
    margin: float = 0.005           # reach: sdf >= margin (m); use >= the grid's h
    alpha: float = 0.525            # centre of mass: a leg's mass at alpha shoulder + (1-alpha) foot
    adhesion: float = 0.0           # max pull per foot (N)
    mu: float = 0.5                 # friction coefficient
    foot_r: float = 0.011           # ankle pivot above the pad face (m)
    body_clearance: float = 0.02    # body box corners above the floor (m)
    xyz_bound: float = 0.25         # body translation bound around B (m)
    rot_bound: float = 0.35         # roll / pitch / yaw bound (rad)
    foot_bound: float = 0.30        # foothold f within this of leg i's old foothold, in x, y (m)
    force_bound: float = 100.0      # |force component| (N)
    min_normal: float = 0.1         # each supporting foot carries >= this x its share M g / 3:
                                    # keeps the centre of mass off the support triangle's edges
    foot_sep: float = 0.08          # f at least this far (x, y) from the other footholds (m)
    air_foot: tuple = (0.15, 0.0, -0.08)   # a leg in the air at B': this point, shoulder frame
    push_frac: float = 0.2          # solve_angles: min-norm push score >= this x the weight (N),
                                    # at every body (0: off)
    tau_max: float = 5.0            # solve_angles: |servo torque| <= this (N m), every body
    reg_torque: float = 1e-3        # solve_angles: weight of sum tau^2 (N^2 m^2) in the objective
    reg_body: float = 0.1           # sum |translation step|^2 between consecutive bodies
    reg_force: float = 1e-5         # sum |F|^2 (internal forces are free otherwise)
    maxiter: int = 1000
    ftol: float = 1e-9


@dataclass
class Transfer:
    """The fixed start of a transfer.

    Args:
        body: B (SE3), supporting T.
        footholds: (L, 3) the current footholds, world (leg i's: where it lifted from, used
            only for the centre of mass while it is in the air).
        normals: (L, 3) their unit normals.
        i: the planting leg (in the air at B).
        i_prime: the lifting leg (lifts at B_lift).
    """
    body: SE3
    footholds: jax.Array
    normals: jax.Array
    i: int
    i_prime: int


def com_approx(body: SE3, shoulders, feet, m_body, m_leg, alpha):
    """Centre of mass without joint angles: the body's mass at its origin, each leg's at
    ``alpha`` of the way from its foot to its shoulder (fit against the kinematic centre of
    mass: median error ~6 mm over random postures).

    Args:
        body: SE3.
        shoulders: (L, 3) shoulder positions, world.
        feet: (L, 3) foot points, world.
        m_body, m_leg: body and per-leg mass (kg).
        alpha: the leg's mass fraction placed at the shoulder.

    Returns:
        (3,) centre of mass, world.
    """
    L = feet.shape[0]
    legs = alpha * shoulders + (1.0 - alpha) * feet
    return (m_body * body.translation() + m_leg * legs.sum(0)) / (m_body + L * m_leg)


def _body(v):
    return SE3.from_te(v[:3], v[3:], "xyz")


def solve(robot, sdf, mass_model, g, tr: Transfer, cfg: TowrCfg, warm=None) -> dict:
    """Solve the transfer NLP (SLSQP, gradients and Jacobians from JAX).

    Args:
        robot: the :class:`Robot` (mounts, leg).
        sdf: the leg's :class:`ReachSDF` (shoulder frame; one for all legs).
        mass_model: :class:`.statics.MassModel`.
        g: (3,) gravity, world.
        tr: the fixed start.
        cfg: the knobs.
        warm: optional initial ``z`` (else every body at B, f at leg i's old foothold).

    Returns:
        dict with the bodies, f, forces, the objective, constraint violations, solver info.
    """
    L = robot.num_legs
    i, ip = tr.i, tr.i_prime
    T = [j for j in range(L) if j != i]                 # supports B, B_plant
    Tp = [j for j in range(L) if j != ip]               # supports B_lift, B'
    m_leg = float(jnp.sum(mass_model.links) + mass_model.foot)
    m_body = float(mass_model.body)
    M = m_body + L * m_leg
    n_up = jnp.array([0.0, 0.0, 1.0])
    fh0 = jnp.asarray(tr.footholds, jnp.float32)
    nrm = jnp.asarray(tr.normals, jnp.float32)
    corners = jnp.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
    box_c, box_h = robot.mount_box(half_height=0.03)
    B = tr.body
    assert np.allclose(np.asarray(B.rotation().as_matrix()), np.eye(3), atol=1e-6), \
        "v0: B must be level (identity rotation); the bodies' rpy are absolute"
    b0 = jnp.concatenate([B.translation(), jnp.zeros(3)])

    def unpack(z):
        return z[0:6], z[6:12], z[12:18], z[18:21], z[21:57].reshape(4, 3, 3)

    def feet_of(f):
        # footholds for S (leg i at f) and the pivots the FK feet must reach
        S = fh0.at[i].set(f)
        return S, S + cfg.foot_r * nrm.at[i].set(n_up)

    n_min = cfg.min_normal * M * float(jnp.linalg.norm(g)) / 3

    def pieces(body, feet_all, support, F):
        sh = robot.shoulders(body)
        com = com_approx(body, sh.translation(), feet_all, m_body, m_leg, cfg.alpha)
        st = point_mass_statics(com, feet_all[jnp.array(support)], M, g)
        G = jnp.transpose(st.J_base, (1, 0, 2)).reshape(6, -1)
        eq = G @ F.reshape(-1) - st.bias_base
        n_sup = nrm[jnp.array(support)]
        lim = -limit_rows(F, jnp.zeros(0), n_sup, cfg.adhesion, cfg.mu, 0.0, PYRAMID_FACES)
        lim = jnp.concatenate([lim, jnp.sum(F * n_sup, -1) - n_min])
        return sh, eq, lim

    def reach_rows(sh, targets, legs):
        return jnp.stack([sdf.sdf(sh[j].inverse().apply(targets[j])) - cfg.margin for j in legs])

    def clearance(body):
        pts = jax.vmap(body.apply)(box_c + corners * box_h)
        return pts[:, 2] - cfg.body_clearance

    def model(z):
        vp, vl, ve, f, F = unpack(z)
        Bp, Bl, Be = _body(vp), _body(vl), _body(ve)
        S, tg = feet_of(f)
        _, eq0, lim0 = pieces(B, fh0 + cfg.foot_r * nrm, T, F[0])
        shp, eq1, lim1 = pieces(Bp, tg, T, F[1])
        shl, eq2, lim2 = pieces(Bl, tg, Tp, F[2])
        air = robot.shoulders(Be)[ip].apply(jnp.asarray(cfg.air_foot, jnp.float32))
        she, eq3, lim3 = pieces(Be, tg.at[ip].set(air), Tp, F[3])   # leg i' in the air
        reach = jnp.concatenate([reach_rows(shp, tg, range(L)), reach_rows(shl, tg, range(L)),
                                 reach_rows(she, tg, Tp)])
        others = jnp.array([j for j in range(L) if j != i])
        sep = jnp.linalg.norm(fh0[others, :2] - f[None, :2], axis=-1) - cfg.foot_sep
        eq = jnp.concatenate([eq0, eq1, eq2, eq3, f[2:3]])
        ineq = jnp.concatenate([reach, sep, lim0, lim1, lim2, lim3,
                                clearance(Bp), clearance(Bl), clearance(Be)])
        return eq, ineq

    def objective(z):
        vp, vl, ve, f, F = unpack(z)
        steps = jnp.stack([vp[:3] - b0[:3], vl[:3] - vp[:3], ve[:3] - vl[:3]])
        return -(ve[0] - b0[0]) + cfg.reg_body * jnp.sum(steps ** 2) + cfg.reg_force * jnp.sum(F ** 2)

    eq_f = jax.jit(lambda z: model(z)[0])
    in_f = jax.jit(lambda z: model(z)[1])
    eq_j, in_j = jax.jit(jax.jacfwd(lambda z: model(z)[0])), jax.jit(jax.jacfwd(lambda z: model(z)[1]))
    ob_f, ob_g = jax.jit(objective), jax.jit(jax.grad(objective))
    np64 = lambda fn: (lambda z: np.asarray(fn(jnp.asarray(z, jnp.float32)), np.float64))

    if warm is None:
        F0 = np.zeros((4, 3, 3)); F0[..., 2] = M * 9.81 / 3
        warm = np.concatenate([np.tile(np.asarray(b0), 3), np.asarray(fh0[i]), F0.reshape(-1)])
    tb = [(float(b0[k]) - cfg.xyz_bound, float(b0[k]) + cfg.xyz_bound) for k in range(3)] + \
         [(-cfg.rot_bound, cfg.rot_bound)] * 3
    fb = [(float(fh0[i, 0]) - cfg.foot_bound, float(fh0[i, 0]) + cfg.foot_bound),
          (float(fh0[i, 1]) - cfg.foot_bound, float(fh0[i, 1]) + cfg.foot_bound), (-0.05, 0.05)]
    bounds = tb * 3 + fb + [(-cfg.force_bound, cfg.force_bound)] * 36

    t0 = time.perf_counter()
    jax.block_until_ready((eq_j(jnp.asarray(warm, jnp.float32)), in_j(jnp.asarray(warm, jnp.float32))))
    t_compile = time.perf_counter() - t0
    t0 = time.perf_counter()
    res = minimize(np64(ob_f), warm, jac=np64(ob_g), method="SLSQP", bounds=bounds,
                   constraints=[dict(type="eq", fun=np64(eq_f), jac=np64(eq_j)),
                                dict(type="ineq", fun=np64(in_f), jac=np64(in_j))],
                   options=dict(maxiter=cfg.maxiter, ftol=cfg.ftol))
    dt = time.perf_counter() - t0
    vp, vl, ve, f, F = (np.asarray(a) for a in unpack(jnp.asarray(res.x, jnp.float32)))
    return dict(z=res.x, B_plant=_body(jnp.asarray(vp)), B_lift=_body(jnp.asarray(vl)),
                B_end=_body(jnp.asarray(ve)), f=f, F=F, progress=float(ve[0] - b0[0]),
                success=bool(res.success), message=str(res.message), nit=int(res.nit),
                eq_viol=float(np.abs(np64(eq_f)(res.x)).max()),
                ineq_viol=float(np.maximum(0.0, -np64(in_f)(res.x)).max()),
                t=dt, t_compile=t_compile, T=T, Tp=Tp)


def hold_exact(com, feet, normals, mass, g, adhesion, mu, faces=PYRAMID_FACES):
    """Exact check: do some forces hold a point mass at ``com`` on these feet within adhesion
    and friction? (An LP, HiGHS; for verifying solutions with the real centre of mass.)

    Returns:
        A tuple ``(feasible, normal_forces)``: bool, (p,) one feasible solution's normal
        forces (N; NaN if infeasible).
    """
    st = point_mass_statics(jnp.asarray(com), jnp.asarray(feet), mass, g)
    p = feet.shape[0]
    G = np.asarray(jnp.transpose(st.J_base, (1, 0, 2)).reshape(6, 3 * p), np.float64)
    rows = jax.jacfwd(lambda F: limit_rows(F.reshape(-1, 3), jnp.zeros(0), jnp.asarray(normals),
                                           adhesion, mu, 0.0, faces))(jnp.zeros(3 * p))
    A = np.asarray(rows, np.float64)
    b = -np.asarray(limit_rows(jnp.zeros((p, 3)), jnp.zeros(0), jnp.asarray(normals), adhesion,
                               mu, 0.0, faces), np.float64)
    r = linprog(np.zeros(3 * p), A_ub=A, b_ub=b, A_eq=G, b_eq=np.asarray(st.bias_base, np.float64),
                bounds=[(None, None)] * (3 * p), method="highs")
    if r.status != 0:
        return False, np.full(p, np.nan)
    return True, np.sum(r.x.reshape(p, 3) * np.asarray(normals), -1)


# --------------------------------------------------------------------------- #
# The same transfer with joint angles as variables (no reach grid)            #
# --------------------------------------------------------------------------- #
def solve_angles(robot, mass_model, g, tr: Transfer, cfg: TowrCfg, theta_B, ankle_deg: float,
                 warm_thetas=None) -> dict:
    """The transfer NLP with the joint angles as variables instead of the reach signed
    distance (as the first ``refine.py``), and **T's footholds variables** too (``tr.footholds``
    only warm-start them). B is fixed.

    Variables (114): B_plant, B_lift, B' (6 each); the four footholds P (12: T's three and
    leg i's new one, f; each on the floor by z = 0); the 12 joint angles at each of the four
    bodies (48); the supporting feet's forces at each body (36). Per body: forward-kinematics
    equalities (every foot it must reach on its foothold, at the pivot: B reaches T, B_plant
    and B_lift reach S, B' reaches T'); joint limits as bounds; the ankle cone
    ``v . n >= cos(ankle)`` (v: foot-to-knee direction) per planted foot; equilibrium with
    the exact centre of mass (:func:`.statics.kinematic_com`); the force limits and the
    minimum normal force; **servo torques** ``|tau| <= tau_max`` for all servos, with
    ``tau = b_a - J^T F`` from the force variables (:func:`.statics.leg_torques`, no MuJoCo);
    the footholds pairwise at least ``foot_sep`` apart; body clearance. Legs in the air (leg
    i at B, leg i' at B') have free angles. Objective: max ``x(B') - x(B)``, minus
    ``reg_torque`` x the sum of squared torques over the four bodies.

    Push margin (``cfg.push_frac > 0``): the min-norm push score (:func:`.statics.
    push_score_min_norm`, adhesion and friction) >= ``push_frac`` x the weight at every body,
    as smooth per-row constraints ``s_i - lam |h_i| >= 0`` (equivalent to the ball radius
    >= lam, without the min). These use their own minimum-norm forces, not the force
    variables (which prove the stance holds; the rows ask the min-norm split to have push
    margin).

    Args:
        robot, mass_model, g, tr, cfg: as :func:`solve` (``cfg.margin``, ``cfg.alpha`` and
            ``cfg.air_foot`` unused). ``tr.body`` is B (fixed).
        theta_B: (L, n_joints) warm-start angles (B reaching ``tr.footholds``).
        ankle_deg: the ankle cone's half-angle (deg).
        warm_thetas: (4, L, n_joints) initial angles at B, B_plant, B_lift, B'; None:
            theta_B at all four.

    Returns:
        As :func:`solve`, plus ``P`` (L, 3) the footholds (``f = P[i]``) and ``thetas``
        (4, L, n_joints).
    """
    from controlkit.kinematics.types import Posture
    from .statics import kinematic_com, leg_torques

    L, nj = robot.num_legs, robot.leg.num_joints
    i, ip = tr.i, tr.i_prime
    T = [j for j in range(L) if j != i]
    Tp = [j for j in range(L) if j != ip]
    M = float(mass_model.body + L * (jnp.sum(mass_model.links) + mass_model.foot))
    nrm = jnp.asarray(tr.normals, jnp.float32)
    P0 = jnp.asarray(tr.footholds, jnp.float32)
    corners = jnp.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
    box_c, box_h = robot.mount_box(half_height=0.03)
    cos_a = float(np.cos(np.radians(ankle_deg)))
    B = tr.body
    assert np.allclose(np.asarray(B.rotation().as_matrix()), np.eye(3), atol=1e-6), \
        "v0: B must be level (identity rotation); the bodies' rpy are absolute"
    b0 = jnp.concatenate([B.translation(), jnp.zeros(3)])
    n_min = cfg.min_normal * M * float(jnp.linalg.norm(g)) / 3
    lam = cfg.push_frac * M * float(jnp.linalg.norm(g))
    nv = 18 + 3 * L + 4 * L * nj + 4 * 9

    def unpack(z):
        o = 18 + 3 * L + 4 * L * nj
        return (z[0:18].reshape(3, 6), z[18:18 + 3 * L].reshape(L, 3),
                z[18 + 3 * L:o].reshape(4, L, nj), z[o:o + 36].reshape(4, 3, 3))

    def legs(body, theta):
        sh = robot.shoulders(body)
        feet = jax.vmap(lambda j: sh[j].apply(robot.leg.forward(theta[j]).translation()[-1]))(jnp.arange(L))
        cvec = jax.vmap(lambda j: sh[j].rotation().apply(robot.leg.contact_vector(theta[j])))(jnp.arange(L))
        return feet, cvec, kinematic_com(robot, Posture(body, theta), mass_model)

    def body_rows(body, theta, F, support, reach, targets):
        feet, cvec, com = legs(body, theta)
        sup = jnp.array(support)
        st = point_mass_statics(com, feet[sup], M, g)
        G = jnp.transpose(st.J_base, (1, 0, 2)).reshape(6, -1)
        n_sup = nrm[sup]
        lim = -limit_rows(F, jnp.zeros(0), n_sup, cfg.adhesion, cfg.mu, 0.0, PYRAMID_FACES)
        ineq = [lim, jnp.sum(F * n_sup, -1) - n_min]
        r = jnp.array(reach, dtype=jnp.int32)
        fk = (feet[r] - targets[r]).reshape(-1)
        ineq.append(jnp.sum(cvec[r] * nrm[r], -1) - cos_a)
        if lam > 0:
            sl, h = min_norm_slacks(com, feet[sup], n_sup, M, g, cfg.adhesion, cfg.mu)
            ineq.append(sl - lam * jnp.sqrt(jnp.sum(h ** 2, -1) + 1e-12))
        tau = leg_torques(robot, mass_model, g, body, theta, jnp.zeros((L, 3)).at[sup].set(F)).reshape(-1)
        ineq += [cfg.tau_max - tau, cfg.tau_max + tau]
        return jnp.concatenate([G @ F.reshape(-1) - st.bias_base, fk]), jnp.concatenate(ineq), tau

    def clearance(body):
        pts = jax.vmap(body.apply)(box_c + corners * box_h)
        return pts[:, 2] - cfg.body_clearance

    # per body (B, B_plant, B_lift, B'): (support, reach)
    plan = [(T, T), (T, list(range(L))), (Tp, list(range(L))), (Tp, Tp)]
    pairs = [(a, b) for a in range(L) for b in range(a + 1, L)]

    def model(z):
        V, P, th, F = unpack(z)
        tg = P + cfg.foot_r * nrm
        bodies = [B] + [_body(V[k]) for k in range(3)]
        eqs, ins, taus = [], [], []
        for k, (support, reach) in enumerate(plan):
            e, q, tau = body_rows(bodies[k], th[k], F[k], support, reach, tg)
            eqs.append(e); ins.append(q); taus.append(tau)
            if k > 0:
                ins.append(clearance(bodies[k]))
        ins.append(jnp.stack([jnp.linalg.norm(P[a, :2] - P[b, :2]) for a, b in pairs]) - cfg.foot_sep)
        return jnp.concatenate(eqs + [P[:, 2]]), jnp.concatenate(ins), jnp.stack(taus)

    def objective(z):
        V, P, th, F = unpack(z)
        steps = jnp.concatenate([V[:1, :3] - b0[None, :3], V[1:, :3] - V[:-1, :3]])
        tau = model(z)[2]
        return (-(V[2, 0] - b0[0]) + cfg.reg_body * jnp.sum(steps ** 2) + cfg.reg_force * jnp.sum(F ** 2)
                + cfg.reg_torque * jnp.sum(tau ** 2))

    eq_f = jax.jit(lambda z: model(z)[0])
    in_f = jax.jit(lambda z: model(z)[1])
    eq_j, in_j = jax.jit(jax.jacfwd(lambda z: model(z)[0])), jax.jit(jax.jacfwd(lambda z: model(z)[1]))
    ob_f, ob_g = jax.jit(objective), jax.jit(jax.grad(objective))
    np64 = lambda fn: (lambda z: np.asarray(fn(jnp.asarray(z, jnp.float32)), np.float64))

    th0 = np.broadcast_to(np.asarray(theta_B), (4, L, nj)) if warm_thetas is None else np.asarray(warm_thetas)
    F0 = np.zeros((4, 3, 3)); F0[..., 2] = M * 9.81 / 3
    warm = np.concatenate([np.tile(np.asarray(b0), 3), np.asarray(P0).reshape(-1), th0.reshape(-1),
                           F0.reshape(-1)])
    lim = np.asarray(robot.leg.limits)
    tb = [(float(b0[k]) - cfg.xyz_bound, float(b0[k]) + cfg.xyz_bound) for k in range(3)] + \
         [(-cfg.rot_bound, cfg.rot_bound)] * 3
    pb = []
    for j in range(L):
        pb += [(float(P0[j, 0]) - cfg.foot_bound, float(P0[j, 0]) + cfg.foot_bound),
               (float(P0[j, 1]) - cfg.foot_bound, float(P0[j, 1]) + cfg.foot_bound), (-0.05, 0.05)]
    thb = [tuple(map(float, lim[k % nj])) for k in range(4 * L * nj)]
    bounds = tb * 3 + pb + thb + [(-cfg.force_bound, cfg.force_bound)] * 36
    assert len(bounds) == nv == warm.shape[0]

    t0 = time.perf_counter()
    jax.block_until_ready((eq_j(jnp.asarray(warm, jnp.float32)), in_j(jnp.asarray(warm, jnp.float32))))
    t_compile = time.perf_counter() - t0
    t0 = time.perf_counter()
    res = minimize(np64(ob_f), warm, jac=np64(ob_g), method="SLSQP", bounds=bounds,
                   constraints=[dict(type="eq", fun=np64(eq_f), jac=np64(eq_j)),
                                dict(type="ineq", fun=np64(in_f), jac=np64(in_j))],
                   options=dict(maxiter=cfg.maxiter, ftol=cfg.ftol))
    dt = time.perf_counter() - t0
    V, P, th, F = (np.asarray(a) for a in unpack(jnp.asarray(res.x, jnp.float32)))
    tau = np.asarray(jax.jit(lambda z: model(z)[2])(jnp.asarray(res.x, jnp.float32))).reshape(4, L, nj)
    return dict(z=res.x, B_plant=_body(jnp.asarray(V[0])), B_lift=_body(jnp.asarray(V[1])),
                B_end=_body(jnp.asarray(V[2])), P=P, f=P[i], F=F, thetas=th, tau=tau,
                progress=float(V[2, 0] - b0[0]),
                success=bool(res.success), message=str(res.message), nit=int(res.nit),
                eq_viol=float(np.abs(np64(eq_f)(res.x)).max()),
                ineq_viol=float(np.maximum(0.0, -np64(in_f)(res.x)).max()),
                t=dt, t_compile=t_compile, T=T, Tp=Tp)
