"""Statics of a planted posture: foot forces and actuator torques, with the internal
forces chosen for the least torque.

See ``docs/planner-design.md``, "Actuator torques for a given stance and body pose". For
a posture P with some feet planted, static equilibrium fixes the foot forces only up to
the internal forces (feet squeezing / pulling apart):

    F = F_0 + N z        (N spans the null space of the equilibrium map)
    tau(z) = tau_0 + K z

:func:`least_torque` picks z minimizing ``|tau|^2``; ``z = 0`` (:func:`min_norm`) is
what :func:`controlkit.forces.stance_forces` returns. Neither knows any limits.

:func:`hold_margin` adds them -- adhesion, friction (an inner pyramid), torque bounds --
and returns the largest gravity factor s* at which the planted feet still hold the
posture: s* >= 1 means it holds under g (docs, "The hold check"). An LP, solved with
qpax; :func:`hold_lp` builds it (also for a reference solver, see ``test_statics.py``).

Two sources of :class:`Statics`, both for :func:`hold_margin`:

- :func:`statics`: the full robot (one ``mjx.forward``): equilibrium and the joint
  torques, with the legs' masses where they are.
- :func:`point_mass_statics`: the robot as a point mass at its centre of mass, point
  feet; equilibrium only, no torques, no mjx. Cheap and needs no IK: screens body poses
  before the legs are solved. Misses the servo torque limits, and the centre of mass is
  only as good as the caller's estimate.

Built on the same equations as :mod:`controlkit.forces` (MuJoCo generalized coordinates,
one ``mjx.forward``), plus a gravity vector ``g`` of our choice (a wall: tilted g).
Everything is per posture and vmappable. Run under ``uv run --extra mjx``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import qpax
from mujoco import mjx

NULL_TOL = 1e-9     # singular values below this (relative to the largest) span the null space
PYRAMID_FACES = 8   # friction cone -> inner pyramid with this many faces
S_CAP = 10.0        # upper bound on the gravity factor s (keeps the LP bounded)


@jax.tree_util.register_dataclass
@dataclass
class Statics:
    """The linear pieces of the statics of one posture (no forces chosen yet).

    Equilibrium in generalized coordinates, at rest: ``sum_i J_i^T f_i = bias``, split
    into the 6 free-joint (base) rows -- the net force and moment on the robot -- and the
    joint rows, which give the actuator torques ``tau = bias_joint - sum_i J_i,joint^T f_i``.

    Args:
        J_base: (nf, 6, 3) per foot, the base rows of the foot Jacobian (transposed: maps
            a foot force to its generalized force on the base dofs).
        J_joint: (nf, nj_total, 3) per foot, the same for the leg joint dofs.
        bias_base: (6,) gravity's generalized force on the base dofs.
        bias_joint: (nj_total,) gravity's generalized force on the leg joints.
    """
    J_base: jax.Array
    J_joint: jax.Array
    bias_base: jax.Array
    bias_joint: jax.Array


def statics(mx, foot_ids, qpos, g, joint_dofs=None) -> Statics:
    """The statics of posture ``qpos`` under gravity ``g``.

    Args:
        mx: the ``put_model``'d model (floating base on a free joint, then the legs).
        foot_ids: (nf,) foot body ids.
        qpos: (nq,) configuration.
        g: (3,) gravity vector, world frame (m/s^2).
        joint_dofs: (nj,) the dofs that are actuated joints (their torques are bounded
            in :func:`hold_lp`); None: every dof after the free joint. The climb robot's
            passive ankles are dofs but not actuated (:mod:`.climb_statics`).

    Returns:
        The :class:`Statics` of the posture.
    """
    mx = mx.replace(opt=mx.opt.replace(gravity=jnp.asarray(g, float)))
    d = mjx.forward(mx, mjx.make_data(mx).replace(qpos=qpos, qvel=jnp.zeros(mx.nv)))
    # (nf, nv, 3): column-wise Jacobian of each foot point (body origin), transposed
    J = jax.vmap(lambda fid: mjx.jac(mx, d, d.xpos[fid], fid)[0])(foot_ids)
    dofs = jnp.arange(6, mx.nv) if joint_dofs is None else jnp.asarray(joint_dofs)
    return Statics(J_base=J[:, :6, :], J_joint=J[:, dofs, :],
                   bias_base=d.qfrc_bias[:6], bias_joint=d.qfrc_bias[dofs])


def point_mass_statics(com, feet, mass, g) -> Statics:
    """The statics of a point mass ``mass`` at ``com``, standing on point feet.

    The equilibrium of the full model, reduced to geometry: net force
    ``sum_j F_j + mass g = 0`` and net moment about the centre of mass
    ``sum_j (p_j - com) x F_j = 0``. No joints, so no torques: the torque blocks are empty
    and :func:`hold_margin` checks only adhesion and friction (pass ``tau_max=None``).

    Args:
        com: (3,) centre of mass, world frame (m).
        feet: (nf, 3) foot contact points p_j, world frame (m).
        mass: total mass (kg).
        g: (3,) gravity vector, world frame (m/s^2).

    Returns:
        The :class:`Statics`, in the same convention as :func:`statics`: per foot,
        ``J_base`` maps its force to (force, moment about ``com``); ``bias_base`` is
        ``(-mass g, 0)``.
    """
    feet = jnp.asarray(feet, float)
    r = feet - jnp.asarray(com, float)                                  # (nf, 3)
    zero = jnp.zeros(r.shape[0])
    cross = jnp.stack([jnp.stack([zero, -r[:, 2], r[:, 1]], -1),        # [r]_x, per foot
                       jnp.stack([r[:, 2], zero, -r[:, 0]], -1),
                       jnp.stack([-r[:, 1], r[:, 0], zero], -1)], 1)     # (nf, 3, 3)
    eye = jnp.broadcast_to(jnp.eye(3), cross.shape)
    J_base = jnp.concatenate([eye, cross], axis=1)                      # (nf, 6, 3)
    bias_base = jnp.concatenate([-mass * jnp.asarray(g, float), jnp.zeros(3)])
    return Statics(J_base=J_base, J_joint=jnp.zeros((r.shape[0], 0, 3)),
                   bias_base=bias_base, bias_joint=jnp.zeros(0))


def _equilibrium(st: Statics, planted):
    """The equilibrium map A F = b over all feet's forces, with unplanted feet pinned
    to zero force.

    Returns:
        A tuple ``(A, b)``:

        A: (6 + 3 nf, 3 nf) the base rows (net force and moment), then one 3x3 block
            per unplanted foot forcing its force to zero (zero rows for planted feet).
        b: (6 + 3 nf,) right-hand side: the base gravity term, then zeros.
    """
    nf = st.J_base.shape[0]
    G = jnp.transpose(st.J_base, (1, 0, 2)).reshape(6, 3 * nf)          # (6, 3 nf)
    free = jnp.repeat(~planted, 3).astype(G.dtype)                       # 1: force pinned to 0
    A = jnp.concatenate([G, jnp.diag(free)])
    b = jnp.concatenate([st.bias_base, jnp.zeros(3 * nf)])
    return A, b


def _torques(st: Statics, F):
    """Actuator torques (nj_total,) for foot forces F (nf, 3)."""
    return st.bias_joint - jnp.einsum("inc,ic->n", st.J_joint, F)


def min_norm(st: Statics, planted):
    """The minimum-norm foot forces (zero internal forces) and their torques.

    The same answer as :func:`controlkit.forces.stance_forces`.

    Args:
        st: the posture's statics (:func:`statics`).
        planted: (nf,) bool per foot. True: the foot bears load. False: lifted, zero
            force.

    Returns:
        A tuple ``(F, tau)``:

        F: (nf, 3) foot forces, world frame (N).
        tau: (nj_total,) actuator torques (N m), legs in order.
    """
    A, b = _equilibrium(st, planted)
    F = jnp.linalg.lstsq(A, b)[0].reshape(-1, 3)
    return F, _torques(st, F)


def least_torque(st: Statics, planted):
    """The foot forces that hold the posture with the least actuator torque.

    Among all equilibrium forces ``F = F_0 + N z`` (F_0 the minimum-norm solution, N the
    null space of the equilibrium map restricted to the planted feet), the z minimizing
    ``|tau(z)|^2`` with ``tau(z) = tau_0 + K z``: ``z* = -K^+ tau_0``. Fixed shapes for
    vmap: N is the full right-singular basis, its non-null columns masked to zero.

    Args:
        st: the posture's statics (:func:`statics`).
        planted: (nf,) bool per foot. True: the foot bears load. False: lifted, zero
            force.

    Returns:
        A tuple ``(F, tau)``:

        F: (nf, 3) foot forces, world frame (N).
        tau: (nj_total,) actuator torques (N m), legs in order.
    """
    A, b = _equilibrium(st, planted)
    F0 = jnp.linalg.lstsq(A, b)[0]
    _, s, Vt = jnp.linalg.svd(A, full_matrices=True)                    # Vt: (3 nf, 3 nf)
    s = jnp.concatenate([s, jnp.zeros(Vt.shape[0] - s.shape[0])])
    null = s <= NULL_TOL * s[0]                                          # null-space directions
    N = Vt.T * null[None, :]                                             # (3 nf, 3 nf), masked
    tau0 = _torques(st, F0.reshape(-1, 3))
    K = -jnp.einsum("inc,ick->nk", st.J_joint, N.reshape(-1, 3, N.shape[1]))
    z = -jnp.linalg.pinv(K) @ tau0
    F = (F0 + N @ z).reshape(-1, 3)
    return F, _torques(st, F)


def residual(st: Statics, planted, F):
    """Equilibrium residual ``|A F - b|`` (N, N m): ~0 for a valid force solution."""
    A, b = _equilibrium(st, planted)
    return jnp.linalg.norm(A @ F.reshape(-1) - b)


# --------------------------------------------------------------------------- #
# Hold check: the margin LP                                                   #
# --------------------------------------------------------------------------- #
def _tangents(n):
    """Two unit tangents (3,) each, orthogonal to the unit normal n and to each other."""
    a = jnp.where(jnp.abs(n[0]) < 0.9, jnp.array([1.0, 0.0, 0.0]), jnp.array([0.0, 1.0, 0.0]))
    t1 = jnp.cross(n, a)
    t1 = t1 / jnp.linalg.norm(t1)
    return t1, jnp.cross(n, t1)


def hold_lp(st: Statics, planted_idx, normals, adhesion, mu, tau_max,
            faces=PYRAMID_FACES, s_cap=S_CAP):
    """The hold-margin LP, in ``x = (F_P, s)``: the planted feet's forces and the
    gravity factor s.

        max s  subject to
          equilibrium, gravity scaled by s:      G_P F_P - s bias_base = 0
          adhesion, per planted foot j:          n_j . F_j >= -adhesion
          friction, inner pyramid, per face k:   d_k . F_j <= mu_p (n_j . F_j + adhesion)
          torques, every joint (lifted leg too): |s bias_joint - J_P^T F_P| <= tau_max
          bounded:                               s <= s_cap

    mu_p = mu cos(pi / faces): the pyramid lies inside the cone (conservative).
    Gravity enters equilibrium and the leg weights linearly, so scaling it by s keeps
    everything linear; s = 0, F = 0 is always feasible.

    Args:
        st: the posture's statics (:func:`statics`), with the gravity g to scale.
        planted_idx: tuple of the planted feet's indices (static, e.g. ``(1, 2, 3)``).
        normals: (p, 3) unit surface normals at the planted feet, world frame.
        adhesion: max pull per foot, A (N); 0 without magnets.
        mu: friction coefficient.
        tau_max: actuator torque limit (N m), the same for every joint; None: no torque
            constraints. Do not use a huge number instead: constraints far from active
            (slack ~1e9) derail qpax's interior-point steps, and it returns a feasible
            but far-from-optimal s* with no violation to show it.
        faces: pyramid faces per friction cone.
        s_cap: upper bound on s.

    Returns:
        A tuple ``(c, A_eq, b_eq, G, h)``: minimize ``c . x`` subject to
        ``A_eq x = b_eq`` and ``G x <= h``.

        c: (3p + 1,) the objective (-1 on s: maximize s).
        A_eq, b_eq: (6, 3p + 1), (6,) equilibrium.
        G, h: (m, 3p + 1), (m,) adhesion, friction, torques and the cap on s.
    """
    return _lp(st, planted_idx, normals, adhesion, mu, tau_max, faces,
               eq_col=-st.bias_base, eq_rhs=jnp.zeros(6),
               tau_const=jnp.zeros_like(st.bias_joint), tau_col=st.bias_joint, cap=s_cap)


def _lp(st: Statics, planted_idx, normals, adhesion, mu, tau_max, faces, *, eq_col,
        eq_rhs, tau_const, tau_col, cap, slack=False):
    """The shared LP of :func:`hold_lp`, :func:`disturbance_lp` and :func:`foot_forces`, in
    ``x = (F_P, t)``: the planted feet's forces and one scalar t to maximize.

        max t  subject to
          equilibrium:   G_P F_P + t eq_col = eq_rhs
          adhesion:      n_j . F_j >= -adhesion          (+ t, if slack)
          friction:      d_k . F_j <= mu_p (n_j . F_j + adhesion), inner pyramid (- t, if slack)
          torques:       |tau_const + t tau_col - J_P^T F_P| <= tau_max (if not None)
          bounded:       t <= cap

    With ``slack``, t is a common margin (N) on every adhesion and friction row.

    Returns:
        ``(c, A_eq, b_eq, G, h)`` as in :func:`hold_lp`.
    """
    idx = jnp.asarray(planted_idx)
    p = len(planted_idx)
    nx = 3 * p + 1
    normals = jnp.asarray(normals, float)

    G_eq = jnp.transpose(st.J_base[idx], (1, 0, 2)).reshape(6, 3 * p)
    A_eq = jnp.concatenate([G_eq, jnp.asarray(eq_col)[:, None]], axis=1)
    b_eq = jnp.asarray(eq_rhs)

    # torques tau = tau_const + t tau_col - J_P^T F_P, bounded on both sides
    rows, rhs = [], []
    if tau_max is not None:
        T = jnp.concatenate([-jnp.transpose(st.J_joint[idx], (1, 0, 2)).reshape(-1, 3 * p),
                             jnp.asarray(tau_col)[:, None]], axis=1)
        rows += [T, -T]
        rhs += [tau_max - tau_const, tau_max + tau_const]

    mu_p = mu * math.cos(math.pi / faces)
    angles = 2 * math.pi * jnp.arange(faces) / faces
    for j in range(p):
        n = normals[j]
        t1, t2 = _tangents(n)
        d = jnp.cos(angles)[:, None] * t1 + jnp.sin(angles)[:, None] * t2   # (faces, 3)
        foot = jnp.zeros((faces + 1, nx))
        foot = foot.at[0, 3 * j: 3 * j + 3].set(-n)                          # -n.F <= A
        foot = foot.at[1:, 3 * j: 3 * j + 3].set(d - mu_p * n)               # friction
        if slack:
            foot = foot.at[:, -1].set(1.0)                                   # + t <= rhs
        rows.append(foot)
        rhs.append(jnp.concatenate([jnp.array([adhesion]),
                                    jnp.full(faces, mu_p * adhesion)]))
    rows.append(jnp.zeros((1, nx)).at[0, -1].set(1.0))
    rhs.append(jnp.array([cap]))

    c = jnp.zeros(nx).at[-1].set(-1.0)
    return c, A_eq, b_eq, jnp.concatenate(rows), jnp.concatenate(rhs)


def _solve(c, A_eq, b_eq, G, h, max_iter):
    """Solve the LP with qpax (QR); returns ``(x, violation)``."""
    Q = 1e-6 * jnp.eye(c.shape[0])                    # an LP; qpax wants a QP
    x = qpax.solve_qp(Q, c, A_eq, b_eq, G, h, max_iter=max_iter,
                      linear_solver=qpax.LinearSolver.QR)[0]
    violation = jnp.maximum(jnp.abs(A_eq @ x - b_eq).max(), jnp.maximum(G @ x - h, 0.0).max())
    return x, violation


def hold_margin(st: Statics, planted_idx, normals, adhesion, mu, tau_max,
                faces=PYRAMID_FACES, s_cap=S_CAP, max_iter=100):
    """The largest gravity factor s* at which the planted feet hold the posture.

    Solves :func:`hold_lp` with qpax (QR linear solver: the default Cholesky fails on
    the degenerate LPs where s* = 0 or s* = s_cap). qpax rarely reports convergence on
    these LPs although the answer is accurate (checked against HiGHS in
    ``test_statics.py``), so the result comes with its own constraint violation: trust
    s* when ``violation`` is small (e.g. < 1e-3).

    Args:
        st, planted_idx, normals, adhesion, mu, tau_max, faces, s_cap: see
            :func:`hold_lp`.
        max_iter: qpax iterations.

    Returns:
        A tuple ``(s, F, violation)``:

        s: the gravity factor s*. >= 1: the posture holds under g, with margin s - 1.
            0: it does not hold under any gravity of this direction.
        F: (p, 3) the planted feet's forces at s*, world frame (N).
        violation: the largest constraint violation of the solution (equality residual
            or inequality excess, in N or N m).
    """
    x, violation = _solve(*hold_lp(st, planted_idx, normals, adhesion, mu, tau_max, faces,
                                   s_cap), max_iter)
    return x[-1], x[:-1].reshape(-1, 3), violation


# --------------------------------------------------------------------------- #
# Foot forces under the real gravity                                          #
# --------------------------------------------------------------------------- #
FORCE_CAP = 100.0   # upper bound on the slack t (N)


def foot_forces_lp(st: Statics, planted_idx, normals, adhesion, mu, tau_max,
                   faces=PYRAMID_FACES, cap=FORCE_CAP):
    """The LP of :func:`foot_forces` (``(c, A_eq, b_eq, G, h)`` as in :func:`hold_lp`), for
    another solver: qpax is unreliable on it (see :func:`foot_forces`)."""
    return _lp(st, planted_idx, normals, adhesion, mu, tau_max, faces, eq_col=jnp.zeros(6),
               eq_rhs=st.bias_base, tau_const=st.bias_joint,
               tau_col=jnp.zeros_like(st.bias_joint), cap=cap, slack=True)


def foot_forces(st: Statics, planted_idx, normals, adhesion, mu, tau_max, faces=PYRAMID_FACES,
                cap=FORCE_CAP, max_iter=100):
    """Foot forces that hold the posture under the real gravity, as far from the adhesion
    and friction limits as possible.

    The forces are not unique (internal forces); this picks the most central ones: the
    largest common slack t (N) on every adhesion and friction row, at gravity factor 1,
    with the torque limits as hard constraints. t < 0: the posture does not hold; the
    forces then break the limits by about |t|.

    **Unreliable with qpax** (2026-10-06): on wall postures a third of the solutions miss
    equilibrium by > 0.5 N, and it fails where the posture does not hold; HiGHS solves
    the same LP exactly. For exact forces, solve :func:`foot_forces_lp` with HiGHS
    (``scipy.optimize.linprog(method="highs")``), as the scores experiment does.

    Args:
        st, planted_idx, normals, adhesion, mu, tau_max, faces: see :func:`hold_lp`.
        cap: upper bound on t (N).
        max_iter: qpax iterations.

    Returns:
        A tuple ``(F, t, violation)``:

        F: (p, 3) the planted feet's forces, world frame (N): what surface and magnet
            exert on each foot. ``n_j . F_j > 0`` pushes, ``< 0`` pulls (at most A).
        t: the common slack (N).
        violation: the LP solution's constraint violation.
    """
    x, violation = _solve(*_lp(st, planted_idx, normals, adhesion, mu, tau_max, faces,
                               eq_col=jnp.zeros(6), eq_rhs=st.bias_base,
                               tau_const=st.bias_joint, tau_col=jnp.zeros_like(st.bias_joint),
                               cap=cap, slack=True), max_iter)
    return x[:-1].reshape(-1, 3), x[-1], violation


# --------------------------------------------------------------------------- #
# Disturbance margin                                                          #
# --------------------------------------------------------------------------- #
DIST_LENGTH = 0.1   # a moment M counts as a force M / DIST_LENGTH (m)
DIST_CAP = 200.0    # upper bound on the disturbance (N)


def disturbance_directions(length=DIST_LENGTH):
    """The 12 unit disturbances, as generalized forces on the base dofs (6,): +/- a force
    of 1 N along each axis, +/- a moment of ``length`` N m about each axis (a force of
    1 N at ``length``). A direction e is the extra load the feet must carry, i.e. an
    external push of -e on the body (the set is symmetric, so the minimum is the same). Axes as the statics' base rows: for :func:`statics` the forces
    in the world frame and the moments in the body frame (MuJoCo's free joint), for
    :func:`point_mass_statics` both in the world frame.

    Returns:
        (12, 6) array.
    """
    e = jnp.concatenate([jnp.eye(3), jnp.zeros((3, 3))], 1)
    m = jnp.concatenate([jnp.zeros((3, 3)), length * jnp.eye(3)], 1)
    return jnp.concatenate([e, -e, m, -m])


def disturbance_lp(st: Statics, planted_idx, normals, adhesion, mu, tau_max, direction,
                   faces=PYRAMID_FACES, cap=DIST_CAP):
    """The disturbance LP for one direction, in ``x = (F_P, lam)``: how large an extra
    wrench ``lam * direction`` on the body the planted feet still resist, under the real
    gravity (factor 1).

        max lam  subject to  G_P F_P - lam direction = bias_base, the limits of
        :func:`hold_lp` with the torques at gravity factor 1, lam <= cap.

    Args:
        st, planted_idx, normals, adhesion, mu, tau_max, faces: see :func:`hold_lp`.
        direction: (6,) the disturbance's direction (:func:`disturbance_directions`).
        cap: upper bound on lam.

    Returns:
        ``(c, A_eq, b_eq, G, h)`` as in :func:`hold_lp`.
    """
    return _lp(st, planted_idx, normals, adhesion, mu, tau_max, faces,
               eq_col=-jnp.asarray(direction, float), eq_rhs=st.bias_base,
               tau_const=st.bias_joint, tau_col=jnp.zeros_like(st.bias_joint), cap=cap)


def disturbance_margin(st: Statics, planted_idx, normals, adhesion, mu, tau_max,
                       directions=None, faces=PYRAMID_FACES, cap=DIST_CAP, max_iter=100):
    """How large a push or twist on the body the stance resists, in its worst direction.

    Per direction e (default :func:`disturbance_directions`: 6 forces, 6 moments as
    force x ``DIST_LENGTH``), the largest lam such that the feet still hold the posture
    under the real gravity plus ``lam e`` on the body (:func:`disturbance_lp`). Unlike
    s*, this depends on where the feet are and how high the body is even when friction
    caps s* (a vertical wall): peel moments, pushes towards a weak foot.

    Only meaningful where the posture holds at all (s* >= 1): otherwise the LP is
    infeasible and lam is meaningless; check s* first.

    Args:
        st, planted_idx, normals, adhesion, mu, tau_max, faces: see :func:`hold_lp`.
        directions: (K, 6) disturbance directions.
        cap: upper bound on lam (N).
        max_iter: qpax iterations.

    Returns:
        A tuple ``(margin, lam, violation)``:

        margin: min over the directions of lam (N; moments as force x ``DIST_LENGTH``).
        lam: (K,) lam per direction.
        violation: (K,) the solutions' constraint violations.
    """
    directions = disturbance_directions() if directions is None else directions

    def one(e):
        x, viol = _solve(*disturbance_lp(st, planted_idx, normals, adhesion, mu, tau_max, e,
                                         faces, cap), max_iter)
        return x[-1], viol

    lam, viol = jax.vmap(one)(directions)
    return lam.min(), lam, viol
