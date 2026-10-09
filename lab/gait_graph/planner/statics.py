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

NULL_TOL = 1e-5     # singular values below this (relative to the largest) span the null space
                    # (float32: the numerically zero ones come out around 1e-7, not 1e-9)
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
# Least torque within the limits                                              #
# --------------------------------------------------------------------------- #
def cones_ok(F, normals, adhesion, mu, faces=PYRAMID_FACES, tol=1e-6):
    """Whether the planted feet's forces are within their contact limits: adhesion
    (``n . F >= -A``) and the friction pyramid around the adhesion-shifted normal force,
    as in :func:`hold_lp`.

    Args:
        F: (p, 3) the planted feet's forces (N).
        normals: (p, 3) unit surface normals.
        adhesion, mu, faces: see :func:`hold_lp`.
        tol: slack allowed on every row (N).

    Returns:
        A tuple ``(ok, per_foot)``: scalar bool, (p,) bool.
    """
    mu_p = mu * math.cos(math.pi / faces)
    angles = 2 * math.pi * jnp.arange(faces) / faces

    def foot(f, n):
        t1, t2 = _tangents(n)
        d = jnp.cos(angles)[:, None] * t1 + jnp.sin(angles)[:, None] * t2
        fn = f @ n
        return (fn >= -adhesion - tol) & jnp.all(d @ f <= mu_p * (fn + adhesion) + tol)

    per = jax.vmap(foot)(F, jnp.asarray(normals, float))
    return jnp.all(per), per


def cone_margin(F, normals, adhesion, mu):
    """How far inside its contact cone each planted foot's force is: the Euclidean
    distance of the adhesion-shifted force ``F + A n`` to the surface of the round cone of
    half-angle ``atan(mu)`` (apex at the origin, axis ``n``).

    With ``a = n . (F + A n)`` and ``t`` the size of the tangential part:
    ``d = (mu a - t) / sqrt(1 + mu^2)``. Covers adhesion and friction together; ``d >= 0``
    iff inside the round cone. (The LPs use the inscribed 8-face pyramid, slightly
    stricter: pass ``mu * cos(pi / 8)`` for a margin that is ``>= 0`` only inside it.)
    Behind the apex (``a < 0``, pulled harder than A) ``d`` is negative but not the exact
    distance.

    Args:
        F: (p, 3) the planted feet's forces (N).
        normals: (p, 3) unit surface normals.
        adhesion: A per foot (N).
        mu: friction coefficient (of the cone used).

    Returns:
        A tuple ``(min, per_foot)``: the smallest margin (N), (p,) margins (N).
    """
    n = jnp.asarray(normals, float)
    Fs = F + adhesion * n
    a = jnp.sum(Fs * n, -1)
    t = jnp.linalg.norm(Fs - a[:, None] * n, axis=-1)
    d = (mu * a - t) / math.sqrt(1.0 + mu ** 2)
    return d.min(), d


def within_limits(F, tau, normals, adhesion, mu, tau_max, faces=PYRAMID_FACES, tol=1e-6):
    """Whether the planted feet's forces and the torques satisfy the limits of :func:`hold_lp`:
    the contact limits (:func:`cones_ok`) and ``|tau| <= tau_max`` (if not None).

    Args:
        F: (p, 3) the planted feet's forces (N).
        tau: (nj,) the servo torques (N m).
        normals, adhesion, mu, tau_max, faces: see :func:`hold_lp`.
        tol: slack allowed on every row (N, N m).

    Returns:
        Scalar bool: True if every limit holds.
    """
    ok = cones_ok(F, normals, adhesion, mu, faces, tol)[0]
    if tau_max is not None:
        ok = ok & jnp.all(jnp.abs(tau) <= tau_max + tol)
    return ok


def least_torque_kkt(st: Statics, planted_idx, reg=1e-8):
    """The least-torque forces of the planted feet and their response to a body push, from
    one linear solve: smooth in the posture (for the NLP refine, ``docs/trajopt-nlp.md``).

    The KKT (Karush-Kuhn-Tucker) conditions of ``min |b_a - M F|^2 s.t. G F = b_b``
    (``M``: forces to servo torques, ``G``: forces to the base wrench), with a multiplier
    ``mu`` (6,) for equilibrium:

        [[M'M + reg I, G'], [G, 0]] [F; mu] = [M' b_a; b_b]

    The same matrix with right-hand side ``[0; w]`` gives the extra forces carrying an extra
    base load ``w`` (a push): ``F(b_b + w) = F + H w``. Equals :func:`least_torque` (up to
    ``reg``) where the matrix is invertible (feet not on a line, no kinematic singularity).

    Args:
        st: the posture's statics.
        planted_idx: tuple of the planted feet (static).
        reg: Tikhonov term on the forces (keeps the matrix invertible).

    Returns:
        A tuple ``(F, tau, H, M)``: (p, 3) forces (N), (nj,) servo torques (N m),
        (3p, 6) push map (``F_e = F + (H @ w).reshape(p, 3)``), (nj, 3p) force-to-torque
        map (``tau_e = b_a - M F_e``).
    """
    idx = jnp.asarray(planted_idx)
    p = len(planted_idx)
    G = jnp.transpose(st.J_base[idx], (1, 0, 2)).reshape(6, 3 * p)
    M = jnp.transpose(st.J_joint[idx], (1, 0, 2)).reshape(-1, 3 * p)
    K = jnp.block([[M.T @ M + reg * jnp.eye(3 * p), G.T], [G, jnp.zeros((6, 6))]])
    rhs = jnp.concatenate([jnp.concatenate([M.T @ st.bias_joint, st.bias_base])[:, None],
                           jnp.concatenate([jnp.zeros((3 * p, 6)), jnp.eye(6)])], 1)
    sol = jnp.linalg.solve(K, rhs)                                   # (3p + 6, 7)
    F = sol[: 3 * p, 0]
    H = sol[: 3 * p, 1:]
    return F.reshape(p, 3), st.bias_joint - M @ F, H, M


def limit_rows(F, tau, normals, adhesion, mu, tau_max, faces=PYRAMID_FACES):
    """The limits of :func:`hold_lp` as smooth rows ``g <= 0`` (for the NLP): per planted
    foot adhesion and the friction pyramid, per servo ``|tau| <= tau_max``.

    Args:
        F: (p, 3) forces (N); tau: (nj,) torques (N m).
        normals, adhesion, mu, tau_max, faces: see :func:`hold_lp`.

    Returns:
        (p (faces + 1) + 2 nj,) row values, <= 0 where the limit holds (N, N m).
    """
    mu_p = mu * math.cos(math.pi / faces)
    angles = 2 * math.pi * jnp.arange(faces) / faces

    def foot(f, n):
        t1, t2 = _tangents(n)
        d = jnp.cos(angles)[:, None] * t1 + jnp.sin(angles)[:, None] * t2
        fn = f @ n
        return jnp.concatenate([jnp.array([-fn - adhesion]), d @ f - mu_p * (fn + adhesion)])

    rows = jax.vmap(foot)(F, jnp.asarray(normals, float)).reshape(-1)
    return jnp.concatenate([rows, tau - tau_max, -tau - tau_max])


def push_check_fast(st: Statics, planted_idx, normals, adhesion, mu, tau_max, lam_min,
                    directions=None, faces=PYRAMID_FACES, tol=1e-6):
    """Does the stance resist a push or twist of ``lam_min`` in every direction, with the
    internal forces the servos settle at (no LP)?

    The least-torque forces (:func:`least_torque`) are linear in the load: per direction e
    the feet carry gravity plus ``lam_min e`` with the least-torque split of that load (the
    compliance answer, ``docs/statics.md`` section 5), and those forces and torques must
    satisfy the limits (:func:`within_limits`; the pyramid, as the LPs). The undisturbed
    solution must too. Sufficient for :func:`disturbance_check` (that LP may choose any
    internal forces per direction), not necessary: a stance can fail here and pass there
    (it then holds the push only with tuned internal forces).

    Args:
        st, planted_idx, normals, adhesion, mu, tau_max, faces: see :func:`hold_lp`.
        lam_min: the push to resist (N; moments as force x ``DIST_LENGTH``).
        directions: (K, 6) push directions (default :func:`disturbance_directions`).
        tol: slack allowed on every limit row (N, N m).

    Returns:
        A tuple ``(ok, per_direction, margin)``: scalar bool (the undisturbed solution
        and every direction within the limits), (K,) bool, the smallest round-cone margin
        over the directions and feet (N, :func:`cone_margin`).
    """
    directions = disturbance_directions() if directions is None else directions
    L = st.J_base.shape[0]
    idx = jnp.asarray(planted_idx)
    planted = jnp.zeros(L, bool).at[idx].set(True)
    A, _ = _equilibrium(st, planted)
    A_pinv = jnp.linalg.pinv(A)
    _, s, Vt = jnp.linalg.svd(A, full_matrices=True)
    s = jnp.concatenate([s, jnp.zeros(Vt.shape[0] - s.shape[0])])
    N = Vt.T * (s <= NULL_TOL * s[0])[None, :]
    K = -jnp.einsum("inc,ick->nk", st.J_joint, N.reshape(-1, 3, N.shape[1]))
    K_pinv = jnp.linalg.pinv(K)

    def least(load):
        """Least-torque forces (L, 3) and torques for the base load ``load`` (6,)."""
        F0 = A_pinv @ jnp.concatenate([load, jnp.zeros(3 * L)])
        z = -K_pinv @ _torques(st, F0.reshape(-1, 3))
        F = (F0 + N @ z).reshape(-1, 3)
        return F, _torques(st, F)

    F, tau = least(st.bias_base)
    ok0 = within_limits(F[idx], tau, normals, adhesion, mu, tau_max, faces, tol)
    Fe, tau_e = jax.vmap(least)(st.bias_base + lam_min * jnp.asarray(directions, float))
    per = jax.vmap(lambda f, t: within_limits(f[idx], t, normals, adhesion, mu, tau_max,
                                              faces, tol))(Fe, tau_e)
    margin = jax.vmap(lambda f: cone_margin(f[idx], normals, adhesion, mu)[0])(Fe).min()
    return ok0 & per.all(), per, margin


def least_torque_fast(st: Statics, planted_idx, normals, adhesion, mu, tau_max,
                      faces=PYRAMID_FACES, tol=1e-6):
    """The least-torque forces without limits (:func:`least_torque`, closed form), and
    whether they satisfy the limits anyway.

    If they do, they are also the optimum of :func:`least_torque_qp` (constraints that the
    unconstrained optimum satisfies do not change it; up to the QP's small regularization),
    so the QP is needed only where ``within`` is False.

    Returns:
        A tuple ``(F, tau, within)``: (p, 3) the planted feet's forces (N), (nj,) torques
        (N m), scalar bool.
    """
    L = st.J_base.shape[0]
    planted = jnp.zeros(L, bool).at[jnp.asarray(planted_idx)].set(True)
    F, tau = least_torque(st, planted)
    F = F[jnp.asarray(planted_idx)]
    return F, tau, within_limits(F, tau, normals, adhesion, mu, tau_max, faces, tol)


def least_torque_qp(st: Statics, planted_idx, normals, adhesion, mu, tau_max,
                    faces=PYRAMID_FACES, reg=1e-6, max_iter=100):
    """The foot forces that hold the posture with the least actuator torque, within the
    limits: a QP.

        min_F  |tau(F)|^2 + reg |F|^2,   tau(F) = bias_joint - J_P^T F
        s.t.   equilibrium (gravity factor 1), adhesion, friction (inner pyramid),
               |tau| <= tau_max (if not None)

    Unlike :func:`least_torque` (no limits: it may pull with feet that cannot) and the
    "most central" forces of :func:`foot_forces_lp` (many optimal distributions with four
    feet; the solver's pick is arbitrary), the answer is unique (``reg``) and within the
    limits. Infeasible where the posture does not hold: check ``violation``.

    Args:
        st, planted_idx, normals, adhesion, mu, tau_max, faces: see :func:`hold_lp`.
        reg: Tikhonov weight on the forces (uniqueness, N m^2 per N^2).
        max_iter: qpax iterations.

    Returns:
        A tuple ``(F, tau, violation)``:

        F: (p, 3) the planted feet's forces, world frame (N).
        tau: (nj,) the actuator torques (N m).
        violation: the solution's largest constraint violation (N or N m).
    """
    p = len(planted_idx)
    # the limits and equilibrium of the LP builder, without its scalar variable
    _, A_eq, b_eq, G, h = _lp(st, planted_idx, normals, adhesion, mu, tau_max, faces,
                              eq_col=jnp.zeros(6), eq_rhs=st.bias_base,
                              tau_const=st.bias_joint, tau_col=jnp.zeros_like(st.bias_joint),
                              cap=0.0)
    A_eq, G, h = A_eq[:, :-1], G[:-1, :-1], h[:-1]          # drop t (and its cap row)
    M = jnp.transpose(st.J_joint[jnp.asarray(planted_idx)], (1, 0, 2)).reshape(-1, 3 * p)
    Q = 2.0 * (M.T @ M) + 2.0 * reg * jnp.eye(3 * p)       # |c - M F|^2 = F'M'MF - 2c'MF + ..
    q = -2.0 * (M.T @ st.bias_joint)
    x = qpax.solve_qp(Q, q, A_eq, b_eq, G, h, max_iter=max_iter,
                      linear_solver=qpax.LinearSolver.QR)[0]
    violation = jnp.maximum(jnp.abs(A_eq @ x - b_eq).max(), jnp.maximum(G @ x - h, 0.0).max())
    return x.reshape(-1, 3), st.bias_joint - M @ x, violation


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


def disturbance_check(st: Statics, planted_idx, normals, adhesion, mu, tau_max, lam_min,
                      directions=None, faces=PYRAMID_FACES, max_iter=100, tol=1e-3):
    """Does the stance resist a push or twist of ``lam_min`` in every direction?

    Per direction e, :func:`disturbance_lp` capped at ``lam_min``: it passes if the
    largest push reaches the cap (within ``tol``, relative) with a small constraint
    violation. Unlike :func:`disturbance_margin` it only asks for ``lam_min``, not the
    largest push. False where the posture does not hold at all.

    Args:
        st, planted_idx, normals, adhesion, mu, tau_max, faces: see :func:`hold_lp`.
        lam_min: the push to resist (N; moments as force x ``DIST_LENGTH``).
        directions: (K, 6) disturbance directions (default :func:`disturbance_directions`).
        max_iter: qpax iterations.
        tol: relative tolerance on reaching ``lam_min``, and the violation bound (N).

    Returns:
        A tuple ``(ok, per_direction)``: scalar bool (every direction passes), (K,) bool.
    """
    directions = disturbance_directions() if directions is None else directions

    def one(e):
        x, viol = _solve(*disturbance_lp(st, planted_idx, normals, adhesion, mu, tau_max, e,
                                         faces, lam_min), max_iter)
        return (x[-1] >= lam_min * (1.0 - tol)) & (viol < tol)

    per = jax.vmap(one)(directions)
    return per.all(), per


# --------------------------------------------------------------------------- #
# Push score: the ball radius (docs/push-score.md)                             #
# --------------------------------------------------------------------------- #
def push_slacks(st: Statics, planted_idx, normals, adhesion, mu, tau_max,
                length=DIST_LENGTH, faces=PYRAMID_FACES, reg=1e-8):
    """The limit slacks at rest and their sensitivity to a body push (``docs/push-score.md``).

    The forces respond linearly to a push w (6,), ``F(w) = F + H S w`` (the least-torque
    response, :func:`least_torque_kkt`; ``S = diag(I, length I)``: w's moment part is
    moment / length, in N). Each limit row of :func:`limit_rows` then has the slack
    ``s_i(w) = s_i - h_i . w``.

    Args:
        st, planted_idx, normals, adhesion, mu, tau_max, faces: see :func:`hold_lp`.
        length: lever length L (m) of the push's moment part.
        reg: see :func:`least_torque_kkt`.

    Returns:
        A tuple ``(s, h)``: (m,) slacks at rest (N, N m), (m, 6) sensitivities h_i.
    """
    F, _, H, M = least_torque_kkt(st, planted_idx, reg)
    HS = H * jnp.array([1.0, 1.0, 1.0, length, length, length])
    F = F.reshape(-1)

    def rows(w):
        Fw = F + HS @ w
        return limit_rows(Fw.reshape(-1, 3), st.bias_joint - M @ Fw, normals, adhesion, mu,
                          tau_max, faces)

    w0 = jnp.zeros(6)
    return -rows(w0), jax.jacfwd(rows)(w0)


def push_score(st: Statics, planted_idx, normals, adhesion, mu, tau_max,
               length=DIST_LENGTH, faces=PYRAMID_FACES, beta=None, h_tol=1e-9):
    """How large a push the stance withstands in every direction: the ball radius
    ``r* = min_i s_i / |h_i|`` over the limits a push affects (``docs/push-score.md``).

    No LP: one linear solve (:func:`least_torque_kkt`) and one ratio per limit row. Negative
    when a limit already fails at rest (then not a distance, only the worst violated
    limit, normalized).

    Args:
        st, planted_idx, normals, adhesion, mu, tau_max, faces: see :func:`hold_lp`.
        length: lever length L (m) of the push's moment part.
        beta: None for the hard minimum; otherwise the sharpness of a soft minimum
            ``-log(sum exp(-beta r_i)) / beta`` (smooth, <= r*).
        h_tol: rows with ``|h_i|`` below this ignore the push (e.g. a lifted leg's servos).

    Returns:
        A tuple ``(r, i_star, holds)``: the score (N; moments as moment / length), the
        index of the limit row that breaks first, scalar bool: every slack >= 0 at rest
        (including the rows the push does not affect).
    """
    s, h = push_slacks(st, planted_idx, normals, adhesion, mu, tau_max, length, faces)
    return _ball_radius(s, h, beta, h_tol)


def _ball_radius(s, h, beta=None, h_tol=1e-9):
    """``(r, i_star, holds)`` from slacks s (m,) and sensitivities h (m, 6): see
    :func:`push_score`."""
    hn = jnp.linalg.norm(h, axis=-1)
    affected = hn > h_tol
    ratio = jnp.where(affected, s / jnp.where(affected, hn, 1.0), jnp.inf)
    i_star = jnp.argmin(ratio)
    if beta is None:
        r = ratio[i_star]
    else:
        r = -jax.nn.logsumexp(-beta * jnp.where(affected, ratio, 1e6)) / beta
    return r, i_star, jnp.all(s >= 0.0)


# --------------------------------------------------------------------------- #
# Push score without the servos: minimum-norm forces (docs/push-score.md)      #
# --------------------------------------------------------------------------- #
@dataclass
class MassModel:
    """Where a robot's mass sits, for its centre of mass from the kinematics alone.

    Args:
        body: mass of the body (kg), at the body frame's origin.
        links: (n_joints,) mass of each leg link (kg), per leg.
        link_com: (n_joints,) each link's centre of mass along its frame's x-axis (m),
            from its joint.
        foot: mass at the foot point (kg), per leg (the pad, approximated at the pivot).
    """
    body: float
    links: jax.Array
    link_com: jax.Array
    foot: float


def total_mass(mm: MassModel, num_legs: int) -> jax.Array:
    """The robot's total mass (kg)."""
    return mm.body + num_legs * (jnp.sum(mm.links) + mm.foot)


def kinematic_com(robot, posture, mm: MassModel) -> jax.Array:
    """The robot's centre of mass from its kinematics (no MuJoCo), world frame (m).

    The body's mass at the body origin, each link's along its frame's x-axis, the foot's at
    the foot point.

    Args:
        robot: a :class:`controlkit.kinematics.Robot` (mounts, leg).
        posture: body pose and (L, n_joints) angles.
        mm: the mass model.

    Returns:
        (3,) centre of mass.
    """
    sh = robot.shoulders(posture.body)
    L = posture.thetas.shape[0]

    def leg(i):
        fr = robot.leg.forward(posture.thetas[i])                       # (n+1,) base frame
        n = mm.links.shape[0]
        link = jax.vmap(lambda k: fr[k].apply(jnp.array([1.0, 0.0, 0.0]) * mm.link_com[k]))(
            jnp.arange(n))                                              # (n, 3)
        pts = jnp.concatenate([link, fr.translation()[-1:]], 0)        # links, foot
        w = jnp.concatenate([mm.links, jnp.array([mm.foot])])
        return jax.vmap(sh[i].apply)(pts), w

    pts, w = jax.vmap(leg)(jnp.arange(L))                               # (L, n+1, 3), (L, n+1)
    num = mm.body * posture.body.translation() + jnp.einsum("lk,lkc->c", w, pts)
    return num / (mm.body + jnp.sum(w))


def leg_torques(robot, mm: MassModel, g, body, thetas, forces):
    """Servo torques from the kinematics alone (no MuJoCo): ``tau_j = b_a,j - J_j^T f_j`` per
    leg, the torques that hold the legs' own weight and the foot forces.

    ``J_j`` (3, n_joints): the Jacobian of leg j's foot point (world) w.r.t. its joint angles;
    ``b_a,j``: the gravity torque of the leg's links, the gradient of their potential energy
    ``-sum_k m_k g . c_k(theta)`` (link masses at ``mm.link_com`` along each link, the foot
    mass at the foot point, as :func:`kinematic_com`). Same convention as :func:`statics`:
    ``tau = bias_joint - J^T F``, F the force the surface exerts on the foot.

    Args:
        robot: the :class:`Robot`.
        mm: the mass model.
        g: (3,) gravity, world (m/s^2).
        body: SE3 body pose.
        thetas: (L, n_joints) joint angles.
        forces: (L, 3) foot forces, world (zero for legs in the air).

    Returns:
        (L, n_joints) torques (N m).
    """
    sh = robot.shoulders(body)
    n = mm.links.shape[0]
    g = jnp.asarray(g, float)

    def leg(j):
        def foot(th):
            return sh[j].apply(robot.leg.forward(th).translation()[-1])

        def potential(th):
            fr = robot.leg.forward(th)
            link = jax.vmap(lambda k: fr[k].apply(jnp.array([1.0, 0.0, 0.0]) * mm.link_com[k]))(
                jnp.arange(n))
            pts = jax.vmap(sh[j].apply)(jnp.concatenate([link, fr.translation()[-1:]], 0))
            w = jnp.concatenate([mm.links, jnp.array([mm.foot])])
            return -jnp.sum(w * (pts @ g))

        th = thetas[j]
        return jax.grad(potential)(th) - jax.jacfwd(foot)(th).T @ forces[j]

    return jax.vmap(leg)(jnp.arange(thetas.shape[0]))


def min_norm_forces(com, feet, mass, g, length=DIST_LENGTH, eps=1e-10):
    """Minimum-norm foot forces in equilibrium, and their response to a body push.

    The robot as a point mass at ``com`` (:func:`point_mass_statics`): ``G F = b_b``, ``G``
    (6, 3p) the planted feet's forces to the net force and moment about ``com``. Of all
    forces in equilibrium, the one with the smallest sum of squared foot forces ``|F|^2``
    (**not** the least servo torque, which needs the servo rows); likewise for a push w (6,):

        F = G^T (G G^T)^-1 b_b,    H = G^T (G G^T)^-1 S,    F(w) = F + H w

    (``S = diag(I, length I)``: w's moment part is moment / length, in N).

    Args:
        com: (3,) centre of mass, world frame (m) (e.g. :func:`kinematic_com`).
        feet: (p, 3) planted foot points (where the force acts: the ankle pivots), world.
        mass: total mass (kg).
        g: (3,) gravity, world frame (m/s^2).
        length: lever length L (m) of the push's moment part.
        eps: Tikhonov term on ``G G^T`` (feet on a line make it singular).

    Returns:
        A tuple ``(F, H)``: (p, 3) forces the surface exerts on the feet (N), (3p, 6) push
        response.
    """
    st = point_mass_statics(com, feet, mass, g)
    G = jnp.transpose(st.J_base, (1, 0, 2)).reshape(6, -1)             # (6, 3p)
    S = jnp.diag(jnp.array([1.0, 1.0, 1.0, length, length, length]))
    X = jnp.linalg.solve(G @ G.T + eps * jnp.eye(6),
                         jnp.concatenate([st.bias_base[:, None], S], 1))   # (6, 7)
    FH = G.T @ X                                                        # (3p, 7)
    return FH[:, 0].reshape(-1, 3), FH[:, 1:]


def hold_min_norm(com, feet, normals, mass, g, adhesion, mu, faces=PYRAMID_FACES):
    """Does the stance hold under gravity with the minimum-norm foot forces, within adhesion
    and friction (:func:`min_norm_forces`)? No pushes, no servos, no torque limits.

    A **sufficient** equilibrium check: if it holds, some forces hold the stance; if not,
    other internal forces might still (exact: an LP over all F, :func:`hold_lp` without
    torques). One 6 x 6 solve and the foot rows.

    Args:
        com, feet, mass, g: see :func:`min_norm_forces`.
        normals: (p, 3) unit surface normals at the planted feet.
        adhesion, mu, faces: see :func:`hold_lp`.

    Returns:
        A tuple ``(holds, F, margin)``: scalar bool; (p, 3) the forces (N); the smallest
        slack over the foot rows (N; < 0: some row fails).
    """
    F, _ = min_norm_forces(com, feet, mass, g)
    s = -limit_rows(F, jnp.zeros(0), normals, adhesion, mu, 0.0, faces)
    return jnp.all(s >= 0.0), F, s.min()


def min_norm_slacks(com, feet, normals, mass, g, adhesion, mu, length=DIST_LENGTH,
                    faces=PYRAMID_FACES, eps=1e-10):
    """Foot-row slacks and their sensitivity to a body push, with minimum-norm forces.

    Only the base rows of the statics, for the robot as a point mass at ``com``
    (:func:`point_mass_statics`): ``G F = b_b``, with ``G`` (6, 3p) the planted feet's
    forces to the net force and moment about ``com``. The forces are the **minimum-norm**
    solution, the smallest sum of squared foot forces ``|F|^2`` in equilibrium (not the
    least servo torque, which would need the servo rows), and so is their response to a
    push w (6,):

        F = G^T (G G^T)^-1 b_b,    H = G^T (G G^T)^-1 S,    F(w) = F + H w

    (``S = diag(I, length I)``: w's moment part is moment / length, in N). The limits are
    the foot rows only, adhesion and the friction pyramid (:func:`limit_rows` without
    torques); no servo torques, no torque limits.

    Args:
        com: (3,) centre of mass, world frame (m) (e.g. :func:`kinematic_com`).
        feet: (p, 3) planted foot points (where the force acts: the ankle pivots), world.
        normals: (p, 3) unit surface normals at the planted feet.
        mass: total mass (kg).
        g: (3,) gravity, world frame (m/s^2).
        adhesion, mu, faces: see :func:`hold_lp`.
        length: lever length L (m) of the push's moment part.
        eps: Tikhonov term on ``G G^T`` (feet on a line make it singular).

    Returns:
        A tuple ``(s, h)``: (p (faces + 1),) slacks at rest (N), (p (faces + 1), 6)
        sensitivities.
    """
    F, H = min_norm_forces(com, feet, mass, g, length, eps)
    F = F.reshape(-1)

    def rows(w):
        return limit_rows((F + H @ w).reshape(-1, 3), jnp.zeros(0), normals, adhesion, mu,
                          0.0, faces)

    w0 = jnp.zeros(6)
    return -rows(w0), jax.jacfwd(rows)(w0)


def push_score_min_norm(com, feet, normals, mass, g, adhesion, mu, length=DIST_LENGTH,
                        faces=PYRAMID_FACES, beta=None, h_tol=1e-9):
    """The push score (the ball radius, :func:`push_score`) without the servos: minimum-norm
    foot forces, adhesion and friction only (:func:`min_norm_slacks`).

    Needs no MuJoCo statics, only the planted feet and the centre of mass, so it is cheap.
    It differs from :func:`push_score` in two ways: the force split (minimum ``|F|^2``,
    not least servo torque; another split of the internal forces) and no torque limits
    (a stance the servos cannot hold can score well here). ``holds``: the minimum-norm
    forces satisfy adhesion and friction, a **sufficient** equilibrium check (other
    internal forces might hold where these do not).

    Args:
        com, feet, normals, mass, g, adhesion, mu, length, faces: see
            :func:`min_norm_slacks`.
        beta, h_tol: see :func:`push_score`.

    Returns:
        A tuple ``(r, i_star, holds)`` as :func:`push_score` (rows: foot rows only).
    """
    s, h = min_norm_slacks(com, feet, normals, mass, g, adhesion, mu, length, faces)
    return _ball_radius(s, h, beta, h_tol)


def push_lambda(s, h, directions):
    """Largest push ``lam`` along each direction e (unit, in w's units) that the stance
    withstands: ``min_{i: h_i . e > 0} s_i / (h_i . e)`` (inf if no row limits it).

    Args:
        s, h: from :func:`push_slacks`.
        directions: (K, 6) unit push directions.

    Returns:
        (K,) push sizes (N).
    """
    he = h @ jnp.asarray(directions).T                               # (m, K)
    pos = he > 0.0
    return jnp.min(jnp.where(pos, s[:, None] / jnp.where(pos, he, 1.0), jnp.inf), axis=0)
