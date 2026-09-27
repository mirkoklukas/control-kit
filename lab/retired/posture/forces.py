"""Static stance forces: joint torques + foot reaction forces for a planted posture.

Scenario A -- given a body pose + joint angles + which feet are planted, solve the
statics (no stepping, no welds): the planted feet must supply the gravity wrench on
the body. This is one ``mjx.forward`` + a small least-squares, differentiable and
vmappable over N postures. (A weld only matters if you also want to run dynamics;
for the force *readout* the planted set just selects which feet's Jacobians bear
load. A single forward of a weld does NOT give equilibrium forces -- this does.)

Method -- static equilibrium at qvel = qacc = 0:

    sum_i  Jp_i^T f_i  =  qfrc_bias            (per dof; qfrc_bias = gravity gen. force)

  * base (free-joint) rows -> solve the foot forces f_i (feet support the body),
  * joint rows             -> tau = qfrc_bias_joint - sum_i Jp_i,joint^T f_i.

A planted foot with normal force <= 0 means the stance can't be held there (the
foot would have to pull / it lifts) -- a tip-over signal.

This is the ``lab/posture`` variant of ``lab/stance_graph/mjx_stance``, wired to the
4-DOF-per-leg model ``6x4DOF.xml`` (24 joint DOF). Everything below is written in
terms of ``nf`` legs and ``(nv - 6) / nf`` joints per leg, so it does not assume a
particular joint count.

Run under ``uv run --extra mjx``.
"""
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

# The model-agnostic statics core now lives in the library; this module keeps the
# lab-local glue (its own Posture/Stance, the 6x4DOF model, self-collision) on top.
from controlkit.forces import (
    stance_forces, stance_wrenches, stance_sensitivity,
    stance_forces_batch, stance_wrenches_batch, stance_sensitivity_batch,
    stance_sigma_min,
)

from .kinematics import (
    Posture, RADIUS, SHOULDERS, _forward_model, NUM_LEGS as LEGS,
)

MODEL = Path(__file__).resolve().parent / "6x4DOF.xml"


def load_model(path=MODEL):
    """Load the model and locate the foot bodies.

    Args:
        path: model XML path.

    Returns:
        mjx_model: the ``put_model``'d model.
        foot_ids: (LEGS,) foot body ids, leg order 0..LEGS-1.
    """
    m = mujoco.MjModel.from_xml_path(str(path))
    mx = mjx.put_model(m)
    foot_ids = jnp.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"foot{i}")
                          for i in range(LEGS)])
    return mx, foot_ids


def to_qpos(posture: Posture):
    """Assemble a model ``qpos`` from a posture.

    Reorders the base pose from jaxlie's ``wxyz_xyz`` (quat-first) to MuJoCo's
    free-joint layout (position-first) and appends the flattened joint angles.
    vmap over a batched ``Posture`` for the ``(N, nq)`` stack.

    Args:
        posture: a :class:`..kinematics.Posture` (single; ``body`` SE3 +
            ``(NUM_LEGS, NUM_JOINTS)`` ``thetas``).

    Returns:
        (nq,) qpos ``[x y z qw qx qy qz | NUM_LEGS*NUM_JOINTS joint angles]``.
    """
    wxyz_xyz = posture.body.wxyz_xyz                             # (7,): qw qx qy qz x y z
    free = jnp.concatenate([wxyz_xyz[4:], wxyz_xyz[:4]])        # -> x y z qw qx qy qz
    return jnp.concatenate([free, jnp.asarray(posture.thetas).reshape(-1)])


def posture_forces(mjx_model, foot_ids, posture: Posture):
    """:func:`stance_forces` for a :class:`..kinematics.Posture`.

    Builds ``qpos`` via :func:`to_qpos` and uses ``posture.stance.support`` as the
    planted mask. vmap over a batched ``Posture`` for N postures.

    Args:
        mjx_model: the ``put_model``'d model.
        foot_ids: (LEGS,) foot body ids.
        posture: a :class:`..kinematics.Posture` (single).

    Returns:
        foot_forces: (LEGS, 3) world-frame reaction force per foot (0 for unplanted
            legs); joint_torques: (LEGS, JOINTS) actuator torque per leg.
    """
    return stance_forces(mjx_model, foot_ids, to_qpos(posture), posture.stance.support)


def posture_wrenches(mjx_model, foot_ids, posture: Posture):
    """:func:`stance_wrenches` for a :class:`..kinematics.Posture`.

    Builds ``qpos`` via :func:`to_qpos` and uses ``posture.stance.support`` as the
    planted mask. vmap over a batched ``Posture`` for N postures.

    Args:
        mjx_model: the ``put_model``'d model.
        foot_ids: (LEGS,) foot body ids.
        posture: a :class:`..kinematics.Posture` (single).

    Returns:
        foot_forces: (LEGS, 3), foot_moments: (LEGS, 3), joint_torques: (LEGS, JOINTS).
    """
    return stance_wrenches(mjx_model, foot_ids, to_qpos(posture), posture.stance.support)


def posture_sigma_min(mjx_model, foot_ids, posture: Posture, *, length=RADIUS):
    """:func:`stance_sigma_min` for a :class:`..kinematics.Posture`."""
    return stance_sigma_min(mjx_model, foot_ids, to_qpos(posture),
                            posture.stance.support, length=length)


# hexbody AABB half-extents in the body frame: hexagon circumradius RADIUS
# (corners at 30/90/... deg -> x = R cos30, y = R), prism thickness 0.05.
BODY_HALF = (RADIUS * 0.8660254, RADIUS, 0.025)


def _seg_aabb_hit(a, b, half):
    """Do segments ``a->b`` intersect the origin-centred AABB of half-extents ``half``?

    Slab test clipped to the segment, requiring a *positive-length* overlap (so a
    segment merely touching a face -- e.g. at a leg root -- does not count).

    Args:
        a, b: (..., 3) segment endpoints (body frame).
        half: (3,) box half-extents.

    Returns:
        (...,) bool -- True where the open segment interior meets the box interior.
    """
    d = b - a
    eps = 1e-9
    safe = jnp.where(jnp.abs(d) < eps, eps, d)
    t1 = (-half - a) / safe
    t2 = (half - a) / safe
    lo = jnp.minimum(t1, t2)
    hi = jnp.maximum(t1, t2)
    par = jnp.abs(d) < eps                                     # axis parallel to a slab
    inside = jnp.abs(a) <= half
    lo = jnp.where(par, jnp.where(inside, -jnp.inf, jnp.inf), lo)
    hi = jnp.where(par, jnp.where(inside, jnp.inf, -jnp.inf), hi)
    t_enter = jnp.maximum(jnp.max(lo, axis=-1), 0.0)
    t_exit = jnp.minimum(jnp.min(hi, axis=-1), 1.0)
    return t_enter < t_exit


def _leg_segments(thetas):
    """(LEGS, 3, 2, 3) femur/tibia segments per leg, in the body frame (coxa skipped).

    Joint positions from ``_forward_model`` lifted by ``SHOULDERS``; the base pose is
    irrelevant, so it isn't used. The coxa stub (shoulder -> first joint) always
    points radially outward, so it is dropped. Last two axes are ``[a, b]`` endpoints.
    """
    pts = jax.vmap(lambda sh, th: sh.apply(_forward_model(th)))(SHOULDERS, jnp.asarray(thetas))  # (LEGS,5,3)
    return jnp.stack([pts[:, 1:-1, :], pts[:, 2:, :]], axis=-2)   # (LEGS,3,2,3)


def _seg_seg_dist(p1, q1, p2, q2):
    """Closest distance between segments ``p1->q1`` and ``p2->q2`` (broadcasts over
    leading axes). Clamped closest-point solve (Ericson)."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a = (d1 * d1).sum(-1); e = (d2 * d2).sum(-1); f = (d2 * r).sum(-1)
    b = (d1 * d2).sum(-1); c = (d1 * r).sum(-1)
    eps = 1e-12
    denom = a * e - b * b
    s = jnp.where(denom > eps, jnp.clip((b * f - c * e) / jnp.where(denom > eps, denom, 1.0), 0.0, 1.0), 0.0)
    t = jnp.clip((b * s + f) / jnp.where(e > eps, e, 1.0), 0.0, 1.0)
    s = jnp.clip((t * b - c) / jnp.where(a > eps, a, 1.0), 0.0, 1.0)
    c1 = p1 + s[..., None] * d1
    c2 = p2 + t[..., None] * d2
    return jnp.linalg.norm(c1 - c2, axis=-1)


def legs_intersect_body(posture: Posture, *, box=BODY_HALF, margin=0.0):
    """True if any leg segment penetrates the body's bounding box.

    The body is an axis-aligned box in the body frame (default the hexbody AABB,
    :data:`BODY_HALF`). The coxa stubs are skipped; the femur/tibia segments are
    tested (see :func:`_leg_segments`).

    Args:
        posture: a :class:`..kinematics.Posture` (single; only ``thetas`` are used).
        box: (3,) body-frame AABB half-extents.
        margin: added to ``box`` (e.g. the leg-capsule radius) for clearance.

    Returns:
        scalar bool -- True if any tested segment intersects the (inflated) box.
    """
    segs = _leg_segments(posture.thetas)                      # (LEGS,3,2,3)
    return _seg_aabb_hit(segs[..., 0, :], segs[..., 1, :], jnp.asarray(box) + margin).any()


def self_collision(posture: Posture, *, box=BODY_HALF, margin=0.0,
                   leg_radius=0.015, foot_radius=0.025):
    """True if the posture self-collides.

    Checks, on the femur/tibia segments + foot spheres (coxa stubs skipped):
      * segment vs the body bounding box (see :func:`legs_intersect_body`);
      * segment vs another leg's segment (centerlines within ``2 * leg_radius``);
      * foot sphere vs the body box (center within ``box + foot_radius``);
      * foot sphere vs another leg's foot (centers within ``2 * foot_radius``).
    Same-leg (within-a-leg) pairs are excluded -- adjacent links share joints.

    Args:
        posture: a :class:`..kinematics.Posture` (single; only ``thetas`` are used).
        box: (3,) body-frame AABB half-extents.
        margin: added to ``box`` for the segment/body test.
        leg_radius: leg-capsule radius; inter-leg collision if centerlines are within
            ``2 * leg_radius``.
        foot_radius: foot-sphere radius.

    Returns:
        scalar bool.
    """
    segs = _leg_segments(posture.thetas)                      # (LEGS, S, 2, 3)
    a, b = segs[..., 0, :], segs[..., 1, :]
    feet = b[:, -1, :]                                        # (LEGS, 3) foot = last tibia endpoint

    # segment vs body box, and segment vs other legs' segments
    body = _seg_aabb_hit(a, b, jnp.asarray(box) + margin).any()
    S = segs.shape[1]
    P, Q = a.reshape(LEGS * S, 3), b.reshape(LEGS * S, 3)      # segments flattened, leg-major
    dist = _seg_seg_dist(P[:, None, :], Q[:, None, :], P[None, :, :], Q[None, :, :])  # (M, M)
    leg = jnp.repeat(jnp.arange(LEGS), S)                      # leg id per segment
    seg_pairs = jnp.triu(jnp.ones((LEGS * S, LEGS * S), bool), 1) & (leg[:, None] != leg[None, :])
    inter = ((dist < 2.0 * leg_radius) & seg_pairs).any()

    # foot sphere vs body box, and foot vs foot (different legs)
    foot_body = (jnp.abs(feet) <= (jnp.asarray(box) + foot_radius)).all(-1).any()
    D = jnp.linalg.norm(feet[:, None, :] - feet[None, :, :], axis=-1)   # (LEGS, LEGS)
    foot_foot = ((D < 2.0 * foot_radius) & jnp.triu(jnp.ones((LEGS, LEGS), bool), 1)).any()
    # (could also add foot-vs-other-leg-segment: _seg_seg_dist(feet, P, Q) < foot_radius + leg_radius)

    return body | inter | foot_body | foot_foot


if __name__ == "__main__":
    import numpy as np

    m = mujoco.MjModel.from_xml_path(str(MODEL))
    mx, foot_ids = load_model()
    weight = float(m.body_mass.sum() * abs(m.opt.gravity[2]))
    home = jnp.array(m.key_qpos[0])

    supports = jnp.array([[1, 0, 1, 0, 1, 0],       # tripod
                         [1, 1, 1, 1, 1, 1],       # all six
                         [1, 0, 0, 1, 0, 0]], bool)  # two legs
    qpos = jnp.broadcast_to(home, (supports.shape[0], home.shape[0]))
    ff, tau = jax.jit(stance_forces_batch)(mx, foot_ids, qpos, supports)
    ff, tau = np.asarray(ff), np.asarray(tau)
    print(f"weight = {weight:.1f} N | nq={m.nq} nv={m.nv} tau/leg={tau.shape[-1]}")
    for i, name in enumerate(["tripod 0,2,4", "all six", "two 0,3"]):
        print(f"  {name:14s} sum Fz = {ff[i,:,2].sum():6.1f} N | "
              f"max |tau| = {np.abs(tau[i]).max():5.2f} Nm | "
              f"min planted Fz = {ff[i,:,2][ff[i,:,2] != 0].min():6.2f} N")
