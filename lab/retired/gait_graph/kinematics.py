"""Single-leg hexapod kinematics + the flat-ground stability heuristic.

Copied (and lightly cleaned) from ``lab.ik.core`` so ``gait_graph`` stays
self-contained -- this is the churny leg-level layer the gait graph is built on.
Everything is pure jax on jaxlie ``SE3``/``SO3``; ``jax.vmap`` handles batches
(the 6 legs, or many body poses). The leg is a 3R chain -- coxa (yaw about local
+z) -> femur (lift about +y) -> tibia (knee about +y). The *shoulder frame* is
the frame in which, at zero angles, the leg extends along +x.

Constants match ``models/weld0.xml``. Run under ``uv run --extra mjx``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxlie import SE3, SO3


# Let a batched SE3 be indexed/sliced like an array (jaxlie doesn't support it).
def _se3_getitem(self, index) -> SE3:
    return SE3(self.wxyz_xyz[index])
SE3.__getitem__ = _se3_getitem

def _se3_shape(self) -> tuple:
    return self.wxyz_xyz.shape
SE3.shape = property(_se3_shape)


# --- hexapod model constants (from models/weld0.xml) --------------------------
# Six legs on a hexagon, circumradius 0.15, at 30/90/.../330 deg. Each shoulder
# is yawed by its angle so the leg's local +x points radially outward. SHOULDERS
# is a batched SE3 of shape (6,), expressed in the base (body) frame.
_SHOULDER_ANGLES = jnp.deg2rad(jnp.arange(30.0, 360.0, 60.0))   # 30,90,...,330
SHOULDERS = SE3.from_rotation_and_translation(
    SO3.from_z_radians(_SHOULDER_ANGLES),
    0.15 * jnp.stack([jnp.cos(_SHOULDER_ANGLES),
                      jnp.sin(_SHOULDER_ANGLES),
                      jnp.zeros(6)], axis=-1),
)

# Segment lengths [coxa, femur, tibia] -- local +x offsets down the chain.
LENGTHS = jnp.array([0.025, 0.2, 0.206])

# Joint limits [coxa, femur, tibia], radians, from weld0.xml (<default> classes).
JOINT_RANGES = jnp.deg2rad(jnp.array([
    [-45.0,  45.0],    # coxa   (yaw)
    [-80.0,  75.0],    # femur  (lift)
    [  0.0, 155.0],    # tibia  (knee)
]))


# Euler / RPY convention.
#   Intrinsic = XYZ (uppercase, moving axes) ; Extrinsic = xyz (lowercase, fixed).
# jaxlie's SO3.from_rpy_radians(r, p, y) == Rz(y) @ Ry(p) @ Rx(r) (ZYX), and is
# exactly scipy's from_euler("xyz", [r, p, y]) / from_euler("ZYX", [y, p, r]).
def from_te(t, e) -> SE3:
    """Create an SE3 from a translation ``t`` and Euler angles ``e`` (rpy=xyz)."""
    return SE3.from_rotation_and_translation(SO3.from_rpy_radians(*e), t)


def foot_transform(shoulder: SE3, theta: jax.Array, lengths: jax.Array) -> SE3:
    """Forward kinematics for one leg: world foot frame from joint angles."""
    coxa = from_te(jnp.array([0.0, 0.0, 0.0]), jnp.array([0., 0., theta[0]]))
    femur = from_te(jnp.array([lengths[0], 0.0, 0.0]), jnp.array([0., theta[1], 0.]))
    tibia = from_te(jnp.array([lengths[1], 0.0, 0.0]), jnp.array([0., theta[2], 0.]))
    foot = from_te(jnp.array([lengths[2], 0.0, 0.0]), jnp.zeros(3))
    return shoulder @ coxa @ femur @ tibia @ foot


# Planted feet at the home stance: base at height 0.241185, identity orientation,
# all legs at the home joint angles. Shape (6, 3), indexed by leg like SHOULDERS.
_HOME_BASE = SE3.from_translation(jnp.array([0.0, 0.0, 0.241185]))
_HOME_THETA = jnp.array([0.0, 0.0523599, 1.46608])
PLANTED = jax.vmap(foot_transform, in_axes=(0, None, None))(
    _HOME_BASE @ SHOULDERS, _HOME_THETA, LENGTHS
).translation()


def infer_theta(shoulder: SE3, foot: jax.Array, lengths: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Inverse of ``foot_transform``: joint angles that place the foot at ``foot``.

    foot     : (3,)  target foot position, world frame.
    shoulder : SE3   world pose of the shoulder frame.
    lengths  : (3,)  segment lengths [Lc, Lf, Lt].

    Returns ``(reachable, theta)`` where ``theta`` is ``(2, 3)`` -- the two knee
    branches [coxa, femur, tibia], elbow-"down" (+arccos) first. Joint limits are
    NOT applied here; ``reachable`` is only whether the foot lies in the reach
    annulus (see ``infer_posture`` for the limit-aware branch selection).
    """
    Lc, Lf, Lt = lengths
    p = shoulder.inverse().apply(foot)          # foot in the shoulder frame
    theta1 = jnp.arctan2(p[1], p[0])            # coxa yaw to the foot azimuth
    u = jnp.hypot(p[0], p[1]) - Lc              # radial reach past the coxa
    xi = -p[2]                                  # +rot about y tilts +x -> -z
    c3 = (u * u + xi * xi - Lf * Lf - Lt * Lt) / (2 * Lf * Lt)
    reachable = (c3 >= -1.0) & (c3 <= 1.0)
    theta3 = jnp.array([1.0, -1.0]) * jnp.arccos(jnp.clip(c3, -1.0, 1.0))   # (2,) down, up
    theta2 = jnp.arctan2(xi, u) - jnp.arctan2(Lt * jnp.sin(theta3), Lf + Lt * jnp.cos(theta3))
    theta = jnp.stack([jnp.broadcast_to(theta1, theta3.shape), theta2, theta3], axis=-1)  # (2,3)
    return reachable, theta





def tipover_score(com: jax.Array, feet_xy: jax.Array, max_angle: float = jnp.pi / 2) -> jax.Array:
    """Tip-over stability score in [0, 1] for a support polygon and a CoM.

    feet_xy : (N, 2)  foot xy on the flat support plane (z = 0). Requiring xy
                      makes the coplanar / flat-ground assumption explicit.
    com     : (3,)    center of mass -- xy for the margin, z for height.

    Score is the tip-over angle -- how far you could tilt before the CoM crosses
    the nearest support edge -- normalized by ``max_angle``:

        margin = signed horizontal distance from CoM to the nearest edge (>0 inside)
        theta  = atan2(margin, com_z)      # low CoM -> larger theta
        score  = clip(theta / max_angle, 0, 1)

    ``0`` = at/beyond an edge (unstable); higher = more tip-resistant. Assumes
    feet in convex position. Flat-ground only -- to be replaced by an MJX-based
    check for non-planar stances.
    """
    c = feet_xy.mean(axis=0)                     # (2,) foot centroid
    poly = feet_xy[jnp.argsort(jnp.arctan2(feet_xy[:, 1] - c[1], feet_xy[:, 0] - c[0]))]  # CCW
    a, b = poly, jnp.roll(poly, -1, axis=0)
    e = b - a
    d = (e[:, 0] * (com[1] - a[:, 1]) - e[:, 1] * (com[0] - a[:, 0])) / jnp.linalg.norm(e, axis=1)
    margin = jnp.min(d)                          # nearest-edge horizontal margin
    return jnp.clip(jnp.arctan2(margin, com[2]) / max_angle, 0.0, 1.0)
