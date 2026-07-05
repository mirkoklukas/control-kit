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

import dataclasses
from functools import partial

import jax
import jax.numpy as jnp
from jaxlie import SE3, SO3


# Let a batched SE3 be indexed/sliced like an array (jaxlie doesn't support it).
def _se3_getitem(self, index) -> SE3:
    if not isinstance(index, tuple):
        index = (index,)
    return SE3(self.wxyz_xyz[index + (slice(None),)])  # keep the trailing 7 axis
SE3.__getitem__ = _se3_getitem

def _se3_shape(self) -> tuple:
    return self.wxyz_xyz.shape[:-1]
SE3.shape = property(_se3_shape)

def _se3_reshape(self, shape) -> SE3:
    return SE3(self.wxyz_xyz.reshape(tuple(shape) + (7,)))
SE3.reshape = _se3_reshape

def _se3_broadcast_to(self, shape) -> SE3:
    return SE3(jnp.broadcast_to(self.wxyz_xyz, tuple(shape) + (7,)))
SE3.broadcast_to = _se3_broadcast_to


# Euler / RPY convention.
#   Intrinsic = XYZ (uppercase, moving axes) ; Extrinsic = xyz (lowercase, fixed).
# jaxlie's SO3.from_rpy_radians(r, p, y) == Rz(y) @ Ry(p) @ Rx(r) (ZYX), and is
# exactly scipy's from_euler("xyz", [r, p, y]) / from_euler("ZYX", [y, p, r]).
def from_te(t, e) -> SE3:
    """Create an SE3 from a translation ``t`` and Euler angles ``e`` (rpy=xyz)."""
    return SE3.from_rotation_and_translation(SO3.from_rpy_radians(*e), t)



# Segment lengths [coxa, femur, tibia] -- local +x offsets down the chain.
LENGTHS = jnp.array([0.025, 0.2, 0.206])

ALL_PLANTED = jnp.arange(6)  # all legs, for indexing convenience

# Joint limits [coxa, femur, tibia], radians, from weld0.xml (<default> classes).
JOINT_RANGES = jnp.deg2rad(jnp.array([
    [-65.0,  65.0],    # coxa   (yaw)
    [-80.0,  75.0],    # femur  (lift - pitch
    [  0.0, 155.0],    # tibia  (knee - pitch
]))

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


FEET_LIFTED = SHOULDERS.apply(jnp.array([LENGTHS[0]+0.2, 0.0, LENGTHS[1]-LENGTHS[2]]))






def complement(subset, N=6):
    """Legs in ``range(N)`` not in ``subset``: ``(..., K) -> (..., N-K)``.

    The "free" legs of a stance (not currently planted). Vectorized and
    jit-friendly. E.g. ``complement([0, 2, 4]) -> [1, 3, 5]``.
    """
    subset = jnp.asarray(subset)
    K = subset.shape[-1]
    in_subset = jax.nn.one_hot(subset, N).sum(axis=-2) > 0          # (..., N)
    order = jnp.argsort(in_subset, axis=-1, stable=True)            # not-in-subset first
    return jnp.asarray(order[..., : N - K])


def _infer_feet(shoulder: SE3, theta: jax.Array, lengths: jax.Array) -> SE3:
    """Forward kinematics for one leg: world foot frame from joint angles."""
    coxa = from_te(jnp.array([0.0, 0.0, 0.0]), jnp.array([0., 0., theta[0]]))
    femur = from_te(jnp.array([lengths[0], 0.0, 0.0]), jnp.array([0., theta[1], 0.]))
    tibia = from_te(jnp.array([lengths[1], 0.0, 0.0]), jnp.array([0., theta[2], 0.]))
    foot = from_te(jnp.array([lengths[2], 0.0, 0.0]), jnp.zeros(3))
    return shoulder @ coxa @ femur @ tibia @ foot


def infer_feet(body: SE3, thetas: jax.Array) -> jax.Array:
    shoulders = body @ SHOULDERS
    thetas = jnp.broadcast_to(thetas, (6, 3))   # (6, 3)
    return jax.vmap(_infer_feet, in_axes=(0, 0, None))(shoulders, thetas, LENGTHS).translation()

def _infer_joint_xpos(shoulder, theta):
    coxa = from_te(jnp.array([0.0, 0.0, 0.0]), jnp.array([0., 0., theta[0]]))
    femur = from_te(jnp.array([LENGTHS[0], 0.0, 0.0]), jnp.array([0., theta[1], 0.]))
    tibia = from_te(jnp.array([LENGTHS[1], 0.0, 0.0]), jnp.array([0., theta[2], 0.]))
    foot = from_te(jnp.array([LENGTHS[2], 0.0, 0.0]), jnp.zeros(3))
    coxa_pos = (shoulder @ coxa).translation()
    femur_pos = (shoulder @ coxa @ femur).translation()
    tibia_pos = (shoulder @ coxa @ femur @ tibia).translation() 
    foot_pos = (shoulder @ coxa @ femur @ tibia @ foot).translation()
    return jnp.stack([coxa_pos, femur_pos, tibia_pos, foot_pos], axis=0)  # (4, 3)

def infer_joint_xpos(body: SE3, thetas: jax.Array) -> jax.Array:
    """Return the world positions of the joints (coxa, femur, tibia) for each leg."""
    shoulders = body @ SHOULDERS
    thetas = jnp.broadcast_to(thetas, (6, 3))   # (6, 3)

    return jax.vmap(_infer_joint_xpos)(shoulders, thetas)  # (6, 4, 3)

# Planted feet at the home stance: base at height 0.241185, identity orientation,
# all legs at the home joint angles. Shape (6, 3), indexed by leg like SHOULDERS.
_HOME_BASE = SE3.from_translation(jnp.array([0.0, 0.0, 0.241185]))
_HOME_THETA = jnp.array([0.0, 0.0523599, 1.46608])
PLANTED = jax.vmap(_infer_feet, in_axes=(0, None, None))(
    _HOME_BASE @ SHOULDERS, _HOME_THETA, LENGTHS
).translation()


def _infer_theta(shoulder: SE3, foot: jax.Array, lengths: jax.Array) -> tuple[jax.Array, jax.Array]:
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
    TOL = 1e-3
    reachable = (c3 >= -1.0 - TOL) & (c3 <= 1.0 + TOL)
    theta3 = jnp.array([1.0, -1.0]) * jnp.arccos(jnp.clip(c3, -1.0, 1.0))   # (2,) down, up
    theta2 = jnp.arctan2(xi, u) - jnp.arctan2(Lt * jnp.sin(theta3), Lf + Lt * jnp.cos(theta3))
    theta = jnp.stack([jnp.broadcast_to(theta1, theta3.shape), theta2, theta3], axis=-1)  # (2,3)
    return reachable, theta


def infer_theta(body: SE3, feet: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Vectorized version of ``_infer_theta`` for all legs.

    shoulders : (6,) SE3   world poses of the shoulder frames.
    feet      : (6, 3)     world foot positions.
    lengths   : (3,)       segment lengths [Lc, Lf, Lt].

    Returns:
        reachable : (6,) bool     whether each foot is reachable (in the annulus)
        theta     : (6, 2, 3)     joint angles for each leg

    ``(reachable, theta)`` where ``theta`` is ``(6, 2, 3)`` -- the two
    knee branches [coxa, femur, tibia], elbow-"down" (+arccos) first. Joint
    limits are NOT applied here; ``reachable`` is only whether the foot lies in
    the reach annulus (see ``infer_posture`` for the limit-aware branch selection).
    """
    shoulders = body @ SHOULDERS
    return jax.vmap(_infer_theta, (0, 0, None))(shoulders, feet, LENGTHS)

def _infer_posture(body, feet, stance) -> jax.Array:
    """Joint angles that plant ``stance``'s feet under body pose ``body``.

    Args:
        body   : SE3       world body (base) pose.
        feet   : (6, 3)    world foot positions of all legs.
        stance : (F,) int   which legs are planted (indices into 0..5)
    
    Returns:
        valid : bool      whether all legs are reachable and in-limits
        theta : (6, 3)    chosen angles for legs; unplanted legs are set to the home angles.
        feet: (6, 3)      world foot positions of the legs; unplanted legs are set to the home angles.
    """
    lifted = complement(stance)
    feet = feet.at[lifted].set((body@FEET_LIFTED)[lifted])                         
    shoulders = body @ SHOULDERS
    reachable, theta = jax.vmap(_infer_theta, (0, 0, None))(  
        shoulders, feet, LENGTHS)
    # Choose the elbow-down branch for planted legs, and the home angles for lifted legs.
    theta = theta[:,0]

    in_limits = (JOINT_RANGES[:,0] <= theta) & (theta <= JOINT_RANGES[:,1])  

    valid = jnp.all(reachable[stance])  & jnp.all(in_limits[stance])
    return valid, theta, feet

def infer_posture(bodies, feet, stance) -> jax.Array:
    """Joint angles that plant ``stance``'s feet under body pose ``body``.

    Args:
        bodies : (B,) SE3       World body (base) poses.
        feet   : (S, 6, 3)      Stance foot positions, world foot positions of all legs.
        stance : (F,) int       Stance ids, i.e. which legs are planted (indices into 0..

    Returns:
        valid : (B, S) bool     Whether all legs are reachable and in-limits
        theta : (B, S, 6, 3)    Chosen angles for
    """
    return jax.vmap(
                jax.vmap(_infer_posture, (None, 0, None)), 
            (0, None, None))(bodies, feet, stance)



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



# grid means a BxS family, where B num bodies and S is the number of stances
def stability_scorer(bodies, thetas_grid, feet_grid, stance):
    """Tip-over score of ``posture``'s body over ``stance``'s support polygon.

    The support feet are given explicitly by ``stance`` (rather than inferred
    from the posture) since which feet bear load is a choice, not a geometric
    fact. CoM proxy is the base origin. Flat-ground heuristic; scalar in [0, 1].
    """
    com = bodies.translation()
    return jax.vmap(jax.vmap(tipover_score,(None, 0)), (0, 0))(
        com, feet_grid[..., stance, :2])

#     """Tip-over score of ``posture``'s body over ``stance``'s support polygon.

#     The support feet are given explicitly by ``stance`` (rather than inferred
#     from the posture) since which feet bear load is a choice, not a geometric
#     fact. CoM proxy is the base origin. Flat-ground heuristic; scalar in [0, 1].
#     """
#     return tipover_score(posture.body.translation(), stance.foot_positions[:, :2])


# def transition_stability(posture: Posture, old_stance: Stance, new_stance: Stance, tau=0.05):
#     """Stability of a foot-swap: the body must stand on BOTH supports.

#     A transition lifts the free feet and plants the new ones; the body has to be
#     statically stable on the old footprint (before) and the new one (after).
#     Returns ``(stb_old, stb_new, ok)`` -- both raw scores plus ``ok = both > tau``
#     (raw scores kept so callers see margins, not just the boolean).
#     """
#     stb_old = stability_scorer(posture, old_stance)
#     stb_new = stability_scorer(posture, new_stance)
#     return stb_old, stb_new, (stb_old > tau) & (stb_new > tau)
