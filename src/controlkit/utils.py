"""Small JAX/MJX helpers shared across experiments."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx


def keyframe(mj_model: mujoco.MjModel, name: str = "home") -> tuple[jax.Array, jax.Array]:
    """Return ``(qpos, qvel)`` of a named ``<keyframe>`` as JAX arrays.

    MJX has no in-place ``mj_resetDataKeyframe`` -- reset is functional. Read the
    keyframe off the host ``mj_model`` once (it's static), then build the initial
    state with ``mjx.make_data(model).replace(qpos=qpos, qvel=qvel)`` followed by
    ``mjx.forward``. ``key_qvel`` is zeros unless the keyframe sets velocities.

    Raises ``KeyError`` (via ``mj_model.key``) if no keyframe with ``name`` exists.
    """
    kid = mj_model.key(name).id
    return jnp.asarray(mj_model.key_qpos[kid]), jnp.asarray(mj_model.key_qvel[kid])


# --------------------------------------------------------------------------- #
# Base-frame kinematics (pure JAX)                                            #
# --------------------------------------------------------------------------- #


def _base_rot(data: mjx.Data, base_body: int) -> jax.Array:
    """3x3 rotation matrix, world <- base, for the base body."""
    return data.xmat[base_body].reshape(3, 3)

def to_base_frame(data: mjx.Data, base_body: int, vec: jax.Array) -> jax.Array:
    """Rotate a vector from world frame to base frame."""
    rot = _base_rot(data, base_body)
    return rot.T @ vec

def base_lin_vel(data: mjx.Data, base_body: int) -> jax.Array:
    """Base linear velocity in the BASE frame (heading-invariant). Shape (3,).

    ``cvel`` is the spatial velocity in world axes, laid out [angular; linear];
    we take the linear part and rotate it into the base frame.
    """
    rot = _base_rot(data, base_body)
    v_world = data.cvel[base_body, 3:6]
    return rot.T @ v_world


def base_ang_vel(data: mjx.Data, base_body: int) -> jax.Array:
    """Base angular velocity in the BASE frame. Shape (3,) = [roll, pitch, yaw]."""
    rot = _base_rot(data, base_body)
    w_world = data.cvel[base_body, 0:3]
    return rot.T @ w_world


def projected_gravity(data: mjx.Data, base_body: int, world_up: jax.Array) -> jax.Array:
    """World up-axis expressed in the base frame; xy is the tilt signal. Shape (3,).

    Kept the conventional name ``projected_gravity``; the input is the world up
    direction (gravity's sign/scale are irrelevant to the ``xy^2`` tilt penalty).
    """
    rot = _base_rot(data, base_body)
    return rot.T @ world_up


# --------------------------------------------------------------------------- #
# Contact forces (pure JAX)                                                    #
# --------------------------------------------------------------------------- #
def contact_normal_forces(data: mjx.Data, condim=3) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Per-contact normal force and geom pair, vmap-safe (no loop over contacts).

    Pyramidal cone with condim 3 (this model): the normal force is the sum of the
    4 pyramid rows in ``efc_force`` starting at the contact's ``efc_address``.
    Inactive / zero-padded contacts (``efc_address < 0``) contribute zero. Returns
    ``(normal, geom1, geom2)``, each shape ``(ncon,)``.
    """
    c = data._impl.contact
    efc = data._impl.efc_force
    addr = c.efc_address  # (ncon,), -1 for inactive
    valid = addr >= 0
    rows = jnp.arange(2 * (condim - 1))  # 2 * (condim - 1)
    idx = jnp.clip(addr[:, None] + rows[None, :], 0, efc.shape[0] - 1)
    normal = jnp.where(valid, jnp.sum(efc[idx], axis=1), 0.0)
    return normal, c.geom[:, 0], c.geom[:, 1]


def get_contacts(
    data: mjx.Data, geoms: jax.Array, force_thresh: float = 0.0
) -> jax.Array:
    """``(n_geoms,)`` bool: is each geom in active contact with normal force
    above ``force_thresh``?

    The threshold is essential -- without it, soft-contact grazings register as
    touchdowns and the air-time timer never accumulates a real swing. Vectorized
    over the fixed contact axis (no Python loop over contacts).
    """
    normal, g1, g2 = contact_normal_forces(data)
    strong = normal > force_thresh  # (ncon,); inactive contacts have normal 0
    involves = (g1[None, :] == geoms[:, None]) | (
        g2[None, :] == geoms[:, None]
    )  # (n_geoms, ncon)
    return jnp.any(involves & strong[None, :], axis=1)  # (n_geoms,)


def foot_contacts(
    data: mjx.Data, foot_geoms: jax.Array, force_thresh: float = 0.0
) -> jax.Array:
    """``(n_feet,)`` bool: is each foot geom in active contact with normal force
    above ``force_thresh``?

    The threshold is essential -- without it, soft-contact grazings register as
    touchdowns and the air-time timer never accumulates a real swing. Vectorized
    over the fixed contact axis (no Python loop over contacts).
    """
    normal, g1, g2 = contact_normal_forces(data)
    strong = normal > force_thresh  # (ncon,); inactive contacts have normal 0
    involves = (g1[None, :] == foot_geoms[:, None]) | (
        g2[None, :] == foot_geoms[:, None]
    )  # (n_feet, ncon)
    return jnp.any(involves & strong[None, :], axis=1)  # (n_feet,)


def foot_normal_forces(
    data: mjx.Data, foot_geoms: jax.Array
) -> jax.Array:
    """``(n_feet,)`` normal contact force [N] each foot carries (0 when airborne).

    The continuous, force-valued counterpart of :func:`foot_contacts`: instead of
    "is this foot down?" it answers "how much vertical load is this foot bearing?"
    -- i.e. the ground-reaction normal force per leg, the load-distribution signal.

    Per foot, this sums the normal force over *every* contact that geom touches, so
    a foot resting on two surfaces (or two contact points) is counted once, fully.
    Inactive / separated contacts carry zero normal force (see
    :func:`_contact_normal_forces`), so airborne feet read exactly 0 and no force
    threshold is needed here -- threshold downstream if you want a "planted" mask.

    Sign / units: the normal force is along the contact normal and is always >= 0
    (contacts push apart, never pull), in newtons. Sanity check: in a static stance
    the six values sum to roughly the robot's weight ``m * g``; their fractions are
    each leg's share of support.

    Vectorized over the fixed contact axis (no Python loop over contacts), so it is
    jit-/vmap-safe inside an env step or a scanned rollout.

    Args:
        data: MJX data *after* ``mjx.step`` / ``mjx.forward`` (needs the solved
            contact forces in ``efc_force``).
        foot_geoms: ``(n_feet,)`` geom ids of the feet (e.g. ``ids.feet()``).

    Returns:
        ``(n_feet,)`` array of per-foot normal force in newtons, aligned with
        ``foot_geoms``.
    """
    normal, g1, g2 = contact_normal_forces(data)  # (ncon,) each; normal >= 0
    # (n_feet, ncon): does contact c involve foot f (as either geom of the pair)?
    involves = (g1[None, :] == foot_geoms[:, None]) | (
        g2[None, :] == foot_geoms[:, None]
    )
    # Sum each foot's contact normals; non-involved contacts contribute 0.
    return jnp.sum(jnp.where(involves, normal[None, :], 0.0), axis=1)  # (n_feet,)


def foot_velocities(
    data: mjx.Data, foot_geoms: jax.Array, foot_bodies: jax.Array, root_body: int
) -> jax.Array:
    """``(n_feet, 3)`` world-frame linear velocity of each foot geom.

    Computed from the bodies' spatial velocities (``cvel``) by rigid-body transport
    to the foot point. The subtlety: ``cvel`` is ``[angular; linear]`` in *world*
    axes, but its linear part is the velocity of the point at the kinematic tree's
    **root subtree CoM** (``subtree_com[root_body]``) -- a single reference shared by
    every body, NOT each body's own com. So the foot-point velocity is::

        v_foot = cvel_lin + omega x (p_foot - subtree_com[root])

    (verified against ``mj_objectVelocity`` to ~1e-16). Vectorized over feet, so it
    is jit-/vmap-safe inside an env step or a scanned rollout.

    Args:
        data: MJX data after ``mjx.step`` / ``mjx.forward``.
        foot_geoms: ``(n_feet,)`` foot geom ids (e.g. ``ids.feet()``).
        foot_bodies: ``(n_feet,)`` body id each foot geom belongs to
            (e.g. ``ids.feet_bodies()``).
        root_body: the floating-base root body id (e.g. ``ids.base``); its
            ``subtree_com`` is the reference point for ``cvel``.

    Returns:
        ``(n_feet, 3)`` world-frame linear velocities, aligned with ``foot_geoms``.
        For slip detection, the in-plane (xy) magnitude on *loaded* feet should be
        ~0 -- a planted foot that is sliding reads nonzero here.
    """
    p = data.geom_xpos[foot_geoms]          # (n_feet, 3) foot positions in world
    w = data.cvel[foot_bodies, 0:3]         # (n_feet, 3) body angular vel (world)
    v = data.cvel[foot_bodies, 3:6]         # (n_feet, 3) body linear vel at root com
    return v + jnp.cross(w, p - data.subtree_com[root_body])


def bad_body_contacts(
    data: mjx.Data, bad_geoms: jax.Array, force_thresh: float = 1.0
) -> jax.Array:
    """Count of contacts involving any bad geom (trunk/head/legs) with normal force
    above ``force_thresh``.

    The force threshold -- not ``efc_address >= 0`` -- is the activity check: in
    this MJX version the contact buffer assigns an efc address to *every* buffered
    geom pair, so separated pairs would otherwise be counted. Real collisions
    carry force.
    """
    normal, g1, g2 = _contact_normal_forces(data)
    strong = normal > force_thresh  # (ncon,)
    in_bad = jnp.any(g1[None, :] == bad_geoms[:, None], axis=0) | jnp.any(
        g2[None, :] == bad_geoms[:, None], axis=0
    )  # (ncon,)
    return jnp.sum(strong & in_bad)
