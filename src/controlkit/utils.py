"""Small JAX/MJX helpers shared across experiments."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx


def keyframe(mj_model: mujoco.MjModel, name: str = "home"):
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
def _contact_normal_forces(data: mjx.Data) -> tuple[jax.Array, jax.Array, jax.Array]:
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
    rows = jnp.arange(4)  # 2 * (condim - 1), condim == 3
    idx = jnp.clip(addr[:, None] + rows[None, :], 0, efc.shape[0] - 1)
    normal = jnp.where(valid, jnp.sum(efc[idx], axis=1), 0.0)
    return normal, c.geom[:, 0], c.geom[:, 1]


def foot_contacts(
    data: mjx.Data, foot_geoms: jax.Array, force_thresh: float = 1.0
) -> jax.Array:
    """``(n_feet,)`` bool: is each foot geom in active contact with normal force
    above ``force_thresh``?

    The threshold is essential -- without it, soft-contact grazings register as
    touchdowns and the air-time timer never accumulates a real swing. Vectorized
    over the fixed contact axis (no Python loop over contacts).
    """
    normal, g1, g2 = _contact_normal_forces(data)
    strong = normal > force_thresh  # (ncon,); inactive contacts have normal 0
    involves = (g1[None, :] == foot_geoms[:, None]) | (
        g2[None, :] == foot_geoms[:, None]
    )  # (n_feet, ncon)
    return jnp.any(involves & strong[None, :], axis=1)  # (n_feet,)


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
