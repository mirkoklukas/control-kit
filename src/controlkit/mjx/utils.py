"""Small JAX/MJX helpers shared across experiments."""
from __future__ import annotations

import jax.numpy as jnp
import mujoco


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
