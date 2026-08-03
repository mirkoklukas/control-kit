"""General reward-term kernels for MJX locomotion (pure JAX).

Reusable, experiment-agnostic building blocks. Each term is a pure JAX function
(jit-able, vmap-able over a batch of envs); cross-step state is threaded through
arguments and returned, never mutated. Experiments compose these into a concrete
per-step reward (see e.g. ``lab/hexapod_ppo/v0/reward.py``).

The base-frame kinematics and contact extractors these build on live in
``controlkit.utils``; the command / foot-state pytrees in
``controlkit.hexapod``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from controlkit.hexapod import FootState


def feet_air_time(
    foot_state: FootState,
    contact: jax.Array,
    dt: float,
    air_time_target: float = 0.4,
    gate: jax.Array = 1.0,
) -> tuple[jax.Array, FootState]:
    """Gait-shaping term, paid ONLY on the touchdown step.

    Rewards each foot for swinging ~``air_time_target`` before it lands::

        reward = gate * Σ_feet (air_at_landing - target) * first_contact

    where ``first_contact`` is the touchdown rising edge (in contact now, not last
    step). ``gate`` (0/1) suppresses the term when no motion is commanded, so the
    robot doesn't step in place while told to stand; it scales the *reward* only,
    the timer keeps running. Returns ``(reward, new_foot_state)``.
    """
    new_air = foot_state.air_time + dt
    first_contact = contact & ~foot_state.last_contact  # touchdown (rising edge)
    air_at_landing = new_air  # swing duration at landing
    reset_air = jnp.where(contact, 0.0, new_air)  # zero the timer for grounded feet
    new_state = foot_state.replace(air_time=reset_air, last_contact=contact)

    reward = gate * jnp.sum((air_at_landing - air_time_target) * first_contact)
    return reward, new_state
