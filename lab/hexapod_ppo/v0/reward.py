"""Per-step reward for hexapod_ppo v0 -- this experiment's concrete reward.

Command-conditioned velocity tracking (base-frame ``vx, vy, yaw_rate``) plus
regularization terms that shape a clean, tripod-like gait. The general pieces it
builds on are stable-core: base-frame kinematics + contact extractors from
``controlkit.utils``, the ``feet_air_time`` gait term from
``controlkit.rewards``, and the ``Command`` / ``HexapodIds`` / ``FootState``
descriptors from ``controlkit.hexapod``. This module owns the *composition*: the
term set (``RewardWeights``) and how they sum (``compute_reward``).

Called ONCE per env step, AFTER ``mjx.step`` (see ``env.HexapodEnv.step``). The
trajectory return is the plain sum of step rewards: no summation/discounting/
normalization happens here. ``compute_reward`` is a pure JAX function (jit-able,
vmap-able over a batch of envs); all cross-step state (``foot_state``,
``last_action``) is threaded through arguments and returned, never mutated.

See ``docs/hexapod-rewards.md`` for the full per-term inventory.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
from mujoco import mjx

from controlkit.hexapod import Command, FootState, HexapodIds
from controlkit.utils import (
    bad_body_contacts,
    base_ang_vel,
    base_lin_vel,
    foot_contacts,
    projected_gravity,
)
from controlkit.rewards import feet_air_time

# Gaussian kernel width^2 for the velocity-tracking rewards.
_TRACK_SIGMA2 = 0.25
# Gaussian kernel width^2 (meters^2) for the base-height-keeping reward.
_HEIGHT_SIGMA2 = 0.0025


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RewardWeights:
    """One weight per term. The weight carries the sign (penalties negative)."""

    lin_vel: float = 1.0
    ang_vel: float = 0.5
    base_height: float = 1.0
    lin_vel_z: float = -2.0
    ang_vel_xy: float = -0.05
    orientation: float = -0.2
    torques: float = -1e-4
    dof_acc: float = -2.5e-7
    action_rate: float = -0.01
    feet_air_time: float = 1.0
    collision: float = -1.0

    def replace(self, **kw) -> "RewardWeights":
        return dataclasses.replace(self, **kw)


def compute_reward(
    data: mjx.Data,
    *,
    cmd: Command = Command.stand(),
    ids: HexapodIds,
    w: RewardWeights = RewardWeights(),
    world_up: jax.Array = (0.0, 0.0, 1.0),
    foot_state: FootState,
    last_action: jax.Array,
    action: jax.Array,
    air_time_target: float = 0.4,
    contact_force_thresh: float = 1.0,
    move_cmd_eps: float = 0.05,
    height_target: float | None = None,
    dt: float,
) -> tuple[jax.Array, dict[str, jax.Array], FootState]:
    """Full per-step reward (all 11 terms). Returns ``(total, terms, new_foot_state)``.

    ``total`` is a scalar, ``terms`` maps each term name to its *unweighted* scalar
    value (for logging), and ``new_foot_state`` carries the updated foot timer.

    ``ids`` is a :class:`HexapodIds` closed over at jit time (it holds the host
    ``MjModel`` and is not a traceable arg); the concrete index arrays each term
    needs are built explicitly from it in the body. No name lookups happen here --
    those ran once in ``HexapodIds.__init__``. ``world_up`` is the world up-axis
    (default +z unit; only its xy projection in the base frame matters for the
    orientation penalty). ``feet_air_time`` is gated by whether any motion is
    commanded (linear or yaw, deadband ``move_cmd_eps``), so the robot doesn't step
    in place while told to stand.
    """
    base = ids.base
    v_xy_cmd = cmd.v[:2]
    yaw_rate_cmd = cmd.omega[2]

    # --- task tracking (base frame, heading-invariant) ---
    v = base_lin_vel(data, base)
    omega = base_ang_vel(data, base)
    up_b = projected_gravity(data, base, jnp.asarray(world_up))
    lin_vel = jnp.exp(-jnp.sum((v_xy_cmd - v[:2]) ** 2) / _TRACK_SIGMA2)
    ang_vel = jnp.exp(-((yaw_rate_cmd - omega[2]) ** 2) / _TRACK_SIGMA2)

    # --- base height keeping: dense "alive" bonus, 1.0 at the target trunk height,
    # falling off to 0 as it sinks. Disabled (0) when no height_target is given. ---
    if height_target is None:
        base_height = jnp.zeros(())
    else:
        val = jnp.clip(data.qpos[2] - height_target, min=-jnp.inf, max=0.0)
        base_height = jnp.exp(-(val ** 2) / _HEIGHT_SIGMA2)

    # --- base-stability penalties ---
    lin_vel_z = v[2] ** 2
    ang_vel_xy = omega[0] ** 2 + omega[1] ** 2
    orientation = jnp.sum(up_b[:2] ** 2)

    # --- action smoothness ---
    action_rate = jnp.sum((action - last_action) ** 2)

    # --- effort / regularization over the leg dofs ---
    leg_dofs = ids.leg_dofs()
    torques = jnp.sum(data.qfrc_actuator[leg_dofs] ** 2)
    dof_acc = jnp.sum(data.qacc[leg_dofs] ** 2)

    # --- contacts: collision + gait shaping ---
    collision = bad_body_contacts(
        data, jnp.asarray(ids.bad_geoms), contact_force_thresh
    ).astype(jnp.float32)
    contact = foot_contacts(data, jnp.asarray(ids.feet()), contact_force_thresh)
    moving = (jnp.linalg.norm(cmd.v[:2]) > move_cmd_eps) | (
        jnp.abs(cmd.omega[2]) > move_cmd_eps
    )
    feet_air, new_foot_state = feet_air_time(
        foot_state, contact, dt, air_time_target, gate=moving.astype(jnp.float32)
    )

    terms = {
        "lin_vel": lin_vel,
        "ang_vel": ang_vel,
        "base_height": base_height,
        "lin_vel_z": lin_vel_z,
        "ang_vel_xy": ang_vel_xy,
        "orientation": orientation,
        "action_rate": action_rate,
        "torques": torques,
        "dof_acc": dof_acc,
        "collision": collision,
        "feet_air_time": feet_air,
    }
    total = (
        w.lin_vel * lin_vel
        + w.ang_vel * ang_vel
        + w.base_height * base_height
        + w.lin_vel_z * lin_vel_z
        + w.ang_vel_xy * ang_vel_xy
        + w.orientation * orientation
        + w.action_rate * action_rate
        + w.torques * torques
        + w.dof_acc * dof_acc
        + w.collision * collision
        + w.feet_air_time * feet_air
    )
    return total, terms, new_foot_state
