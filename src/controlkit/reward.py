"""Per-step reward for hexapod locomotion in MuJoCo MJX (JAX).

Command-conditioned velocity tracking (base-frame ``vx, vy, yaw_rate``) plus
regularization terms that shape a clean, tripod-like gait, intended for Brax PPO
on MJX.

Called ONCE per env step, AFTER ``mjx.step``. The trajectory return is the plain
sum of step rewards: no summation/discounting/normalization happens here.
``compute_reward`` is a pure JAX function (jit-able, vmap-able over a batch of
envs); all cross-step state (``foot_state``, ``last_action``) is threaded through
arguments and returned, never mutated.

Built term by term -- see ``docs/hexapod-rewards.md`` for the full inventory and
per-term status. Implemented (slices 1-2): ``lin_vel``, ``ang_vel``, ``lin_vel_z``,
``ang_vel_xy``, ``orientation``, ``action_rate``, ``torques``, ``dof_acc``.
"""

from __future__ import annotations

import dataclasses
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import mujoco
from mujoco import mjx

# Gaussian kernel width^2 for the velocity-tracking rewards.
_TRACK_SIGMA2 = 0.25
# Gaussian kernel width^2 (meters^2) for the base-height-keeping reward.
_HEIGHT_SIGMA2 = 0.0025


# --------------------------------------------------------------------------- #
# Weights & cross-step state (registered JAX pytrees, no flax dependency)      #
# --------------------------------------------------------------------------- #
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



@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Command:
    """Commanded base-frame twist: target linear velocity ``v`` (3,) and angular
    velocity ``omega`` (3,).

    Velocity tracking uses ``v[:2]`` (vx, vy) and ``omega[2]`` (yaw rate); the
    other components are nominally zero and shaped by the ``lin_vel_z`` /
    ``ang_vel_xy`` penalties. Use the factory constructors for the common cases.
    """

    v: jax.Array
    omega: jax.Array

    @classmethod
    def stand(cls) -> "Command":
        """Hold still (zero twist)."""
        return cls(v=jnp.zeros(3), omega=jnp.zeros(3))

    @classmethod
    def straight(cls, vx: float) -> "Command":
        """Walk forward at ``vx`` m/s."""
        return cls(v=jnp.array([vx, 0.0, 0.0]), omega=jnp.zeros(3))

    @classmethod
    def turn(cls, yaw_rate: float) -> "Command":
        """Turn in place at ``yaw_rate`` rad/s."""
        return cls(v=jnp.zeros(3), omega=jnp.array([0.0, 0.0, yaw_rate]))

    @classmethod
    def planar(cls, vx: float = 0.0, vy: float = 0.0, yaw_rate: float = 0.0) -> "Command":
        """General planar command: forward ``vx``, lateral ``vy``, yaw ``yaw_rate``."""
        return cls(v=jnp.array([vx, vy, 0.0]), omega=jnp.array([0.0, 0.0, yaw_rate]))

    def replace(self, **kw) -> "Command":
        return dataclasses.replace(self, **kw)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class FootState:
    """Cross-step foot bookkeeping, threaded through the env pytree."""

    air_time: jax.Array  # (n_feet,) float: seconds since last touchdown
    last_contact: jax.Array  # (n_feet,) bool: contact mask at the previous step

    @classmethod
    def init(cls, n_feet: int = 6) -> "FootState":
        return cls(
            air_time=jnp.zeros(n_feet),
            last_contact=jnp.zeros(n_feet, dtype=bool),
        )

    def replace(self, **kw) -> "FootState":
        return dataclasses.replace(self, **kw)


# --------------------------------------------------------------------------- #
# Env-init id descriptor (host side; name lookups OK here, NOT in compute_reward)
# --------------------------------------------------------------------------- #
# Leg labels in the order the MJCF <replicate count="6"> emits them, i.e. leg i
# carries names coxa{i}/femur{i}/tibia{i}/foot{i} (see models/hexapod.xml):
#   0=FL(30)  1=ML(90)  2=BL(150)  3=BR(210)  4=MR(270)  5=FR(330)
LEGS = ("FL", "ML", "BL", "BR", "MR", "FR")


class Leg(NamedTuple):
    """Ids for one leg: its three hinge joints and the foot geom."""

    coxa: int
    femur: int
    tibia: int
    foot: int


class HexapodIds:
    """Model-specific id descriptor with labeled access -- one per model.

    The leg labels (``FL..FR``) are baked in from the ``<replicate>`` order; the
    concrete ids are resolved from the model by name (``coxa{i}/femur{i}/tibia{i}/
    foot{i}``), so no raw id literals appear here and the descriptor stays correct
    if the MJCF shifts. Build it once at env init; call :meth:`reward_ids` for the
    concrete :class:`RewardIds` bundle threaded into ``compute_reward``.

    Examples
    --------
    >>> ids = HexapodIds(model)
    >>> ids.leg_joints("FL")   # (coxa, femur, tibia) ids for the front-left leg
    >>> ids.leg_joints()       # all 18 hinge ids, in leg order
    >>> ids.foot("BR")         # back-right foot geom id
    >>> ids.reward_ids()       # -> RewardIds for compute_reward
    """

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        # base = the body carrying the freejoint (name-independent).
        free = int(np.argmax(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE))
        self.base = int(model.jnt_bodyid[free])
        # per-leg ids, keyed by label, resolved by name from the model.
        self._legs: dict[str, Leg] = {
            label: Leg(
                coxa=self._joint(f"coxa{i}"),
                femur=self._joint(f"femur{i}"),
                tibia=self._joint(f"tibia{i}"),
                foot=self._geom(f"foot{i}"),
            )
            for i, label in enumerate(LEGS)
        }

    # --- internal name lookups (raise on a missing name) ---
    def _joint(self, name: str) -> int:
        i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if i < 0:
            raise KeyError(f"no joint named {name!r} in model")
        return i

    def _geom(self, name: str) -> int:
        i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if i < 0:
            raise KeyError(f"no geom named {name!r} in model")
        return i

    # --- labeled access ---
    def leg(self, label: str) -> Leg:
        """All ids for one leg (coxa, femur, tibia, foot)."""
        try:
            return self._legs[label]
        except KeyError:
            raise KeyError(f"unknown leg {label!r}; expected one of {LEGS}") from None

    def leg_joints(self, label: str | None = None) -> np.ndarray:
        """Hinge joint ids: one leg's ``(coxa, femur, tibia)`` if ``label`` is
        given, else all 18 in leg order."""
        if label is None:
            return np.array(
                [j for lbl in LEGS for j in self._legs[lbl][:3]], dtype=np.int32
            )
        lg = self.leg(label)
        return np.array([lg.coxa, lg.femur, lg.tibia], dtype=np.int32)

    def leg_dofs(self, label: str | None = None) -> np.ndarray:
        """DOF addresses of the leg hinge joints (qvel/qacc/qfrc_actuator indices):
        one leg's three if ``label`` is given, else all 18 in leg order."""
        return self.model.jnt_dofadr[self.leg_joints(label)].astype(np.int32)

    def foot(self, label: str) -> int:
        """Foot geom id for one leg."""
        return self.leg(label).foot

    def feet(self) -> np.ndarray:
        """All foot geom ids, in leg order."""
        return np.array([self._legs[lbl].foot for lbl in LEGS], dtype=np.int32)

    @property
    def bad_geoms(self) -> np.ndarray:
        """Every geom except the floor plane and the feet (must not touch floor)."""
        plane = int(mujoco.mjtGeom.mjGEOM_PLANE)
        feet = set(self.feet().tolist())
        return np.array(
            [
                g
                for g in range(self.model.ngeom)
                if g not in feet and int(self.model.geom_type[g]) != plane
            ],
            dtype=np.int32,
        )


# --------------------------------------------------------------------------- #
# Helpers (pure JAX)                                                           #
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


# --------------------------------------------------------------------------- #
# Reward                                                                       #
# --------------------------------------------------------------------------- #
# reward(action, data | data_prev, action_prev, command; world_up, hexapod_ids, weights)
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
    dt: float,
) -> tuple[jax.Array, dict[str, jax.Array], FootState]:
    """One step of reward. Returns ``(total, terms, new_foot_state)``.

    ``total`` is a scalar, ``terms`` maps each term name to its *unweighted*
    scalar value (for logging), and ``new_foot_state`` is the functionally
    updated foot state. See module docstring for the calling contract.

    ``ids`` is a :class:`HexapodIds` closed over at jit time (it holds the host
    ``MjModel`` and is not a traceable arg); the concrete index arrays each term
    needs are built explicitly from it in the body. No name lookups happen here --
    those ran once in ``HexapodIds.__init__``. ``world_up`` is the world up-axis
    (default +z unit; only its xy projection in the base frame matters for the
    orientation penalty).
    """
    base = ids.base
    v_xy_cmd = cmd.v[:2]
    yaw_rate_cmd = cmd.omega[2]

    # --- task tracking (base frame, heading-invariant) ---
    v = base_lin_vel(data, base)
    omega = base_ang_vel(data, base)
    up_b = projected_gravity(data, base, jnp.asarray(world_up))

    v_xy = v[:2] #dxy 
    yaw_rate = omega[2] #dyaw

    # GaussianKernel(dxy_cmd, dxy)
    # GaussianKernel(dyaw_cmd, dyaw)
    lin_vel = jnp.exp(-jnp.sum((v_xy_cmd - v_xy) ** 2) / _TRACK_SIGMA2)
    ang_vel = jnp.exp(-((yaw_rate_cmd - yaw_rate) ** 2) / _TRACK_SIGMA2)

    # --- base-stability penalties (quantities; sign lives in the weights) ---
    # dz**2, ||omega[:2]||**2, ||grav[:2]||**2
    lin_vel_z = v[2]**2
    ang_vel_xy = omega[0]**2 + omega[1]**2
    orientation = jnp.sum(up_b[:2] ** 2)    # align gravity with base z-axis
    
    # --- action smoothness ---
    action_rate = jnp.sum((action - last_action) ** 2)

    # --- effort / regularization over the leg dofs ---
    # qfrc_actuator is the generalized (joint-coord) actuator force, NOT
    # actuator_force; qacc is the joint acceleration. Both indexed by dof address.
    leg_dofs = ids.leg_dofs()
    torques = jnp.sum(data.qfrc_actuator[leg_dofs] ** 2)
    dof_acc = jnp.sum(data.qacc[leg_dofs] ** 2)

    terms = {
        "lin_vel": lin_vel,
        "ang_vel": ang_vel,
        "lin_vel_z": lin_vel_z,
        "ang_vel_xy": ang_vel_xy,
        "orientation": orientation,
        "action_rate": action_rate,
        "torques": torques,
        "dof_acc": dof_acc,
    }

    total = (
        w.lin_vel * lin_vel
        + w.ang_vel * ang_vel
        + w.lin_vel_z * lin_vel_z
        + w.ang_vel_xy * ang_vel_xy
        + w.orientation * orientation
        + w.action_rate * action_rate
        + w.torques * torques
        + w.dof_acc * dof_acc
    )

    # Foot timer is wired in with the gait slice; thread state through unchanged.
    new_foot_state = foot_state
    return total, terms, new_foot_state


def compute_reward_2(
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
    """Full per-step reward (all 10 terms): slices 1-2 plus the contact-driven
    ``collision`` and ``feet_air_time``.

    Same contract as :func:`compute_reward`. ``feet_air_time`` is gated by whether
    any motion is commanded (linear or yaw, deadband ``move_cmd_eps``), so the
    robot doesn't step in place while told to stand. Returns ``(total, terms,
    new_foot_state)`` with ``new_foot_state`` carrying the updated foot timer.
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
        base_height = jnp.exp(-((data.qpos[2] - height_target) ** 2) / _HEIGHT_SIGMA2)

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
