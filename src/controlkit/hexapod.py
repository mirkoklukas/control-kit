from __future__ import annotations

import dataclasses
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import mujoco
from mujoco import mjx


# --------------------------------------------------------------------------- #
# Command & cross-step state (registered JAX pytrees, no flax dependency)      #
# --------------------------------------------------------------------------- #
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

    def feet_bodies(self) -> np.ndarray:
        """Body id (the tibia) each foot geom belongs to, aligned with feet()."""
        return self.model.geom_bodyid[self.feet()].astype(np.int32)

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

