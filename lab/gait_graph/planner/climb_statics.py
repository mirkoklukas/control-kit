"""Statics of the climb robot: its MuJoCo model, postures -> qpos, the servo dofs.

:func:`.statics.statics` needs a model and a ``qpos``. The climb robot's model
(:func:`.climb_model.mjmodel.build`: true masses, pads, passive Cardan ankles) has the
ankles as extra joints, interleaved with the legs' in ``qpos``, and only the 12 leg joints
are servos. :class:`ClimbModel` bundles that:

- :meth:`ClimbModel.qpos`: a stance-code posture (body pose + 12 leg joint angles) ->
  the model's ``qpos``, with each pad laid flat on its surface (normal given), the ankle
  angles clipped to their range. JAX, jittable.
- :meth:`ClimbModel.statics`: :func:`.statics.statics` with torques on the servos only.

The model is the climb model without its adhesion actuators (MJX's JAX backend has none,
and the statics needs no actuators): the magnets enter the hold check as the adhesion
limit A, not as actuators.

The feet are the ankle pivots (bodies ``foot{i}``): a pad carries no moment through its
passive ankle, so its force acts at the pivot (docs, "The hold check").
"""
from __future__ import annotations

import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from controlkit.kinematics.types import Posture

from .climb_model.config import MjModelCfg
from .climb_model.mjmodel import build
from .statics import Statics, statics


class ClimbModel:
    """The climb robot's MuJoCo model, set up for the statics.

    Args:
        mj: the robot's model config.
    """

    def __init__(self, mj: MjModelCfg = None):
        self.mj = mj or MjModelCfg()
        spec, _ = build(self.mj, write=False)
        # MJX (JAX) has no adhesion actuators (body transmission); the statics needs no
        # actuators at all (forces come in through the feet, torques are computed), so
        # drop the adhesion ones and keep the keyframes' ctrl to the remaining servos
        adh = [a for a in spec.actuators if a.trntype == mujoco.mjtTrn.mjTRN_BODY]
        for a in adh:
            spec.delete(a)
        n_servo = len(spec.actuators)
        for key in spec.keys:
            key.ctrl = list(key.ctrl)[:n_servo]
        m = spec.compile()
        self.model = m
        self.mx = mjx.put_model(m)
        L = self.mj.num_legs
        self.foot_ids = jnp.array([m.body(f"foot{i}").id for i in range(L)])
        self.leg_qadr = jnp.array([[m.joint(f"leg{i}_j{k}").qposadr[0] for k in range(3)]
                                   for i in range(L)])                  # (L, 3)
        self.ank_qadr = jnp.array([[m.joint(f"ankle{i}_{s}").qposadr[0] for s in "ab"]
                                   for i in range(L)])                  # (L, 2)
        self.servo_dofs = jnp.array([m.joint(f"leg{i}_j{k}").dofadr[0]
                                     for i in range(L) for k in range(3)])  # (12,)
        self.mass = float(m.body_subtreemass[1])                       # body 1: the base
        self.ankle_range = np.radians(self.mj.ankle_range_deg)

    def qpos(self, posture: Posture, normals) -> jnp.ndarray:
        """The model's ``qpos`` for a posture, pads laid flat on their surfaces.

        Ankle angles from the foot frame (as :func:`.climb_model.poses.ankle_angles`, for
        any normal): the pad's face normal is its x-axis, ``R_foot (cos a cos b, sin b,
        -sin a cos b)``; setting it to ``-n`` gives ``b = asin(d_y)``, ``a = atan2(-d_z,
        d_x)`` with ``d = R_foot^T (-n)``. Clipped to the ankle range (past it, the pad
        would land on an edge).

        Args:
            posture: body pose and (L, 3) leg joint angles.
            normals: (L, 3) unit surface normals at the feet, world frame (a lifted foot's
                entry: any unit vector; its pad is laid against it too).

        Returns:
            (nq,) qpos.
        """
        w = posture.body.wxyz_xyz                                     # qw qx qy qz x y z
        q = jnp.zeros(self.model.nq)
        q = q.at[:7].set(jnp.concatenate([w[4:], w[:4]]))           # free joint: xyz, wxyz
        q = q.at[self.leg_qadr.reshape(-1)].set(jnp.asarray(posture.thetas).reshape(-1))
        d = mjx.kinematics(self.mx, mjx.make_data(self.mx).replace(qpos=q))
        R = d.xmat[self.foot_ids]                                     # (L, 3, 3)
        dvec = jnp.einsum("lji,lj->li", R, -jnp.asarray(normals, float))
        a = jnp.arctan2(-dvec[:, 2], dvec[:, 0])
        b = jnp.arcsin(jnp.clip(dvec[:, 1], -1.0, 1.0))
        ank = jnp.clip(jnp.stack([a, b], -1), -self.ankle_range, self.ankle_range)
        return q.at[self.ank_qadr.reshape(-1)].set(ank.reshape(-1))

    def statics(self, qpos, g) -> Statics:
        """:func:`.statics.statics` of ``qpos`` under ``g``, torques on the 12 servos."""
        return statics(self.mx, self.foot_ids, qpos, g, joint_dofs=self.servo_dofs)
