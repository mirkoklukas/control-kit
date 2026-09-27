"""Static-force scoring of planted postures (as in notebooks/staged/10_stable_postures).

Per posture, from :mod:`controlkit.forces` (one ``mjx.forward`` + a least-squares):

- ``force``:  ``|f|``, Frobenius norm of the (min-norm) foot reaction forces.
  Bounded below by ``W / sqrt(k)`` for ``k`` loaded feet (even load), so its term
  rewards spreading the load evenly rather than small forces as such.
- ``torque``: ``|tau|``, norm of the joint torques that hold the pose.
- ``sigma``:  smallest singular value of the base map (stance degeneracy; 0 = a
  body wrench no foot force can resist).

    score = w_force * exp(-|f| / W) + w_torque * exp(-|tau| / (W * l)) + w_sigma * sigma

with ``W`` the robot's weight and ``l`` its total leg length. The notebook used
``exp(-|f|)`` / ``exp(-|tau|)`` unscaled, which saturate at 0 for this robot
(``|f| ~ 20 N``); dividing by ``W`` / ``W * l`` puts both terms in a usable range.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

import controlkit.forces as forces
from controlkit.kinematics.robot import Robot
from controlkit.kinematics.types import Posture

from .config import Cfg


class Scorer:
    """Scores batches of planted postures; the scoring function is jitted.

    Args:
        robot: the robot.
        cfg: densities (mass model) and the score weights.
    """

    def __init__(self, robot: Robot, cfg: Cfg):
        mx, foot_ids = forces.model_from_robot(
            robot, body_density=cfg.body_density, leg_density=cfg.leg_density,
            foot_density=cfg.foot_density, link_radius=cfg.link_radius,
            foot_radius=cfg.foot_radius, body_half_height=cfg.body_half_height)
        self.mass = float(mx.body_subtreemass[1])            # body 1 = the free base
        self.weight = self.mass * 9.81
        self.length = float(sum(cfg.leg_lengths))
        W, ell = self.weight, self.length

        def score(postures: Posture, planted: jax.Array) -> dict:
            """Score N postures.

            Args:
                postures: (N,) batched postures.
                planted: (N, num_legs) bool, which feet bear load.

            Returns:
                dict of (N,) arrays: ``force``, ``torque``, ``sigma``, ``score``.
            """
            qpos = jax.vmap(forces.to_qpos)(postures)
            f, tau = forces.stance_forces_batch(mx, foot_ids, qpos, planted)
            sigma = forces.stance_sigma_min_batch(mx, foot_ids, qpos, planted, length=ell)
            fn = jnp.linalg.norm(f, axis=(-2, -1))
            tn = jnp.linalg.norm(tau, axis=(-2, -1))
            total = (cfg.w_force * jnp.exp(-fn / W) + cfg.w_torque * jnp.exp(-tn / (W * ell))
                     + cfg.w_sigma * sigma)
            return dict(force=fn, torque=tn, sigma=sigma, score=total)

        self.score_fn = score                 # un-jitted, for use inside other jits
        self.score = jax.jit(score)
