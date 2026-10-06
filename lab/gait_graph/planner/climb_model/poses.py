"""(Copied from ``lab/rl/climb/poses.py`` for the planner, 2026-10-05.)

(Copied from ``lab/rl/poses.py``, unchanged.)

The ``rest`` and ``stand`` poses, and the body-height sweep between them.

Both poses plant the feet at the same world points (radius ``stand_foot_radius``,
pad flat on the ground) and differ only in body height, so rest -> stand is a pure
vertical lift: :func:`leg_angles` solves the leg IK at any height along the way.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from controlkit.kinematics import Robot, candidates
from controlkit.se3 import SE3

from .config import MjModelCfg


def rest_height(cfg: MjModelCfg) -> float:
    """Base height with the trunk lying on the ground (plus ``rest_clearance``)."""
    return cfg.body_half_height + cfg.rest_clearance


def foot_targets(robot: Robot, cfg: MjModelCfg) -> jax.Array:
    """(num_legs, 3) world ankle-pivot targets: radially out along each mount,
    at the height a flat pad puts the pivot."""
    t = robot.mounts.translation()
    dirs = t / jnp.linalg.norm(t[:, :2], axis=-1, keepdims=True)
    z = cfg.pivot_height + cfg.pad_thickness
    return cfg.stand_foot_radius * dirs.at[:, 2].set(0.0) + jnp.array([0.0, 0.0, z])


def leg_angles(robot: Robot, cfg: MjModelCfg, height, lift=None) -> tuple[jax.Array, jax.Array]:
    """Leg IK with the body level at ``height`` and feet on :func:`foot_targets`.

    Branch choice: reachable, within limits, and the knee highest (spider stance).

    Args:
        robot, cfg: the robot and config.
        height: scalar base height (jit/vmap friendly).
        lift: optional (num_legs,) raise of each foot target along world z (m), e.g.
            one leg lifted off the floor (``magnet_test``). None: all on the floor.

    Returns:
        ``(ok, thetas)`` -- (num_legs,) bool and (num_legs, 3).
    """
    body = SE3.from_translation(jnp.array([0.0, 0.0, 1.0]) * height)
    shoulders = robot.shoulders(body)
    feet = foot_targets(robot, cfg)
    if lift is not None:
        feet = feet.at[:, 2].add(jnp.asarray(lift))
    leg = robot.leg

    def one(shoulder, foot):
        ok, thetas = leg.ik_from_foot(shoulder.inverse().apply(foot))
        ok = ok & candidates.in_limits(thetas, leg.limits)
        knee_z = jax.vmap(lambda th: leg.forward(th)[2].translation()[2])(thetas)
        i = candidates.best(knee_z, ok)
        return ok[i], thetas[i]

    return jax.vmap(one)(shoulders, feet)


def ankle_angles(model: mujoco.MjModel, data: mujoco.MjData, i: int) -> np.ndarray:
    """Ankle angles ``(a, b)`` that lay pad ``i`` flat (face normal = world -z).

    Uses the foot frame from ``data`` (after ``mj_forward``). The pad's rotation is
    ``R_foot Ry(a) Rz(b)``, whose x-axis is ``R_foot (cos a cos b, sin b, -sin a cos b)``;
    solving for it to equal ``-z`` gives ``b = asin(d_y)``, ``a = atan2(-d_z, d_x)``
    with ``d = R_foot^T (-z)``. Not clipped to the ankle range -- the caller checks.
    """
    fid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"foot{i}")
    d = data.xmat[fid].reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
    return np.array([np.arctan2(-d[2], d[0]), np.arcsin(np.clip(d[1], -1, 1))])


def pose_qpos(model: mujoco.MjModel, robot: Robot, cfg: MjModelCfg, height: float):
    """Full ``qpos`` (base, legs, ankles laid flat) and servo targets at ``height``.

    Returns:
        ``(qpos, servo_ctrl, ankles)`` -- (nq,), (num_legs * 3,), (num_legs, 2) rad.
        ``ankles`` are the *required* flat-pad angles (possibly past the stops);
        ``qpos`` holds them clipped to ``ankle_range_deg``.

    Raises:
        RuntimeError: if a foot target is unreachable at this height.
    """
    ok, thetas = leg_angles(robot, cfg, height)
    if not bool(ok.all()):
        raise RuntimeError(f"feet unreachable at height={height:.3f}: ok={np.asarray(ok)}")
    thetas = np.asarray(thetas)

    data = mujoco.MjData(model)
    data.qpos[:7] = [0.0, 0.0, height, 1.0, 0.0, 0.0, 0.0]
    for i in range(cfg.num_legs):
        for k in range(3):
            data.qpos[model.joint(f"leg{i}_j{k}").qposadr[0]] = thetas[i, k]
    mujoco.mj_kinematics(model, data)
    ankles = np.stack([ankle_angles(model, data, i) for i in range(cfg.num_legs)])
    lim = np.radians(cfg.ankle_range_deg)
    ankles_q = np.clip(ankles, -lim, lim)     # past the stop: pad lands on an edge
    for i in range(cfg.num_legs):
        data.qpos[model.joint(f"ankle{i}_a").qposadr[0]] = ankles_q[i, 0]
        data.qpos[model.joint(f"ankle{i}_b").qposadr[0]] = ankles_q[i, 1]
    return data.qpos.copy(), thetas.reshape(-1), ankles


def keyframe(model: mujoco.MjModel, robot: Robot, cfg: MjModelCfg, name: str):
    """``(qpos, ctrl)`` for keyframe ``rest`` or ``stand`` (adhesion off)."""
    height = {"rest": rest_height(cfg), "stand": cfg.stand_height}[name]
    qpos, servo, _ = pose_qpos(model, robot, cfg, height)
    ctrl = np.zeros(model.nu)
    ctrl[: servo.size] = servo
    return qpos, ctrl
