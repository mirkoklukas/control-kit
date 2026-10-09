"""Hand-written motions and static tests, as model/actuator sanity checks.

- :func:`standup` -- rest -> stand -> rest. Body height follows a cosine blend;
  servo targets come from the leg IK at each height, so the feet ideally stay put.
- :func:`hold` -- start in ``stand`` with adhesion on and hold at a fixed gravity
  tilt, or (``sweep``) tilt gravity from 0 up to ``tilt_deg`` and then hold.

All run through :func:`simulate`, which logs what the foot/actuator design needs:
servo torques, per-pad contact force (world frame), ankle angles, gravity.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from .config import MjModelCfg, ScriptedCfg
from .mjmodel import adhesion_actuators, gravity, make_robot, pad_cell_bodies
from .poses import leg_angles, rest_height


def height_schedule(cfg: MjModelCfg, scfg: ScriptedCfg, dt: float) -> np.ndarray:
    """Base-height targets per sim step: hold, lift, hold, lower, hold."""
    h0, h1 = rest_height(cfg), cfg.stand_height
    hold_ = np.full(int(scfg.hold_time / dt), 0.0)
    s = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, int(scfg.transition_time / dt)))
    u = np.concatenate([hold_, s, hold_ + 1, 1 - s, hold_])
    return h0 + (h1 - h0) * u


def pad_forces(model: mujoco.MjModel, data: mujoco.MjData, body_leg: np.ndarray,
               num_legs: int) -> np.ndarray:
    """(num_legs, 3) total contact force on each foot's pad cells, world frame (N).

    MuJoCo's contact force is exerted by geom1 on geom2, in the contact frame
    (rows of ``contact.frame``); flip it when the pad is geom1.

    Args:
        body_leg: (nbody,) leg index of each pad-cell body, -1 for other bodies.
    """
    out = np.zeros((num_legs, 3))
    f = np.zeros(6)
    for j in range(data.ncon):
        c = data.contact[j]
        if c.efc_address < 0:          # generated within margin but not active
            continue
        b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
        for b, sign in ((b2, 1.0), (b1, -1.0)):
            if body_leg[b] >= 0:
                mujoco.mj_contactForce(model, data, j, f)
                out[body_leg[b]] += sign * (c.frame.reshape(3, 3).T @ f[:3])
    return out


def simulate(model: mujoco.MjModel, cfg: MjModelCfg, start: str, servo: np.ndarray,
             adhesion, grav: np.ndarray | None = None) -> dict:
    """Step the model from keyframe ``start`` under open-loop commands.

    Args:
        model: compiled model (from :func:`.mjmodel.build`). Its gravity is modified
            in place when ``grav`` is given.
        cfg: model config.
        start: keyframe name.
        servo: (T, num_legs * 3) servo targets, one row per sim step.
        adhesion: scalar or (T, num_legs) adhesion ctrl in [0, 1].
        grav: optional (T, 3) gravity per step; None keeps the model's.

    Returns:
        Dict of per-step arrays: ``time``, ``qpos``, ``qvel``, ``ctrl``, ``torque``
        (servo forces), ``pad_force`` (T, num_legs, 3, world), ``ankle``
        (T, num_legs, 2), ``gravity`` (T, 3).
    """
    n, T = cfg.num_legs, len(servo)
    adhesion = np.broadcast_to(np.asarray(adhesion, dtype=float), (T, n))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key(start).id)
    body_leg = np.full(model.nbody, -1)
    for i in range(n):
        body_leg[pad_cell_bodies(model, i)] = i
    adh = [adhesion_actuators(model, i) for i in range(n)]
    ankle_adr = np.array([[model.joint(f"ankle{i}_{s}").qposadr[0] for s in "ab"]
                          for i in range(n)])

    log = {k: [] for k in ("time", "qpos", "qvel", "ctrl", "torque", "pad_force",
                           "ankle", "gravity")}
    for t in range(T):
        if grav is not None:
            model.opt.gravity[:] = grav[t]
        data.ctrl[: 3 * n] = servo[t]
        for i in range(n):
            data.ctrl[adh[i]] = adhesion[t, i]
        mujoco.mj_step(model, data)
        log["time"].append(data.time)
        log["qpos"].append(data.qpos.copy())
        log["qvel"].append(data.qvel.copy())
        log["ctrl"].append(data.ctrl.copy())
        log["torque"].append(data.actuator_force[: 3 * n].copy())
        log["pad_force"].append(pad_forces(model, data, body_leg, n))
        log["ankle"].append(data.qpos[ankle_adr].copy())
        log["gravity"].append(model.opt.gravity.copy())
    return {k: np.asarray(v) for k, v in log.items()}


def standup(model: mujoco.MjModel, cfg: MjModelCfg, scfg: ScriptedCfg, *,
            adhesion: float = 0.0) -> dict:
    """rest -> stand -> rest from the ``rest`` keyframe. Adds ``height_target``.

    Args:
        model: compiled model (from :func:`.mjmodel.build`).
        cfg: model config.
        scfg: timing (``transition_time``, ``hold_time``).
        adhesion: constant adhesion ctrl in [0, 1] on every foot.
    """
    robot = make_robot(cfg)
    heights = height_schedule(cfg, scfg, model.opt.timestep)
    ok, thetas = jax.jit(jax.vmap(lambda h: leg_angles(robot, cfg, h)))(jnp.asarray(heights))
    if not bool(ok.all()):
        raise RuntimeError("scripted heights include unreachable foot targets")
    log = simulate(model, cfg, "rest", np.asarray(thetas).reshape(len(heights), -1), adhesion)
    log["height_target"] = heights
    return log


def hold(model: mujoco.MjModel, cfg: MjModelCfg, scfg: ScriptedCfg, *, tilt_deg: float,
         adhesion: float = 1.0, sweep: bool = False) -> dict:
    """Hold ``stand`` under tilted gravity. Adds ``tilt_deg`` (T,).

    Without ``sweep``: gravity at ``tilt_deg`` throughout, for ``hold_duration``.
    With ``sweep``: tilt ramps linearly 0 -> ``tilt_deg`` over ``sweep_time``,
    then holds for ``hold_duration``.

    Args:
        model: compiled model (from :func:`.mjmodel.build`).
        cfg: model config.
        scfg: timing (``hold_duration``, ``sweep_time``).
        tilt_deg: final gravity tilt (0 floor, 90 wall, 180 ceiling).
        adhesion: constant adhesion ctrl in [0, 1] on every foot.
        sweep: ramp the tilt up from 0 instead of starting at ``tilt_deg``.
    """
    dt = model.opt.timestep
    n_hold = int(scfg.hold_duration / dt)
    if sweep:
        tilt = np.concatenate([np.linspace(0.0, tilt_deg, int(scfg.sweep_time / dt)),
                               np.full(n_hold, tilt_deg)])
    else:
        tilt = np.full(n_hold, tilt_deg)
    grav = np.array([gravity(a) for a in tilt])
    servo = np.broadcast_to(model.key("stand").ctrl[: 3 * cfg.num_legs], (len(tilt), 3 * cfg.num_legs))
    log = simulate(model, cfg, "stand", servo, adhesion, grav)
    log["tilt_deg"] = tilt
    return log
