"""Test: hexapod on the vertical wall (sticky0.xml) in a tripod stance.

Rigidly rotate the home stance 90 deg so the legs face the wall (x=1 plane),
plant a tripod on the wall with the other legs lifted, turn on adhesion for the
tripod, settle under gravity, and report whether it stays stuck.

Plain MuJoCo (base venv): ``uv run python -m lab.ik.sticky_test``.
Renders before/after PNGs to scratch/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models" / "sticky0.xml"
SCRATCH = Path(__file__).resolve().parent / "scratch"   # experiment-local, gitignored
SCRATCH.mkdir(exist_ok=True)


def _jadr(model, seg, i):
    return model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{seg}{i}")]


FOOT_RADIUS = 0.025   # foot sphere radius -- the site (sphere center) sits this far behind the surface


def wall_tripod_qpos(model, tripod=(0, 2, 4), into_wall=0.0, lift_tibia=0.6, lift_femur=-0.3):
    """Home stance rotated onto the wall; ``into_wall`` is the foot *surface*
    penetration (0 = touching, negative = in the margin band), free legs folded."""
    home = model.key_qpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")].copy()
    q = home.copy()
    # R_y(-90): home leg-down (-z) -> +x (toward wall). base pos = R@home_pos + t.
    # -FOOT_RADIUS so the sphere SURFACE (not its center) lands at the wall.
    q[0:3] = [0.7838 - FOOT_RADIUS + into_wall, 0.0, 0.6]
    q[3:7] = [np.sqrt(0.5), 0.0, -np.sqrt(0.5), 0.0]        # quat for R_y(-90)
    for i in range(6):
        if i not in tripod:                                 # fold free legs off the wall
            q[_jadr(model, "femur", i)] += lift_femur
            q[_jadr(model, "tibia", i)] += lift_tibia
    return q


def _foot_x(model, data):
    return [float(data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"foot{i}")][0])
            for i in range(6)]


def _render(model, data, path, cam):
    r = mujoco.Renderer(model, 600, 800)
    r.update_scene(data, cam)
    mpimg.imsave(path, r.render())
    r.close()


def run(tripod=(0, 2, 4), seconds=1.0):
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    model.vis.global_.offwidth, model.vis.global_.offheight = 800, 600
    data = mujoco.MjData(model)

    q = wall_tripod_qpos(model, tripod)
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    print("foot x (wall at x=1):", [round(x, 3) for x in _foot_x(model, data)],
          "| tripod", list(tripod))
    base0 = data.qpos[:3].copy()

    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.85, 0.0, 0.6]
    cam.distance, cam.azimuth, cam.elevation = 1.6, 140.0, -8.0
    _render(model, data, SCRATCH / "sticky_before.png", cam)

    # hold joints with the position servos; adhesion on for the tripod only
    theta = q[7:].copy()
    for i in range(6):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"stick{i}")
        data.ctrl[sid] = 1.0 if i in tripod else 0.0
    data.ctrl[:18] = theta

    jdof = np.arange(6, model.nv)
    for _ in range(int(seconds / model.opt.timestep)):
        data.qfrc_applied[jdof] = data.qfrc_bias[jdof]      # gravity comp on joints
        data.ctrl[:18] = theta
        mujoco.mj_step(model, data)

    drift = data.qpos[:3] - base0
    _render(model, data, SCRATCH / "sticky_after.png", cam)
    print(f"base drift after {seconds}s:  dx={drift[0]*1000:+.1f} mm  "
          f"dy={drift[1]*1000:+.1f} mm  dz={drift[2]*1000:+.1f} mm")
    stuck = np.linalg.norm(drift) < 0.05
    print("STUCK" if stuck else "SLID / FELL", f"(|drift|={np.linalg.norm(drift)*1000:.1f} mm)")


def _apply_fixes(model, gain, rigid_damping=1e4):
    """Solver + adhesion gain + rigid legs. The *contact* params (margin/gap,
    solimp, condim, friction) are baked into models/sticky0.xml -- setting
    geom_margin at RUNTIME does not take effect (it's compiled into the collision
    setup), so the feet never register the band contact and adhesion has nothing
    to grab."""
    model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    model.opt.iterations, model.opt.ls_iterations = 100, 50
    for i in range(6):
        model.actuator_gainprm[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"stick{i}"), 0] = gain
    model.dof_damping[6:] = rigid_damping                     # freeze the 18 leg joints (rigid)


def required_torque(qpos, planted=(0, 2, 4), gain=200.0, secs=1.0, limit=8.0):
    """Per-joint actuator torque needed to hold ``qpos`` on the wall.

    Holds the pose with rigid legs (so the sim is stable) and reads off the torque
    the *real actuators* would have to supply via inverse dynamics
    (``qfrc_bias - qfrc_constraint`` = gravity/Coriolis minus the contact forces
    projected to the joints). Independent of how the pose is actually held.

    qpos    : (nq,) full configuration (feet placed on/into the wall).
    planted : leg indices whose adhesion is on.
    Returns ``(tau, feasible)``: ``tau`` is a ``(6, 3)`` array of max |torque| over
    the settle, indexed [leg, (coxa, femur, tibia)]; ``feasible`` = every joint
    within +/- ``limit`` Nm.
    """
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    _apply_fixes(model, gain)
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    theta = np.asarray(qpos)[7:].copy()
    for i in range(6):
        data.ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"stick{i}")] = 1.0 if i in planted else 0.0
    jdof = np.arange(6, model.nv)
    peak = np.zeros(model.nv)
    for _ in range(int(secs / model.opt.timestep)):
        data.qfrc_applied[jdof] = data.qfrc_bias[jdof]
        data.ctrl[:18] = theta
        mujoco.mj_step(model, data)
        peak = np.maximum(peak, np.abs(data.qfrc_bias - data.qfrc_constraint))
    dofs = np.array([[model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{s}{i}")]
                      for s in ("coxa", "femur", "tibia")] for i in range(6)])
    tau = peak[dofs]                                    # (6, 3)
    return tau, bool(np.all(tau <= limit))


def run_rigid(tripod=(0, 2, 4), seconds=2.0, gain=200.0):
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    model.vis.global_.offwidth, model.vis.global_.offheight = 800, 600
    _apply_fixes(model, gain)
    data = mujoco.MjData(model)

    q = wall_tripod_qpos(model, tripod)
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    base0 = data.qpos[:3].copy()
    for i in range(6):
        data.ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"stick{i}")] = 1.0 if i in tripod else 0.0

    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.85, 0.0, 0.6]
    cam.distance, cam.azimuth, cam.elevation = 1.6, 140.0, -8.0
    _render(model, data, SCRATCH / "rigid_before.png", cam)

    jdof = np.arange(6, model.nv)
    qs, vs = [], []
    for _ in range(int(seconds / model.opt.timestep)):
        data.qfrc_applied[jdof] = data.qfrc_bias[jdof]        # gravity-comp the (frozen) joints
        mujoco.mj_step(model, data)
        qs.append(data.qpos.copy()); vs.append(data.qvel.copy())

    _render(model, data, SCRATCH / "rigid_after.png", cam)
    drift = data.qpos[:3] - base0
    print(f"rigid + adhesion-fix, {seconds}s: base drift dx={drift[0]*1000:+.1f} dy={drift[1]*1000:+.1f} "
          f"dz={drift[2]*1000:+.1f} mm  -> {'STUCK' if np.linalg.norm(drift)<0.03 else 'MOVED'} "
          f"(|drift|={np.linalg.norm(drift)*1000:.1f} mm)")

    out = SCRATCH / "sticky_rigid.npz"
    np.savez(out, qpos=np.array(qs), qvel=np.array(vs),
             timestep=model.opt.timestep, model="models/sticky0.xml")
    print("rollout saved ->", out, " | replay:  uv run ctk play lab/ik/scratch/sticky_rigid.npz")


if __name__ == "__main__":
    run_rigid()
