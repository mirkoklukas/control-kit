"""Simple stability test in the climb world (floor + box).

Place the robot, hold the legs rigid, and just watch whether the body moves/tips
over a short settle. Ground feet rely on gravity + floor contact (no adhesion);
wall feet use adhesion. Also reads the joint torque the actuators would need to
hold the pose (via inverse dynamics) -- returned but not used yet.

Test cases: stable + unstable tripod, on the ground and on the box wall, with the
free legs lifted ~10 cm.  ``uv run python -m lab.ik.debug.climb_stability``.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import mujoco

from lab.ik.sticky_test import _apply_fixes, wall_tripod_qpos

ROOT = Path(__file__).resolve().parents[3]
CLIMB = ROOT / "models" / "climb0.xml"


def _qadr(model, seg, i):
    return model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{seg}{i}")]


def ground_pose(model, base_xy, planted, lift_femur=-0.45):
    """Standing on the floor; planted feet at home, free feet raised (~10 cm)."""
    q = model.key_qpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")].copy()
    q[:3] = [base_xy[0], base_xy[1], 0.241185]
    q[3:7] = [1, 0, 0, 0]
    for i in range(6):
        if i not in planted:
            q[_qadr(model, "femur", i)] += lift_femur       # raise the free foot
    return q


def stability_test(qpos, planted, adhesion=False, gain=100.0, secs=1.5):
    """Place the robot, hold legs rigid, settle, and report if the body moved.

    Returns dict: moved (bool), drift_mm, tilt_deg, tau (6,3) required joint torque.
    """
    model = mujoco.MjModel.from_xml_path(str(CLIMB))
    _apply_fixes(model, gain)                                # Newton, rigid legs, adhesion gain
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    theta = np.asarray(qpos)[7:].copy()
    for i in range(6):
        data.ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"stick{i}")] = \
            1.0 if (adhesion and i in planted) else 0.0

    jdof = np.arange(6, model.nv)
    p0 = data.qpos[:3].copy(); q0 = data.qpos[3:7].copy()
    peak = np.zeros(model.nv); frames = []
    for _ in range(int(secs / model.opt.timestep)):
        data.qfrc_applied[jdof] = data.qfrc_bias[jdof]       # gravity comp
        data.ctrl[:18] = theta                               # servos hold joint targets
        mujoco.mj_step(model, data)
        peak = np.maximum(peak, np.abs(data.qfrc_bias - data.qfrc_constraint))
        frames.append(data.qpos.copy())

    drift = float(np.linalg.norm(data.qpos[:3] - p0))
    dq = np.zeros(4)
    mujoco.mju_mulQuat(dq, data.qpos[3:7], np.array([q0[0], -q0[1], -q0[2], -q0[3]]))
    tilt = float(2.0 * np.arccos(np.clip(abs(dq[0]), 0.0, 1.0)))
    dofs = np.array([[model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{s}{i}")]
                      for s in ("coxa", "femur", "tibia")] for i in range(6)])
    return {
        "moved": bool(drift > 0.03 or tilt > np.radians(8)),
        "drift_mm": drift * 1000, "tilt_deg": np.degrees(tilt),
        "tau": peak[dofs], "frames": np.array(frames), "timestep": model.opt.timestep,
    }


def _cases(m):
    return [
        ("ground  [0,2,4]", ground_pose(m, (0.0, 0.0), (0, 2, 4)), (0, 2, 4), False),
        ("ground  [0,1,2]", ground_pose(m, (0.0, 0.0), (0, 1, 2)), (0, 1, 2), False),
        ("wall    [0,2,4]", wall_tripod_qpos(m, (0, 2, 4), into_wall=-0.005), (0, 2, 4), True),
        ("wall    [0,1,2]", wall_tripod_qpos(m, (0, 1, 2), into_wall=-0.005), (0, 1, 2), True),
        ("wall    [0,5]  ", wall_tripod_qpos(m, (0, 5), into_wall=-0.005), (0, 5), True),
    ]


def _run_all(m):
    """Run every case; return list of (name, frames) and print the verdicts."""
    out, dt = [], m.opt.timestep
    for name, q, planted, adh in _cases(m):
        r = stability_test(q, planted, adhesion=adh)
        print(f"{name}: {'MOVED ' if r['moved'] else 'stable'}  drift={r['drift_mm']:6.1f}mm "
              f"tilt={r['tilt_deg']:5.1f}deg  max|tau|={float(r['tau'].max()):5.1f}Nm")
        out.append((name, r["frames"])); dt = r["timestep"]
    return out, dt


def _demo():
    m = mujoco.MjModel.from_xml_path(str(CLIMB))
    frames, dt = _run_all(m)
    q = np.concatenate([f for _, f in frames])
    scratch = Path(__file__).resolve().parents[1] / "scratch"; scratch.mkdir(exist_ok=True)
    np.savez(scratch / "climb_stability.npz", qpos=q, qvel=np.zeros((len(q), m.nv)),
             timestep=dt, model="models/climb0.xml")
    print("rollout -> lab/ik/scratch/climb_stability.npz  (or cycle with: mjpython -m lab.ik.debug.climb_stability view)")


def _view():
    """Passive viewer: left/right arrows cycle cases; each plays its settle on loop."""
    import time
    import mujoco.viewer
    m = mujoco.MjModel.from_xml_path(str(CLIMB))
    frames, _ = _run_all(m)
    data = mujoco.MjData(m)
    st = {"i": 0, "f": 0}

    def key_cb(keycode):
        if keycode == 262:  st["i"] = (st["i"] + 1) % len(frames); st["f"] = 0   # right
        elif keycode == 263: st["i"] = (st["i"] - 1) % len(frames); st["f"] = 0  # left
        print("->", frames[st["i"]][0])

    with mujoco.viewer.launch_passive(m, data, key_callback=key_cb) as viewer:
        viewer.cam.lookat[:] = [0.5, 0.0, 0.4]; viewer.cam.distance = 3.0
        viewer.cam.azimuth, viewer.cam.elevation = 130.0, -12.0
        print("LEFT/RIGHT arrows cycle cases. current:", frames[0][0])
        while viewer.is_running():
            name, fr = frames[st["i"]]
            data.qpos[:] = fr[st["f"] % len(fr)]; mujoco.mj_forward(m, data)
            viewer.sync(); st["f"] += 1; time.sleep(0.01)


if __name__ == "__main__":
    import sys
    _view() if "view" in sys.argv else _demo()
