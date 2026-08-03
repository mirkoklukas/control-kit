"""M1b: fly the target box with the DualSense; MPPI drives the robot to chase it.

The four feet are pinned to the ground with ``connect`` equalities, so the body
moves only through the leg servos and the planner absorbs the IK: each tick it
reads the mocap **target box** (sculpted by the controller), plans the leg-servo
targets with :class:`~lab.controller.policy.MPPI`, applies the head, and steps.

Run (macOS needs the MuJoCo viewer under mjpython):

    uv run --extra mjx mjpython -m lab.controller.run

MPC starts **off** (the robot just holds the rest posture via the servos, feet
pinned) so the rest stance can be inspected; L1 toggles the planner on/off. Foot
positions and their world-in-foot pose are printed to the terminal. The exact
model that is run is written to ``scene.xml`` beside this module on startup.

Controls: L1 = toggle MPC, left stick = x/y (along heading), right stick =
pitch/yaw, R2/L2 = +/- z, Options = reset target to rest.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import mujoco

from controlkit.mujoco_viz import to_qpos

from .config import Cfg
from .robot import make_robot, rest_posture, scene_xml, pin_all_feet
from .policy import MPPI
from .target import TargetBox

_FP = mujoco.mjtState.mjSTATE_FULLPHYSICS


def _print_feet(model, data, num_legs, p0):
    """Print each foot's world position (+ drift from rest) and its world-in-foot pose.

    Args:
        model, data: the compiled model and its data.
        num_legs: leg count.
        p0: (num_legs, 3) rest foot positions, for the drift column.
    """
    print("feet:")
    for i in range(num_legs):
        fid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"foot{i}")
        p, q = data.xpos[fid], data.xquat[fid]
        inv_q = np.zeros(4); mujoco.mju_negQuat(inv_q, q)
        inv_p = np.zeros(3); mujoco.mju_rotVecQuat(inv_p, -p, inv_q)
        drift = np.linalg.norm(p - p0[i]) * 1e3
        print(f"  foot{i}: world {np.round(p, 4)}  drift {drift:6.1f}mm  "
              f"| world-in-foot p {np.round(inv_p, 4)} q {np.round(inv_q, 3)}")


def _relaunch_under_mjpython():
    """On macOS, re-exec as a module under mjpython (the passive viewer needs it)."""
    from mujoco import viewer as mj_viewer
    if sys.platform != "darwin" or getattr(mj_viewer, "_MJPYTHON", None) is not None:
        return
    mjpython = Path(sys.executable).with_name("mjpython")
    if not mjpython.exists():
        raise RuntimeError(
            f"macOS viewer needs `mjpython`, not found next to {sys.executable}. "
            "Run: uv run --extra mjx mjpython -m lab.controller.run")
    print(f"relaunching under mjpython: {mjpython}", flush=True)
    os.execv(str(mjpython), [str(mjpython), "-m", __spec__.name])


def main():
    _relaunch_under_mjpython()
    from mujoco import viewer as mj_viewer
    # imported after any relaunch so it lives in the mjpython process
    from .input import DualSense

    cfg = Cfg()
    robot = make_robot(cfg)
    rest = rest_posture(robot, cfg)
    rest_qpos = to_qpos(rest)[0]
    rest_ctrl = np.asarray(rest.thetas).reshape(-1)

    # Build and compile the scene, and save the exact MJCF we run to scene.xml so
    # it always reflects the live config.
    xml = scene_xml(robot, cfg)
    Path(__file__).with_name("scene.xml").write_text(xml)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    # Activate the foot connect pins, anchored at the rest foot positions.
    pin_all_feet(model, data, cfg.num_legs, rest_qpos)
    data.qpos[:] = rest_qpos
    data.ctrl[:] = rest_ctrl
    mujoco.mj_forward(model, data)

    mocap = model.body_mocapid[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target")]

    planner = MPPI(model, cfg, rest_ctrl)
    state = np.zeros(mujoco.mj_stateSize(model, _FP))

    # Rest foot positions, for the drift column in the printout.
    rest_feet = np.array([data.xpos[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, f"foot{i}")].copy() for i in range(cfg.num_legs)])

    pad = DualSense()
    target = TargetBox(cfg)
    # Start with the planner off -- just hold rest so the pinned stance is visible.
    mpc_on = False
    print(f"connected: {pad.name}\n"
          "L1=toggle MPC (default OFF)  left stick=x/y  right stick=pitch/yaw  "
          "R2/L2=+/-z  Options=reset")
    print("\n[MPC OFF] holding rest posture -- feet should stay put:")
    _print_feet(model, data, cfg.num_legs, rest_feet)

    dt = 1.0 / cfg.control_hz
    tick = 0
    with mj_viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            s = pad.poll()
            if "l1" in s.pressed:
                mpc_on = not mpc_on
                print(f"\n[MPC {'ON' if mpc_on else 'OFF'}]")
            if "start" in s.pressed:
                target.reset()
            target.update(s, dt)
            tpos, tquat = target.pose()
            data.mocap_pos[mocap], data.mocap_quat[mocap] = tpos, tquat

            if mpc_on:
                # plan the leg-servo targets that bring the body onto the target box
                mujoco.mj_getState(model, data, state, _FP)
                data.ctrl[:] = planner(state, tpos, tquat)
            else:
                # planner off: just hold the rest posture
                data.ctrl[:] = rest_ctrl
            for _ in range(cfg.decimation):
                mujoco.mj_step(model, data)

            # print the feet roughly once a second
            tick += 1
            if tick % 60 == 0:
                _print_feet(model, data, cfg.num_legs, rest_feet)
            viewer.sync()
            time.sleep(dt)
    pad.close()


if __name__ == "__main__":
    main()
