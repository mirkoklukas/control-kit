"""Debug experiments: adhesion on a SINGLE foot (models/sticky_test.xml).

These are the isolation tests that untangled the sticky-wall behavior. Findings:
  * adhesion holds the NORMAL direction fine (foot doesn't peel);
  * the sphere foot with condim=3 has no rolling resistance -> it ROLLS down the
    wall; sliding friction never engages. condim=6 (rolling friction) fixes it;
  * margin==gap makes a band where the contact stays active with no repulsion, so
    adhesion can grab across a small gap without the deep-penetration fight.

Plain MuJoCo (base venv): ``uv run python -m lab.ik.debug.single_foot``.
"""
from __future__ import annotations

import numpy as np
import mujoco

MODEL = "models/sticky_test.xml"


def _model(condim=3, roll_fric=0.5, margin=0.0, gap=0.0, gain=30.0):
    m = mujoco.MjModel.from_xml_path(MODEL)
    for name in ("foot", "wall"):
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
        m.geom_condim[g] = condim
        m.geom_friction[g] = [2.0, roll_fric, roll_fric]   # [slide, torsional, rolling]
    fg = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "foot")
    m.geom_margin[fg], m.geom_gap[fg] = margin, gap
    m.actuator_gainprm[0, 0] = gain
    return m


def wall_stick(condim=3, roll_fric=0.5, margin=0.0, gap=0.0, gain=30.0, x_start=0.97, secs=2.0):
    """Foot near the wall, adhesion on. Returns base dz (mm); ~0 = stuck."""
    m = _model(condim, roll_fric, margin, gap, gain); d = mujoco.MjData(m)
    d.qpos[0] = x_start
    mujoco.mj_forward(m, d); z0 = float(d.qpos[2]); d.ctrl[0] = 1.0
    for _ in range(int(secs / m.opt.timestep)):
        mujoco.mj_step(m, d)
    return (float(d.qpos[2]) - z0) * 1000.0


def floor_pull(Fx, gain=30.0, secs=1.0):
    """Foot on the floor, adhesion on, pushed sideways by Fx. Returns x slide (mm).
    Shows the sphere ROLLS (condim=3): even a tiny push moves it far."""
    m = _model(gain=gain); d = mujoco.MjData(m)
    fb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "foot")
    d.qpos[:3] = [0.0, 0.0, 0.024]
    mujoco.mj_forward(m, d); d.ctrl[0] = 1.0; x0 = float(d.qpos[0])
    for _ in range(int(secs / m.opt.timestep)):
        d.xfrc_applied[fb, 0] = Fx
        mujoco.mj_step(m, d)
    return (float(d.qpos[0]) - x0) * 1000.0


def main():
    print("wall stick (base dz, mm; ~0 = stuck):")
    print(f"  condim=3 (sphere rolls)      -> {wall_stick(condim=3):+8.1f}")
    print(f"  condim=6 (rolling friction)  -> {wall_stick(condim=6):+8.1f}")
    print(f"  condim=6 + margin/gap band   -> {wall_stick(condim=6, margin=0.01, gap=0.01):+8.1f}")
    print("floor pull, adhesion on (x slide, mm) -- sphere rolls, so tiny force moves it:")
    for F in (2, 5, 20):
        print(f"  Fx={F:2d} N -> {floor_pull(F):+8.1f}")


if __name__ == "__main__":
    main()
