"""Debug experiments: the hexapod tripod on the wall (models/sticky0.xml).

Uses the fixes from ``lab.ik.sticky_test._apply_fixes`` (stiff contact, condim=6
rolling friction, margin/gap band, rigid legs). Findings:
  * a placement bug put the foot *sphere center* at the wall -> 25 mm penetration
    -> ~260 N repulsion spike -> the robot was ejected off the wall. Fixed by
    offsetting the base by the foot radius (see ``wall_tripod_qpos``);
  * after the fix the robot stays at the wall (dx~0) but still settles ~270 mm
    down, catching on the folded free legs on the floor -- the tripod-wall
    adhesion grab across the band isn't fully holding yet.

Plain MuJoCo (base venv): ``uv run python -m lab.ik.debug.hexapod_wall``.
"""
from __future__ import annotations

import numpy as np
import mujoco

from lab.ik.sticky_test import wall_tripod_qpos, _apply_fixes

MODEL = "models/sticky0.xml"


def _setup(into_wall, gain=200.0, tripod=(0, 2, 4)):
    m = mujoco.MjModel.from_xml_path(MODEL); _apply_fixes(m, gain); d = mujoco.MjData(m)
    d.qpos[:] = wall_tripod_qpos(m, tripod, into_wall=into_wall)
    mujoco.mj_forward(m, d)
    for i in range(6):
        d.ctrl[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"stick{i}")] = 1.0 if i in tripod else 0.0
    return m, d


def drift(into_wall, gain=200.0, secs=2.0):
    """Base drift (dx, dz, |drift|) in mm after settling with adhesion on."""
    m, d = _setup(into_wall, gain); jdof = np.arange(6, m.nv); b0 = d.qpos[:3].copy()
    for _ in range(int(secs / m.opt.timestep)):
        d.qfrc_applied[jdof] = d.qfrc_bias[jdof]
        mujoco.mj_step(m, d)
    dr = d.qpos[:3] - b0
    return dr[0] * 1000, dr[2] * 1000, float(np.linalg.norm(dr)) * 1000


def trace(into_wall=0.0, gain=200.0, secs=1.5, tripod=(0, 2, 4)):
    """Print base pose + tripod wall-contact normals + which free feet hit the floor."""
    m, d = _setup(into_wall, gain, tripod); jdof = np.arange(6, m.nv)
    wall = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "wall")
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    footg = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"foot{i}"): i for i in range(6)}
    f6 = np.zeros(6)
    for step in range(int(secs / m.opt.timestep)):
        d.qfrc_applied[jdof] = d.qfrc_bias[jdof]; mujoco.mj_step(m, d)
        if step % int(0.25 / m.opt.timestep) == 0:
            wn = {i: 0.0 for i in tripod}; onfloor = set()
            for c in range(d.ncon):
                gs = {d.contact[c].geom1, d.contact[c].geom2}
                mujoco.mj_contactForce(m, d, c, f6)
                if wall in gs and (g := (gs - {wall}).pop()) in footg and footg[g] in tripod:
                    wn[footg[g]] += f6[0]
                if floor in gs and (g := (gs - {floor}).pop()) in footg:
                    onfloor.add(footg[g])
            print(f"t={step*m.opt.timestep:.2f} z={float(d.qpos[2]):.3f} x={float(d.qpos[0]):.3f} "
                  f"wallN={[round(wn[i],1) for i in tripod]} free_on_floor={sorted(onfloor)}")


def main():
    print("placement sweep (into_wall = foot-surface penetration), base drift mm:")
    for iw in (-0.005, 0.0, 0.004):
        dx, dz, n = drift(iw)
        print(f"  into_wall={iw*1000:+5.1f}mm -> dx={dx:+7.1f} dz={dz:+7.1f} |drift|={n:6.1f}")
    print("\ntrace (into_wall=0.0):")
    trace(0.0)


if __name__ == "__main__":
    main()
