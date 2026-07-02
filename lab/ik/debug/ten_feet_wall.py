"""Experiment: 10 feet stuck to the wall with a gain gradient.

Each foot is an independent geom with its own adhesion actuator, laid out
left-to-right with increasing gain. Turn all magnets on and watch: low-gain feet
slide down, high-gain feet hold. Confirms "stronger magnet -> more press -> more
friction -> slower slide".

Two foot options:
  * box                       -- flat faces, can't roll (freejoint).
  * sphere with no_roll=True  -- 3 slide joints (translation only, no rotation),
                                 so the sphere can't roll; motion is pure sliding.

Generates the model, runs it, prints gain->drop, saves a replay to scratch/.
``uv run python -m lab.ik.debug.ten_feet_wall``.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[3]
SCRATCH = Path(__file__).resolve().parents[1] / "scratch"
SCRATCH.mkdir(exist_ok=True)

GAINS = [0, 5, 10, 20, 40, 80, 160, 320, 640, 1280]
YS = np.linspace(-0.45, 0.45, len(GAINS))

_SLIDE = ('      <joint type="slide" axis="1 0 0"/>\n'
          '      <joint type="slide" axis="0 1 0"/>\n'
          '      <joint type="slide" axis="0 0 1"/>')


def write_model(foot_type="sphere", no_roll=True):
    geom = ('type="box" size="0.02 0.02 0.02"' if foot_type == "box"
            else 'type="sphere" size="0.02"')
    joint = _SLIDE if (foot_type == "sphere" and no_roll) else "      <freejoint/>"
    bodies, acts = [], []
    for i, y in enumerate(YS):
        bodies.append(
            f'    <body name="foot{i}" pos="0.975 {y:.3f} 0.6">\n{joint}\n'
            f'      <geom name="foot{i}" {geom} material="foot"\n'
            f'            contype="0" conaffinity="1" friction="2 0.5 0.5" margin="0.01" gap="0.01"\n'
            f'            solimp="0.99 0.9999 0.0001 0.5 2"/>\n'   # stiff: no deep penetration
            f'    </body>')
        acts.append(f'    <adhesion name="stick{i}" body="foot{i}" ctrlrange="0 1" gain="{GAINS[i]}"/>')
    tag = f"{foot_type}s10" + ("_noroll" if (foot_type == "sphere" and no_roll) else "")
    path = ROOT / "models" / f"sticky_{tag}.xml"
    path.write_text(f'''<mujoco model="sticky_{tag}">
  <option timestep="0.004" integrator="implicitfast" iterations="10" ls_iterations="8"/>
  <visual>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.65 0.65 0.65" specular="0.1 0.1 0.1"/>
    <map haze="0"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.2 0.3 0.4" rgb2="0.1 0.15 0.2" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="10 10" reflectance="0"/>
    <material name="foot" rgba="1 0.258824 0.976471 1"/>
  </asset>
  <worldbody>
    <light pos="0 0 2.0" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="grid" contype="1" conaffinity="1" priority="1" friction="2 0.5 0.5"/>
    <geom name="wall" type="plane" pos="1 0 0.6" zaxis="-1 0 0" size="0.8 2 0.05" material="grid" contype="1" conaffinity="1" friction="2 0.5 0.5"/>
{chr(10).join(bodies)}
  </worldbody>
  <actuator>
{chr(10).join(acts)}
  </actuator>
</mujoco>
''')
    return path, tag


def run(foot_type="sphere", no_roll=True, secs=3.0):
    path, tag = write_model(foot_type, no_roll)
    m = mujoco.MjModel.from_xml_path(str(path)); d = mujoco.MjData(m)
    fb = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"foot{i}") for i in range(len(GAINS))]
    mujoco.mj_forward(m, d)
    z0 = np.array([float(d.xpos[b][2]) for b in fb])   # body world z (any joint layout)
    d.ctrl[:] = 1.0
    qs = []
    for _ in range(int(secs / m.opt.timestep)):
        mujoco.mj_step(m, d); qs.append(d.qpos.copy())
    drops = (np.array([float(d.xpos[b][2]) for b in fb]) - z0) * 1000
    print(f"[{tag}]  gain -> drop after {secs:.0f}s (mm; ~0 = held):")
    for g, dz in zip(GAINS, drops):
        print(f"  gain={g:5d}: {dz:+8.1f} mm")
    out = SCRATCH / f"{tag}.npz"
    np.savez(out, qpos=np.array(qs), qvel=np.zeros((len(qs), m.nv)),
             timestep=m.opt.timestep, model=str(path.relative_to(ROOT)))
    print(f"replay: uv run ctk play {out.relative_to(ROOT)}")


if __name__ == "__main__":
    run(foot_type="sphere", no_roll=True)
