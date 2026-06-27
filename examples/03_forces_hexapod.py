"""Hexapod tripod stance with a live viewer, contact-force arrows, and a
femur-torque plot.

A sphere is overlaid at each femur joint and colored by that joint's actuator
torque: red for negative, white near zero, blue for positive (diverging scale,
saturating at +/- FORCE_SCALE N*m).

Holds the rest pose but lifts three legs -- mid-left, front-right, back-right --
so the robot stands on the opposite tripod (front-left, back-left, mid-right).
The planted feet sit at bearings 30 / 150 / 270 deg, an equilateral triangle
around the center, so the stance is stable. With three feet down instead of six,
each planted foot now carries ~1/3 of the weight (~5.8 N) instead of ~1/6
(~2.9 N), which the contact-force arrows make visible.

While it stands, the femur actuator torque of every leg is recorded; closing the
viewer pops up a plot (planted legs solid, lifted legs dashed) and saves it to
runs/femur_forces.png. The plot is drawn *after* the viewer closes on purpose: a
concurrent matplotlib window fights the viewer for the macOS main thread.

Live / local only (the viewer needs a display -> mjpython on macOS):

    uv run mjpython examples/03_forces_hexapod.py
"""

import time
from collections import deque
from pathlib import Path

import mujoco
import numpy as np
from mujoco import viewer as mj_viewer

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "hexapod.xml"
OUT = ROOT / "runs" / "femur_forces.png"

LIFT = -0.6                                   # femur target (rad); negative raises the foot
RAMP = 0.6                                    # seconds to ramp the lift in (avoids a servo-force spike)
LEGS = ["FL", "ML", "BL", "BR", "MR", "FR"]   # <replicate> index 0..5 (CCW from the +x corner)
LIFTED_LEGS = {"ML", "FR", "BR"}              # which legs to lift (the rest form the tripod)
FEMURS = [f"femur{i}" for i in range(6)]      # femur actuator/joint name per leg index
PLANTED = [i for i, lab in enumerate(LEGS) if lab not in LIFTED_LEGS]   # planted leg indices
PLANTED_FEET = [f"foot{i}" for i in PLANTED]  # planted-foot geom names (LEGS[i] for the label)
REC_EVERY = 5                                 # record a femur-force sample every N sim steps
SPHERE_R = 0.043                              # radius of the femur-force marker spheres (m)
FORCE_SCALE = 1.0                             # |actuator torque| (N*m) that saturates the color


def geom_force_world(m, d, geom_id):
    """Net contact force on geom_id, in world coordinates (N)."""
    total = np.zeros(3)
    for i in range(d.ncon):
        c = d.contact[i]
        if geom_id not in (c.geom1, c.geom2):
            continue
        f = np.zeros(6)
        mujoco.mj_contactForce(m, d, i, f)
        fw = c.frame.reshape(3, 3).T @ f[:3]          # contact frame -> world
        total += fw if geom_id == c.geom2 else -fw    # force ON our geom
    return total


def draw_force_spheres(scn, anchors, fvals, scale, radius):
    """Overlay a sphere at each joint anchor, colored by actuator torque:
    negative -> red, ~0 -> white, positive -> blue (saturating at +/- scale)."""
    scn.ngeom = 0
    eye = np.eye(3).flatten()
    for pos, f in zip(anchors, fvals):
        t = float(np.clip(f / scale, -1.0, 1.0))
        if t >= 0.0:                                   # positive -> blue
            rgba = np.array([1.0 - t, 1.0 - t, 1.0, 1.0])
        else:                                          # negative -> red
            a = -t
            rgba = np.array([1.0, 1.0 - a, 1.0 - a, 1.0])
        mujoco.mjv_initGeom(
            scn.geoms[scn.ngeom],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([radius, 0.0, 0.0]),
            pos=np.ascontiguousarray(pos, dtype=float),
            mat=eye,
            rgba=rgba,
        )
        scn.ngeom += 1


def plot_femur(times, forces, out):
    """Plot femur actuator torque per leg; planted solid, lifted dashed."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, leg in enumerate(LEGS):
        lifted = leg in LIFTED_LEGS
        ax.plot(times, forces[:, i], "--" if lifted else "-",
                label=f"{leg} ({'lifted' if lifted else 'planted'})")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("femur actuator torque (N*m)")
    ax.set_title("Femur actuator torque per leg -- tripod stance")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"saved plot -> {out}")
    plt.show()


def main():
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    d = mujoco.MjData(m)

    # make the contact-force arrows visible (tune to taste, or adjust live in the viewer)
    m.vis.map.force = 0.02
    m.vis.scale.contactwidth = 0.03
    m.vis.scale.contactheight = 0.06
    m.vis.scale.forcewidth = 0.02

    aid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
    gid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
    jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)

    # all femurs (for logging) and just the lifted ones (ramped up in the loop)
    fem_ids = [aid(n) for n in FEMURS]
    fem_jids = [jid(n) for n in FEMURS]                     # joint == actuator name; xanchor -> world pos
    lifted_ids = [aid(f"femur{i}") for i, lab in enumerate(LEGS) if lab in LIFTED_LEGS]
    feet = [gid(n) for n in PLANTED_FEET]

    times, forces = deque(maxlen=6000), deque(maxlen=6000)   # ~2 min rolling buffer

    with mj_viewer.launch_passive(m, d) as viewer:
        from mujoco import mjtVisFlag as VF
        viewer.opt.flags[VF.mjVIS_CONTACTPOINT] = True
        viewer.opt.flags[VF.mjVIS_CONTACTFORCE] = True
        viewer.opt.flags[VF.mjVIS_CONTACTSPLIT] = True   # split normal vs friction

        print("standing on tripod " + " / ".join(LEGS[i] for i in PLANTED) +
              "; close the viewer (or Ctrl-C) to plot femur torques")
        step = 0
        try:
            while viewer.is_running():
                t0 = time.time()
                d.ctrl[lifted_ids] = LIFT * min(1.0, d.time / RAMP)   # ramp the lift in
                mujoco.mj_step(m, d)
                draw_force_spheres(viewer.user_scn, d.xanchor[fem_jids],
                                   d.actuator_force[fem_ids], FORCE_SCALE, SPHERE_R)
                viewer.sync()
                step += 1
                if step % REC_EVERY == 0:
                    times.append(d.time)
                    forces.append(d.actuator_force[fem_ids].copy())
                if step % 250 == 0:                          # ~ once per second
                    fz = [geom_force_world(m, d, g)[2] for g in feet]
                    print("planted Fz (N): " +
                          "  ".join(f"{LEGS[i]}={v:5.2f}" for i, v in zip(PLANTED, fz)) +
                          f"   sum={sum(fz):5.2f}")
                dt = m.opt.timestep - (time.time() - t0)
                if dt > 0:
                    time.sleep(dt)
        except KeyboardInterrupt:
            pass

    if times:                                              # viewer closed -> plot
        plot_femur(np.array(times), np.array(forces), OUT)


if __name__ == "__main__":
    main()
