"""Step through the key postures of a transfer (or several plans of it), no animation.

Reads ``.npz`` files with ``key_qpos`` (the key postures), ``key_phase`` (their names) and
``key_support`` (per posture: which feet support it), as written by
:mod:`.experiments.refine` (``before.npz``, ``after.npz``): B (tripod T), B_plant (T),
B_plant planted (the full stance S), B_lift (T'), B_lift with the lifting leg up (T'), B'
(T', the next node; the leg still up). Per posture
it draws, on the ground:

- the **support polygon**: the supporting feet's contact points (``pad_drop`` below the
  ankle pivots) joined, **green** for a tripod, **blue** for the full stance;
- the **centre of mass**, projected straight down: a vertical line and a marker, **white**
  inside the support polygon, **red** outside.

Keys: **left / right** previous / next posture (wrapping around), **up / down** previous / next file, **esc**
close. Run from ``lab/gait_graph/planner`` (macOS: relaunches itself under ``mjpython``):

    uv run --extra mjx python -m lab.gait_graph.planner.view_transfer experiments/runs/planner_refine/latest/out/before.npz experiments/runs/planner_refine/latest/out/after.npz
"""
import os
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
from mujoco import viewer as mj_viewer

from .view import ESC, LEFT, REPO, RIGHT, UP, DOWN, _add, _line

TRIPOD, FULL = (0.1, 0.8, 0.2, 1.0), (0.2, 0.45, 0.95, 1.0)
INSIDE, OUTSIDE = (1.0, 1.0, 1.0, 1.0), (0.9, 0.15, 0.1, 1.0)


def _relaunch_under_mjpython():
    """macOS: the passive viewer needs ``mjpython``; re-run this module (not :mod:`.view`)
    under it."""
    if sys.platform == "darwin" and getattr(mj_viewer, "_MJPYTHON", None) is None:
        mjpython = Path(sys.executable).with_name("mjpython")
        if not mjpython.exists():
            raise SystemExit("the viewer needs `mjpython` next to python; not found")
        os.execv(str(mjpython), [str(mjpython), "-m", __spec__.name, *sys.argv[1:]])


def main(argv):
    if not argv:
        raise SystemExit(__doc__)
    _relaunch_under_mjpython()
    files = [Path(a) for a in argv]
    data = [np.load(f) for f in files]
    model_path = Path(str(data[0]["model"]))
    model_path = model_path if model_path.is_absolute() else REPO / model_path
    pad_drop = float(data[0]["pad_drop"]) if "pad_drop" in data[0].files else 0.011
    m = mujoco.MjModel.from_xml_path(str(model_path))
    d = mujoco.MjData(m)
    L = data[0]["key_support"].shape[1]
    feet = [m.body(f"foot{i}").id for i in range(L)]
    st = {"file": 0, "k": 0, "dirty": True, "quit": False}

    def on_key(key):
        n = len(data[st["file"]]["key_qpos"])
        if key == RIGHT:
            st["k"] = (st["k"] + 1) % n                        # wraps: last -> first
        elif key == LEFT:
            st["k"] = (st["k"] - 1) % n                        # first -> last
        elif key == DOWN:
            st["file"] = min(st["file"] + 1, len(data) - 1)
        elif key == UP:
            st["file"] = max(st["file"] - 1, 0)
        elif key == ESC:
            st["quit"] = True
        st["dirty"] = True

    print(f"{len(files)} file(s): " + ", ".join(f.name for f in files)
          + "\nkeys: left / right = posture, up / down = file, esc = close")
    with mj_viewer.launch_passive(m, d, key_callback=on_key) as v:
        v.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
        while v.is_running() and not st["quit"]:
            if st["dirty"]:
                st["dirty"] = False
                z = data[st["file"]]
                k = min(st["k"], len(z["key_qpos"]) - 1)
                d.qpos[:] = z["key_qpos"][k]
                mujoco.mj_forward(m, d)
                support = np.asarray(z["key_support"][k], bool)
                phase = str(z["key_phase"][k])
                v.user_scn.ngeom = 0
                inside = draw_support(v.user_scn, d, feet, support, pad_drop)
                legs = " ".join(str(j) for j in np.nonzero(support)[0])
                what = "full stance" if support.all() else "tripod"
                v.set_texts((mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                             "file\nposture\nsupport\ncentre of mass",
                             f"{files[st['file']].stem}\n{k + 1}/{len(z['key_qpos'])} {phase}\n"
                             f"{what} (legs {legs})\n{'inside' if inside else 'OUTSIDE'}"))
                print(f"{files[st['file']].stem}: {k + 1} {phase}, {what} legs {legs}, "
                      f"centre of mass {'inside' if inside else 'outside'} the support polygon",
                      flush=True)
            v.sync()
            time.sleep(0.02)


def draw_support(scn, d, feet, support, pad_drop):
    """The support polygon of the supporting feet on the ground and the projected centre of
    mass. Returns whether the centre of mass lies inside the polygon (xy)."""
    pts = np.array([d.xpos[feet[j]] - np.array([0, 0, pad_drop]) for j in np.nonzero(support)[0]])
    pts[:, 2] += 0.002                                         # just above the ground
    c = pts[:, :2].mean(0)
    order = np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))
    poly = pts[order]
    color = FULL if support.all() else TRIPOD
    for a in range(len(poly)):
        _line(scn, poly[a], poly[(a + 1) % len(poly)], color, width=4.0)
    com = d.subtree_com[1].copy()                              # body 1: the whole robot
    ground = np.array([com[0], com[1], poly[:, 2].mean()])
    inside = _inside(ground[:2], poly[:, :2])
    col = INSIDE if inside else OUTSIDE
    _line(scn, com, ground, col, width=2.0)
    g = _add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, col)
    if g is not None:
        g.size[:] = 0.008
        g.pos[:] = ground
    return inside


def _inside(p, poly):
    """Whether 2D point ``p`` is inside the convex polygon ``poly`` (counterclockwise)."""
    for a in range(len(poly)):
        e = poly[(a + 1) % len(poly)] - poly[a]
        w = p - poly[a]
        if e[0] * w[1] - e[1] * w[0] < 0:
            return False
    return True


if __name__ == "__main__":
    main(sys.argv[1:])
