"""Browse scored postures in MuJoCo: up / down switch between the good and the bad ones,
left / right step through them.

Reads an ``.npz`` written by :mod:`.experiments.scores` (frames best first, ``plot`` = the
scores per frame) and shows one posture at a time:

- **up**: the good list, from the best posture down;
- **down**: the bad list, from the worst posture up (each list remembers where you were);
- **left / right**: the previous / next posture in the current list;
- **, / .**: 10 back / 10 ahead (the viewer reports no modifier keys, so no shift + arrow);
- **esc**: close.

The current posture's rank and scores are shown top left in the viewer and printed in
the terminal. If the file has foot forces (``foot_force``, from the scores experiment),
each planted foot gets, drawn in force space at its contact point (the pad's face
centre on the surface, ``pad_drop`` below the ankle pivot along the normal;
``FORCE_SCALE`` m per N):

- an arrow: the force surface and magnet exert on it, **green** where it pushes (normal
  force > 0), **red** where it pulls (< 0, the magnet holds it);
- its friction cone with adhesion (wireframe): around the surface normal, half-angle
  arctan(mu), tip at ``-A n`` (the magnet's full pull behind the surface), drawn up to a
  push of ``CONE_REACH A``. A force inside it holds; at its mantle the foot slips. The
  cone is large against the forces: each magnet holds ~1.45 x the robot's weight, and
  three feet share that weight.

The terminal lists each foot's normal force N and sideways force T. Terrain boxes other
than the ground (the wall) are drawn as wireframes, so the cones behind them stay
visible; MuJoCo's own contact-force arrows are off. Run from ``lab/gait_graph/planner`` (macOS: relaunches itself under
``mjpython`` for the viewer):

    uv run --extra mjx python -m lab.gait_graph.planner.view experiments/runs/planner_scores/<run>/out/wall_lifted_samples.npz
"""
import os
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
from mujoco import viewer as mj_viewer

REPO = Path(__file__).resolve().parents[3]
UP, DOWN, LEFT, RIGHT, ESC = 265, 264, 263, 262, 256      # GLFW key codes
COMMA, PERIOD = 44, 46
FORCE_SCALE = 0.02                                        # force space: m per N (10 N = 20 cm)
ARROW_WIDTH = 0.008
PUSH, PULL = (0.1, 0.8, 0.2, 1.0), (0.9, 0.15, 0.1, 1.0)
CONE = (0.3, 0.5, 0.9, 0.8)                               # friction cone wireframe
CONE_RAYS = 16                                            # generator lines per cone
CONE_REACH = 0.5                                          # cone drawn up to N = CONE_REACH * A
WIRE = (0.6, 0.6, 0.65, 1.0)                              # wall wireframe
JUMP = 10                                                 # , / . step this many


def _relaunch_under_mjpython():
    """macOS: the passive viewer needs ``mjpython``; re-run this module under it."""
    if sys.platform == "darwin" and getattr(mj_viewer, "_MJPYTHON", None) is None:
        mjpython = Path(sys.executable).with_name("mjpython")
        if not mjpython.exists():
            raise SystemExit("the viewer needs `mjpython` next to python; not found")
        os.execv(str(mjpython), [str(mjpython), "-m", __spec__.name, *sys.argv[1:]])


def main(argv):
    if not argv:
        raise SystemExit(__doc__)
    _relaunch_under_mjpython()
    path = Path(argv[0])
    npz = np.load(path)
    qpos = npz["qpos"]
    scores = npz["plot"] if "plot" in npz.files else np.zeros((len(qpos), 0))
    labels = [str(x) for x in npz["plot_labels"]] if "plot_labels" in npz.files else []
    title = str(npz["plot_title"]) if "plot_title" in npz.files else path.stem
    forces = npz["foot_force"] if "foot_force" in npz.files else None       # (T, L, 3)
    normals = npz["foot_normal"] if "foot_normal" in npz.files else None    # (T, L, 3)
    planted = npz["foot_planted"] if "foot_planted" in npz.files else None  # (L,)
    adhesion = float(npz["adhesion"]) if "adhesion" in npz.files else 40.0  # older runs
    mu = float(npz["mu"]) if "mu" in npz.files else 0.5
    pad_drop = float(npz["pad_drop"]) if "pad_drop" in npz.files else 0.011
    model_path = Path(str(npz["model"]))
    model_path = model_path if model_path.is_absolute() else REPO / model_path
    n = len(qpos)

    m = mujoco.MjModel.from_xml_path(str(model_path))
    d = mujoco.MjData(m)
    feet = [m.body(f"foot{i}").id for i in range(forces.shape[1])] if forces is not None else []
    # terrain boxes (the wall): invisible, drawn as wireframes (line boxes) instead
    walls = [g for g in range(m.ngeom) if m.geom(g).name.startswith("terrain")]
    m.geom_rgba[walls, 3] = 0.0
    # the current list, and the position in each list (kept when switching lists)
    st = {"list": "good", "pos": {"good": 0, "bad": 0}, "dirty": True, "quit": False}

    def step(k):
        pos = st["pos"]
        pos[st["list"]] = int(np.clip(pos[st["list"]] + k, 0, n - 1))
        st["dirty"] = True

    def on_key(key):
        if key == UP:
            st.update(list="good", dirty=True)
        elif key == DOWN:
            st.update(list="bad", dirty=True)
        elif key == RIGHT:
            step(1)
        elif key == LEFT:
            step(-1)
        elif key == PERIOD:
            step(JUMP)
        elif key == COMMA:
            step(-JUMP)
        elif key == ESC:
            st["quit"] = True

    print(f"{title}: {n} postures (frames best first) from {path}\n"
          f"keys: up = good list, down = bad list, left / right = previous / next, "
          f", / . = 10 back / ahead, esc = close")
    with mj_viewer.launch_passive(m, d, key_callback=on_key) as v:
        v.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
        while v.is_running() and not st["quit"]:
            if st["dirty"]:
                st["dirty"] = False
                which, i = st["list"], st["pos"][st["list"]]
                frame = i if which == "good" else n - 1 - i
                d.qpos[:] = qpos[frame]
                mujoco.mj_forward(m, d)
                head = f"{title}  [{which} {i + 1}/{n}, rank {frame + 1}]"
                vals = "  ".join(f"{l} {s:.2f}" for l, s in zip(labels, scores[frame]))
                print(f"{head}  {vals}", flush=True)
                left = "\n".join([title, "list", "rank", *labels])
                right = "\n".join(["", f"{which} {i + 1}/{n}", f"{frame + 1}/{n}",
                                    *(f"{s:.2f}" for s in scores[frame])])
                v.set_texts((mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                             left, right))
                v.user_scn.ngeom = 0
                draw_walls(v.user_scn, m, d, walls)
                if forces is not None:
                    draw_forces(v.user_scn, d, feet, forces[frame], normals[frame], planted,
                                adhesion, mu, pad_drop)
            v.sync()
            time.sleep(0.02)                      # nothing moves between key presses


def _add(scn, kind, rgba):
    """A fresh geom of ``kind`` in the user scene, or None if it is full."""
    if scn.ngeom >= scn.maxgeom:
        return None
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, kind, np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                        np.array(rgba, np.float32))
    scn.ngeom += 1
    return g


def _line(scn, p0, p1, rgba, width=1.5):
    g = _add(scn, mujoco.mjtGeom.mjGEOM_LINE, rgba)
    if g is not None:
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, width, p0, p1)


def draw_walls(scn, m, d, walls):
    """The terrain boxes as line boxes (their edges only)."""
    for gid in walls:
        g = _add(scn, mujoco.mjtGeom.mjGEOM_LINEBOX, WIRE)
        if g is not None:
            g.size[:] = m.geom_size[gid]
            g.pos[:] = d.geom_xpos[gid]
            g.mat[:] = d.geom_xmat[gid].reshape(3, 3)


def draw_cone(scn, origin, n, adhesion, mu):
    """The friction cone with adhesion in force space at ``origin``: tip at ``origin -
    A n`` (scaled), opening arctan(mu) about n, drawn up to N = CONE_REACH A."""
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    t1 = np.cross(n, a)
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    tip = origin - FORCE_SCALE * adhesion * n
    h = FORCE_SCALE * adhesion * (1.0 + CONE_REACH)              # tip to rim, along n
    rim = [tip + h * n + mu * h * (np.cos(a) * t1 + np.sin(a) * t2)
           for a in 2 * np.pi * np.arange(CONE_RAYS) / CONE_RAYS]
    for k, p in enumerate(rim):
        _line(scn, tip, p, CONE)
        _line(scn, p, rim[(k + 1) % CONE_RAYS], CONE)
    _line(scn, tip, tip + h * n, CONE, width=1.0)               # the axis (the normal)


def draw_forces(scn, d, feet, F, normals, planted, adhesion, mu, pad_drop):
    """Per planted foot: its friction cone and its force arrow (green: pushes, red:
    pulls), in force space at its contact point (``pad_drop`` below the ankle pivot along
    the normal: the pad's face centre). Prints N and T per foot."""
    rows = []
    for i, body in enumerate(feet):
        if planted is not None and not planted[i]:
            rows.append(f"foot {i}: lifted")
            continue
        n = normals[i] / np.linalg.norm(normals[i])
        N = float(F[i] @ n)
        T = float(np.linalg.norm(F[i] - N * n))
        rows.append(f"foot {i}: N {N:+6.1f} N ({'push' if N >= 0 else 'pull'}), T {T:5.1f} N")
        p0 = d.xpos[body] - pad_drop * n                         # the contact point
        draw_cone(scn, p0, n, adhesion, mu)
        if np.linalg.norm(F[i]) > 1e-6:
            g = _add(scn, mujoco.mjtGeom.mjGEOM_ARROW, PUSH if N >= 0 else PULL)
            if g is not None:
                mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, ARROW_WIDTH, p0,
                                     p0 + FORCE_SCALE * F[i])
    print("    " + "   ".join(rows), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
