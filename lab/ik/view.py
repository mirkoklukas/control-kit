"""CLI: open the interactive MuJoCo viewer with overlaid stable body poses.

Samples reachable + statically-stable body poses for a planted tripod (see
:mod:`lab.ik.stance`) and superimposes them as translucent "ghost" robots in the
live viewer -- the first pose solid, the rest translucent -- so you can orbit the
body-motion envelope over the fixed feet.

Needs a display. On macOS the passive viewer must run under ``mjpython``:

    uv run mjpython -m lab.ik.view --planted 0,2,4 --n 20 --hide-other-legs

Options: ``--n`` poses, ``--alpha`` ghost transparency, ``--hide-other-legs`` to
drop the non-planted legs, ``--seed``, ``--reach`` (box half-widths). Use
``--dry-run`` to build the scene and print stats without opening a window.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import mujoco
import mujoco.viewer

from .stance import Stance, sample_reachable
from .viz import _leg_geom_ids, no_sites_option


def _sample(stance, planted, n, seed, reach):
    """Return up to ``n`` reachable + stable qpos for the planted set."""
    home = stance.home_qpos
    base0 = np.array([home[0], home[1], home[2], 0.0, 0.0, 0.0])
    lo = base0 - np.array(reach)
    hi = base0 + np.array(reach)
    poses, qpos = sample_reachable(stance, planted, lo, hi, n=40 * n, seed=seed)
    if len(poses) == 0:
        return qpos
    stable = np.array([stance.stability(p, planted)[1] for p in poses]) > 0
    qpos = qpos[stable]
    order = np.lexsort((poses[stable][:, 0], poses[stable][:, 5]))  # yaw then x
    return qpos[order][:n]


def _hidden_geom_ids(model, planted, hide_other_legs):
    """Geoms to hide when ``hide_other_legs`` is set: the whole non-planted legs,
    including their feet. The planted legs (and their feet) stay visible."""
    if not hide_other_legs:
        return set()
    return _leg_geom_ids(model, [i for i in range(6) if i not in planted])


def _apply_hide(model, ids):
    """Make geoms invisible at the model level (geom_rgba alpha 0 + clear
    material), so both the live robot and the ``mjv_addGeoms`` ghosts drop them."""
    for g in ids:
        model.geom_rgba[g] = [0.5, 0.5, 0.5, 0.0]
        model.geom_matid[g] = -1


def _set_cam(cam):
    cam.lookat[:] = [0.0, 0.0, 0.10]
    cam.distance, cam.azimuth, cam.elevation = 1.1, 120.0, -18.0


def _populate_ghosts(scn, model, ghost_qpos, alpha, hide_ids):
    """Append each ghost qpos as robot geoms in ``scn`` and set transparency."""
    opt, pert = no_sites_option(), mujoco.MjvPerturb()
    tmp = mujoco.MjData(model)
    for q in ghost_qpos:
        tmp.qpos[:] = q
        mujoco.mj_forward(model, tmp)
        mujoco.mjv_addGeoms(model, tmp, opt, pert, int(mujoco.mjtCatBit.mjCAT_DYNAMIC), scn)
    for k in range(scn.ngeom):
        g = scn.geoms[k]
        if g.objtype == mujoco.mjtObj.mjOBJ_GEOM and model.geom_bodyid[g.objid] != 0:
            g.rgba[3] = 0.0 if g.objid in hide_ids else alpha


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--planted", default="0,2,4", help="planted leg indices, e.g. 0,2,4")
    p.add_argument("--n", type=int, default=20, help="number of poses")
    p.add_argument("--step", action="store_true",
                   help="cycle poses one at a time on the live robot (instead of overlaying)")
    p.add_argument("--dt", type=float, default=0.4, help="seconds per pose in --step mode")
    p.add_argument("--alpha", type=float, default=1.0, help="overlay transparency (1 = opaque)")
    p.add_argument("--hide-other-legs", action="store_true",
                   help="hide non-planted legs and all foot markers")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--reach", type=float, nargs=6, metavar=("X", "Y", "Z", "R", "P", "YAW"),
                   default=[0.10, 0.10, 0.10, 0.35, 0.35, 0.6],
                   help="pose-box half-widths around home (x y z roll pitch yaw)")
    p.add_argument("--model", default="models/hexapod.xml")
    p.add_argument("--dry-run", action="store_true", help="build scene, print stats, no window")
    args = p.parse_args(argv)

    planted = [int(i) for i in args.planted.split(",")]
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    model = mujoco.MjModel.from_xml_path(str(root / args.model))
    stance = Stance.from_model(model)

    qpos = _sample(stance, planted, args.n, args.seed, args.reach)
    mode = "stepping" if args.step else "overlaying"
    print(f"planted {planted}: {mode} {len(qpos)} reachable+stable poses")
    if len(qpos) == 0:
        return

    hide_ids = _hidden_geom_ids(model, planted, args.hide_other_legs)
    _apply_hide(model, hide_ids)

    data = mujoco.MjData(model)
    data.qpos[:] = qpos[0]
    mujoco.mj_forward(model, data)

    if args.dry_run:  # headless verification path (no window)
        if args.step:
            print(f"dry-run OK: step mode, {len(qpos)} poses @ {args.dt}s")
            return
        scn = mujoco.MjvScene(model, mujoco.viewer._Simulate.MAX_GEOM)
        mujoco.mjv_updateScene(model, data, no_sites_option(), None,
                               mujoco.MjvCamera(), mujoco.mjtCatBit.mjCAT_ALL, scn)
        _populate_ghosts(scn, model, qpos[1:], args.alpha, hide_ids)
        print(f"dry-run OK: scene has {scn.ngeom} geoms (cap {scn.maxgeom})")
        return

    with mujoco.viewer.launch_passive(model, data) as viewer:
        _set_cam(viewer.cam)
        viewer.opt.sitegroup[:] = 0  # no site markers on the live robot either
        if args.step:
            k = 0  # cycle poses on the single live robot, looping
            while viewer.is_running():
                data.qpos[:] = qpos[k % len(qpos)]
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(args.dt)
                k += 1
        else:
            _populate_ghosts(viewer.user_scn, model, qpos[1:], args.alpha, hide_ids)
            viewer.sync()
            while viewer.is_running():
                time.sleep(0.02)


if __name__ == "__main__":
    main()
