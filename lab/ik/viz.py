"""Render hexapod configurations to images (offscreen, headless-safe).

Small helper for eyeballing a batch of ``qpos`` -- e.g. the reachable/stable body
poses from :mod:`lab.ik.stance`. Uses MuJoCo's offscreen ``Renderer`` (system GL
on macOS, no display needed) and tiles the frames into a PNG montage.

Output defaults to the repo's gitignored ``scratch/`` dir so images land
somewhere you can open but that never gets committed.

    from lab.ik.viz import render_montage
    render_montage(model, qpos)                 # -> scratch/poses_montage.png

Run ``uv run python -m lab.ik.viz`` for a demo (renders 100 stable poses).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[2]
SCRATCH = ROOT / "scratch"


def _default_cam():
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.0, 0.0, 0.10]
    cam.distance, cam.azimuth, cam.elevation = 0.95, 120.0, -18.0
    return cam


def no_sites_option():
    """MjvOption with site markers off (the ``<site>`` spheres). The foot *geoms*
    already show the feet, so the site markers are just clutter here."""
    opt = mujoco.MjvOption()
    opt.sitegroup[:] = 0
    return opt


def _fit_framebuffer(model, width, height):
    """Grow the model's offscreen framebuffer so a WxH render fits (default 640x480)."""
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)


def _leg_geom_ids(model, legs):
    """Geom ids belonging to the given legs (coxa/femur/tibia bodies, incl. foot)."""
    bodies = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{seg}{i}")
              for i in legs for seg in ("coxa", "femur", "tibia")}
    return {g for g in range(model.ngeom) if model.geom_bodyid[g] in bodies}


def render_frames(model, qpos, width=200, height=200, cam=None):
    """Render each ``qpos`` (a list/array of full configurations) to an RGB frame.

    Returns an ``(N, height, width, 3)`` uint8 array. ``cam`` is an optional
    ``mujoco.MjvCamera``; if None a sensible three-quarter view is used.
    """
    qpos = np.atleast_2d(np.asarray(qpos, float))
    cam = cam or _default_cam()
    _fit_framebuffer(model, width, height)
    renderer = mujoco.Renderer(model, height, width)
    opt = no_sites_option()
    data = mujoco.MjData(model)
    frames = np.empty((len(qpos), height, width, 3), np.uint8)
    for k, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, cam, opt)
        frames[k] = renderer.render()
    renderer.close()
    return frames


def tile(frames, cols=None, pad=3, bg=255):
    """Tile ``(N, H, W, 3)`` frames into one montage image (row-major)."""
    frames = np.asarray(frames)
    n, H, W, _ = frames.shape
    cols = cols or int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    grid = np.full((rows * H + (rows + 1) * pad, cols * W + (cols + 1) * pad, 3), bg, np.uint8)
    for k in range(n):
        i, j = divmod(k, cols)
        y, x = pad + i * (H + pad), pad + j * (W + pad)
        grid[y:y + H, x:x + W] = frames[k]
    return grid


def render_montage(model, qpos, out=None, cols=None, width=200, height=200, cam=None):
    """Render ``qpos`` batch and save a tiled montage PNG.

    ``out`` defaults to ``scratch/poses_montage.png``. Returns the output Path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.image as mpimg

    out = Path(out) if out is not None else SCRATCH / "poses_montage.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    grid = tile(render_frames(model, qpos, width, height, cam), cols=cols)
    mpimg.imsave(out, grid)
    return out


def overlay_frame(model, qpos, planted=None, hide_other_legs=False, alpha=0.28,
                  width=700, height=700, cam=None):
    """Render several configurations superimposed in one scene (ghosted robots).

    Each ``qpos`` is drawn as a semi-transparent robot in the same 3D scene (via
    ``mjv_addGeoms``), so overlapping is real geometry, not blended pixels -- good
    for seeing the body-motion envelope over a fixed planted tripod.

    ``alpha`` sets the per-robot transparency. If ``hide_other_legs`` is True the
    non-planted legs are hidden (needs ``planted``, the list of planted leg
    indices). Returns an ``(height, width, 3)`` uint8 image.
    """
    if hide_other_legs and planted is None:
        raise ValueError("hide_other_legs=True needs planted (which legs are kept)")
    qpos = np.atleast_2d(np.asarray(qpos, float))
    cam = cam or _default_cam()
    _fit_framebuffer(model, width, height)

    renderer = mujoco.Renderer(model, height, width)
    opt, pert = no_sites_option(), mujoco.MjvPerturb()
    data = mujoco.MjData(model)

    # First config seeds the scene (floor, lights, robot 1); the rest append only
    # their dynamic (robot) geoms so we don't stack duplicate floors.
    data.qpos[:] = qpos[0]
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, cam, opt)
    scn = renderer.scene
    for q in qpos[1:]:
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        mujoco.mjv_addGeoms(model, data, opt, pert, int(mujoco.mjtCatBit.mjCAT_DYNAMIC), scn)

    hide = _leg_geom_ids(model, [i for i in range(6) if i not in planted]) if hide_other_legs else set()
    # Robot geoms have a non-world body (id != 0); tint them, or hide swing legs.
    for k in range(scn.ngeom):
        g = scn.geoms[k]
        if g.objtype == mujoco.mjtObj.mjOBJ_GEOM and model.geom_bodyid[g.objid] != 0:
            g.rgba[3] = 0.0 if g.objid in hide else alpha

    img = renderer.render()
    renderer.close()
    return img


def render_overlay(model, qpos, out=None, planted=None, hide_other_legs=False,
                   alpha=0.28, width=700, height=700, cam=None):
    """Save an overlay of superimposed configurations. See :func:`overlay_frame`.

    ``out`` defaults to ``scratch/poses_overlay.png``. Returns the output Path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.image as mpimg

    out = Path(out) if out is not None else SCRATCH / "poses_overlay.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    img = overlay_frame(model, qpos, planted, hide_other_legs, alpha, width, height, cam)
    mpimg.imsave(out, img)
    return out


def _demo():
    from .stance import Stance, sample_reachable

    model = mujoco.MjModel.from_xml_path(str(ROOT / "models" / "hexapod.xml"))
    stance = Stance.from_model(model)
    planted = [0, 2, 4]

    home = stance.home_qpos
    base0 = np.array([home[0], home[1], home[2], 0.0, 0.0, 0.0])
    lo = base0 + np.array([-0.10, -0.10, -0.10, -0.35, -0.35, -0.6])
    hi = base0 + np.array([0.10, 0.10, 0.05, 0.35, 0.35, 0.6])

    poses, qpos = sample_reachable(stance, planted, lo, hi, n=8000, seed=1)
    margins = np.array([stance.stability(p, planted)[1] for p in poses])
    poses, qpos = poses[margins > 0], qpos[margins > 0]
    order = np.lexsort((poses[:, 0], poses[:, 5]))  # sweep by yaw then x
    qpos = qpos[order]

    out = render_montage(model, qpos[:100], cols=10)
    print(f"montage of 100 poses          -> {out}")
    out = render_overlay(model, qpos[:20], planted=planted)
    print(f"overlay of 20 poses           -> {out}")
    out = render_overlay(model, qpos[:20], planted=planted, hide_other_legs=True,
                         out=SCRATCH / "poses_overlay_tripod.png")
    print(f"overlay of 20, swing hidden   -> {out}")


if __name__ == "__main__":
    _demo()
