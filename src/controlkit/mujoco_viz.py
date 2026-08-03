"""Minimal MuJoCo-viewer visualization of a :class:`~controlkit.kinematics.Robot`.

The MuJoCo-native counterpart to :mod:`controlkit.rerun_viz`: rather than logging
glyphs, it builds the robot's own MJCF (:meth:`Robot.to_mujoco`) and shows it in the
passive viewer at a given :class:`~controlkit.kinematics.Posture`.

Entry points:

- :func:`render` -- offscreen to RGB arrays. Works anywhere, including Jupyter and
  headless; this is the notebook path.
- :func:`show` -- the interactive passive viewer. Local only, needs a display, and on
  macOS requires ``mjpython`` (it relaunches under it), so it **cannot** run inside a
  Jupyter kernel -- use :func:`render` there. Pass ``support=`` to weld the planted
  feet and simulate the stance.
- ``ctk posture show FILE I`` -- the CLI over a saved
  ``np.savez(file, postures=..., supports=..., robot=...)`` archive. Saving the
  ``robot`` in the file makes it self-contained; otherwise pass ``--robot pkg:OBJ``.

    from controlkit import mujoco_viz
    img = mujoco_viz.render(robot, posture)               # (H,W,3) -> plt.imshow / mediapy
    mujoco_viz.show(robot, postures, foot_radius=0.025)   # window; scrub a batch
    mujoco_viz.show(robot, posture, support=support)      # weld planted feet, simulate
"""
import importlib
import os
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import typer


def _in_ipython() -> bool:
    """True inside any IPython/Jupyter shell, where :func:`show`'s process relaunch
    would replace (and kill) the kernel."""
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except Exception:
        return False


def load(file, key: str = "postures"):
    """Load a (batched) pytree saved with ``np.savez(file, <key>=obj)``.

    ``np.savez`` stores a registered dataclass (``Posture``, ``Support``, ...) as a
    pickled 0-d object array; this unwraps it. Index the result (``obj[i]``) for a
    single element.

    Args:
        file: path to the ``.npz``.
        key: array name inside the archive.

    Returns:
        The stored object (single or batched).
    """
    a = np.load(file, allow_pickle=True)[key]
    return a.item() if a.dtype == object and a.ndim == 0 else a


def to_qpos(postures) -> np.ndarray:
    """A ``Posture`` (single or batched) -> model ``qpos`` rows.

    Reorders the base pose from jaxlie's ``wxyz_xyz`` (quat-first) to MuJoCo's
    free-joint layout (position-first) and appends the flattened joint angles --
    matching the model :meth:`Robot.to_mujoco` emits.

    Args:
        postures: a ``Posture`` with body ``shape`` ``()`` or ``(N,)``.

    Returns:
        (N, nq) float array; ``N = 1`` for a single posture.
    """
    wx = np.asarray(postures.body.wxyz_xyz)                     # (..., 7) qw qx qy qz x y z
    th = np.asarray(postures.thetas)                           # (..., num_legs, num_joints)
    lead = wx.shape[:-1]
    free = np.concatenate([wx[..., 4:], wx[..., :4]], axis=-1)  # (..., 7) x y z qw qx qy qz
    q = np.concatenate([free, th.reshape(*lead, -1)], axis=-1)  # (..., nq)
    return q.reshape(-1, q.shape[-1])


def render(robot, postures, *, width: int = 640, height: int = 480, camera=-1,
           **mjcf_kwargs) -> np.ndarray:
    """Render posture(s) to RGB image array(s) offscreen -- the notebook path.

    Unlike :func:`show`, this opens no window and needs no ``mjpython``, so it works
    in Jupyter and headless. Display the result yourself, e.g. ``plt.imshow(img)`` or
    ``mediapy.show_image(img)`` / ``show_video(frames)``.

    Args:
        robot: the robot to draw.
        postures: a ``Posture`` (single or ``(N,)`` batched).
        width: image width in pixels.
        height: image height in pixels.
        camera: MuJoCo camera id/name, or -1 for the free camera.
        **mjcf_kwargs: forwarded to :meth:`Robot.to_mjcf`.

    Returns:
        ``(H, W, 3)`` uint8 for a single posture, or ``(N, H, W, 3)`` for a batch.
    """
    model = robot.to_mujoco(**mjcf_kwargs)
    # The exported model has no <light>; lean on the headlight and make sure it
    # actually lights the scene (a too-dim headlight renders near-black offscreen).
    model.vis.headlight.ambient[:] = 0.5
    model.vis.headlight.diffuse[:] = 0.7
    data = mujoco.MjData(model)
    qpos = to_qpos(postures)                                   # (N, nq)
    renderer = mujoco.Renderer(model, height, width)
    try:
        frames = []
        for q in qpos:
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            frames.append(renderer.render())
    finally:
        renderer.close()
    frames = np.stack(frames)
    return frames[0] if len(frames) == 1 else frames


def _weld_support(model, data, num_legs: int, qpos: np.ndarray, support):
    """Activate the ``foot{i}`` welds for the planted legs, pinned at ``qpos``.

    Places the robot at ``qpos``, reads each planted foot's world pose, and writes
    the weld target (``eq_data`` relpose = world-in-foot = the *inverse* foot pose,
    per :mod:`lab.stance_graph.mjx_weld`) before toggling ``eq_active``. Needs the
    model to carry the welds (``Robot.to_mjcf(weld_feet=True)``, the default).

    Args:
        model, data: the compiled model and its data.
        num_legs: leg count (``robot.num_legs``).
        qpos: (nq,) the posture to pin at.
        support: a ``Support`` (``.mask(num_legs)``) or a ``(num_legs,)`` bool mask.
    """
    mask = np.asarray(support.mask(num_legs) if hasattr(support, "mask") else support, bool)
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)                             # foot world poses at the posture
    for i in range(num_legs):
        eid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, f"weld_foot{i}")
        fid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"foot{i}")
        if eid < 0:                                            # model built without welds
            continue
        if bool(mask[i]):
            p, q = data.xpos[fid].copy(), data.xquat[fid].copy()   # foot world pose (wxyz)
            inv_q = np.zeros(4); mujoco.mju_negQuat(inv_q, q)      # conjugate = inverse rotation
            inv_p = np.zeros(3); mujoco.mju_rotVecQuat(inv_p, -p, inv_q)
            model.eq_data[eid, 3:6] = inv_p                    # relpose: world-in-foot
            model.eq_data[eid, 6:10] = inv_q
        data.eq_active[eid] = bool(mask[i])


def show(robot, postures, *, support=None, loop: bool = True, fps: float = 30.0,
         **mjcf_kwargs):
    """Open the MuJoCo passive viewer on ``robot`` at the given posture(s).

    Draws the posture(s) kinematically (``mj_forward`` -- the exact configuration,
    no dynamics). A single posture is held (start paused); a batch animates. Keys:
    space = pause/resume, left/right = step one frame (scrub while paused), up/down =
    slower/faster.

    With ``support`` (a single posture), the planted feet's welds are **activated**
    (:func:`_weld_support`), so the model is set up as that stance -- useful for
    handing to a dynamics pipeline. It is *not* simulated here: the standalone export
    (default geom density, no damping, no contact surface) does not settle cleanly;
    physical settling is :mod:`lab.stance_graph.mjx_weld`'s job, with a tuned model.

    On macOS this relaunches under ``mjpython``; it cannot run in a Jupyter kernel
    (use :func:`render` there).

    Args:
        robot: the robot to draw.
        postures: a ``Posture`` (single, or ``(N,)`` batched; single when ``support``).
        support: a ``Support`` / ``(num_legs,)`` bool mask whose planted feet to weld;
            or None.
        loop: restart the animation at the end (batch only).
        fps: playback frames per second.
        **mjcf_kwargs: forwarded to :meth:`Robot.to_mjcf`.
    """
    from mujoco import viewer as mj_viewer

    # In a Jupyter/IPython kernel the relaunch below would replace the kernel
    # process and kill it -- and the interactive viewer can't run in a kernel
    # anyway. Point at the offscreen path instead of crashing.
    if _in_ipython():
        raise RuntimeError(
            "mujoco_viz.show() opens an interactive window and cannot run inside a "
            "Jupyter/IPython kernel (it would relaunch under mjpython and kill the "
            "kernel). Use mujoco_viz.render(...) for inline images, or run show() "
            "from a plain script via `uv run mjpython your_script.py`."
        )

    # macOS passive viewer must run under `mjpython`; relaunch ourselves if needed.
    if sys.platform == "darwin" and getattr(mj_viewer, "_MJPYTHON", None) is None:
        mjpython = Path(sys.executable).with_name("mjpython")
        if not mjpython.exists():
            raise RuntimeError(
                "on macOS the viewer needs `mjpython`, not found next to "
                f"{sys.executable}. Run your script with `uv run mjpython ...`."
            )
        print(f"relaunching under mjpython for the macOS viewer: {mjpython}", flush=True)
        os.execv(str(mjpython), [str(mjpython), *sys.argv])

    model = robot.to_mujoco(**mjcf_kwargs)
    model.vis.headlight.ambient[:] = 0.5
    model.vis.headlight.diffuse[:] = 0.7
    data = mujoco.MjData(model)
    qpos = to_qpos(postures)                                   # (N, nq)

    if support is not None:                                    # activate planted-foot welds
        _weld_support(model, data, robot.num_legs, qpos[0], support)

    # --- kinematic view: draw / scrub the posture(s) ---
    n = len(qpos)
    print(f"showing {n} posture(s); keys: space=pause left/right=step up/down=speed")
    SPACE, RIGHT, LEFT, UP, DOWN = 32, 262, 263, 265, 264
    st = {"paused": n == 1, "idx": 0, "step": 0, "speed": 1.0}  # single frame -> hold

    def on_key(key):
        if key == SPACE:
            st["paused"] = not st["paused"]
        elif key == RIGHT:
            st["paused"], st["step"] = True, 1
        elif key == LEFT:
            st["paused"], st["step"] = True, -1
        elif key == UP:
            st["speed"] = min(st["speed"] * 1.5, 16.0)
        elif key == DOWN:
            st["speed"] = max(st["speed"] / 1.5, 1 / 16.0)

    with mj_viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        while viewer.is_running():
            i = st["idx"] % n
            data.qpos[:] = qpos[i]
            mujoco.mj_forward(model, data)                    # reconstruct poses for rendering
            viewer.sync()

            if st["paused"]:
                if st["step"]:                                # one-shot scrub while paused
                    st["idx"], st["step"] = (i + st["step"]) % n, 0
                time.sleep(1 / 60)
                continue

            time.sleep(1 / (fps * st["speed"]))
            st["idx"] = i + 1
            if st["idx"] >= n:
                if not loop:
                    break
                st["idx"] = 0
                time.sleep(0.3)


# --------------------------------------------------------------------------- #
# CLI: show a saved posture (with its support welded)                         #
# --------------------------------------------------------------------------- #
app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="Show saved postures in the MuJoCo viewer.")


def _import_object(spec: str):
    """``'package.module:ATTR'`` -> the named object."""
    module, _, attr = spec.partition(":")
    if not attr:
        raise typer.BadParameter("robot spec must be 'module:attribute'")
    return getattr(importlib.import_module(module), attr)


@app.command("show")
def _show_cmd(
    file: str = typer.Argument(..., help="npz from np.savez(file, postures=..., supports=..., robot=...)."),
    i: int = typer.Argument(..., help="Posture index to show."),
    robot: str = typer.Option(
        None, "--robot", "-r",
        help="Robot as 'module:attribute' (e.g. myrobots:QUAD_4DOF). Omit to use the "
             "'robot' saved in the file."),
    weld: bool = typer.Option(
        True, help="Weld the planted feet from supports[i]."),
):
    """Show posture ``i`` from ``file`` in the MuJoCo viewer, welding its support.

    The robot comes from ``--robot`` if given, else from a ``robot`` array saved in
    the file (``np.savez(file, postures=..., supports=..., robot=robot)``).
    """
    if robot is not None:
        r = _import_object(robot)
    else:
        try:
            r = load(file, "robot")
        except KeyError:
            raise typer.BadParameter(
                f"no 'robot' in {file}; pass --robot module:attribute, or save it with "
                "np.savez(..., robot=robot).")
    postures = load(file, "postures")
    support = load(file, "supports")[i] if weld else None
    print(f"{file}: {postures.shape[0]} postures; showing #{i}"
          + (" (support welded)" if weld else ""))
    show(r, postures[i], support=support)


if __name__ == "__main__":
    app()
