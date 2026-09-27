"""Replay a saved state trajectory (.npz) in the MuJoCo passive viewer.

Shared by experiments that record qpos/qvel rollouts (the MPC examples, the PPO
sample episodes). The .npz carries `qpos`, `qvel`, `timestep`, and `model` (an
MJCF path relative to the repo root); `mj_forward` reconstructs every geom/site
pose from the state at playback, so the file stays tiny and replays at any
camera/resolution.

Local only: needs a display, and on macOS the passive viewer requires `mjpython`.
"""
import os
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import typer

_ROOT = Path(__file__).resolve().parents[2]   # src/controlkit/viz.py -> repo root


app = typer.Typer(
    add_completion=False, no_args_is_help=True,
    help="Plot a saved npz state trajectory (headless-safe). Replay lives at the "
         "top level: `ctk play <file>`.",
)


def play(
    file: str,
    model: str = typer.Option(None, help="MJCF path; default taken from the npz."),
    loop: bool = typer.Option(True, help="Restart playback when it reaches the end."),
    contacts: bool = typer.Option(True, help="Draw contact forces."),
    gravity: bool = typer.Option(True, help="Draw the gravity vector as an arrow."),
    plot_window: float = typer.Option(2.0, help="Seconds shown in the live plot (npz `plot`)."),
):
    """Replay an .npz trajectory (qpos/qvel/timestep, optional model) in the passive viewer.

    file:  path to the saved trajectory.
    model: MJCF path; if None, taken from the npz `model` field, resolved
           against the repo root.
    loop:  restart the replay when it reaches the end (until the window closes).
    contacts: draw contact forces (contact points are off).
    gravity: draw the gravity direction as an arrow, fixed above the robot's start.

    Optional npz fields: `ctrl` (T, nu) and `qfrc_applied` (T, nv) are applied per
    frame so the replayed constraint (contact) forces match the recording; `gravity` (T, 3) overrides the
    model's gravity per frame (runs that tilt gravity over time); `force_pos` /
    `force_vec` (T, K, 3) are K external forces per frame (application point, force
    in N), drawn as red arrows at `force_scale` m/N (optional scalar; default the
    model's contact-force scale, `vis.map.force`); `plot` (T, K) is drawn as a live
    line plot (bottom left, last `plot_window` seconds up to the current frame), with
    optional `plot_labels` (K,) and `plot_title`.

    Playback keys: space = pause/resume, left/right = step one frame (scrub while
    paused), up/down = slower/faster. The viewer's time readout tracks the frame.
    """
    from mujoco import viewer as mj_viewer

    # macOS passive viewer must run under `mjpython`; relaunch ourselves if needed.
    if sys.platform == "darwin" and getattr(mj_viewer, "_MJPYTHON", None) is None:
        mjpython = Path(sys.executable).with_name("mjpython")
        if not mjpython.exists():
            raise typer.BadParameter(
                "on macOS the viewer needs `mjpython`, which was not found next to "
                f"{sys.executable}. Install mujoco's viewer extra, or run "
                "`uv run mjpython -m controlkit.viz play ...`."
            )
        print(f"relaunching under mjpython for the macOS viewer: {mjpython}", flush=True)
        os.execv(str(mjpython), [str(mjpython), *sys.argv])

    file = Path(file)
    npz = np.load(file)
    qpos, qvel, dt = npz["qpos"], npz["qvel"], float(npz["timestep"])
    ctrl = npz["ctrl"] if "ctrl" in npz.files else None
    qfrc = npz["qfrc_applied"] if "qfrc_applied" in npz.files else None
    grav = npz["gravity"] if "gravity" in npz.files else None
    f_pos = npz["force_pos"] if "force_pos" in npz.files else None
    f_vec = npz["force_vec"] if "force_vec" in npz.files else None
    f_scale = float(npz["force_scale"]) if "force_scale" in npz.files else None
    plot = npz["plot"] if "plot" in npz.files else None
    fig = None
    if plot is not None:
        labels = [str(x) for x in npz["plot_labels"]] if "plot_labels" in npz.files \
            else [f"{k}" for k in range(plot.shape[1])]
        title = str(npz["plot_title"]) if "plot_title" in npz.files else ""
        fig = _make_figure(title, labels, float(np.max(plot)))
    n = len(qpos)
    model_path = Path(model) if model is not None else _ROOT / str(npz["model"])
    print(f"loaded {n} frames from {file} (dt={dt}s); model={model_path}")
    print("keys: space=pause/resume  left/right=step frame  up/down=speed")

    # GLFW key codes the passive viewer hands to key_callback.
    SPACE, RIGHT, LEFT, UP, DOWN = 32, 262, 263, 265, 264
    st = {"paused": False, "idx": 0, "step": 0, "speed": 1.0}

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

    mj_model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(mj_model)
    # fixed anchor for the gravity arrow: above the robot's COM in the first frame
    data.qpos[:] = qpos[0]
    mujoco.mj_forward(mj_model, data)
    arrow_origin = data.subtree_com[min(1, mj_model.nbody - 1)] + np.array([0.0, 0.0, 0.35])
    # contact arrows are forcewidth * meansize wide (model-dependent, often fat); pin
    # them to the drawn force arrows' width. Must be set before the viewer launches.
    mj_model.vis.scale.forcewidth = ARROW_WIDTH / mj_model.stat.meansize
    with mj_viewer.launch_passive(mj_model, data, key_callback=on_key) as viewer:
        if contacts:
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        while viewer.is_running():
            i = st["idx"] % n
            data.qpos[:] = qpos[i]
            data.qvel[:] = qvel[i]
            if ctrl is not None:
                data.ctrl[:] = ctrl[i]
            if qfrc is not None:
                data.qfrc_applied[:] = qfrc[i]
            if grav is not None:
                mj_model.opt.gravity[:] = grav[i]
            data.time = i * dt                       # viewer clock tracks the trajectory
            mujoco.mj_forward(mj_model, data)        # reconstruct poses (and forces)
            viewer.user_scn.ngeom = 0
            if gravity:
                _draw_gravity(viewer.user_scn, mj_model, arrow_origin)
            if f_pos is not None:
                _draw_forces(viewer.user_scn, mj_model, f_pos[i], f_vec[i], scale=f_scale)
            if fig is not None:
                _update_figure(fig, plot, i, dt, plot_window)
                vp = viewer.viewport
                w, h = (vp.width, vp.height) if vp is not None else (1200, 900)
                viewer.set_figures((mujoco.MjrRect(0, 0, w // 3, h // 3), fig))
            viewer.sync()

            if st["paused"]:
                if st["step"]:                       # one-shot scrub while paused
                    st["idx"], st["step"] = (i + st["step"]) % n, 0
                time.sleep(1 / 60)                   # idle redraw, stay responsive
                continue

            time.sleep(dt / st["speed"])             # real-time pacing, speed-scaled
            st["idx"] = i + 1
            if st["idx"] >= n:
                if not loop:
                    break
                st["idx"] = 0
                time.sleep(0.3)                      # brief pause, then loop



ARROW_WIDTH = 0.01   # m; drawn force arrows and (in replay) contact-force arrows


_LINE_RGB = [(0.95, 0.35, 0.3), (0.3, 0.75, 0.95), (0.4, 0.85, 0.4), (0.95, 0.8, 0.25),
             (0.8, 0.45, 0.95), (0.95, 0.55, 0.15)]


def _make_figure(title: str, labels, ymax: float):
    """An MjvFigure for a live line plot: one line per label, y fixed to [0, ymax]."""
    fig = mujoco.MjvFigure()
    mujoco.mjv_defaultFigure(fig)
    fig.title, fig.xlabel = title, "time (s)"
    fig.flg_legend, fig.flg_extend = 1, 0
    fig.figurergba[3] = 0.6
    fig.gridsize = [4, 4]
    fig.range[1] = [0.0, 1.1 * max(ymax, 1e-6)]
    for k, name in enumerate(labels):
        fig.linename[k] = name
        fig.linergb[k] = _LINE_RGB[k % len(_LINE_RGB)]
    return fig


def _update_figure(fig, data, i: int, dt: float, window: float) -> None:
    """Show ``data[:i+1]`` over the last ``window`` seconds (at most 1000 points)."""
    n = min(i + 1, max(2, int(window / dt)), 1000)
    t = (np.arange(i + 1 - n, i + 1)) * dt
    for k in range(data.shape[1]):
        fig.linedata[k, 0:2 * n:2] = t
        fig.linedata[k, 1:2 * n:2] = data[i + 1 - n:i + 1, k]
        fig.linepnt[k] = n
    fig.range[0] = [t[-1] - window, t[-1]]


def _add_arrow(scn, tail, tip, width: float, rgba) -> None:
    """Append one arrow geom (tail -> tip) to ``scn``, if there is room."""
    if scn.ngeom >= scn.maxgeom:
        return
    geom = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3),
                        np.zeros(9), np.asarray(rgba, dtype=np.float32))
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, width,
                         np.asarray(tail, dtype=float), np.asarray(tip, dtype=float))
    scn.ngeom += 1


def _draw_gravity(scn, model, origin, length: float = 0.2, width: float = 0.01):
    """Append one arrow along gravity to ``scn``, starting at ``origin``.

    The arrow only rotates (with ``model.opt.gravity``) about the fixed world point
    ``origin`` (its tail), so a tilting gravity reads as a turning arrow.
    """
    g = np.asarray(model.opt.gravity, dtype=float)
    norm = np.linalg.norm(g)
    if norm == 0.0:
        return
    origin = np.asarray(origin, dtype=float)
    _add_arrow(scn, origin, origin + length * g / norm, width, (1.0, 0.8, 0.1, 1.0))


def _draw_forces(scn, model, pos, vec, width: float = ARROW_WIDTH, scale: float = None):
    """Append one red arrow per force: from ``pos[k]``, length ``scale * |F|``.

    Zero forces are skipped. ``scale`` (m/N) defaults to the model's contact-force
    arrow scale, ``vis.map.force``.
    """
    if scale is None:
        scale = model.vis.map.force
    for p, f in zip(np.asarray(pos, float), np.asarray(vec, float)):
        if np.linalg.norm(f) > 1e-9:
            _add_arrow(scn, p, p + scale * f, width, (0.9, 0.1, 0.1, 1.0))


def plot_episodes(episodes, out=None, control_dt=1.0, cmd_vx=None, z_min=None, step=0):
    """Render a 4-panel base-state plot for a batch of sampled episodes.

    episodes: list of ``(qpos, qvel, kind)``; the 'deterministic' one is drawn
              bold, the rest light, so the exploration spread is visible.
    out:      path to save a PNG (headless-safe); if None, show interactively.
    cmd_vx / z_min: optional reference lines (commanded forward speed, fall height).
    Panels: top-down xy path, height z(t), forward velocity vx(t), forward x(t).
    """
    import matplotlib
    if out is not None:
        matplotlib.use("Agg")            # headless save (e.g. on the GPU box)
    import matplotlib.pyplot as plt

    fig, ((ax_xy, ax_z), (ax_vx, ax_x)) = plt.subplots(2, 2, figsize=(11, 7))
    for qpos, qvel, kind in episodes:
        qpos, qvel = np.asarray(qpos), np.asarray(qvel)
        t = np.arange(qpos.shape[0]) * control_dt
        style = (dict(color="black", lw=2.0, zorder=3) if kind == "deterministic"
                 else dict(lw=1.0, alpha=0.6))
        ax_xy.plot(qpos[:, 0], qpos[:, 1], **style)
        ax_z.plot(t, qpos[:, 2], **style)
        ax_vx.plot(t, qvel[:, 0], **style)
        ax_x.plot(t, qpos[:, 0], **style)

    ax_xy.scatter([0], [0], c="k", s=20, zorder=4)
    ax_xy.set(title="top-down path", xlabel="x [m]", ylabel="y [m]")
    ax_xy.axis("equal"); ax_xy.grid(alpha=0.3)
    if z_min is not None:
        ax_z.axhline(z_min, color="r", ls="--", lw=1, label=f"z_min={z_min:.2f}")
        ax_z.legend(loc="best")
    ax_z.set(title="trunk height", xlabel="t [s]", ylabel="z [m]")
    ax_z.grid(alpha=0.3)
    if cmd_vx is not None:
        ax_vx.axhline(cmd_vx, color="g", ls="--", lw=1, label=f"cmd vx={cmd_vx:.2f}")
        ax_vx.legend(loc="best")
    ax_vx.set(title="forward velocity", xlabel="t [s]", ylabel="vx [m/s]")
    ax_vx.grid(alpha=0.3)
    ax_x.set(title="forward position", xlabel="t [s]", ylabel="x [m]")
    ax_x.grid(alpha=0.3)

    fig.suptitle(f"sample episodes @ step {step:,}  "
                 "(black = deterministic, thin = stochastic)")
    fig.tight_layout()
    if out is not None:
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(f"saved plot -> {out}")
    else:
        plt.show()


@app.command()
def plot(files: list[str], out: str = None):
    """Plot base states from one or more sample-episode .npz files.

    Overlays the given episodes (deterministic bold, stochastic thin). With --out
    saves a PNG (headless-safe); otherwise opens an interactive window. Reads
    cmd_vx / z_min / kind / timestep from each npz when present.

        uv run python -m controlkit.viz plot <run>/results/sample_episode_005_*.npz
    """
    eps, dt, cmd_vx, z_min, step = [], 1.0, None, None, 0
    for f in files:
        npz = np.load(f)
        kind = str(npz["kind"]) if "kind" in npz.files else "deterministic"
        eps.append((npz["qpos"], npz["qvel"], kind))
        if "timestep" in npz.files:
            dt = float(npz["timestep"])
        if "cmd_vx" in npz.files:
            cmd_vx = float(npz["cmd_vx"])
        if "z_min" in npz.files:
            z_min = float(npz["z_min"])
        if "step" in npz.files:
            step = int(npz["step"])
    plot_episodes(eps, out=out, control_dt=dt, cmd_vx=cmd_vx, z_min=z_min, step=step)


if __name__ == "__main__":
    app()