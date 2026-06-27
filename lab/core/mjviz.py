"""Replay a saved state trajectory (.npz) in the MuJoCo passive viewer.

Shared by experiments that record qpos/qvel rollouts (the MPC examples, the PPO
sample episodes). The .npz carries `qpos`, `qvel`, `timestep`, and `model` (an
MJCF path relative to the repo root); `mj_forward` reconstructs every geom/site
pose from the state at playback, so the file stays tiny and replays at any
camera/resolution.

Local only: needs a display, and on macOS the passive viewer requires `mjpython`.
"""
import time
from pathlib import Path

import mujoco
import numpy as np
import typer

_ROOT = Path(__file__).resolve().parents[2]   # lab/core/mjviz.py -> repo root


app = typer.Typer(
    add_completion=False, no_args_is_help=True,
    help="Replay (play) or plot a saved npz state trajectory. The viewer is local "
         "only: needs a display, and on macOS requires `mjpython`; the plot is "
         "headless-safe. Example: uv run mjpython -m lab.core.mjviz play runs/hexapod.npz",
)

@app.command()
def play(file, model=None, loop=True):
    """Replay an .npz trajectory (qpos/qvel/timestep, optional model) in the passive viewer.

    file:  path to the saved trajectory.
    model: MJCF path; if None, taken from the npz `model` field, resolved
           against the repo root.
    loop:  restart the replay when it reaches the end (until the window closes).
    """
    from mujoco import viewer as mj_viewer

    file = Path(file)
    npz = np.load(file)
    qpos, qvel, dt = npz["qpos"], npz["qvel"], float(npz["timestep"])
    model_path = Path(model) if model is not None else _ROOT / str(npz["model"])
    print(f"loaded {len(qpos)} frames from {file} (dt={dt}s); model={model_path}")

    mj_model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(mj_model)
    with mj_viewer.launch_passive(mj_model, data) as viewer:
        while viewer.is_running():
            for q, v in zip(qpos, qvel):
                if not viewer.is_running():
                    break
                data.qpos[:] = q
                data.qvel[:] = v
                mujoco.mj_forward(mj_model, data)   # reconstruct poses for rendering
                viewer.sync()
                time.sleep(dt)                       # real-time pacing
            if not loop:
                break
            time.sleep(0.5)                          # pause, then loop the replay



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

        uv run python -m lab.core.mjviz plot <run>/results/sample_episode_005_*.npz
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