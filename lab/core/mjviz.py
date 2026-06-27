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

_ROOT = Path(__file__).resolve().parents[2]   # lab/core/mjviz.py -> repo root


def play(file, model=None, loop=True):
    """Replay an .npz trajectory (qpos/qvel/timestep[/model]) in the passive viewer.

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
