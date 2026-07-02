"""Visualize the hexapod in rerun from a body pose + per-leg joint angles.

Simple primitives (no mesh): a flat cube for the base, line-strips for the leg
segments, spheres for the feet. Geometry comes from :mod:`lab.ik.core`
(``SHOULDERS``/``LENGTHS``), so this is pure jax + rerun, no MuJoCo.

Three concerns, kept separate:
  * ``init_viewer(...)`` -- pick the sink (connect / save / spawn) once.
  * ``log_robot(...)``   -- draw one configuration.
  * ``log_poses(...)``   -- draw N configurations on a scrubbable timeline.

Run under ``uv run --extra mjx``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import rerun as rr

from lab.ik.core import SHOULDERS, LENGTHS, from_te
from jaxlie import SE3

ROOT = Path(__file__).resolve().parents[2]   # repo root, for CWD-independent paths


# --------------------------------------------------------------------------- #
# Connection / sink                                                           #
# --------------------------------------------------------------------------- #
def _grpc_url(connect) -> str:
    """gRPC url from a port int (localhost) or a full url string."""
    return connect if isinstance(connect, str) else f"rerun+http://127.0.0.1:{int(connect)}/proxy"


def init_viewer(app: str = "hexapod", *, spawn=True, save=None, connect=None):
    """Set up the rerun recording + sink. First set option wins:
    ``connect`` (port/url of a running viewer), ``save`` (.rrd path), ``spawn``.
    """
    rr.init(app)
    if connect is not None:
        rr.connect_grpc(_grpc_url(connect))
    elif save is not None:
        save = Path(save)
        save.parent.mkdir(parents=True, exist_ok=True)
        rr.save(save)
    elif spawn:
        rr.spawn()
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)   # z is up


# --------------------------------------------------------------------------- #
# Plotting                                                                    #
# --------------------------------------------------------------------------- #
def leg_points(shoulder: SE3, theta, lengths) -> jnp.ndarray:
    """World joint chain for one leg: (4, 3) points [shoulder, femur, tibia, foot]."""
    Lc, Lf, Lt = lengths
    coxa = from_te(jnp.zeros(3), jnp.array([0.0, 0.0, theta[0]]))
    femur = from_te(jnp.array([Lc, 0.0, 0.0]), jnp.array([0.0, theta[1], 0.0]))
    tibia = from_te(jnp.array([Lf, 0.0, 0.0]), jnp.array([0.0, theta[2], 0.0]))
    foot = from_te(jnp.array([Lt, 0.0, 0.0]), jnp.zeros(3))
    T0 = shoulder
    T1 = T0 @ coxa @ femur
    T2 = T1 @ tibia
    T3 = T2 @ foot
    return jnp.stack([T0.translation(), T1.translation(), T2.translation(), T3.translation()])


def log_robot(body_pose: SE3, theta, fids, *, entity="world",
              body_size=(0.30, 0.30, 0.05), foot_radius=0.025, leg_radius=0.01):
    """Draw one configuration under ``entity``.

    body_pose : SE3     world pose of the base.
    theta     : (k, 3)  joint angles [coxa, femur, tibia] for the shown legs.
    fids      : (k,)    leg ids (0..5) the rows of ``theta`` refer to.
    """
    t = np.asarray(body_pose.translation())
    q_xyzw = np.asarray(body_pose.rotation().as_quaternion_xyzw())
    rr.log(f"{entity}/base", rr.Transform3D(translation=t, rotation=rr.Quaternion(xyzw=q_xyzw)))
    rr.log(f"{entity}/base/box",
           rr.Boxes3D(half_sizes=[np.array(body_size) / 2.0], fill_mode="solid",
                      colors=[(200, 200, 200)]))

    shoulders = body_pose @ SHOULDERS                      # (6,) SE3
    theta = jnp.asarray(theta)
    fids = np.asarray(fids).astype(int)
    for row, fid in enumerate(fids):
        pts = np.asarray(leg_points(shoulders[int(fid)], theta[row], LENGTHS))
        rr.log(f"{entity}/leg{fid}", rr.LineStrips3D([pts], radii=leg_radius, colors=[(90, 90, 90)]))
        rr.log(f"{entity}/leg{fid}/foot",
               rr.Points3D(pts[-1:], radii=foot_radius, colors=[(255, 66, 249)]))


def log_poses(body_poses: SE3, thetas, fids, *, timeline="pose", **kw):
    """Draw N configurations on a scrubbable timeline.

    body_poses : SE3 batch (N,)   one base pose per frame.
    thetas     : (N, k, 3)        per-frame joint angles for the shown legs.
    fids       : (k,)             leg ids the columns of ``thetas`` refer to.

    Each pose is logged at time index i on ``timeline`` (scrub/play in the
    viewer). For a simultaneous overlay instead, log to distinct ``entity`` paths.
    """
    thetas = jnp.asarray(thetas)
    for i in range(thetas.shape[0]):
        rr.set_time(timeline, sequence=i)
        log_robot(body_poses[i], thetas[i], fids, **kw)


def _demo(connect=8812):
    init_viewer(connect=connect)
    n = 8
    zs = jnp.linspace(0.18, 0.28, n)                                  # sweep body height
    body_poses = SE3.from_translation(jnp.stack([jnp.zeros(n), jnp.zeros(n), zs], axis=-1))
    thetas = jnp.broadcast_to(jnp.array([0.0, 0.0523599, 1.46608]), (n, 6, 3))
    log_poses(body_poses, thetas, jnp.arange(6))
    print(f"streamed {n} poses -> rerun on port {connect}")


if __name__ == "__main__":
    _demo()
