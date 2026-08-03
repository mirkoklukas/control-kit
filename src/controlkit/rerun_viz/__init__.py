"""Visualize a robot in rerun, from a body pose plus per-leg joint angles.

Simple primitives, no meshes from disk: a prism for the body, disks and boxes for
the joints and links, spheres for the feet. Joint frames come from
``controlkit.kinematics`` -- pure jax + rerun, no MuJoCo.

Layering, bottom up -- each layer knows nothing of the one above:

- ``colors``, ``shapes``  pure numpy. No rerun; these lift out if a second
  drawing backend ever wants the same palette and geometry.
- ``draw``     rerun primitives in world coordinates. Knows nothing of robots.
- ``robot``    body, legs, postures. Takes a ``Robot`` explicitly, so a
  quadruped and a hexapod are the same call with a different argument.
- ``session``  where the logs go (connect / save / spawn), picked once.

Every ``log_*`` takes the rerun entity ``path`` first, as ``rr.log(path, ...)``.

Run under ``uv run --extra mjx``.
"""

from controlkit.rerun_viz import colors, shapes
from controlkit.rerun_viz import draw, robot, session
from controlkit.rerun_viz.colors import colors_from_values
from controlkit.rerun_viz.draw import (
    log_box,
    log_boxes,
    log_contacts,
    log_joint,
    log_mesh,
    log_points,
    log_poses,
    log_spheres,
)
from controlkit.rerun_viz.robot import log_body, log_leg, log_posture
from controlkit.rerun_viz.session import init_viewer, set_time

__all__ = [
    "colors",
    "shapes",
    "draw",
    "robot",
    "session",
    "colors_from_values",
    "init_viewer",
    "set_time",
    "log_mesh",
    "log_points",
    "log_spheres",
    "log_contacts",
    "log_joint",
    "log_box",
    "log_boxes",
    "log_poses",
    "log_body",
    "log_leg",
    "log_posture",
]
