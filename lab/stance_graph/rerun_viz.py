"""Visualize the hexapod in rerun from a body pose + per-leg joint angles.

Simple primitives (no mesh): a flat cube for the base, line-strips for the leg
segments, spheres for the feet. World joint positions come from
:func:`lab.gg.kinematics.infer_joint_xpos`, so this is pure jax + rerun, no
MuJoCo.

Three concerns, kept separate:
  * ``init_viewer(...)`` -- pick the sink (connect / save / spawn) once.
  * ``log_robot(...)``   -- draw one configuration.
  * ``log_poses(...)``   -- draw N configurations on a scrubbable timeline.

Run under ``uv run --extra mjx``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import jax.numpy as jnp
import rerun as rr
from jaxlie import SE3

from lab.stance_graph.kinematics import infer_joint_xpos, ALL_PLANTED

FREE_COLOR = (90, 90, 90)         # leg segments
FOOT_FREE_COLOR = (120, 120, 120) # swing feet
FOOT_STANCE_COLOR = (255, 66, 249)  # planted feet


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
def set_time(i, *, timeline="t"):
    """Place subsequent logs at sequence index ``i`` on ``timeline``.

    Thin wrapper over ``rr.set_time`` so callers can drive their own timeline
    (e.g. interleaving ``log_body`` / ``log_robot`` per frame) without importing
    ``rerun`` directly.
    """
    rr.set_time(timeline, sequence=int(i))


def log_body(body: SE3, *, color=(200, 200, 200), entity="world", body_size=(0.2, 0.05, 0.0125)):
    """Draw just the base: a flat box at the ``body`` pose, no legs."""
    t = np.asarray(body.translation())
    q_xyzw = np.asarray(body.rotation().as_quaternion_xyzw())
    rr.log(f"{entity}/base", rr.Transform3D(translation=t, rotation=rr.Quaternion(xyzw=q_xyzw)))
    rr.log(f"{entity}/base/box",
           rr.Boxes3D(half_sizes=[np.array(body_size) / 2.0], fill_mode="solid",
                      colors=[color]))


def colors_from_values(values, *, cmap="viridis", vmin=None, vmax=None):
    """Map scalars ``values`` (N,) to (N, 3) uint8 RGB via a matplotlib colormap.

    ``vmin``/``vmax`` default to the data min/max (constant input -> all mid-map).
    """
    import matplotlib
    from matplotlib.colors import Normalize

    v = np.asarray(values, dtype=float).reshape(-1)
    lo = v.min() if vmin is None else vmin
    hi = v.max() if vmax is None else vmax
    rgba = matplotlib.colormaps[cmap](Normalize(lo, hi)(v))   # (N, 4) float in [0, 1]
    return (rgba[:, :3] * 255).astype(np.uint8)


def log_bodies(bodies: SE3, values=None, *, cmap="viridis", vmin=None, vmax=None,
               entity="bodies", body_size=(0.20, 0.20, 0.05)):
    """Draw a stack of body boxes (no legs) at one time, colored by ``values``.

    bodies : SE3 batch (N,)   base poses.
    values : (N,) or None     per-body scalar mapped through ``cmap`` to RGB
                              (None -> a flat grey).

    All boxes are logged as a single batched ``Boxes3D`` at ``entity``.
    """
    t = np.asarray(bodies.translation())                                 # (N, 3)
    q_xyzw = np.asarray(bodies.rotation().as_quaternion_xyzw())          # (N, 4)
    n = t.shape[0]
    if values is None:
        colors = np.tile(np.array([160, 160, 160], np.uint8), (n, 1))
    else:
        colors = colors_from_values(values, cmap=cmap, vmin=vmin, vmax=vmax)
    rr.log(entity, rr.Boxes3D(
        centers=t,
        half_sizes=np.tile(np.array(body_size) / 2.0, (n, 1)),
        quaternions=[rr.Quaternion(xyzw=q) for q in q_xyzw],
        colors=colors,
        fill_mode="solid",
    ))


def log_mesh(vertices, faces, *, color=(160, 160, 160), normals=None,
             edge_color=(40, 40, 40), edge_radius=0.002, entity="mesh"):
    """Draw a triangle mesh in a single flat color (e.g. a sampling surface).

    Args:
        vertices: (V, 3) mesh vertices.
        faces: (F, 3) int triangle vertex indices.
        color: RGB(A) uint8 color applied to the whole mesh.
        normals: optional (V, 3) vertex normals (for shading).
        edge_color: RGB(A) color for the triangle edges (wireframe overlay under
            ``{entity}/edges``); None to skip drawing edges.
        edge_radius: edge line radius (metres).
        entity: rerun entity path.
    """
    verts = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.uint32)
    color = np.asarray(color, dtype=np.uint8)
    rr.log(entity, rr.Mesh3D(
        vertex_positions=verts,
        triangle_indices=faces,
        vertex_normals=None if normals is None else np.asarray(normals, np.float32),
        vertex_colors=np.tile(color, (verts.shape[0], 1)),
    ))
    if edge_color is not None:
        e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
        e = np.unique(np.sort(e, axis=1), axis=0)              # unique undirected edges
        segs = verts[e]                                        # (E, 2, 3)
        ec = np.tile(np.asarray(edge_color, np.uint8), (segs.shape[0], 1))
        rr.log(f"{entity}/edges", rr.LineStrips3D(segs, radii=edge_radius, colors=ec))


def log_points(points, values=None, *, cmap="viridis", vmin=None, vmax=None,
               radius=0.01, color=(120, 120, 120), entity="points"):
    """Draw a point cloud, colored per-point by ``values`` through ``cmap``.

    Args:
        points: (N, 3) point positions.
        values: (N,) per-point scalars mapped through ``cmap``; None -> flat ``color``.
        cmap: matplotlib colormap name.
        vmin: lower color bound (default data min).
        vmax: upper color bound (default data max).
        radius: point radius (metres).
        color: flat RGB used when ``values`` is None.
        entity: rerun entity path.
    """
    pts = np.asarray(points, dtype=np.float32)
    if values is None:
        colors = np.tile(np.asarray(color, np.uint8), (pts.shape[0], 1))
    else:
        colors = colors_from_values(values, cmap=cmap, vmin=vmin, vmax=vmax)
    rr.log(entity, rr.Points3D(pts, radii=radius, colors=colors))


def log_spheres(centers, radii, values=None, *, cmap="viridis", vmin=None, vmax=None,
                color=(120, 120, 120), fill_mode="solid", entity="spheres"):
    """Draw solid spheres, colored per-sphere by ``values`` through ``cmap``.

    Args:
        centers: (N, 3) sphere centers.
        radii: (N,) or scalar sphere radii (metres).
        values: (N,) per-sphere scalars mapped through ``cmap``; None -> flat ``color``.
        cmap: matplotlib colormap name.
        vmin: lower color bound (default data min).
        vmax: upper color bound (default data max).
        color: flat RGB used when ``values`` is None.
        fill_mode: rerun fill mode ("solid" or "majorwireframe").
        entity: rerun entity path.
    """
    c = np.asarray(centers, dtype=np.float32)
    n = c.shape[0]
    r = np.broadcast_to(np.asarray(radii, np.float32), (n,))
    half = np.repeat(r[:, None], 3, axis=1)                   # sphere -> equal half-sizes
    if values is None:
        colors = np.tile(np.asarray(color, np.uint8), (n, 1))
    else:
        colors = colors_from_values(values, cmap=cmap, vmin=vmin, vmax=vmax)
    rr.log(entity, rr.Ellipsoids3D(centers=c, half_sizes=half, colors=colors,
                                   fill_mode=fill_mode))


def log_robot(body: SE3, theta, *, stance=ALL_PLANTED, only_stance=False,
              entity="world", body_size=(0.20, 0.20, 0.05),
              foot_radius=0.0125, leg_radius=0.005):
    """Draw one configuration under ``entity``.

    body        : SE3       world pose of the base.
    theta       : (6, 3)    joint angles [coxa, femur, tibia] per leg.
    stance      : (F,) int  ids (0..5) of the planted legs; their feet are drawn
                            in ``FOOT_STANCE_COLOR``, the rest in ``FOOT_FREE_COLOR``.
    only_stance : bool      if True, draw only the ``stance`` legs.
    """
    log_body(body, entity=entity, body_size=body_size)

    xpos = np.asarray(infer_joint_xpos(body, jnp.asarray(theta)))   # (6, 4, 3)
    stance = np.asarray(stance).astype(int).reshape(-1)
    planted = set(stance.tolist())
    legs = stance if only_stance else np.arange(6)

    for fid in legs:
        fid = int(fid)
        pts = xpos[fid]                                             # (4, 3): shoulder..foot
        rr.log(f"{entity}/leg{fid}",
               rr.LineStrips3D([pts], radii=leg_radius, colors=[FREE_COLOR]))
        rr.log(f"{entity}/leg{fid}/foot",
               rr.Points3D(pts[-1:], radii=foot_radius,
                           colors=[FOOT_STANCE_COLOR if fid in planted else FOOT_FREE_COLOR]))


def log_poses(bodies: SE3, thetas, *, timeline="pose", **kw):
    """Draw N configurations on a scrubbable timeline.

    bodies : SE3 batch (N,)   one base pose per frame.
    thetas : (N, 6, 3)        per-frame joint angles.

    Each frame is logged at time index i on ``timeline`` (scrub/play in the
    viewer). Extra kwargs (``stance``, ``only_stance``, ...) forward to
    :func:`log_robot`. For a simultaneous overlay instead, log to distinct
    ``entity`` paths.
    """
    thetas = jnp.asarray(thetas)
    for i in range(thetas.shape[0]):
        set_time(i, timeline=timeline)
        log_robot(bodies[i], thetas[i], **kw)


def log_trajectory(bodies: SE3, thetas, stances, *, timeline="t", **kw):
    """Draw a trajectory on a scrubbable timeline, with a per-frame stance.

    bodies  : SE3 batch (N,)   one base pose per frame.
    thetas  : (N, 6, 3)        per-frame joint angles.
    stances : (N, F) int, or a length-N sequence of (F_i,) int arrays -- the
              planted leg ids at each frame (F may vary per frame if a sequence).

    Extra kwargs (``only_stance``, ``entity``, ...) forward to :func:`log_robot`.
    """
    thetas = jnp.asarray(thetas)
    for i in range(thetas.shape[0]):
        set_time(i, timeline=timeline)
        log_robot(bodies[i], thetas[i], stance=stances[i], **kw)


def log_stack(bodies: SE3, thetas, stances=None, *, entity="stack", **kw):
    """Overlay N configurations at a single time, each on its own entity path.

    bodies  : SE3 batch (N,)   one base pose per robot.
    thetas  : (N, 6, 3)        per-robot joint angles.
    stances : (N, F) int / length-N sequence / None   per-robot planted leg ids
              (None -> all legs planted for every robot).

    Unlike :func:`log_trajectory`, all robots are drawn at once (distinct
    ``{entity}/{i}`` paths) rather than across a timeline. Extra kwargs
    (``only_stance``, ``body_size``, ...) forward to :func:`log_robot`.
    """
    thetas = jnp.asarray(thetas)
    for i in range(thetas.shape[0]):
        stance = ALL_PLANTED if stances is None else stances[i]
        log_robot(bodies[i], thetas[i], stance=stance, entity=f"{entity}/{i}", **kw)


def _demo(connect=8812):
    init_viewer(connect=connect)
    n = 8
    zs = jnp.linspace(0.18, 0.28, n)                                  # sweep body height
    bodies = SE3.from_translation(jnp.stack([jnp.zeros(n), jnp.zeros(n), zs], axis=-1))
    thetas = jnp.broadcast_to(jnp.array([0.0, 0.0523599, 1.46608]), (n, 6, 3))
    log_poses(bodies, thetas, stance=jnp.array([0, 2, 4]))
    print(f"streamed {n} poses -> rerun on port {connect}")


if __name__ == "__main__":
    _demo()
