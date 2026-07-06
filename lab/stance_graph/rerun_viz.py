"""Visualize the hexapod in rerun from a body pose + per-leg joint angles.

Simple primitives (no mesh): a flat cube for the base, line-strips for the leg
segments, spheres for the feet. World joint positions come from
:func:`lab.gg.kinematics.infer_joint_xpos`, so this is pure jax + rerun, no
MuJoCo.

Three concerns, kept separate:
  * ``init_viewer(...)`` -- pick the sink (connect / save / spawn) once.
  * ``log_robot(...)``   -- draw one configuration.
  * ``log_poses(...)``   -- draw N configurations on a scrubbable timeline.

Every ``log_*`` helper takes the rerun entity ``path`` as its first argument
(as with ``rr.log(path, ...)``, e.g. ``"world/base/box"``).

Run under ``uv run --extra mjx``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import jax.numpy as jnp
import rerun as rr
from jaxlie import SE3

from lab.stance_graph.kinematics import infer_joint_xpos

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


def log_body(path, body: SE3, *, color=(200, 200, 200), body_size=(0.2, 0.05, 0.0125)):
    """Draw just the base: a flat box at the ``body`` pose, no legs."""
    t = np.asarray(body.translation())
    q_xyzw = np.asarray(body.rotation().as_quaternion_xyzw())
    rr.log(f"{path}/base", rr.Transform3D(translation=t, rotation=rr.Quaternion(xyzw=q_xyzw)))
    rr.log(f"{path}/base/box",
           rr.Boxes3D(half_sizes=[np.array(body_size) / 2.0], fill_mode="solid",
                      colors=[color]))


def _hex_prism(radius=0.15, thickness=0.05):
    """Vertices/faces of a hexagonal prism (the robot body), centred at origin.

    Corners are at angles 30, 90, ... 330 deg and circumradius ``radius`` --
    matching the ``hexbody`` mesh (and the shoulders) in the weld0 model.

    Args:
        radius: hexagon circumradius (metres).
        thickness: prism thickness in z (metres).

    Returns:
        vertices: (12, 3) prism vertices (top hexagon then bottom hexagon).
        faces: (20, 3) int triangle indices (2 caps + 6 side quads).
    """
    ang = np.deg2rad(np.arange(30.0, 360.0, 60.0))
    hx, hy = radius * np.cos(ang), radius * np.sin(ang)
    h = thickness / 2.0
    top = np.stack([hx, hy, np.full(6, h)], axis=1)
    bot = np.stack([hx, hy, np.full(6, -h)], axis=1)
    verts = np.concatenate([top, bot], axis=0)               # (12, 3)
    faces = []
    for i in range(1, 5):                                    # top cap fan
        faces.append([0, i, i + 1])
    for i in range(1, 5):                                    # bottom cap fan (reversed)
        faces.append([6, 6 + i + 1, 6 + i])
    for i in range(6):                                       # side quads -> 2 tris each
        j = (i + 1) % 6
        faces += [[i, j, 6 + j], [i, 6 + j, 6 + i]]
    return verts, np.array(faces, dtype=int)


def log_hex(path, body: SE3, *, color=(66, 135, 245), radius=0.15, thickness=0.05,
            edge_color=(40, 40, 40)):
    """Draw the hexagonal robot body (a hexagonal prism) at ``body``'s pose.

    Args:
        path: rerun entity path.
        body: SE3 world pose of the base.
        color: RGB(A) uint8 body color.
        radius: hexagon circumradius (metres); default matches the weld0 model.
        thickness: prism thickness in z (metres).
        edge_color: wireframe edge color; None to skip.
    """
    verts, faces = _hex_prism(radius, thickness)
    t = np.asarray(body.translation())
    q_xyzw = np.asarray(body.rotation().as_quaternion_xyzw())
    rr.log(f"{path}/hex", rr.Transform3D(translation=t, rotation=rr.Quaternion(xyzw=q_xyzw)))
    log_mesh(f"{path}/hex/body", verts, faces, color=color, edge_color=edge_color)


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


def log_bodies(path, bodies: SE3, values=None, *, cmap="viridis", vmin=None, vmax=None,
               body_size=(0.20, 0.20, 0.05)):
    """Draw a stack of body boxes (no legs) at one time, colored by ``values``.

    path   : str              rerun entity path.
    bodies : SE3 batch (N,)   base poses.
    values : (N,) or None     per-body scalar mapped through ``cmap`` to RGB
                              (None -> a flat grey).

    All boxes are logged as a single batched ``Boxes3D`` at ``path``.
    """
    t = np.asarray(bodies.translation())                                 # (N, 3)
    q_xyzw = np.asarray(bodies.rotation().as_quaternion_xyzw())          # (N, 4)
    n = t.shape[0]
    if values is None:
        colors = np.tile(np.array([160, 160, 160], np.uint8), (n, 1))
    else:
        colors = colors_from_values(values, cmap=cmap, vmin=vmin, vmax=vmax)
    rr.log(path, rr.Boxes3D(
        centers=t,
        half_sizes=np.tile(np.array(body_size) / 2.0, (n, 1)),
        quaternions=[rr.Quaternion(xyzw=q) for q in q_xyzw],
        colors=colors,
        fill_mode="solid",
    ))


def log_mesh(path, vertices, faces, *, color=(160, 160, 160), normals=None,
             edge_color=(40, 40, 40), edge_radius=0.002):
    """Draw a triangle mesh in a single flat color (e.g. a sampling surface).

    Args:
        path: rerun entity path.
        vertices: (V, 3) mesh vertices.
        faces: (F, 3) int triangle vertex indices.
        color: RGB(A) uint8 color applied to the whole mesh.
        normals: optional (V, 3) vertex normals (for shading).
        edge_color: RGB(A) color for the triangle edges (wireframe overlay under
            ``{path}/edges``); None to skip drawing edges.
        edge_radius: edge line radius (metres).
    """
    verts = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.uint32)
    color = np.asarray(color, dtype=np.uint8)
    rr.log(path, rr.Mesh3D(
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
        rr.log(f"{path}/edges", rr.LineStrips3D(segs, radii=edge_radius, colors=ec))


def log_points(path, points, values=None, *, cmap="viridis", vmin=None, vmax=None,
               radius=0.01, color=(120, 120, 120)):
    """Draw a point cloud, colored per-point by ``values`` through ``cmap``.

    Args:
        path: rerun entity path.
        points: (N, 3) point positions.
        values: (N,) per-point scalars mapped through ``cmap``; None -> flat ``color``.
        cmap: matplotlib colormap name.
        vmin: lower color bound (default data min).
        vmax: upper color bound (default data max).
        radius: point radius (metres).
        color: flat RGB used when ``values`` is None.
    """
    pts = np.asarray(points, dtype=np.float32)
    if values is None:
        colors = np.tile(np.asarray(color, np.uint8), (pts.shape[0], 1))
    else:
        colors = colors_from_values(values, cmap=cmap, vmin=vmin, vmax=vmax)
    rr.log(path, rr.Points3D(pts, radii=radius, colors=colors))


def log_spheres(path, centers, radii, values=None, colors=None, *, cmap="viridis",
                vmin=None, vmax=None, color=(120, 120, 120), fill_mode="solid"):
    """Draw solid spheres, colored by explicit ``colors`` or ``values`` via ``cmap``.

    Args:
        path: rerun entity path.
        centers: (N, 3) sphere centers.
        radii: (N,) or scalar sphere radii (metres).
        values: (N,) per-sphere scalars mapped through ``cmap`` (ignored if
            ``colors`` is given).
        colors: explicit color(s); takes precedence. Either a single RGB(A)
            tuple applied to all spheres, or a (N, 3/4) per-sphere array.
        cmap: matplotlib colormap name.
        vmin: lower color bound (default data min).
        vmax: upper color bound (default data max).
        color: flat RGB used when neither ``colors`` nor ``values`` is given.
        fill_mode: rerun fill mode ("solid" or "majorwireframe").
    """
    c = np.asarray(centers, dtype=np.float32)
    n = c.shape[0]
    r = np.broadcast_to(np.asarray(radii, np.float32), (n,))
    half = np.repeat(r[:, None], 3, axis=1)                   # sphere -> equal half-sizes
    if colors is not None:
        cols = np.asarray(colors)
        if cols.ndim == 1:                                    # single color -> all spheres
            cols = np.tile(cols, (n, 1))
    elif values is not None:
        cols = colors_from_values(values, cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        cols = np.tile(np.asarray(color, np.uint8), (n, 1))
    rr.log(path, rr.Ellipsoids3D(centers=c, half_sizes=half, colors=cols,
                                 fill_mode=fill_mode))


def _quat_from_z(d):
    """Quaternions (xyzw) rotating +z onto each unit direction in ``d`` (N, 3)."""
    d = np.asarray(d, float)
    d = d / (np.linalg.norm(d, axis=-1, keepdims=True) + 1e-12)
    z = np.array([0.0, 0.0, 1.0])
    xyz = np.cross(np.broadcast_to(z, d.shape), d)             # z x d
    w = 1.0 + d @ z                                            # (N,)
    q = np.concatenate([xyz, w[:, None]], axis=-1)             # xyzw (unnormalized)
    q[w < 1e-8] = np.array([1.0, 0.0, 0.0, 0.0])              # d ~ -z: 180deg about x
    return q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12)


def log_robot(path, body: SE3, theta, *, f=None, tau=None,
              tmin=None, tmax=None, cmap="bwr",
              foot_radius=0.0125, leg_radius=0.008,
              force_scale=0.005, arrow_radius=0.006,
              foot_color=(150, 150, 150)):
    """Draw a hexapod configuration: hex body, legs, feet, optional torque/force.

    Args:
        path: rerun entity path.
        body: SE3 world pose of the base.
        theta: (6, 3) joint angles [coxa, femur, tibia] per leg.
        f: (6, 3) foot forces (world); if given, a 3D arrow is drawn at each foot.
        tau: (6, 3) joint torques [coxa, femur, tibia]; if given, the leg capsules
            are colored per segment by torque through ``cmap`` (else flat gray).
        tmin, tmax: torque color bounds; default symmetric +/- max|tau|.
        cmap: matplotlib colormap for torque (diverging, e.g. "bwr").
        foot_radius, leg_radius: foot-sphere / leg-capsule radii (m).
        force_scale: arrow length per newton (m/N).
        arrow_radius: force-arrow shaft radius (m).
        foot_color: flat RGB color for the foot spheres.
    """
    log_hex(path, body)                                            # hexagonal body

    xpos = np.asarray(infer_joint_xpos(body, jnp.asarray(theta)))  # (6, 4, 3)
    feet = xpos[:, -1, :]                                          # (6, 3)

    a = xpos[:, :3, :].reshape(-1, 3)                             # (18,3) segment starts
    b = xpos[:, 1:, :].reshape(-1, 3)                             # (18,3) segment ends
    if tau is not None:                                           # color per segment by torque
        t = np.asarray(tau).reshape(-1)                          # coxa/femur/tibia per leg
        hi = float(np.abs(t).max()) if tmax is None else tmax
        lo = -hi if tmin is None else tmin
        colors = colors_from_values(t, cmap=cmap, vmin=lo, vmax=hi)
    else:                                                         # flat gray
        colors = np.tile(np.array(FREE_COLOR, np.uint8), (a.shape[0], 1))
    d = b - a
    rr.log(f"{path}/legs", rr.Capsules3D(                        # legs = capsules
        lengths=np.linalg.norm(d, axis=-1), radii=leg_radius, translations=a,
        quaternions=[rr.Quaternion(xyzw=q) for q in _quat_from_z(d)],
        colors=colors, fill_mode="solid"))

    log_spheres(f"{path}/feet", feet, foot_radius, colors=foot_color)   # feet = solid spheres

    if f is not None:                                             # force arrows at the feet
        vecs = np.asarray(f).reshape(6, 3) * force_scale
        rr.log(f"{path}/forces",
               rr.Arrows3D(origins=feet, vectors=vecs, radii=arrow_radius,
                           colors=(255, 180, 0)))


def log_poses(path, bodies: SE3, thetas, *, timeline="pose", **kw):
    """Draw N configurations under ``path`` on a scrubbable timeline.

    path   : str              rerun entity path.
    bodies : SE3 batch (N,)   one base pose per frame.
    thetas : (N, 6, 3)        per-frame joint angles.

    Each frame is logged at time index i on ``timeline`` (scrub/play in the
    viewer). Extra kwargs (``stance``, ``only_stance``, ...) forward to
    :func:`log_robot`. For a simultaneous overlay instead, use distinct paths.
    """
    thetas = jnp.asarray(thetas)
    for i in range(thetas.shape[0]):
        set_time(i, timeline=timeline)
        log_robot(path, bodies[i], thetas[i], **kw)


def log_trajectory(path, bodies: SE3, thetas, *, timeline="t", **kw):
    """Draw a trajectory under ``path`` on a scrubbable timeline.

    path    : str              rerun entity path.
    bodies  : SE3 batch (N,)   one base pose per frame.
    thetas  : (N, 6, 3)        per-frame joint angles.

    Each frame is logged at time index i on ``timeline``. Extra kwargs
    (``tau``, ``f``, ...) forward to :func:`log_robot`.
    """
    thetas = jnp.asarray(thetas)
    for i in range(thetas.shape[0]):
        set_time(i, timeline=timeline)
        log_robot(path, bodies[i], thetas[i], **kw)


def log_stack(path, bodies: SE3, thetas, **kw):
    """Overlay N configurations at a single time, each on its own sub-path.

    path    : str              base rerun entity path.
    bodies  : SE3 batch (N,)   one base pose per robot.
    thetas  : (N, 6, 3)        per-robot joint angles.

    Unlike :func:`log_trajectory`, all robots are drawn at once (distinct
    ``{path}/{i}`` paths) rather than across a timeline. Extra kwargs
    (``tau``, ``f``, ...) forward to :func:`log_robot`.
    """
    thetas = jnp.asarray(thetas)
    for i in range(thetas.shape[0]):
        log_robot(f"{path}/{i}", bodies[i], thetas[i], **kw)


def _demo(connect=8812):
    init_viewer(connect=connect)
    n = 8
    zs = jnp.linspace(0.18, 0.28, n)                                  # sweep body height
    bodies = SE3.from_translation(jnp.stack([jnp.zeros(n), jnp.zeros(n), zs], axis=-1))
    thetas = jnp.broadcast_to(jnp.array([0.0, 0.0523599, 1.46608]), (n, 6, 3))
    log_poses("world", bodies, thetas)
    print(f"streamed {n} poses -> rerun on port {connect}")


if __name__ == "__main__":
    _demo()
