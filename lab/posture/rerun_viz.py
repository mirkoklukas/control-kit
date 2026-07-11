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
from jaxlie import SE3, SO3

from .kinematics import _joint_frames, SHOULDERS, LENGTHS, AXES

BODY_COLOR = (66, 135, 245)
SEGMENT_COLOR = (90, 90, 90)
CONTACT_COLOR = (255, 180, 0)


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


def log_contacts(path, positions, normals=None, *, show_normals=False,
                 size=0.03, thickness=0.004, cube=0.006,
                 normal_len=0.04, normal_radius=0.0025,
                 color=CONTACT_COLOR, normal_color=(255, 255, 255)):
    """Draw a flat box (tile) at each contact, optionally with a normal arrow.

    The tile's thin axis lies along the normal (it sits flat on the surface); the
    arrow points out along the same normal. A contact whose normal has zero length
    (no orientation) is drawn as a tiny cube and gets no arrow. With ``normals=None``
    every contact is drawn as a tiny cube and no arrows are logged.

    Args:
        path: rerun entity path (tiles at ``path``, arrows at ``{path}/normals``).
        positions: (N, 3) contact points.
        normals: (N, 3) surface normal at each contact (the tile lies flat on it);
            None to draw orientation-less cubes.
        show_normals: draw the normal arrows (only when ``normals`` is given).
        size: tile side length (m).
        thickness: tile thickness along the normal (m).
        cube: side length of the tiny cube drawn where the normal is zero (m).
        normal_len: length of the normal arrow (m).
        normal_radius: normal-arrow shaft radius (m).
        color: single RGB(A) applied to all tiles, or a (N, 3/4) per-contact array.
        normal_color: single RGB(A) for the normal arrows.
    """
    p = np.asarray(positions, np.float32).reshape(-1, 3)
    n = (np.zeros_like(p) if normals is None
         else np.asarray(normals, np.float32).reshape(-1, 3))
    mag = np.linalg.norm(n, axis=1)
    zero = mag < 1e-9                                                      # no orientation

    half = np.tile([size / 2, size / 2, thickness / 2], (p.shape[0], 1))   # thin along local z
    half[zero] = cube / 2.0                                                # zero normal -> tiny cube
    cols = np.asarray(color)
    if cols.ndim == 1:
        cols = np.tile(cols, (p.shape[0], 1))
    rr.log(path, rr.Boxes3D(
        centers=p, half_sizes=half,
        quaternions=[rr.Quaternion(xyzw=q) for q in _quat_from_z(n)],   # local z -> normal (identity if 0)
        colors=cols, fill_mode="solid"))

    if normals is not None and show_normals:
        unit = np.zeros_like(n)
        unit[~zero] = n[~zero] / mag[~zero, None]                         # zero normal -> zero-length arrow
        rr.log(f"{path}/normals", rr.Arrows3D(
            origins=p, vectors=unit * normal_len, radii=normal_radius, colors=normal_color))
    # segs = np.stack([p, p + unit * normal_len], axis=1)                   # (N, 2, 3) line per normal
    # rr.log(f"{path}/normals", rr.LineStrips3D(
    #     segs, radii=normal_radius, colors=normal_color))


_AXIS = {"x": 0, "y": 1, "z": 2}


def log_joint(path, tf: SE3, axis="z", link=None, *,
              disk_radius=0.03, disk_thickness=0.008, disk_color=BODY_COLOR,
              link_width=0.01, link_color=SEGMENT_COLOR):
    """Draw a revolute-joint glyph at ``tf``: a flat disk + a thin link box.

    Args:
        path: rerun entity path (disk at ``path``, link at ``{path}/link``).
        tf: SE3 world pose of the joint frame.
        axis: rotation axis in ``tf``'s local frame ("x"/"y"/"z"); the disk (a flat
            cylinder) is flattened along it.
        link: ``(axis, length)`` -- a thin box of ``length`` running from the joint
            origin along ``tf``'s local ``axis``. None to skip the link.
        disk_radius: disk radius (m).
        disk_thickness: disk thickness along ``axis`` (m).
        disk_color: RGB(A) color of the disk.
        link_width: cross-section side length of the link box (m).
        link_color: RGB(A) color of the link box.
    """
    t = np.asarray(tf.translation(), np.float32)
    R = np.asarray(tf.rotation().as_matrix(), np.float32)          # columns = local axes in world

    # flat cylinder: its symmetry axis (local z) -> the joint axis, so it is
    # flattened along ``axis``.
    zdir = R[:, _AXIS[axis]]
    rr.log(path, rr.Cylinders3D(
        lengths=[disk_thickness], radii=[disk_radius], centers=[t],
        quaternions=[rr.Quaternion(xyzw=_quat_from_z(zdir[None])[0])],
        colors=[disk_color], fill_mode="solid"))

    if link is not None:
        lax, length = _AXIS[link[0]], float(link[1])
        ldir = R[:, lax]                                           # world dir of the link axis
        half = np.full(3, link_width / 2, np.float32)
        half[lax] = length / 2                                     # long along the link axis
        center = t + ldir * (length / 2)                          # box spans [0, length] from tf
        rr.log(f"{path}/link", rr.Boxes3D(
            centers=[center], half_sizes=[half],
            quaternions=[rr.Quaternion(xyzw=np.asarray(tf.rotation().as_quaternion_xyzw()))],
            colors=[link_color], fill_mode="solid"))


def _hex_prism(radius=0.15, thickness=0.05):
    """Vertices/faces of a hexagonal prism (the robot body), centred at origin.

    Corners are at angles 30, 90, ... 330 deg and circumradius ``radius`` --
    matching the shoulders (``RADIUS``) and the hexbody mesh in the models.

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
    faces = np.array(faces, dtype=int)

    # orient every triangle outward (convex prism, centroid at origin) so the
    # vertex normals -- hence the lighting -- come out right.
    tris = verts[faces]
    fn = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    inward = np.einsum("ij,ij->i", fn, tris.mean(1)) < 0
    faces[inward] = faces[inward][:, ::-1]
    return verts, faces


def _vertex_normals(verts, faces):
    """Smooth per-vertex normals (area-weighted face normals, accumulated)."""
    verts, faces = np.asarray(verts, float), np.asarray(faces)
    tris = verts[faces]
    fn = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])   # area-weighted (unnormalized)
    n = np.zeros_like(verts)
    np.add.at(n, faces.reshape(-1), np.repeat(fn, 3, axis=0))
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)


def log_hex(path, body: SE3, *, color=BODY_COLOR, radius=0.15, thickness=0.05,
            edge_color=(40, 40, 40)):
    """Draw the hexagonal robot body (a hexagonal prism) at ``body``'s pose.

    Args:
        path: rerun entity path (body under ``{path}/hex``).
        body: SE3 world pose of the base.
        color: RGB(A) uint8 body color.
        radius: hexagon circumradius (metres); default matches ``RADIUS``.
        thickness: prism thickness in z (metres).
        edge_color: wireframe edge color; None to skip.
    """
    verts, faces = _hex_prism(radius, thickness)
    normals = _vertex_normals(verts, faces)                    # so it shades like the legs
    t = np.asarray(body.translation())
    q_xyzw = np.asarray(body.rotation().as_quaternion_xyzw())
    rr.log(f"{path}/hex", rr.Transform3D(translation=t, rotation=rr.Quaternion(xyzw=q_xyzw)))
    log_mesh(f"{path}/hex/body", verts, faces, color=color, normals=normals, edge_color=edge_color)


def log_posture(path, posture, *, hide_free=False, frc=None, tau=None,
                tmin=None, tmax=None, cmap="bwr",
                body_size=(0.2, 0.15, 0.025), body_color=BODY_COLOR,
                foot_radius=0.02, foot_color=SEGMENT_COLOR,
                force_scale=0.005, arrow_radius=0.006, force_color=CONTACT_COLOR,
                **joint_kw):
    """Draw a full posture: a flat-cube body plus each leg's joints and links,
    optionally colored by joint torque and with a reaction force arrow per foot.

    Per-leg joint frames come from :func:`..kinematics._joint_frames` (pure jax),
    lifted to the world by ``body @ SHOULDERS[leg]``. Each joint is drawn with
    :func:`log_joint` (a disk flattened along the joint axis + a link box along the
    leg's local x); the foot is a small sphere at the end-effector frame.

    Args:
        path: rerun entity path.
        posture: :class:`..kinematics.Posture` (``body`` SE3 + ``(NUM_LEGS,
            NUM_JOINTS)`` ``thetas``).
        hide_free: skip the free (non-planted) legs, i.e. draw only legs with
            ``posture.stance.support`` True (their feet/forces too).
        frc: (NUM_LEGS, 3) world-frame foot forces; if given, an arrow is drawn at
            each foot (under ``{path}/forces``).
        tau: (NUM_LEGS, NUM_JOINTS) joint torques; if given, each joint's disk
            (cylinder) is colored by its torque through ``cmap`` (else the
            ``log_joint`` default); links keep their color.
        tmin, tmax: torque color bounds; default symmetric +/- max|tau|.
        cmap: matplotlib colormap for torque (diverging, e.g. "bwr").
        body_size: (x, y, z) full side lengths of the flat body cube (m).
        body_color: RGB(A) color of the body cube.
        foot_radius: foot-sphere radius (m).
        foot_color: RGB(A) color of the foot spheres.
        force_scale: force-arrow length per newton (m/N).
        arrow_radius: force-arrow shaft radius (m).
        force_color: RGB(A) color of the force arrows.
        **joint_kw: forwarded to :func:`log_joint` (``disk_radius``, ``link_width``,
            ``link_color``, ...); ``disk_color`` is overridden when ``tau`` is given.
    """
    body = posture.body
    thetas = jnp.asarray(posture.thetas)
    nleg, njoint = thetas.shape[0], len(AXES)
    shown = (np.asarray(posture.stance.support).astype(bool) if hide_free
             else np.ones(nleg, bool))

    # per-joint torque colors (symmetric bounds, like log_robot)
    if tau is not None:
        tvals = np.asarray(tau).reshape(-1)
        hi = float(np.abs(tvals).max()) if tmax is None else tmax
        lo = -hi if tmin is None else tmin
        jcolors = colors_from_values(tvals, cmap=cmap, vmin=lo, vmax=hi).reshape(nleg, njoint, 3)

    # hexagon body
    log_hex(path, body, color=body_color)

    # flat cube body (kept in case we switch back)
    # t = np.asarray(body.translation(), np.float32)
    # q_xyzw = np.asarray(body.rotation().as_quaternion_xyzw())
    # rr.log(f"{path}/body", rr.Boxes3D(
    #     centers=[t], half_sizes=[np.asarray(body_size, np.float32) / 2],
    #     quaternions=[rr.Quaternion(xyzw=q_xyzw)], colors=[body_color], fill_mode="solid"))

    shoulders = body @ SHOULDERS                                  # (NUM_LEGS,) world poses
    feet, foot_legs = [], []
    for i in range(nleg):
        if not shown[i]:                                         # hidden free leg
            continue
        frames = shoulders[i] @ _joint_frames(thetas[i])         # (NUM_JOINTS+1,) world frames
        for k in range(njoint):                                  # disk + link per joint
            jk = dict(joint_kw)
            if tau is not None:
                jk["disk_color"] = jcolors[i, k]
            log_joint(f"{path}/leg{i}/j{k}", frames[k], AXES[k],
                      ("x", float(LENGTHS[k])), **jk)
        feet.append(np.asarray(frames[njoint].translation()))    # end-effector = foot
        foot_legs.append(i)
    if feet:
        feet = np.stack(feet)
        log_spheres(f"{path}/feet", feet, foot_radius, colors=foot_color)
        if frc is not None:                                      # reaction force arrows at feet
            vecs = np.asarray(frc).reshape(nleg, 3)[foot_legs] * force_scale
            rr.log(f"{path}/forces", rr.Arrows3D(
                origins=feet, vectors=vecs, radii=arrow_radius, colors=force_color))



