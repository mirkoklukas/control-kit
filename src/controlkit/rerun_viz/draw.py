"""Rerun drawing primitives, in world coordinates.

Nothing here knows there is a robot -- these take points, meshes and frames. The
robot lives one layer up, in ``robot``. (Same split as ``kinematics.chain`` vs
``kinematics.leg4dof``.)

Every helper takes the rerun entity ``path`` first, as ``rr.log(path, ...)`` does.
"""

import numpy as np
import rerun as rr

from controlkit.rerun_viz import shapes
from controlkit.rerun_viz.colors import (
    BODY_COLOR,
    CONTACT_COLOR,
    EDGE_COLOR,
    SEGMENT_COLOR,
    as_per_item,
    colors_from_values,
)

_AXIS = {"x": 0, "y": 1, "z": 2}


def log_mesh(path, vertices, faces, *, color=(160, 160, 160), normals=None,
             edge_color=EDGE_COLOR, edge_radius=0.002):
    """Draw a triangle mesh in a single flat color.

    Args:
        path: rerun entity path (edges under ``{path}/edges``).
        vertices: (V, 3) mesh vertices.
        faces: (F, 3) int triangle vertex indices.
        color: RGB(A) uint8 applied to the whole mesh.
        normals: optional (V, 3) vertex normals, for shading.
        edge_color: wireframe overlay color; None to skip edges.
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
    """Draw a point cloud, optionally colored per-point by ``values``.

    Args:
        path: rerun entity path.
        points: (N, 3) positions.
        values: (N,) scalars mapped through ``cmap``; None -> flat ``color``.
        cmap: matplotlib colormap name.
        vmin: lower color bound (default data min).
        vmax: upper color bound (default data max).
        radius: point radius (metres).
        color: flat RGB used when ``values`` is None.
    """
    pts = np.asarray(points, dtype=np.float32)
    if values is None:
        cols = as_per_item(np.asarray(color, np.uint8), pts.shape[0])
    else:
        cols = colors_from_values(values, cmap=cmap, vmin=vmin, vmax=vmax)
    rr.log(path, rr.Points3D(pts, radii=radius, colors=cols))


def log_spheres(path, centers, radii, values=None, colors=None, *, cmap="viridis",
                vmin=None, vmax=None, color=(120, 120, 120), fill_mode="solid"):
    """Draw spheres, colored by explicit ``colors`` or by ``values`` via ``cmap``.

    Args:
        path: rerun entity path.
        centers: (N, 3) sphere centers.
        radii: (N,) or scalar radii (metres).
        values: (N,) scalars mapped through ``cmap``; ignored if ``colors`` given.
        colors: explicit color(s), takes precedence -- one RGB(A) for all, or
            an (N, 3/4) array.
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
        cols = as_per_item(colors, n)
    elif values is not None:
        cols = colors_from_values(values, cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        cols = as_per_item(np.asarray(color, np.uint8), n)
    rr.log(path, rr.Ellipsoids3D(centers=c, half_sizes=half, colors=cols,
                                 fill_mode=fill_mode))


def log_contacts(path, positions, normals=None, *, show_normals=False,
                 size=0.03, thickness=0.004, cube=0.006,
                 normal_len=0.04, normal_radius=0.0025,
                 color=CONTACT_COLOR, normal_color=(255, 255, 255)):
    """Draw a flat tile at each contact, optionally with a normal arrow.

    The tile's thin axis lies along the normal, so it sits flat on the surface.
    A contact whose normal has zero length has no orientation, so it is drawn as
    a tiny cube and gets no arrow; ``normals=None`` makes every contact a cube.

    Args:
        path: rerun entity path (tiles at ``path``, arrows at ``{path}/normals``).
        positions: (N, 3) contact points.
        normals: (N, 3) surface normals; None to draw orientation-less cubes.
        show_normals: draw the normal arrows (only when ``normals`` is given).
        size: tile side length (m).
        thickness: tile thickness along the normal (m).
        cube: side length of the cube drawn where the normal is zero (m).
        normal_len: length of the normal arrow (m).
        normal_radius: normal-arrow shaft radius (m).
        color: one RGB(A) for all tiles, or an (N, 3/4) array.
        normal_color: RGB(A) for the normal arrows.
    """
    p = np.asarray(positions, np.float32).reshape(-1, 3)
    n = (np.zeros_like(p) if normals is None
         else np.asarray(normals, np.float32).reshape(-1, 3))
    mag = np.linalg.norm(n, axis=1)
    zero = mag < 1e-9                                                      # no orientation

    half = np.tile([size / 2, size / 2, thickness / 2], (p.shape[0], 1))   # thin along local z
    half[zero] = cube / 2.0                                                # zero normal -> tiny cube
    rr.log(path, rr.Boxes3D(
        centers=p, half_sizes=half,
        quaternions=[rr.Quaternion(xyzw=q) for q in shapes.quat_from_z(n)],
        colors=as_per_item(color, p.shape[0]), fill_mode="solid"))

    if normals is not None and show_normals:
        unit = np.zeros_like(n)
        unit[~zero] = n[~zero] / mag[~zero, None]                         # zero normal -> zero-length arrow
        rr.log(f"{path}/normals", rr.Arrows3D(
            origins=p, vectors=unit * normal_len, radii=normal_radius, colors=normal_color))


def log_joint(path, tf, axis="z", link=None, *,
              disk_radius=0.03, disk_thickness=0.008, disk_color=BODY_COLOR,
              link_width=0.01, link_color=SEGMENT_COLOR):
    """Draw a revolute-joint glyph: a flat disk, plus a thin link box.

    Generic over robots -- it draws *a hinge*, given a frame and an axis.

    Args:
        path: rerun entity path (disk at ``path``, link at ``{path}/link``).
        tf: SE3 world pose of the joint frame.
        axis: rotation axis in ``tf``'s local frame -- a ``"x"``/``"y"``/``"z"``
            char or a 3-vector (e.g. a leg's stored joint axis); the disk (a flat
            cylinder) is flattened along it.
        link: ``(axis, length)`` -- a thin box of ``length`` running from the
            joint origin along ``tf``'s local ``axis``. None to skip the link.
        disk_radius: disk radius (m).
        disk_thickness: disk thickness along ``axis`` (m).
        disk_color: RGB(A) of the disk.
        link_width: cross-section side of the link box (m).
        link_color: RGB(A) of the link box.
    """
    t = np.asarray(tf.translation(), np.float32)
    R = np.asarray(tf.rotation().as_matrix(), np.float32)          # columns = local axes in world

    # Flat cylinder: its symmetry axis (local z) -> the joint axis, so it ends up
    # flattened along `axis`. `axis` is a principal char or a local 3-vector.
    zdir = R[:, _AXIS[axis]] if isinstance(axis, str) else R @ np.asarray(axis, np.float32)
    rr.log(path, rr.Cylinders3D(
        lengths=[disk_thickness], radii=[disk_radius], centers=[t],
        quaternions=[rr.Quaternion(xyzw=shapes.quat_from_z(zdir[None])[0])],
        colors=[disk_color], fill_mode="solid"))

    if link is not None:
        lax, length = _AXIS[link[0]], float(link[1])
        ldir = R[:, lax]                                           # world dir of the link axis
        half = np.full(3, link_width / 2, np.float32)
        half[lax] = length / 2                                     # long along the link axis
        center = t + ldir * (length / 2)                           # box spans [0, length] from tf
        rr.log(f"{path}/link", rr.Boxes3D(
            centers=[center], half_sizes=[half],
            quaternions=[rr.Quaternion(xyzw=np.asarray(tf.rotation().as_quaternion_xyzw()))],
            colors=[link_color], fill_mode="solid"))


def log_box(path, center, half, *, quaternion=None, color=(200, 200, 200),
            wireframe=True):
    """Draw one (optionally oriented) box.

    Args:
        path: rerun entity path.
        center: (3,) box centre, world frame.
        half: (3,) half-extents.
        quaternion: xyzw orientation of the box; None for axis-aligned.
        color: RGB(A).
        wireframe: draw edges only (majorwireframe) instead of a solid.
    """
    kw = {}
    if quaternion is not None:
        kw["quaternions"] = [rr.Quaternion(xyzw=np.asarray(quaternion, np.float32))]
    rr.log(path, rr.Boxes3D(
        centers=[np.asarray(center, np.float32)],
        half_sizes=[np.asarray(half, np.float32)],
        colors=[color],
        fill_mode="majorwireframe" if wireframe else "solid",
        **kw))


def log_boxes(path, boxes, *, colors=(200, 200, 200), wireframe=True):
    """Draw a batch of AABBs as one entity (e.g. an obstacle scene).

    Args:
        path: rerun entity path.
        boxes: (N, 2, 3) AABBs -- ``[:, 0]`` min corners, ``[:, 1]`` max corners.
        colors: one RGB(A) for all, or an (N, 3/4) per-box array.
        wireframe: edges only (majorwireframe) instead of solid.
    """
    boxes = np.asarray(boxes, np.float32).reshape(-1, 2, 3)
    lo, hi = boxes[:, 0], boxes[:, 1]
    rr.log(path, rr.Boxes3D(
        centers=0.5 * (lo + hi), half_sizes=0.5 * (hi - lo),
        colors=as_per_item(colors, boxes.shape[0]),
        fill_mode="majorwireframe" if wireframe else "solid"))


def log_poses(path, poses, half, values=None, colors=None, *, cmap="viridis",
              vmin=None, vmax=None, color=(200, 200, 200), wireframe=True):
    """Draw a batch of oriented boxes, one per SE3 pose, colored by ``values``.

    The oriented counterpart to :func:`log_boxes`: each pose places and orients a
    box of the given half-extents (e.g. a stack of body poses or OBBs). Coloring
    follows :func:`log_spheres` -- explicit ``colors`` win, else ``values`` through
    ``cmap``, else flat ``color``.

    Args:
        path: rerun entity path.
        poses: (N,) SE3 stack -- box centres (translation) and orientations.
        half: (3,) half-extents shared by all boxes, or (N, 3) per box.
        values: (N,) scalars mapped through ``cmap``; ignored if ``colors`` given.
        colors: explicit color(s), takes precedence -- one RGB(A), or (N, 3/4).
        cmap: matplotlib colormap name.
        vmin: lower color bound (default data min).
        vmax: upper color bound (default data max).
        color: flat RGB used when neither ``colors`` nor ``values`` is given.
        wireframe: edges only (majorwireframe) instead of solid.
    """
    wxyz_xyz = np.asarray(poses.wxyz_xyz, np.float32).reshape(-1, 7)
    n = wxyz_xyz.shape[0]
    centers = wxyz_xyz[:, 4:]
    xyzw = wxyz_xyz[:, [1, 2, 3, 0]]                           # stored wxyz -> rerun xyzw
    half = np.broadcast_to(np.asarray(half, np.float32), (n, 3))

    if colors is not None:
        cols = as_per_item(colors, n)
    elif values is not None:
        cols = colors_from_values(values, cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        cols = as_per_item(np.asarray(color, np.uint8), n)

    rr.log(path, rr.Boxes3D(
        centers=centers, half_sizes=half,
        quaternions=[rr.Quaternion(xyzw=q) for q in xyzw],
        colors=cols,
        fill_mode="majorwireframe" if wireframe else "solid"))
