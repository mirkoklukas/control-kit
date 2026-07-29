"""Geometry helpers: quaternions and generated meshes.

Pure numpy -- no rerun. Lifts out of this package unchanged if a second drawing
backend ever wants the same shapes.
"""

import numpy as np


def quat_from_z(d):
    """Quaternions rotating +z onto each given direction.

    Args:
        d: (N, 3) directions; normalized internally.

    Returns:
        (N, 4) xyzw quaternions. A direction of ~-z maps to a 180-degree turn
        about x (the rotation is genuinely ambiguous there; this picks one).
    """
    d = np.asarray(d, float)
    d = d / (np.linalg.norm(d, axis=-1, keepdims=True) + 1e-12)
    z = np.array([0.0, 0.0, 1.0])
    xyz = np.cross(np.broadcast_to(z, d.shape), d)             # z x d
    w = 1.0 + d @ z                                            # (N,)
    q = np.concatenate([xyz, w[:, None]], axis=-1)             # xyzw (unnormalized)
    q[w < 1e-8] = np.array([1.0, 0.0, 0.0, 0.0])               # d ~ -z: 180deg about x
    return q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12)


def vertex_normals(verts, faces):
    """Smooth per-vertex normals (area-weighted face normals, accumulated).

    Args:
        verts: (V, 3) vertices.
        faces: (F, 3) int triangle indices.

    Returns:
        (V, 3) unit vertex normals.
    """
    verts, faces = np.asarray(verts, float), np.asarray(faces)
    tris = verts[faces]
    fn = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])   # area-weighted
    n = np.zeros_like(verts)
    np.add.at(n, faces.reshape(-1), np.repeat(fn, 3, axis=0))
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)


def polygon_prism(num_sides: int, radius: float = 0.15, thickness: float = 0.05):
    """A regular prism, centred at the origin -- the robot body.

    Corners sit at the same angles :func:`..kinematics.robot.radial_mounts` uses
    (offset by half a step), so with ``num_sides = robot.num_legs`` every corner
    lands on a shoulder: a hexagon for a hexapod, a square for a quadruped.

    Args:
        num_sides: corners of the polygon.
        radius: circumradius (metres).
        thickness: extent in z (metres).

    Returns:
        ``(vertices, faces)`` -- (2*num_sides, 3) top ring then bottom ring, and
        (F, 3) int triangles (2 caps + one quad per side). Triangles are wound
        outward, so vertex normals -- hence lighting -- come out right.
    """
    n = num_sides
    ang = np.deg2rad(np.arange(360.0 / n / 2, 360.0, 360.0 / n))
    hx, hy = radius * np.cos(ang), radius * np.sin(ang)
    h = thickness / 2.0
    top = np.stack([hx, hy, np.full(n, h)], axis=1)
    bot = np.stack([hx, hy, np.full(n, -h)], axis=1)
    verts = np.concatenate([top, bot], axis=0)               # (2n, 3)

    faces = []
    for i in range(1, n - 1):                                # top cap fan
        faces.append([0, i, i + 1])
    for i in range(1, n - 1):                                # bottom cap fan (reversed)
        faces.append([n, n + i + 1, n + i])
    for i in range(n):                                       # side quads -> 2 tris each
        j = (i + 1) % n
        faces += [[i, j, n + j], [i, n + j, n + i]]
    faces = np.array(faces, dtype=int)

    # Orient every triangle outward (convex prism, centroid at the origin).
    tris = verts[faces]
    fn = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    inward = np.einsum("ij,ij->i", fn, tris.mean(1)) < 0
    faces[inward] = faces[inward][:, ::-1]
    return verts, faces
