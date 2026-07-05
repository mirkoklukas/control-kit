"""Pre-sample foot candidates on the scene terrain (``models/weld0.xml``).

Tessellate the *static* collision geometry -- the floor plane + the climbing box
-- into world-space triangles, then draw an area-uniform pool of surface points
(+ outward normals). Plain MuJoCo/numpy: build once, cache to ``.npz``; the jax
``foot_sampler`` (see ``mjx_sample.py``) then draws footholds from this pool.

Only world-body geoms are used (``geom_bodyid == 0``), so the robot's own geoms
are skipped automatically. The floor plane is infinite, so it is bounded to the
``region`` AABB; the box faces are area-sampled and rejected outside ``region``
so the pool concentrates on the climb corridor (floor near the robot + the near
face of the box at x=1), not the far/underside faces.

Run under ``uv run --extra mjx`` (needs only mujoco + numpy).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models" / "weld0.xml"
CACHE = Path(__file__).resolve().parent / "weld0_pool.npz"

# World-space region of interest, [lo, hi] per axis. Keeps only surface points
# inside this AABB: the approach floor and the near/base of the climbing box.
REGION = np.array([[-1.0, 1.0],      # x: origin -> the wall base at x=1 (floor stops here,
                                     #    so no buried footholds under the box footprint)
                   [-1.2, 1.2],      # y
                   [-0.05, 1.2]])    # z: floor up the wall face


# --------------------------------------------------------------------------- #
# Tessellation: one geom -> (tris (T, 3, 3), normals (T, 3))                   #
# --------------------------------------------------------------------------- #
def _plane_triangles(pos, R, region):
    """A bounded rectangle for an (assumed near-horizontal) plane.

    Built directly in world xy over ``region``'s xy bounds at height ``pos[2]``;
    normal is the plane's world +z. Exact for the axis-aligned floor in weld0.
    """
    (xlo, xhi), (ylo, yhi), _ = region
    z = pos[2]
    corners = np.array([[xlo, ylo, z], [xhi, ylo, z], [xhi, yhi, z], [xlo, yhi, z]])
    n = R @ np.array([0.0, 0.0, 1.0])
    tris = np.array([[corners[0], corners[1], corners[2]],
                     [corners[0], corners[2], corners[3]]])
    return tris, np.array([n, n])


def _box_triangles(pos, R, half):
    """12 triangles (2 per face) for a box at pose ``(pos, R)``, half-extents ``half``."""
    tris, normals = [], []
    for a in range(3):                       # face-normal axis
        b, c = (a + 1) % 3, (a + 2) % 3
        for s in (+1.0, -1.0):               # +/- face
            n_local = np.zeros(3); n_local[a] = s
            quad = []
            for sb, sc in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
                p = np.zeros(3)
                p[a] = s * half[a]; p[b] = sb * half[b]; p[c] = sc * half[c]
                quad.append(p)
            quad = np.array(quad) @ R.T + pos
            n = R @ n_local                  # explicit outward normal (winding-agnostic)
            tris += [[quad[0], quad[1], quad[2]], [quad[0], quad[2], quad[3]]]
            normals += [n, n]
    return np.array(tris), np.array(normals)


def _mesh_triangles(model, g, pos, R):
    """Triangles for a MuJoCo ``mesh`` geom, transformed to world."""
    did = int(model.geom_dataid[g])
    vadr, vnum = int(model.mesh_vertadr[did]), int(model.mesh_vertnum[did])
    fadr, fnum = int(model.mesh_faceadr[did]), int(model.mesh_facenum[did])
    verts = model.mesh_vert[vadr:vadr + vnum].reshape(-1, 3) @ R.T + pos
    faces = model.mesh_face[fadr:fadr + fnum].reshape(-1, 3)
    tris = verts[faces]                                          # (F, 3, 3)
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    return tris, n


def scene_triangles(model_path=MODEL, region=REGION):
    """World-space triangle soup (+ normals) of the terrain geoms in the scene."""
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)                              # world geom poses

    tris, normals = [], []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != 0:                          # world body only
            continue
        pos = data.geom_xpos[g].copy()
        R = data.geom_xmat[g].reshape(3, 3).copy()
        gtype = model.geom_type[g]
        if gtype == mujoco.mjtGeom.mjGEOM_PLANE:
            t, n = _plane_triangles(pos, R, region)
        elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
            t, n = _box_triangles(pos, R, model.geom_size[g])
        elif gtype == mujoco.mjtGeom.mjGEOM_MESH:
            t, n = _mesh_triangles(model, g, pos, R)
        else:
            continue
        tris.append(t); normals.append(n)
    return np.concatenate(tris), np.concatenate(normals)


# --------------------------------------------------------------------------- #
# Area-uniform sampling                                                       #
# --------------------------------------------------------------------------- #
def _sample_tris(rng, tris, normals, n):
    """Area-weighted triangle pick + uniform barycentric point (with its normal)."""
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    idx = rng.choice(len(tris), size=n, p=areas / areas.sum())
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1.0                                          # fold into the triangle
    u[flip], v[flip] = 1.0 - u[flip], 1.0 - v[flip]
    pts = v0[idx] + u[:, None] * (v1[idx] - v0[idx]) + v[:, None] * (v2[idx] - v0[idx])
    return pts, normals[idx]


def build_pool(model_path=MODEL, M=20000, region=REGION, seed=0, oversample=4):
    """Area-uniform pool of ``M`` surface points inside ``region``.

    Returns ``(xyz (M, 3), normal (M, 3))``. Points are reject-sampled against
    ``region`` so faces poking outside the AABB (box top, far/side faces) drop out.
    """
    region = np.asarray(region)
    tris, normals = scene_triangles(model_path, region)
    rng = np.random.default_rng(seed)

    xyz, nrm, need = [], [], M
    while need > 0:
        pts, nn = _sample_tris(rng, tris, normals, need * oversample)
        keep = np.all((pts >= region[:, 0]) & (pts <= region[:, 1]), axis=1)
        pts, nn = pts[keep][:need], nn[keep][:need]
        xyz.append(pts); nrm.append(nn); need -= len(pts)
    return np.concatenate(xyz), np.concatenate(nrm)


def load_pool(path=CACHE, rebuild=False, **kw):
    """Load the cached pool, building (and caching) it on first use."""
    path = Path(path)
    if rebuild or not path.exists():
        xyz, nrm = build_pool(**kw)
        np.savez(path, xyz=xyz, normal=nrm)
    d = np.load(path)
    return d["xyz"], d["normal"]


if __name__ == "__main__":
    xyz, nrm = load_pool(rebuild=True)
    print(f"built pool: {xyz.shape[0]} pts -> {CACHE}")
    print(f"  x {xyz[:,0].min():.2f}..{xyz[:,0].max():.2f}  "
          f"y {xyz[:,1].min():.2f}..{xyz[:,1].max():.2f}  "
          f"z {xyz[:,2].min():.2f}..{xyz[:,2].max():.2f}")
    on_wall = (np.abs(xyz[:, 0] - 1.0) < 1e-3).mean()
    print(f"  fraction on the wall face (x=1): {on_wall:.2%}")
