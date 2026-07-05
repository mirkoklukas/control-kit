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

For arbitrary/steppable terrain, ``plane``/``box``/``mesh`` geoms are all
tessellated, so a terrain **mesh** in the model XML is sampled directly:
``pool_from_geom(model_xml, "terrain")`` area-samples a single named geom with no
region clip. ``build_pool(region=None)`` does the same over all world-body geoms.

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
    """Bounded rectangle for an (assumed near-horizontal) plane.

    The rectangle is built directly in world xy over ``region``'s xy bounds at
    height ``pos[2]``. Exact for the axis-aligned floor in weld0.

    Args:
        pos: (3,) world position of the plane geom.
        R: (3, 3) world rotation of the plane (local +z is the normal).
        region: (3, 2) AABB ``[lo, hi]`` per axis; the rectangle spans the xy bounds.

    Returns:
        tris: (2, 3, 3) two triangles over the xy bounds at height ``pos[2]``.
        normals: (2, 3) the plane's world +z, one per triangle.
    """
    (xlo, xhi), (ylo, yhi), _ = region
    z = pos[2]
    corners = np.array([[xlo, ylo, z], [xhi, ylo, z], [xhi, yhi, z], [xlo, yhi, z]])
    n = R @ np.array([0.0, 0.0, 1.0])
    tris = np.array([[corners[0], corners[1], corners[2]],
                     [corners[0], corners[2], corners[3]]])
    return tris, np.array([n, n])


def _box_triangles(pos, R, half):
    """Tessellate a box into 12 triangles (2 per face).

    Args:
        pos: (3,) world position of the box centre.
        R: (3, 3) world rotation of the box.
        half: (3,) half-extents along the local axes.

    Returns:
        tris: (12, 3, 3) triangle vertices in world coordinates.
        normals: (12, 3) outward face normal, one per triangle.
    """
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
    """Tessellate a MuJoCo ``mesh`` geom into world-space triangles.

    Args:
        model: the ``MjModel`` holding the mesh asset.
        g: geom id of the mesh geom.
        pos: (3,) world position of the geom.
        R: (3, 3) world rotation of the geom.

    Returns:
        tris: (F, 3, 3) triangle vertices in world coordinates.
        normals: (F, 3) unit face normals from the vertex winding.
    """
    did = int(model.geom_dataid[g])
    vadr, vnum = int(model.mesh_vertadr[did]), int(model.mesh_vertnum[did])
    fadr, fnum = int(model.mesh_faceadr[did]), int(model.mesh_facenum[did])
    verts = model.mesh_vert[vadr:vadr + vnum].reshape(-1, 3) @ R.T + pos
    faces = model.mesh_face[fadr:fadr + fnum].reshape(-1, 3)
    tris = verts[faces]                                          # (F, 3, 3)
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    return tris, n


def scene_triangles(model_path=MODEL, region=REGION, geom_names=None):
    """World-space triangle soup (+ normals) of selected terrain geoms.

    Args:
        model_path: path to the model XML.
        region: (3, 2) AABB used only to bound the (infinite) plane rectangle. If
            ``None``, planes are skipped (they can't be area-sampled unbounded);
            boxes/meshes are unaffected.
        geom_names: ``None`` -> every world-body geom (``geom_bodyid == 0``); else
            a list of geom names to tessellate (any body), e.g. terrain meshes.

    Returns:
        tris: (T, 3, 3) triangle vertices in world coordinates.
        normals: (T, 3) outward face normal, one per triangle.
    """
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)                              # world geom poses

    if geom_names is not None:
        gids = []
        for n in geom_names:
            g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
            if g < 0:
                raise ValueError(f"no geom named {n!r} in {model_path}")
            gids.append(g)
    else:
        gids = [g for g in range(model.ngeom) if model.geom_bodyid[g] == 0]

    tris, normals = [], []
    for g in gids:
        pos = data.geom_xpos[g].copy()
        R = data.geom_xmat[g].reshape(3, 3).copy()
        gtype = model.geom_type[g]
        if gtype == mujoco.mjtGeom.mjGEOM_PLANE:
            if region is None:
                continue                                       # unbounded -> skip
            t, n = _plane_triangles(pos, R, np.asarray(region))
        elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
            t, n = _box_triangles(pos, R, model.geom_size[g])
        elif gtype == mujoco.mjtGeom.mjGEOM_MESH:
            t, n = _mesh_triangles(model, g, pos, R)
        else:
            continue
        tris.append(t); normals.append(n)
    if not tris:
        raise ValueError("no tessellable terrain geoms selected")
    return np.concatenate(tris), np.concatenate(normals)


# --------------------------------------------------------------------------- #
# Area-uniform sampling                                                       #
# --------------------------------------------------------------------------- #
def _sample_tris(rng, tris, normals, n):
    """Draw ``n`` area-uniform points on a triangle soup.

    Args:
        rng: a numpy ``Generator``.
        tris: (T, 3, 3) triangle vertices.
        normals: (T, 3) per-triangle normals.
        n: number of points to draw.

    Returns:
        pts: (n, 3) sampled points -- triangle picked with probability
            proportional to area, then a uniform barycentric point inside it.
        normals: (n, 3) the normal of each point's source triangle.
    """
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    idx = rng.choice(len(tris), size=n, p=areas / areas.sum())
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1.0                                          # fold into the triangle
    u[flip], v[flip] = 1.0 - u[flip], 1.0 - v[flip]
    pts = v0[idx] + u[:, None] * (v1[idx] - v0[idx]) + v[:, None] * (v2[idx] - v0[idx])
    return pts, normals[idx]


def build_pool(model_path=MODEL, M=20000, region=REGION, geom_names=None,
               drop_underside=True, seed=0, oversample=4):
    """Area-uniform pool of surface points (+ normals) over the terrain.

    Args:
        model_path: path to the model XML.
        M: number of pool points to return.
        region: (3, 2) AABB ``[lo, hi]`` per axis. Points are reject-sampled
            against it (faces poking outside drop out). ``None`` -> no clip: a
            single area-uniform draw over the selected geoms.
        geom_names: geoms to sample; ``None`` -> all world-body geoms. See
            :func:`scene_triangles`.
        drop_underside: drop triangles whose outward normal points downward
            (``n_z < 0``), e.g. a box's bottom face -- never a valid foothold.
            Vertical faces (``n_z == 0``, e.g. a climbable wall) are kept.
        seed: RNG seed.
        oversample: batch factor for reject sampling when ``region`` is set.

    Returns:
        xyz: (M, 3) surface points.
        normal: (M, 3) outward normal at each point.
    """
    tris, normals = scene_triangles(model_path, region, geom_names=geom_names)
    if drop_underside:
        keep = normals[:, 2] >= -1e-6                           # keep up + vertical faces
        tris, normals = tris[keep], normals[keep]
        if len(tris) == 0:
            raise ValueError("drop_underside removed every face")
    rng = np.random.default_rng(seed)

    if region is None:
        return _sample_tris(rng, tris, normals, M)             # unclipped, exactly M

    region = np.asarray(region)
    xyz, nrm, need = [], [], M
    while need > 0:
        pts, nn = _sample_tris(rng, tris, normals, need * oversample)
        keep = np.all((pts >= region[:, 0]) & (pts <= region[:, 1]), axis=1)
        pts, nn = pts[keep][:need], nn[keep][:need]
        xyz.append(pts); nrm.append(nn); need -= len(pts)
    return np.concatenate(xyz), np.concatenate(nrm)


def pool_from_geom(model_path, geom_name, M=20000, drop_underside=True, seed=0):
    """Area-uniform pool from one (or a few) named geom(s), unclipped.

    Convenience for steppable/terrain **mesh** geoms: point it at the terrain
    geom(s) in the model XML and it samples that surface directly (no region AABB,
    other geoms ignored).

    Args:
        model_path: path to the model XML.
        geom_name: a geom name, or a list of geom names, to sample.
        M: number of pool points to return.
        drop_underside: drop downward-facing triangles (e.g. a box's bottom face).
        seed: RNG seed.

    Returns:
        xyz: (M, 3) surface points.
        normal: (M, 3) outward normal at each point.
    """
    names = [geom_name] if isinstance(geom_name, str) else list(geom_name)
    return build_pool(model_path, M=M, region=None, geom_names=names,
                      drop_underside=drop_underside, seed=seed)


# --------------------------------------------------------------------------- #
# Explicit sampling meshes (authored surfaces, not part of the simulation)     #
# --------------------------------------------------------------------------- #
def sample_mesh(vertices, faces, M=20000, seed=0):
    """Area-uniform pool from an explicit triangle mesh.

    A sampling-only surface: hand it any ``(vertices, faces)`` (in-code, or loaded
    from an ``.obj``) and it area-samples the triangles. Nothing to do with the
    physics scene -- useful to define exactly the region footholds may land on.

    Args:
        vertices: (V, 3) mesh vertices.
        faces: (F, 3) int triangle vertex indices.
        M: number of pool points to return.
        seed: RNG seed.

    Returns:
        xyz: (M, 3) surface points.
        normal: (M, 3) unit face normal at each point (from the vertex winding).
    """
    verts = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=int)
    tris = verts[faces]                                         # (F, 3, 3)
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    return _sample_tris(np.random.default_rng(seed), tris, n, M)


def _add_quad(verts, faces, corners):
    """Append a quad (4 corners) as two triangles to ``(verts, faces)`` in place."""
    i = len(verts)
    verts.extend(corners)
    faces.append([i, i + 1, i + 2])
    faces.append([i, i + 2, i + 3])


def climb_surface(model_path=MODEL, floor_x=(-1.0, 1.0), floor_y=(-2.0, 2.0),
                  box_geom="box"):
    """Author the weld0 climb surface as a sampling mesh (not a physics geom).

    The surface is the floor up to the box plus the box's top and four sides --
    no bottom face, no floor buried under the box. The box pose/size is read
    (read-only) from the model, so the surface stays in sync with the geometry.
    Assumes an axis-aligned box (true for weld0).

    Args:
        model_path: model XML to read the box geom from (not modified).
        floor_x: (lo, hi) floor extent in x. Default stops at the box near face
            (x=1) so no floor lands under the box; set hi>1 to extend under it.
        floor_y: (lo, hi) floor extent in y.
        box_geom: name of the box geom to wrap.

    Returns:
        vertices: (V, 3) mesh vertices.
        faces: (F, 3) int triangle indices. Feed to :func:`sample_mesh`.
        normals: (V, 3) outward per-vertex normals (for :func:`log_mesh` shading).
    """
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, box_geom)
    if g < 0:
        raise ValueError(f"no geom named {box_geom!r} in {model_path}")
    c, h = data.geom_xpos[g], model.geom_size[g]               # centre, half-extents
    x0, x1 = c[0] - h[0], c[0] + h[0]
    y0, y1 = c[1] - h[1], c[1] + h[1]
    z0, z1 = c[2] - h[2], c[2] + h[2]

    verts, faces = [], []
    fx0, fx1 = floor_x
    fy0, fy1 = floor_y
    _add_quad(verts, faces, [(fx0, fy0, 0.0), (fx1, fy0, 0.0),      # floor (z=0)
                             (fx1, fy1, 0.0), (fx0, fy1, 0.0)])
    _add_quad(verts, faces, [(x0, y0, z1), (x1, y0, z1),           # box top (z=z1)
                             (x1, y1, z1), (x0, y1, z1)])
    _add_quad(verts, faces, [(x0, y0, z0), (x0, y0, z1),           # side x=x0 (near)
                             (x0, y1, z1), (x0, y1, z0)])
    _add_quad(verts, faces, [(x1, y0, z0), (x1, y1, z0),           # side x=x1 (far)
                             (x1, y1, z1), (x1, y0, z1)])
    _add_quad(verts, faces, [(x0, y0, z0), (x1, y0, z0),           # side y=y0
                             (x1, y0, z1), (x0, y0, z1)])
    _add_quad(verts, faces, [(x0, y1, z0), (x0, y1, z1),           # side y=y1
                             (x1, y1, z1), (x1, y1, z0)])

    vertices = np.array(verts, dtype=float)
    faces = np.array(faces, dtype=int)
    tris = vertices[faces]
    fn = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])   # outward (winding)
    fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
    normals = np.zeros_like(vertices)                                 # accumulate per vertex
    np.add.at(normals, faces.reshape(-1), np.repeat(fn, 3, axis=0))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    return vertices, faces, normals


def load_pool(path=CACHE, rebuild=False, **kw):
    """Load the cached pool, building (and caching) it on first use.

    Args:
        path: ``.npz`` cache path.
        rebuild: force a rebuild even if the cache exists.
        **kw: forwarded to :func:`build_pool` when (re)building.

    Returns:
        xyz: (M, 3) surface points.
        normal: (M, 3) outward normal at each point.
    """
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
