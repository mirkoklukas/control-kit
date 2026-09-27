import jax
import jax.numpy as jnp


def complement(subset, N=6):
    """Legs in ``range(N)`` not in ``subset``: ``(..., K) -> (..., N-K)``.

    The "free" legs of a stance (not currently planted). Vectorized and
    jit-friendly. E.g. ``complement([0, 2, 4]) -> [1, 3, 5]``.
    """
    subset = jnp.asarray(subset)
    K = subset.shape[-1]
    in_subset = jax.nn.one_hot(subset, N).sum(axis=-2) > 0          # (..., N)
    order = jnp.argsort(in_subset, axis=-1, stable=True)            # not-in-subset first
    return jnp.asarray(order[..., : N - K])


def adjust_angle(theta):
    """Adjust angles to be within [-pi, pi] range."""
    return (theta + jnp.pi) % (2 * jnp.pi) - jnp.pi


def angle_between(u, v):
    dot_product = jnp.dot(u, v)
    norm_u = jnp.linalg.norm(u)
    norm_v = jnp.linalg.norm(v)
    
    # Clip to avoid NaNs caused by floating-point inaccuracy
    cosine_angle = jnp.clip(dot_product / (norm_u * norm_v), -1.0, 1.0)
    
    return jnp.arccos(cosine_angle)


def face_normals(vertices, faces):
    """Compute face normals for a triangle mesh."""
    tris = jnp.asarray(vertices)[jnp.asarray(faces)]           # (F, 3, 3)
    nrm = jnp.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    nrm = nrm / (jnp.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12)
    return nrm

def vertex_normals(vertices, faces):
    """Compute vertex normals for a triangle mesh."""
    tris = jnp.asarray(vertices)[jnp.asarray(faces)]           # (F, 3, 3)
    nrm = jnp.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    nrm = nrm / (jnp.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12)

    # Accumulate normals for each vertex
    vertex_nrm = jnp.zeros_like(vertices)
    vertex_nrm = vertex_nrm.at[faces[:, 0]].add(nrm)
    vertex_nrm = vertex_nrm.at[faces[:, 1]].add(nrm)
    vertex_nrm = vertex_nrm.at[faces[:, 2]].add(nrm)

    # Normalize the accumulated normals
    vertex_nrm = vertex_nrm / (jnp.linalg.norm(vertex_nrm, axis=1, keepdims=True) + 1e-12)
    
    return vertex_nrm

def sample_mesh(key, vertices, faces, M=20000):
    """Area-uniform pool from an explicit triangle mesh (jax).

    A sampling-only surface: hand it any ``(vertices, faces)`` (in-code, or loaded
    from an ``.obj``) and it area-samples the triangles. Nothing to do with the
    physics scene -- useful to define exactly the region footholds may land on.

    Args:
        key: jax PRNG key.
        vertices: (V, 3) mesh vertices.
        faces: (F, 3) int triangle vertex indices.
        M: number of pool points to return.

    Returns:
        xyz: (M, 3) surface points.
        normal: (M, 3) unit face normal at each point (from the vertex winding).
    """
    tris = jnp.asarray(vertices)[jnp.asarray(faces)]           # (F, 3, 3)
    nrm = jnp.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    areas = 0.5 * jnp.linalg.norm(nrm, axis=1)
    nrm = nrm / (jnp.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12)

    k_tri, k_bary = jax.random.split(key)
    idx = jax.random.choice(k_tri, tris.shape[0], (M,), p=areas / areas.sum())
    uv = jax.random.uniform(k_bary, (M, 2))
    uv = jnp.where(uv.sum(1, keepdims=True) > 1.0, 1.0 - uv, uv)   # fold into the triangle
    v0, v1, v2 = tris[idx, 0], tris[idx, 1], tris[idx, 2]
    pts = v0 + uv[:, :1] * (v1 - v0) + uv[:, 1:] * (v2 - v0)
    return pts, nrm[idx]