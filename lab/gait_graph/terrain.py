"""Terrain = axis-aligned boxes; footholds sampled on their faces.

The same boxes serve three roles: a :class:`collision.Scene` for the robot-vs-terrain
check, the foothold pool (area-uniform samples on the faces), and ``<geom type="box">``
lines for the MuJoCo scene. Boxes because MuJoCo collides them exactly (a mesh geom
is replaced by its convex hull).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from controlkit.kinematics import collision
from controlkit.kinematics.types import Foothold


def make_scene(boxes) -> collision.Scene:
    """Boxes ``((center, half), ...)`` -> a :class:`collision.Scene`."""
    c = jnp.array([b[0] for b in boxes], dtype=float)
    h = jnp.array([b[1] for b in boxes], dtype=float)
    return collision.Scene(box_center=c, box_half=h)


def _faces(scene: collision.Scene):
    """All 6 faces of every box, as (center, normal, u-axis, v-axis, half-u, half-v).

    Returns:
        Tuple of arrays, each with leading dim ``6 * num_boxes``.
    """
    cs, ns, us, vs, hu, hv = [], [], [], [], [], []
    eye = np.eye(3)
    for c, h in zip(np.asarray(scene.box_center), np.asarray(scene.box_half)):
        for a in range(3):
            b, d = (a + 1) % 3, (a + 2) % 3
            for s in (1.0, -1.0):
                cs.append(c + s * h[a] * eye[a])
                ns.append(s * eye[a])
                us.append(eye[b]); vs.append(eye[d])
                hu.append(h[b]); hv.append(h[d])
    return tuple(jnp.array(x) for x in (cs, ns, us, vs, hu, hv))


def sample_footholds(key, scene: collision.Scene, M: int, *, edge_margin: float = 0.0,
                     max_down: float = -0.5) -> Foothold:
    """Area-uniform footholds on the box faces, with their outward normals.

    Drawn on each face's interior shrunk by ``edge_margin``. Samples that are
    buried (a small step off the face along its normal lands inside some box, e.g.
    a face resting on the ground) and downward-facing ones (``n_z < max_down``) are
    dropped, so fewer than ``M`` come back. Eager, not jittable (ragged output).

    Args:
        key: PRNG key.
        scene: the terrain.
        M: number of samples drawn (before filtering).
        edge_margin: keep this far from face edges.
        max_down: drop faces whose normal z-component is below this.

    Returns:
        (M',) footholds, world frame.
    """
    c, n, u, v, hu, hv = _faces(scene)
    hu, hv = jnp.maximum(hu - edge_margin, 0.0), jnp.maximum(hv - edge_margin, 0.0)
    area = 4 * hu * hv
    k_face, k_uv = jax.random.split(key)
    f = jax.random.choice(k_face, area.shape[0], (M,), p=area / area.sum())
    st = jax.random.uniform(k_uv, (M, 2), minval=-1.0, maxval=1.0)
    pos = c[f] + (st[:, :1] * hu[f, None]) * u[f] + (st[:, 1:] * hv[f, None]) * v[f]
    nrm = n[f]

    exposed = collision.sdf(scene, pos + 1e-3 * nrm) > 0
    keep = np.asarray(exposed & (nrm[:, 2] >= max_down))
    return Foothold(pos[keep], nrm[keep])


def mjcf_geoms(boxes, *, material: str = None, prefix: str = "terrain") -> str:
    """``<geom type="box">`` lines for the worldbody, named ``{prefix}{i}``."""
    mat = f' material="{material}"' if material else ""
    return "".join(
        f'    <geom name="{prefix}{i}" type="box" pos="{c[0]:.6g} {c[1]:.6g} {c[2]:.6g}" '
        f'size="{h[0]:.6g} {h[1]:.6g} {h[2]:.6g}"{mat}/>\n'
        for i, (c, h) in enumerate(boxes))
