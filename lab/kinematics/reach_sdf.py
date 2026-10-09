"""A leg's reach set as a signed distance: the reach constraint for an NLP.

Built from a voxel grid around the shoulder (the leg's base frame, shoulder at the origin):
a yes/no reachability mask per vertex (analytic IK, optionally joint limits and a keep-out
box such as the body), then Euclidean distance transforms. Evaluated by interpolation, in
JAX, so gradients (and the chain rule through a body pose) come from autodiff.

    grid = make_reach_sdf(leg, h=0.005, keepout=Keepout(body_pose, center, half))
    d = grid.sdf(x)                                  # > 0: reachable, metres
    d, g = jax.value_and_grad(grid.sdf)(x)
    ok = grid.sdf(x) >= grid.h                       # the constraint, with a margin

Accuracy (``reach-grid-limits.ipynb``, resolution sweep): the mask loses where the boundary
lies between vertices, so the zero level is only good to about h and sits slightly outside
the true boundary. **Use a margin of at least h**: projecting points onto the set with
SLSQP, every result with ``sdf >= h`` was truly reachable (exact IK, 1 cm / 5 mm / 2 mm);
with ``sdf >= h / 2`` only ~75% were, although random points with ``sdf > h / 2`` all were
(the solver ends on the boundary, where the bias is largest). Prefer :meth:`ReachSDF.sdf` (Catmull-Rom, C1) over
:meth:`ReachSDF.sdf_linear` (trilinear) in a solver: ~17 vs. ~40-80 SLSQP iterations.
"""
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.ndimage import map_coordinates
from scipy.ndimage import distance_transform_edt

from controlkit.kinematics import candidates
from controlkit.kinematics.leg import Leg
from controlkit.se3 import SE3


@jax.tree_util.register_dataclass
@dataclass
class Keepout:
    """An oriented box the foot must stay out of (e.g. the body), in the shoulder frame.

    Args:
        pose: SE3 of the box's frame in the shoulder frame.
        center: (3,) box centre in the box's frame.
        half: (3,) half-extents.
    """
    pose: SE3
    center: jax.Array
    half: jax.Array

    def contains(self, x: jax.Array) -> jax.Array:
        """Whether point ``x`` (3,), shoulder frame, lies inside the box (scalar bool)."""
        return jnp.all(jnp.abs(self.pose.inverse().apply(x) - self.center) <= self.half)


def reachable_fn(leg: Leg, *, use_limits: bool = True, keepout: Keepout | None = None):
    """Exact membership in the reach set: some IK branch passes the enabled filters.

    Args:
        leg: a leg with analytic IK (``ik_from_foot``), e.g. :class:`Leg3DOF`.
        use_limits: gate the branches by ``leg.limits``.
        keepout: drop feet inside this box; None: no box.

    Returns:
        A function ``(3,) -> scalar bool``, jit- and vmap-able.
    """
    def reachable(foot):
        ok, thetas = leg.ik_from_foot(foot)
        if use_limits:
            ok = ok & candidates.in_limits(thetas, leg.limits)
        ok = ok.any()
        if keepout is not None:
            ok = ok & ~keepout.contains(foot)
        return ok

    return reachable


def reach_mask(leg: Leg, h: float, *, use_limits: bool = True, keepout: Keepout | None = None,
               half_width: float | None = None, chunk: int = 1 << 20):
    """Reachability of every vertex of a cubic grid around the shoulder.

    Args:
        leg: see :func:`reachable_fn`.
        h: voxel size (m).
        use_limits, keepout: see :func:`reachable_fn`.
        half_width: the grid spans ``[-half_width, half_width]^3``; None: the leg's full
            length (``leg.lengths.sum()``), the farthest it can reach.
        chunk: vertices per IK batch (bounds memory: 64M vertices at 2 mm).

    Returns:
        A tuple ``(mask, axis)``: (n, n, n) bool, indexed ``[ix, iy, iz]``; (n,) vertex
        coordinates along each axis (m). The spacing is ``2 half_width / (n - 1)``, h
        rounded to fit.
    """
    if half_width is None:
        half_width = float(jnp.sum(leg.lengths))
    n = int(round(2 * half_width / h)) + 1
    axis = np.linspace(-half_width, half_width, n)
    reach = jax.jit(jax.vmap(reachable_fn(leg, use_limits=use_limits, keepout=keepout)))
    mask = np.zeros(n ** 3, bool)
    for k in range(0, n ** 3, chunk):
        idx = np.arange(k, min(k + chunk, n ** 3))
        ijk = np.stack(np.unravel_index(idx, (n, n, n)), -1)
        mask[idx] = np.asarray(reach(jnp.asarray(axis[ijk], jnp.float32)))
    return mask.reshape(n, n, n), axis


def signed_distance_grid(mask: np.ndarray, h: float) -> np.ndarray:
    """Signed distance samples from a vertex mask: > 0 inside, < 0 outside (m).

    Distance transforms on both sides, shifted by half a cell so the vertices next to the
    boundary sit at +-h/2 (slope 1 across the boundary).

    Args:
        mask: (n, n, n) bool, True = reachable.
        h: voxel size (m).

    Returns:
        (n, n, n) float32.
    """
    inside = distance_transform_edt(mask) - 0.5
    outside = distance_transform_edt(~mask) - 0.5
    return (np.where(mask, inside, -outside) * h).astype(np.float32)


def _cr_weights(t):
    """Catmull-Rom weights for the samples at offsets -1, 0, 1, 2; t in [0, 1)."""
    t2, t3 = t * t, t * t * t
    return 0.5 * jnp.stack([-t3 + 2 * t2 - t, 3 * t3 - 5 * t2 + 2, -3 * t3 + 4 * t2 + t, t3 - t2])


def _safe_norm(v):
    """``|v|`` with gradient 0 (not NaN) at v = 0."""
    return jnp.sqrt(jnp.maximum(v @ v, 1e-20))


@jax.tree_util.register_dataclass
@dataclass
class ReachSDF:
    """A signed distance to a leg's reach set, from grid samples. Shoulder frame.

    Defined on all of R^3: interpolated inside the grid, and outside it the value at the
    nearest grid point minus the distance to it (continuous; slope 1 back towards the
    grid). Pass it into jitted functions like any pytree.

    Args:
        D: (m, m, m) signed distance samples (m), > 0 reachable; padded with unreachable
            cells so the reach set stays clear of the grid's faces.
        lo: coordinate of index 0 along every axis (m).
        h: voxel size (m).
    """
    D: jax.Array
    lo: float = field(metadata=dict(static=True))
    h: float = field(metadata=dict(static=True))

    @property
    def size(self) -> int:
        return self.D.shape[0]

    def sdf(self, x: jax.Array) -> jax.Array:
        """Signed distance at ``x`` (3,): tricubic Catmull-Rom (C1). Scalar, metres."""
        q = jnp.clip(x, self.lo + self.h, self.lo + (self.size - 3) * self.h)   # 4x4x4 stencil fits
        u = (q - self.lo) / self.h
        i = jnp.floor(u).astype(jnp.int32)
        t = u - i
        ix = i[:, None] + jnp.arange(-1, 3)                                     # (3, 4)
        patch = self.D[ix[0][:, None, None], ix[1][None, :, None], ix[2][None, None, :]]
        d = jnp.einsum("ijk,i,j,k->", patch, _cr_weights(t[0]), _cr_weights(t[1]),
                       _cr_weights(t[2]))
        return d - _safe_norm(x - q)

    def sdf_linear(self, x: jax.Array) -> jax.Array:
        """Signed distance at ``x`` (3,): trilinear (C0; gradient jumps at cell faces)."""
        q = jnp.clip(x, self.lo, self.lo + (self.size - 1) * self.h)
        u = (q - self.lo) / self.h
        d = map_coordinates(self.D, [u[0], u[1], u[2]], order=1, mode="nearest")
        return d - _safe_norm(x - q)

    def save(self, path) -> None:
        """Write to an ``.npz``."""
        np.savez_compressed(path, D=np.asarray(self.D), lo=self.lo, h=self.h)

    @classmethod
    def load(cls, path) -> "ReachSDF":
        """Read from an ``.npz`` written by :meth:`save`."""
        z = np.load(path)
        return cls(D=jnp.asarray(z["D"]), lo=float(z["lo"]), h=float(z["h"]))


def make_reach_sdf(leg: Leg, h: float, *, use_limits: bool = True,
                   keepout: Keepout | None = None, half_width: float | None = None,
                   pad: int = 4) -> ReachSDF:
    """Build a leg's reach signed distance at voxel size ``h``.

    Args:
        leg: see :func:`reachable_fn`.
        h: voxel size (m). Build time and memory grow as 1/h^3 (1 cm: 2.8 MB, 0.3 s;
            5 mm: 19 MB, 1.5 s; 2 mm: 274 MB, ~20 s on a laptop CPU).
        use_limits, keepout, half_width: see :func:`reach_mask`.
        pad: unreachable cells added on every side.

    Returns:
        A :class:`ReachSDF`.
    """
    mask, axis = reach_mask(leg, h, use_limits=use_limits, keepout=keepout,
                            half_width=half_width)
    step = float(axis[1] - axis[0])
    D = signed_distance_grid(np.pad(mask, pad), step)
    return ReachSDF(D=jnp.asarray(D), lo=float(axis[0]) - pad * step, h=step)
