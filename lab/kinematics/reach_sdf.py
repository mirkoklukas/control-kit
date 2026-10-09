"""A leg's reach set on a voxel grid, with the joint angles, and as a signed distance.

Built around the shoulder (the leg's base frame, shoulder at the origin): per grid vertex,
every IK branch, whether it passes the filters (joint limits, a keep-out box such as the
body), and its joint angles: a :class:`ReachGrid`. From its reachability mask, Euclidean
distance transforms give a :class:`ReachSDF`, evaluated by interpolation in JAX, so
gradients (and the chain rule through a body pose) come from autodiff.

    grid = make_reach_grid(leg, h=0.005, keepout=Keepout(body_pose, center, half))
    grid.ok[i, j, k, b], grid.thetas[i, j, k, b]     # branch b at vertex grid.points()[i, j, k]
    grid.feet[i, j, k, b]                            # FK of that theta (~ the vertex)
    grid.contact[i, j, k, b]                         # its foot-to-knee direction (ankle cone)
    inside, ok, thetas, feet = grid.device().lookup(x_shoulder)   # nearest vertex, in JAX
    sdf = grid.sdf()
    d = sdf.sdf(x)                                   # > 0: reachable, metres
    d, g = jax.value_and_grad(sdf.sdf)(x)
    ok = sdf.sdf(x) >= sdf.h                         # the constraint, with a margin

Accuracy (``reach-grid-limits.ipynb``, resolution sweep): the mask loses where the boundary
lies between vertices, so the zero level is only good to about h and sits slightly outside
the true boundary. **Use a margin of at least h**: projecting points onto the set with
SLSQP, every result with ``sdf >= h`` was truly reachable (exact IK, 1 cm / 5 mm / 2 mm);
with ``sdf >= h / 2`` only ~75% were, although random points with ``sdf > h / 2`` all were
(the solver ends on the boundary, where the bias is largest). Prefer :meth:`ReachSDF.sdf`
(Catmull-Rom, C1) over :meth:`ReachSDF.sdf_linear` (trilinear) in a solver: ~17 vs. ~40-80
SLSQP iterations.
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


def branches_fn(leg: Leg, *, use_limits: bool = True, keepout: Keepout | None = None):
    """Every IK branch at a foot position, and whether it passes the enabled filters.

    Args:
        leg: a leg with analytic IK (``ik_from_foot``), e.g. :class:`Leg3DOF`.
        use_limits: gate the branches by ``leg.limits``.
        keepout: no branch is valid for a foot inside this box; None: no box.

    Returns:
        A function ``(3,) -> (ok, thetas)``, (B,) bool and (B, n_joints), jit- and
        vmap-able. Angles of invalid branches are finite but meaningless.
    """
    def branches(foot):
        ok, thetas = leg.ik_from_foot(foot)
        if use_limits:
            ok = ok & candidates.in_limits(thetas, leg.limits)
        if keepout is not None:
            ok = ok & ~keepout.contains(foot)
        return ok, thetas

    return branches


def reachable_fn(leg: Leg, *, use_limits: bool = True, keepout: Keepout | None = None):
    """Exact membership in the reach set: some IK branch passes the enabled filters.

    Args:
        leg, use_limits, keepout: see :func:`branches_fn`.

    Returns:
        A function ``(3,) -> scalar bool``, jit- and vmap-able.
    """
    branches = branches_fn(leg, use_limits=use_limits, keepout=keepout)
    return lambda foot: branches(foot)[0].any()


def _over_box(f, lower_left, h, shape, chunk):
    """Apply a batched ``f`` to every vertex of a box grid, in chunks; outputs stacked to
    ``shape + out.shape[1:]``."""
    N = int(np.prod(shape))
    outs = []
    for k in range(0, N, chunk):
        ijk = np.stack(np.unravel_index(np.arange(k, min(k + chunk, N)), shape), -1)
        outs.append(jax.tree.map(np.asarray, f(jnp.asarray(lower_left + h * ijk, jnp.float32))))
    return jax.tree.map(lambda *a: np.concatenate(a).reshape(tuple(shape) + a[0].shape[1:]), *outs)


def signed_distance_grid(mask: np.ndarray, h: float) -> np.ndarray:
    """Signed distance samples from a vertex mask: > 0 inside, < 0 outside (m).

    Distance transforms on both sides, shifted by half a cell so the vertices next to the
    boundary sit at +-h/2 (slope 1 across the boundary).

    Args:
        mask: (nx, ny, nz) bool, True = reachable.
        h: voxel size (m).

    Returns:
        (nx, ny, nz) float32.
    """
    inside = distance_transform_edt(mask) - 0.5
    outside = distance_transform_edt(~mask) - 0.5
    return (np.where(mask, inside, -outside) * h).astype(np.float32)


def _cr_weights(t):
    """Catmull-Rom weights for the samples at offsets -1, 0, 1, 2; t in [0, 1)."""
    t2, t3 = t * t, t * t * t
    return 0.5 * jnp.stack([-t3 + 2 * t2 - t, 3 * t3 - 5 * t2 + 2, -3 * t3 + 4 * t2 + t, t3 - t2])


def _catmull_rom(D, lower_left, h, x):
    """Tricubic Catmull-Rom interpolation of grid samples ``D`` (nx, ny, nz, ...) at ``x``
    (3,), clamped to where the 4x4x4 stencil fits. Returns ``(value, clamped x)``."""
    upper = lower_left + h * (jnp.array(D.shape[:3]) - 1)
    q = jnp.clip(x, lower_left + h, upper - 2 * h)
    u = (q - lower_left) / h
    i = jnp.floor(u).astype(jnp.int32)
    t = u - i
    ix = i[:, None] + jnp.arange(-1, 3)                                        # (3, 4)
    patch = D[ix[0][:, None, None], ix[1][None, :, None], ix[2][None, None, :]]
    return jnp.einsum("ijk...,i,j,k->...", patch, _cr_weights(t[0]), _cr_weights(t[1]),
                      _cr_weights(t[2])), q


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
        D: (nx, ny, nz) signed distance samples (m), > 0 reachable; padded with
            unreachable cells so the reach set stays clear of the grid's faces.
        lower_left: (3,) position of sample (0, 0, 0), the corner with the smallest x, y, z
            (m). Sample (i, j, k) sits at ``lower_left + h (i, j, k)``.
        h: voxel size (m).
    """
    D: jax.Array
    lower_left: jax.Array
    h: float = field(metadata=dict(static=True))

    @property
    def upper_right(self) -> jax.Array:
        """(3,) position of the last sample, the corner with the largest x, y, z."""
        return self.lower_left + self.h * (jnp.array(self.D.shape) - 1)

    def sdf(self, x: jax.Array) -> jax.Array:
        """Signed distance at ``x`` (3,): tricubic Catmull-Rom (C1). Scalar, metres."""
        d, q = _catmull_rom(self.D, self.lower_left, self.h, x)
        return d - _safe_norm(x - q)

    def sdf_linear(self, x: jax.Array) -> jax.Array:
        """Signed distance at ``x`` (3,): trilinear (C0; gradient jumps at cell faces)."""
        q = jnp.clip(x, self.lower_left, self.upper_right)
        u = (q - self.lower_left) / self.h
        d = map_coordinates(self.D, [u[0], u[1], u[2]], order=1, mode="nearest")
        return d - _safe_norm(x - q)

    def save(self, path) -> None:
        """Write to an ``.npz``."""
        np.savez_compressed(path, D=np.asarray(self.D), lower_left=np.asarray(self.lower_left),
                            h=self.h)

    @classmethod
    def load(cls, path) -> "ReachSDF":
        """Read from an ``.npz`` written by :meth:`save`."""
        z = np.load(path)
        return cls(D=jnp.asarray(z["D"]), lower_left=jnp.asarray(z["lower_left"]), h=float(z["h"]))


@jax.tree_util.register_dataclass
@dataclass
class VectorField:
    """A 3-vector per grid vertex, interpolated (Catmull-Rom, C1) and normalized; e.g. one IK
    branch's foot-to-knee direction (:meth:`ReachGrid.contact_field`). Shoulder frame.

    Args:
        V: (nx, ny, nz, 3) samples.
        lower_left: (3,) position of sample (0, 0, 0) (m).
        h: voxel size (m).
    """
    V: jax.Array
    lower_left: jax.Array
    h: float = field(metadata=dict(static=True))

    def __call__(self, x: jax.Array) -> jax.Array:
        """The unit vector at ``x`` (3,) (clamped to the grid)."""
        v, _ = _catmull_rom(self.V, self.lower_left, self.h, x)
        return v / _safe_norm(v)


@jax.tree_util.register_dataclass
@dataclass
class ReachGrid:
    """A leg's reach set on a box grid around the shoulder: every IK branch per vertex,
    with the foot each branch's angles put. Shoulder frame. Host (numpy) arrays as built
    (100 B per vertex with 4 branches); :meth:`device` for JAX (:meth:`lookup`).

    Args:
        ok: (nx, ny, nz, B) bool, branch b reaches vertex (i, j, k) and passes the filters.
        thetas: (nx, ny, nz, B, n_joints) float32, the branches' joint angles (meaningless
            where not ``ok``).
        feet: (nx, ny, nz, B, 3) float32, forward kinematics of ``thetas``: the foot the
            stored angles put (the vertex, up to float error, for an IK-built grid; not
            for a grid built from forward kinematics).
        contact: (nx, ny, nz, B, 3) float32, the unit foot-to-knee direction of each branch
            (the leg's ``contact_vector``: compare with a surface normal for the ankle cone).
        lower_left: (3,) position of vertex (0, 0, 0), the corner with the smallest x, y, z
            (m). Vertex (i, j, k) sits at ``lower_left + h (i, j, k)``.
        h: voxel size (m).
    """
    ok: np.ndarray
    thetas: np.ndarray
    feet: np.ndarray
    contact: np.ndarray
    lower_left: np.ndarray
    h: float = field(metadata=dict(static=True))

    @property
    def shape(self) -> tuple:
        """``(nx, ny, nz)``: vertices per axis."""
        return self.ok.shape[:3]

    @property
    def mask(self) -> np.ndarray:
        """(nx, ny, nz) bool: some branch reaches the vertex."""
        return self.ok.any(-1)

    def device(self) -> "ReachGrid":
        """The same grid with JAX arrays (for :meth:`lookup` inside jitted code)."""
        return ReachGrid(ok=jnp.asarray(self.ok), thetas=jnp.asarray(self.thetas),
                         feet=jnp.asarray(self.feet), contact=jnp.asarray(self.contact),
                         lower_left=jnp.asarray(self.lower_left), h=self.h)

    def lookup(self, x: jax.Array):
        """The nearest vertex's branches for a foot position ``x`` (3,), shoulder frame.

        ``round((x - lower_left) / h)``: a few flops and one gather. Use on a
        :meth:`device` grid inside jitted code.

        Returns:
            A tuple ``(inside, ok, thetas, feet)``: scalar bool, x within half a cell of the
            grid; (B,) bool, valid branches (all false outside the grid); (B, n_joints)
            angles; (B, 3) their feet (shoulder frame, up to ``sqrt(3) h / 2`` from x).
        """
        dims = jnp.array(self.ok.shape[:3])
        ijk = jnp.round((x - self.lower_left) / self.h).astype(jnp.int32)
        inside = jnp.all((ijk >= 0) & (ijk < dims))
        i, j, k = jnp.clip(ijk, 0, dims - 1)
        return inside, self.ok[i, j, k] & inside, self.thetas[i, j, k], self.feet[i, j, k]

    def contact_field(self, b: int, pad: int = 4) -> VectorField:
        """Branch ``b``'s foot-to-knee direction as a smooth field (for the ankle cone in an
        NLP, with the branch fixed per foot).

        Where branch b is not valid, its samples are replaced by the nearest valid ones (so
        the interpolation does not mix in meaningless angles near the boundary); padded by
        repeating the edge.

        Args:
            b: the branch.
            pad: cells added on every side (as :meth:`sdf`).

        Returns:
            A :class:`VectorField`.
        """
        valid = self.ok[..., b]
        _, idx = distance_transform_edt(~valid, return_indices=True)
        V = self.contact[..., b, :][idx[0], idx[1], idx[2]]
        V = np.pad(V, [(pad, pad)] * 3 + [(0, 0)], mode="edge")
        return VectorField(V=jnp.asarray(V, jnp.float32),
                           lower_left=jnp.asarray(self.lower_left - pad * self.h), h=self.h)

    def points(self) -> np.ndarray:
        """(nx, ny, nz, 3) vertex positions (m)."""
        ijk = np.stack(np.meshgrid(*[np.arange(n) for n in self.shape], indexing="ij"), -1)
        return self.lower_left + self.h * ijk

    def sdf(self, pad: int = 4) -> ReachSDF:
        """The signed distance to the reach set, from :attr:`mask`.

        Args:
            pad: unreachable cells added on every side (keeps the reach set clear of the
                grid's faces, where the extension outside the grid takes over).

        Returns:
            A :class:`ReachSDF`.
        """
        D = signed_distance_grid(np.pad(self.mask, pad), self.h)
        return ReachSDF(D=jnp.asarray(D), lower_left=jnp.asarray(self.lower_left - pad * self.h),
                        h=self.h)

    def save(self, path) -> None:
        """Write to an ``.npz``."""
        np.savez_compressed(path, ok=self.ok, thetas=self.thetas, feet=self.feet,
                            contact=self.contact, lower_left=self.lower_left, h=self.h)

    @classmethod
    def load(cls, path) -> "ReachGrid":
        """Read from an ``.npz`` written by :meth:`save`."""
        z = np.load(path)
        return cls(ok=z["ok"], thetas=z["thetas"], feet=z["feet"], contact=z["contact"],
                   lower_left=z["lower_left"], h=float(z["h"]))


def make_reach_grid(leg: Leg, h: float, *, use_limits: bool = True,
                    keepout: Keepout | None = None, half_width: float | None = None,
                    crop: int | None = 1, chunk: int = 1 << 20) -> ReachGrid:
    """A leg's :class:`ReachGrid` at voxel size ``h``, from IK.

    Two passes: reachability over the full cube ``[-half_width, half_width]^3``, then every
    branch over the cube cropped to the reachable vertices' bounding box (plus ``crop``
    cells), so the branch arrays only cover where the leg reaches (about half the cube with
    a +-90 deg coxa).

    Args:
        leg, use_limits, keepout: see :func:`branches_fn`.
        h: voxel size (m); rounded so the cube's half-width is a multiple of it. Memory with
            4 branches (angles, feet, contact directions), uncropped: 1 cm ~80 MB, 5 mm
            ~620 MB, 2 mm ~9.5 GB.
        half_width: the cube to search; None: the leg's full length (``leg.lengths.sum()``),
            the farthest it can reach.
        crop: cells kept around the bounding box of the reachable vertices; None: keep the
            full cube.
        chunk: vertices per IK batch (bounds device memory).

    Returns:
        A :class:`ReachGrid`.
    """
    if half_width is None:
        half_width = float(jnp.sum(leg.lengths))
    n = int(round(2 * half_width / h)) + 1
    h = 2 * half_width / (n - 1)
    lower_left = np.full(3, -half_width)
    shape = (n, n, n)
    if crop is not None:
        reach = jax.jit(jax.vmap(reachable_fn(leg, use_limits=use_limits, keepout=keepout)))
        idx = np.argwhere(_over_box(reach, lower_left, h, shape, chunk))
        lo_i = np.maximum(idx.min(0) - crop, 0)
        hi_i = np.minimum(idx.max(0) + crop, n - 1)
        lower_left, shape = lower_left + h * lo_i, tuple(hi_i - lo_i + 1)
    branches = branches_fn(leg, use_limits=use_limits, keepout=keepout)

    def with_feet(x):
        ok, thetas = branches(x)
        xpos = jax.vmap(lambda t: leg.forward(t).translation())(thetas)       # (B, n+1, 3)
        v = xpos[:, -2] - xpos[:, -1]                                          # foot -> knee
        return ok, thetas, xpos[:, -1], v / jnp.linalg.norm(v, axis=-1, keepdims=True)

    ok, thetas, feet, contact = _over_box(jax.jit(jax.vmap(with_feet)), lower_left, h, shape, chunk)
    return ReachGrid(ok=ok, thetas=thetas.astype(np.float32), feet=feet.astype(np.float32),
                     contact=contact.astype(np.float32), lower_left=lower_left, h=h)


def reach_mask(leg: Leg, h: float, *, use_limits: bool = True, keepout: Keepout | None = None,
               half_width: float | None = None, chunk: int = 1 << 20):
    """Reachability of every vertex of the full cube around the shoulder (no angles).

    Args:
        leg, h, use_limits, keepout, half_width, chunk: see :func:`make_reach_grid`.

    Returns:
        A tuple ``(mask, axis)``: (n, n, n) bool, indexed ``[ix, iy, iz]``; (n,) vertex
        coordinates along each axis (m).
    """
    if half_width is None:
        half_width = float(jnp.sum(leg.lengths))
    n = int(round(2 * half_width / h)) + 1
    axis = np.linspace(-half_width, half_width, n)
    reach = jax.jit(jax.vmap(reachable_fn(leg, use_limits=use_limits, keepout=keepout)))
    return _over_box(reach, np.full(3, axis[0]), axis[1] - axis[0], (n, n, n), chunk), axis


def make_reach_sdf(leg: Leg, h: float, *, use_limits: bool = True,
                   keepout: Keepout | None = None, half_width: float | None = None,
                   pad: int = 4) -> ReachSDF:
    """Build a leg's reach signed distance at voxel size ``h`` (mask only, no angles).

    Args:
        leg, h, use_limits, keepout, half_width: see :func:`make_reach_grid`. Build time
            and memory grow as 1/h^3 (1 cm: 2.8 MB, 0.3 s; 5 mm: 19 MB, 1.5 s; 2 mm:
            274 MB, ~20 s on a laptop CPU).
        pad: unreachable cells added on every side.

    Returns:
        A :class:`ReachSDF`.
    """
    mask, axis = reach_mask(leg, h, use_limits=use_limits, keepout=keepout,
                            half_width=half_width)
    step = float(axis[1] - axis[0])
    D = signed_distance_grid(np.pad(mask, pad), step)
    return ReachSDF(D=jnp.asarray(D), lower_left=jnp.full(3, float(axis[0]) - pad * step),
                    h=step)
