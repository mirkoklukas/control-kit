"""Quick-and-dirty collision, in pure jax. For filtering postures fast.

The cheap tier. Signed distance fields for a few primitive shapes (boxes, spheres),
and a posture reduced to a point cloud queried against them. Approximate -- it can
miss a thin obstacle slipping between sampled points -- so use it to *reject* the
obviously-colliding and let an exact check (MuJoCo) confirm the survivors. It
``vmap``s over a whole batch of postures in one call, which the exact engine can't.

A signed distance is negative inside a shape, zero on the surface, positive outside.
So a scene is the ``min`` over its shapes (nearest surface wins), and a
sphere/capsule of radius ``r`` is clear where ``sdf >= r``.

Leg-agnostic, like ``planar`` and ``metrics``: it takes point arrays, not a
:class:`Robot`. Turn a posture into points with ``robot.points(posture)``, then
``densify`` along the links so a bar between two clear joints is not missed.
"""

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp


# --------------------------------------------------------------------------- #
# Primitive signed distance fields                                            #
# --------------------------------------------------------------------------- #
def box_sdf(x: jax.Array, center: jax.Array, half: jax.Array) -> jax.Array:
    """Signed distance from points to an axis-aligned box.

    Args:
        x: (..., 3) query points.
        center: (..., 3) box centre, broadcastable against ``x``.
        half: (..., 3) box half-extents.

    Returns:
        (...,) signed distance; negative inside.
    """
    d = jnp.abs(x - center) - half
    outside = jnp.linalg.norm(jnp.maximum(d, 0.0), axis=-1)
    inside = jnp.minimum(jnp.max(d, axis=-1), 0.0)
    return outside + inside


def sphere_sdf(x: jax.Array, center: jax.Array, radius: jax.Array) -> jax.Array:
    """Signed distance from points to a sphere.

    Args:
        x: (..., 3) query points.
        center: (..., 3) sphere centre, broadcastable against ``x``.
        radius: (...,) sphere radius.

    Returns:
        (...,) signed distance; negative inside.
    """
    return jnp.linalg.norm(x - center, axis=-1) - radius


# --------------------------------------------------------------------------- #
# Box representation helpers                                                   #
# --------------------------------------------------------------------------- #
# Two conventions for the same axis-aligned box: (center, half-extents), which the
# SDF uses, and (lo, hi) corners, which reads naturally for a hand-placed obstacle.
def box_bounds(center: jax.Array, half: jax.Array) -> tuple[jax.Array, jax.Array]:
    """``(center, half)`` -> ``(lo, hi)`` corner form.

    Args:
        center: (..., 3) box centre.
        half: (..., 3) half-extents.

    Returns:
        ``(lo, hi)`` -- the min and max corners.
    """
    return center - half, center + half


def box_center_half(lo: jax.Array, hi: jax.Array) -> tuple[jax.Array, jax.Array]:
    """``(lo, hi)`` corners -> ``(center, half)`` form.

    Args:
        lo: (..., 3) min corner.
        hi: (..., 3) max corner.

    Returns:
        ``(center, half)``.
    """
    return 0.5 * (lo + hi), 0.5 * (hi - lo)


def aabb(points: jax.Array, axis: int = -2) -> tuple[jax.Array, jax.Array]:
    """Axis-aligned bounding box of a set of points.

    Wrap a link, a leg, or the whole robot in one box -- e.g. a coarse
    broad-phase reject, or the body's box from its corner points.

    Args:
        points: (..., k, 3) points; the box is over the ``axis`` (point) dimension.
        axis: which dimension indexes the points.

    Returns:
        ``(center, half)`` of the enclosing box.
    """
    return box_center_half(points.min(axis), points.max(axis))


def box_corners(center: jax.Array, half: jax.Array) -> jax.Array:
    """The 8 corners of a box.

    Handy for the body box: feed its corners in as query points, or use them for a
    box-vs-box test that ``sdf``-at-points would miss (crossing boxes).

    Args:
        center: (..., 3) box centre.
        half: (..., 3) half-extents.

    Returns:
        (..., 8, 3) corner points.
    """
    signs = jnp.array([[sx, sy, sz] for sx in (-1.0, 1.0)
                       for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)])   # (8, 3)
    return center[..., None, :] + signs * half[..., None, :]


@jax.tree_util.register_dataclass
@dataclass
class OBB:
    """Oriented bounding box(es), batched over any leading axes.

    Args:
        center: (..., 3) centre.
        half: (..., 3) half-extents along the box's own axes.
        rot: (..., 3, 3) rotation; its columns are the box axes.
    """

    center: jax.Array
    half: jax.Array
    rot: jax.Array

    @classmethod
    def from_lo_hi(cls, lo: jax.Array, hi: jax.Array, rot: jax.Array = None) -> "OBB":
        """A box from ``(lo, hi)`` corners, optionally oriented by ``rot``.

        The corners set the centre and half-extents; ``rot`` orients the box in
        place about that centre (default: axis-aligned).

        Args:
            lo: (..., 3) min corners.
            hi: (..., 3) max corners.
            rot: (..., 3, 3) rotation, columns are the box axes; None -> identity.

        Returns:
            An :class:`OBB` (batched to match ``lo``/``hi``).
        """
        center = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo)
        if rot is None:
            rot = jnp.broadcast_to(jnp.eye(3), center.shape[:-1] + (3, 3))
        return cls(center, half, rot)

    @property
    def shape(self) -> tuple[int, ...]:
        """Leading (batch) shape -- everything but the trailing 3."""
        return self.center.shape[:-1]

    def __getitem__(self, index) -> "OBB":
        """Index a batched OBB (leading axes) -> an OBB."""
        return OBB(self.center[index], self.half[index], self.rot[index])

    def __iter__(self):
        """Iterate over the leading axis of a batched OBB."""
        for i in range(self.shape[0]):
            yield self[i]

    def reshape(self, *shape) -> "OBB":
        """Reshape the batch (leading) axes; the trailing 3 / 3x3 are kept.

        Accepts ``reshape(2, 3)``, ``reshape((2, 3))`` or ``reshape(-1)`` -- e.g. to
        flatten ``leg_boxes`` from ``(num_legs, n)`` to ``(num_legs * n,)``.
        """
        if len(shape) == 1 and not isinstance(shape[0], int):
            shape = tuple(shape[0])
        return OBB(self.center.reshape(*shape, 3),
                   self.half.reshape(*shape, 3),
                   self.rot.reshape(*shape, 3, 3))


def overlap(a: OBB, b: OBB, *, margin: float = 0.0) -> jax.Array:
    """Do two :class:`OBB` overlap? Thin wrapper over :func:`boxes_overlap`.

    Scalar boxes in, scalar bool out; ``vmap`` for batches.
    """
    return boxes_overlap(a.center, a.half, a.rot, b.center, b.half, b.rot, margin=margin)


def boxes_overlap(center_a, half_a, rot_a, center_b, half_b, rot_b,
                  *, margin: float = 0.0, eps: float = 1e-9) -> jax.Array:
    """Do two oriented boxes overlap? Separating Axis Theorem.

    Unlike a point-vs-box SDF, this is a true solid overlap: it catches boxes that
    interpenetrate with no corner inside the other (crossing like a ``+``), and one
    box fully inside another. The AABB case falls out when both rotations are
    identity.

    Two convex solids are disjoint iff some axis separates their projections. For
    boxes the only candidates are the 3+3 face normals and the 9 pairwise
    edge-edge cross-products (the cross terms are what a face-normal-only test
    misses). If *none* separates, they overlap.

    Args:
        center_a: (3,) centre of box A.
        half_a: (3,) half-extents of A along its own axes.
        rot_a: (3, 3) rotation of A; columns are its local axes.
        center_b, half_b, rot_b: box B, likewise.
        margin: inflate the keep-out -- overlap is reported within this distance.
        eps: cross-products shorter than this (near-parallel edges) are skipped;
            they cannot be a genuine separating axis.

    Returns:
        Scalar bool: True if the boxes overlap (within ``margin``).
    """
    a_ax = rot_a.T                                     # (3, 3) rows = box A's axes
    b_ax = rot_b.T
    crosses = jnp.cross(a_ax[:, None, :], b_ax[None, :, :]).reshape(-1, 3)  # (9, 3)
    axes = jnp.concatenate([a_ax, b_ax, crosses], axis=0)                   # (15, 3)

    norms = jnp.linalg.norm(axes, axis=-1)
    valid = norms > eps
    axes = axes / jnp.where(valid, norms, 1.0)[:, None]                     # unit (or skipped)

    # projection radius of each box onto every axis: sum_i half_i |axis . box_axis_i|
    r_a = jnp.sum(jnp.abs(axes @ a_ax.T) * half_a, axis=-1)                 # (15,)
    r_b = jnp.sum(jnp.abs(axes @ b_ax.T) * half_b, axis=-1)
    sep = jnp.abs(axes @ (center_b - center_a))                            # (15,)

    separated = valid & (sep > r_a + r_b + margin)
    return ~jnp.any(separated)


def segment_box_intersection(p0, p1, lo, hi, *, eps: float = 1e-12):
    """Clip a segment to an axis-aligned box -- the Liang-Barsky slab method.

    Dimension-agnostic: pass 2-vectors for a rectangle, 3-vectors for a box. Returns
    the parameter interval where ``p0 + t (p1 - p0)`` lies inside, so you get both
    the boolean and the entry/exit points (``p0 + t0 d``, ``p0 + t1 d``).

    Args:
        p0: (..., D) segment start.
        p1: (..., D) segment end.
        lo: (..., D) box min corner.
        hi: (..., D) box max corner.
        eps: axes with ``|d| < eps`` are treated as parallel (constrain only if the
            start already lies within that slab).

    Returns:
        ``(hit, t0, t1)``. ``hit`` is a scalar bool; on a hit the intersection is
        ``t in [t0, t1] ⊆ [0, 1]``. When ``hit`` is False, ``t0``/``t1`` are
        meaningless (``t0 > t1``).
    """
    d = p1 - p0
    parallel = jnp.abs(d) < eps
    inside = (p0 >= lo) & (p0 <= hi)                 # for the parallel axes
    inv = jnp.where(parallel, 0.0, 1.0 / jnp.where(parallel, 1.0, d))
    t_lo = (lo - p0) * inv
    t_hi = (hi - p0) * inv
    # a parallel axis inside its slab gives no constraint (-inf, +inf); outside, it
    # kills the intersection (+inf, -inf), forcing t0 > t1.
    near = jnp.where(parallel, jnp.where(inside, -jnp.inf, jnp.inf),
                     jnp.minimum(t_lo, t_hi))
    far = jnp.where(parallel, jnp.where(inside, jnp.inf, -jnp.inf),
                    jnp.maximum(t_lo, t_hi))
    t0 = jnp.maximum(0.0, jnp.max(near, axis=-1))
    t1 = jnp.minimum(1.0, jnp.min(far, axis=-1))
    return t0 <= t1, t0, t1


def point_box_distance(p, lo, hi) -> jax.Array:
    """Signed distance from a point to an axis-aligned box, in ``(lo, hi)`` form.

    The corner-form counterpart of :func:`box_sdf` (which takes centre/half), so it
    pairs with :func:`segment_box_intersection`. Dimension-agnostic: 2-vectors for a
    rectangle, 3-vectors for a box. Negative inside.

    For "distance to the filled region" use ``max(0, .)``; for distance to the
    boundary use ``abs(.)``.

    Args:
        p: (..., D) query points.
        lo: (..., D) box min corner.
        hi: (..., D) box max corner.

    Returns:
        (...,) signed distance; negative inside.
    """
    return box_sdf(p, 0.5 * (lo + hi), 0.5 * (hi - lo))


# --------------------------------------------------------------------------- #
# A scene: a union of primitives                                              #
# --------------------------------------------------------------------------- #
@jax.tree_util.register_dataclass
@dataclass
class Scene:
    """A static environment: some boxes and some spheres.

    The scene's SDF is the ``min`` over its shapes. Both shape sets are stored as
    stacked arrays so :func:`sdf` evaluates all of them at once and ``vmap``s over
    query points; either set may be empty.

    Args:
        box_center: (Mb, 3) box centres.
        box_half: (Mb, 3) box half-extents.
        sphere_center: (Ms, 3) sphere centres.
        sphere_radius: (Ms,) sphere radii.
    """

    box_center: jax.Array = field(default_factory=lambda: jnp.zeros((0, 3)))
    box_half: jax.Array = field(default_factory=lambda: jnp.zeros((0, 3)))
    sphere_center: jax.Array = field(default_factory=lambda: jnp.zeros((0, 3)))
    sphere_radius: jax.Array = field(default_factory=lambda: jnp.zeros((0,)))


def sdf(scene: Scene, x: jax.Array) -> jax.Array:
    """Signed distance from points to the whole scene (nearest surface).

    Args:
        scene: the environment.
        x: (..., 3) query points.

    Returns:
        (...,) signed distance; ``+inf`` for an empty scene.
    """
    d = jnp.full(x.shape[:-1], jnp.inf)
    if scene.box_center.shape[0]:                       # (..., Mb) -> min over boxes
        db = box_sdf(x[..., None, :], scene.box_center, scene.box_half)
        d = jnp.minimum(d, db.min(-1))
    if scene.sphere_center.shape[0]:
        ds = sphere_sdf(x[..., None, :], scene.sphere_center, scene.sphere_radius)
        d = jnp.minimum(d, ds.min(-1))
    return d


# --------------------------------------------------------------------------- #
# Posture side: points along the links                                        #
# --------------------------------------------------------------------------- #
def densify(points: jax.Array, per_link: int = 3) -> jax.Array:
    """Sample points along the links, so a bar between two clear joints is not missed.

    Interpolates ``per_link`` points along each segment between consecutive points.

    Args:
        points: (..., k, 3) joint/foot positions, e.g. ``robot.points(posture)``
            (shape ``(num_legs, n+1, 3)``).
        per_link: samples per segment (>= 2 to keep the endpoints).

    Returns:
        (..., (k-1) * per_link, 3) points along the links.
    """
    a = points[..., :-1, :]                              # (..., k-1, 3) segment starts
    b = points[..., 1:, :]                               # (..., k-1, 3) segment ends
    t = jnp.linspace(0.0, 1.0, per_link)                 # (per_link,)
    pts = a[..., None, :] + t[:, None] * (b - a)[..., None, :]   # (..., k-1, per_link, 3)
    return pts.reshape(points.shape[:-2] + (-1, 3))


def grid_box(center: jax.Array, half: jax.Array, res=(2, 2, 2)) -> jax.Array:
    """A regular grid of points filling a box.

    A cheap point-cloud stand-in for a solid volume (e.g. the body): if an obstacle
    overlaps the box, a grid point lands inside it, so ``sdf < 0`` catches it -- as
    long as the grid is finer than the thinnest obstacle.

    Args:
        center: (3,) box centre.
        half: (3,) half-extents.
        res: points per axis; an int (same for all) or a 3-tuple.

    Returns:
        (prod(res), 3) grid points.
    """
    center = jnp.asarray(center)
    half = jnp.asarray(half)
    r = (res, res, res) if isinstance(res, int) else res
    axes = [jnp.linspace(center[i] - half[i], center[i] + half[i], r[i]) for i in range(3)]
    g = jnp.meshgrid(*axes, indexing="ij")
    return jnp.stack([x.reshape(-1) for x in g], axis=-1)


def clearance(scene: Scene, points: jax.Array, radius: float = 0.0) -> jax.Array:
    """Smallest signed clearance between a set of spheres and the scene.

    Args:
        scene: the environment.
        points: (P, 3) sphere centres (e.g. ``densify(robot.points(posture))``).
        radius: sphere radius (the link/foot radius).

    Returns:
        Scalar: ``min(sdf(points)) - radius``. Negative means collision; the value
        is a usable margin for a soft "keep away" cost.
    """
    return sdf(scene, points).min() - radius


def collides(scene: Scene, points: jax.Array, radius: float = 0.0,
             margin: float = 0.0) -> jax.Array:
    """Whether any sphere is within ``margin`` of the scene. Boolean.

    Args:
        scene: the environment.
        points: (P, 3) sphere centres.
        radius: sphere radius.
        margin: extra keep-out distance (reject near-misses too).

    Returns:
        Scalar bool.
    """
    return clearance(scene, points, radius) < margin
