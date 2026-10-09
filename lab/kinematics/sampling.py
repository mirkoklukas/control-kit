"""Posture sampling with a reach grid (``posture-sampling.ipynb``).

Footholds are given as the points the FK foot must reach (``targets``, world frame: the
foothold plus the pivot height along its normal). A grid lookup (:meth:`ReachGrid.lookup`)
gives every branch's validity, angles and stored foot at the nearest grid vertex: an
approximate posture (the foot up to ``sqrt(3) h / 2`` off the foothold), self-consistent.

- :func:`fixed_body_sampler`: n footholds and angles for one leg at a fixed body; one lookup
  per foothold, then draws.
- :func:`joint_sampler`: n body poses in a region around a centre, each with footholds and
  angles for the given legs (``TRIES`` tries per leg from a superset of footholds).

``method="ik"`` uses the analytic IK (``branches_fn``) instead of the grid, for comparison.
"""
import jax
import jax.numpy as jnp

from controlkit.se3 import SE3

from .reach_sdf import ReachGrid


def draw_body(key, center: SE3, lo, hi) -> SE3:
    """A body pose in the region: offset (x, y, z, m) and roll / pitch / yaw (deg) uniform in
    ``[lo, hi]`` (6,), in the centre pose's frame."""
    u = jax.random.uniform(key, (6,), minval=lo, maxval=hi)
    return center @ SE3.from_te(u[:3], jnp.deg2rad(u[3:]), "xyz")


def draw_masked(key, mask, shape):
    """Indices uniform among ``mask`` (M,), with replacement: inverse CDF, O(log M) each.

    (``jax.random.categorical`` with a sample shape draws shape x M Gumbel noise.)
    """
    c = jnp.cumsum(mask)
    u = jax.random.uniform(key, shape) * c[-1]
    return jnp.minimum(jnp.searchsorted(c, u, side="right"), mask.shape[0] - 1)


def _branch(key, ok):
    """One valid branch per row, uniformly (Gumbel-max); ok (..., B) -> (...)."""
    return jnp.argmax(jnp.where(ok, jax.random.gumbel(key, ok.shape), -jnp.inf), -1)


def _lookup(grid: ReachGrid, branches, method, Xs):
    """``(ok, thetas, feet)`` for shoulder-frame points Xs (..., 3): grid or IK."""
    flat = Xs.reshape(-1, 3)
    if method == "grid":
        _, ok, th, feet = jax.vmap(grid.lookup)(flat)
    else:
        ok, th = jax.vmap(branches)(flat)
        feet = jnp.broadcast_to(flat[:, None], th.shape[:-1] + (3,))       # IK is exact
    lead = Xs.shape[:-1]
    return ok.reshape(lead + ok.shape[1:]), th.reshape(lead + th.shape[1:]), \
        feet.reshape(lead + feet.shape[1:])


def fixed_body_sampler(robot, grid: ReachGrid, targets, leg: int, n: int, *, branches=None,
                       method="grid", reach=None):
    """n footholds and angles for leg ``leg`` at a fixed body pose.

    Every foothold is looked up once (grid or IK), then n draws among the reachable ones.

    Args:
        robot: the :class:`Robot` (shoulders, leg).
        grid: a :meth:`ReachGrid.device` grid.
        targets: (M, 3) points the foot must reach, world frame.
        leg: the leg.
        n: draws (static).
        branches: ``branches_fn(...)`` for ``method="ik"``.
        method: "grid" or "ik".
        reach: ball radius for the grid's pre-filter; None: the leg's full length.

    Returns:
        A jitted ``(key, body) -> (valid, footholds, thetas, feet)``: (n,) bool, (n,) foothold
        indices, (n, n_joints), (n, 3) feet (world).
    """
    R = float(jnp.sum(robot.leg.lengths)) if reach is None else reach

    @jax.jit
    def run(key, body):
        k1, k2 = jax.random.split(key)
        sh = robot.shoulders(body)[leg]
        Xs = jax.vmap(sh.inverse().apply)(targets)                         # (M, 3)
        ok, th, feet = _lookup(grid, branches, method, Xs)
        reachable = ok.any(-1) & (jnp.sum(Xs ** 2, -1) <= R ** 2)
        f = draw_masked(k1, reachable, (n,))
        b = _branch(k2, ok[f])
        return ok[f].any(-1), f, th[f, b], jax.vmap(sh.apply)(feet[f, b])

    return run


def superset(key, robot, targets, center: SE3, lo, hi, n_delta: int, reach: float):
    """Per leg, the footholds within ``reach + delta`` of its shoulder at the centre pose, as
    a cumulative count (L, M) for drawing; and delta.

    delta = max shoulder displacement ``|s(body) - s0|`` over the region. Any foothold x
    reachable from a sampled body has ``|x - s| <= reach``, so ``|x - s0| <= reach + delta``
    (triangle inequality): the ``reach + delta`` ball around s0 holds them all.

    NOTE: delta is the max over ``n_delta`` sampled bodies, which slightly underestimates the
    true max (samples rarely hit the region's corners; for x, y +-6 cm, z +-3 cm, rpy
    +-5 / 5 / 10 deg: ~103 mm sampled vs. a bound ``|t|_max + 2 r sin(phi_max / 2)`` ~ 111 mm).
    Footholds reachable only from the most extreme bodies can be missed. Accepted.
    """
    s0 = robot.shoulders(center).translation()                              # (L, 3)
    shs = jax.vmap(lambda k: robot.shoulders(draw_body(k, center, lo, hi)).translation())(
        jax.random.split(key, n_delta))
    delta = jnp.linalg.norm(shs - s0[None], axis=-1).max()
    mask = jnp.linalg.norm(targets[None] - s0[:, None], axis=-1) <= reach + delta
    return jnp.cumsum(mask, -1), delta


def _leg_tries(key, grid, branches, method, targets, sh, csum, tries):
    """One leg, all n samples: ``tries`` footholds each from the superset (csum (M,)), the
    first reachable one and a random valid branch. sh: (n,) shoulder poses."""
    k1, k2 = jax.random.split(key)
    n = sh.wxyz_xyz.shape[0]
    M = csum.shape[0]
    u = jax.random.uniform(k1, (n, tries)) * csum[-1]
    f = jnp.minimum(jnp.searchsorted(csum, u, side="right"), M - 1)        # (n, tries)
    Xs = jax.vmap(lambda s, x: jax.vmap(s.inverse().apply)(x))(sh, targets[f])
    ok, th, feet = _lookup(grid, branches, method, Xs)
    r = jnp.arange(n)
    t = jnp.argmax(ok.any(-1), -1)                                          # first reachable try
    ok_t = ok[r, t]
    b = _branch(k2, ok_t)
    foot = jax.vmap(lambda s, x: s.apply(x))(sh, feet[r, t, b])
    return ok_t.any(-1), f[r, t], th[r, t, b], foot


def joint_sampler(robot, grid: ReachGrid, targets, legs, n: int, *, branches=None,
                  method="grid", tries: int = 8, n_delta: int = 4096, reach=None):
    """n body poses in a region around a centre, each with footholds and angles for ``legs``.

    Per call: the superset (:func:`superset`), n bodies (:func:`draw_body`), then per leg
    (an outer loop over all samples at once) ``tries`` footholds from its superset, the
    first reachable one. A sample is valid if every leg in ``legs`` found one.

    Args:
        robot, grid, targets, branches, method, reach: see :func:`fixed_body_sampler`.
        legs: the legs to sample (static tuple).
        n: samples (static).
        tries: footholds tried per leg and sample.
        n_delta: bodies sampled for delta.

    Returns:
        A jitted ``(key, center, lo, hi) -> (valid, bodies, footholds, thetas, feet, delta)``:
        (n,) bool, (n,) SE3, (n, len(legs)) indices, (n, len(legs), n_joints), (n, len(legs),
        3) feet (world), scalar. ``center`` and the region are traced (no recompile).
    """
    legs = tuple(legs)
    R = float(jnp.sum(robot.leg.lengths)) if reach is None else reach

    @jax.jit
    def run(key, center, lo, hi):
        k_sup, k_body, *k_legs = jax.random.split(key, 2 + len(legs))
        csum, delta = superset(k_sup, robot, targets, center, lo, hi, n_delta, R)
        bodies = jax.vmap(lambda k: draw_body(k, center, lo, hi))(jax.random.split(k_body, n))
        sh = jax.vmap(robot.shoulders)(bodies)                              # (n, L)
        out = [_leg_tries(k, grid, branches, method, targets, sh[:, i], csum[i], tries)
               for k, i in zip(k_legs, legs)]
        valid = jnp.stack([o[0] for o in out], 1).all(1)
        return (valid, bodies, *(jnp.stack([o[j] for o in out], 1) for j in (1, 2, 3)), delta)

    return run
