"""Samplers for the phases of a crawl step (:mod:`.step`): kinematically valid candidates
only, no scoring, no walking direction.

One step of swing leg i, from stance S with its posture (``docs/planner-design.md``):

1. B_lift: :func:`sample_body` with all four feet planted.
2. (lift leg i: ``Kit.lift_leg``, deterministic)
3. B_plant: :func:`sample_body` with the tripod planted, leg i lifted (it rides along).
4. f': :func:`sample_foothold`, footholds for leg i at the B_plant body.

Each draws ``n`` candidates uniformly at random and returns which are valid (IK reaches,
:func:`.stance.checks` pass) with their postures. Where to sample is the conditioning, a
keyword argument: a :class:`BodyRegion` for the body, nothing for the footholds (leg i's
reach ball). The walking direction is not a sampler input; it enters through the scores
(:func:`.scoring.direction_scores`), or through a region that already lies ahead.

Body heights are absolute: the body's distance to the surface (the planted footholds,
along the body's z) is drawn from the region's height band, so it cannot drift over the
steps.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jaxlie import SO3

from controlkit.kinematics.types import Posture
from controlkit.se3 import SE3

from .stance import Kit, _plant, body_boxes, body_terrain_ok, move_body, spacing_ok


@jax.tree_util.register_dataclass
@dataclass
class BodyRegion:
    """Where to sample a body pose, relative to the current body (traced: changing it
    does not recompile).

    Args:
        xy_lo, xy_hi: (2,) bounds of the offset in the body's x, y (the surface plane) (m).
        rpy_lo, rpy_hi: (3,) bounds of the rotation relative to the current orientation,
            roll / pitch / yaw about the body's axes (rad).
        height_lo, height_hi: bounds of the body's distance to the surface (absolute) (m).
    """
    xy_lo: jax.Array
    xy_hi: jax.Array
    rpy_lo: jax.Array
    rpy_hi: jax.Array
    height_lo: jax.Array
    height_hi: jax.Array


def region(x=(-0.08, 0.08), y=(-0.08, 0.08), height=(0.12, 0.20), roll=(0.0, 0.0),
           pitch=(0.0, 0.0), yaw=(0.0, 0.0)) -> BodyRegion:
    """A :class:`BodyRegion` from per-axis ``(lo, hi)`` bounds (m, rad)."""
    a = lambda *v: jnp.asarray(v, float)
    return BodyRegion(xy_lo=a(x[0], y[0]), xy_hi=a(x[1], y[1]),
                      rpy_lo=a(roll[0], pitch[0], yaw[0]), rpy_hi=a(roll[1], pitch[1], yaw[1]),
                      height_lo=a(height[0]), height_hi=a(height[1]))


def gap(kit: Kit, body: SE3, stance: jax.Array, planted: jax.Array) -> jax.Array:
    """The body's distance to the surface: from the mean planted foothold, along the
    body's z (m)."""
    feet = kit.footholds.position[stance]
    w = planted.astype(feet.dtype)[:, None]
    mean = (w * feet).sum(0) / w.sum()
    return (body.translation() - mean) @ body.rotation().as_matrix()[:, 2]


def sample_body(kit: Kit, key, n: int, posture: Posture, stance, planted, *,
                region: BodyRegion):
    """``n`` body poses in ``region`` around ``posture.body``, the ``planted`` feet held
    (the others keep their joint angles: their feet ride along).

    Offset uniform in the region's xy box (body frame), rotation uniform in its rpy box
    (composed with the current orientation), height uniform in its band.

    Args:
        kit: the stance kit.
        key: PRNG key.
        n: candidates (static).
        posture: the current posture.
        stance: (L,) foothold indices.
        planted: (L,) bool, the feet that stay down.
        region: where to sample.

    Returns:
        ``(ok, postures)``: (n,) valid (IK and every check), (n,) postures.
    """
    k_xy, k_rpy, k_h = jax.random.split(key, 3)
    u = lambda k, lo, hi, shape: lo + (hi - lo) * jax.random.uniform(k, shape)
    xy = u(k_xy, region.xy_lo, region.xy_hi, (n, 2))
    rpy = u(k_rpy, region.rpy_lo, region.rpy_hi, (n, 3))
    h = u(k_h, region.height_lo, region.height_hi, (n,))
    body = posture.body
    dz = h - gap(kit, body, stance, planted)
    R = body.rotation().as_matrix()
    t = body.translation() + jnp.concatenate([xy, dz[:, None]], 1) @ R.T
    rot = body.rotation() @ jax.vmap(SO3.from_rpy_radians)(rpy[:, 0], rpy[:, 1], rpy[:, 2])
    bodies = SE3.from_rotation_and_translation(rot, t)
    move = jax.vmap(lambda b: move_body(kit.robot, kit.scene, kit.cfg, kit.footholds,
                                        posture, stance, planted, b)[:2])
    return move(bodies)


def sample_foothold(kit: Kit, key, n: int, posture: Posture, stance, i):
    """f': ``n`` footholds for leg ``i`` at ``posture.body``, planted.

    Drawn uniformly among leg ``i``'s candidates (:func:`.stance.candidates_at`: the
    footholds in its reach ball; the current foothold excluded), without replacement if
    ``n <= k_per_leg``, else with; each planted on every IK branch and checked
    (:func:`.stance._plant`), plus the foot spacing and the body against the terrain.

    Returns:
        ``(ok, postures, stances)``: (n,) valid, (n,) postures, (n, num_legs) stances.
    """
    robot, cfg, fh = kit.robot, kit.cfg, kit.footholds
    L = robot.num_legs
    cand = kit.candidates(posture.body)
    row = cand.idx[i]
    mask = cand.mask[i] & (row != stance[i])
    k_pick, k_plant = jax.random.split(key)
    logits = jnp.where(mask, 0.0, -jnp.inf)
    if n <= row.shape[0]:                           # n distinct, uniform among the mask
        _, sel = jax.lax.top_k(jax.random.gumbel(k_pick, row.shape) + logits, n)
    else:                                           # more than the candidates: with replacement
        sel = jax.random.categorical(k_pick, logits, shape=(n,))
    f = row[sel]
    shoulder = robot.shoulders(posture.body)[i]
    box = body_boxes(cfg, posture.body)
    planted = jnp.ones(L, bool)

    def one(k, fi):
        ok, theta = _plant(k, robot, kit.scene, cfg, shoulder, box, fh[fi])
        spaced = spacing_ok(cfg, fh[stance.at[i].set(fi)], planted)
        return ok & spaced, theta

    ok, thetas = jax.vmap(one)(jax.random.split(k_plant, n), f)
    ok = ok & mask[sel] & body_terrain_ok(robot, kit.scene, cfg, posture.body)
    th = jnp.broadcast_to(posture.thetas, (n,) + posture.thetas.shape).at[:, i].set(thetas)
    postures = Posture(posture.body.broadcast_to((n,)), th)
    stances = jnp.broadcast_to(stance, (n, L)).at[:, i].set(f)
    return ok, postures, stances


class Samplers:
    """The samplers jitted over one kit (``n`` static: one compile per n; the region is
    traced).

    Args:
        kit: the stance kit.
    """

    def __init__(self, kit: Kit):
        self.kit = kit
        self.body = jax.jit(partial(sample_body, kit), static_argnums=1)
        self.foothold = jax.jit(partial(sample_foothold, kit), static_argnums=1)
