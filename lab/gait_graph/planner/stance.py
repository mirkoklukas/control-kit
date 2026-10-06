"""Stances, their witness postures, and the moves that extend a path of them.

The climb robot's copy of ``lab/gait_graph/stance_3dof.py`` (2026-10-05): 4 radial 3-DOF
legs. Differences to that (hexapod) version:

- The body is the climb robot's square trunk, the mounts' bounding box (one box:
  :func:`body_boxes`, :func:`_body_cloud`), not the hexagon.
- The "foot" is the ankle pivot. ``cfg.foot_radius`` is its height above the pad face
  (:func:`.climb_cfg.climb_cfg`), so :func:`foot_center` puts it that far above the
  foothold along the normal: a flat pad on the surface. The pad itself (40 mm square)
  is not a collision shape here; near a planted foot, terrain contact is ignored within
  ``foot_clearance_radius``.
- ``ankle``: the tibia within ``ankle_limit_deg`` of the normal, a cone standing in for the
  Cardan ankle's two +/- limits.

From the hexapod version, unchanged: :func:`_track_leg` uses the closed-form 3-DOF IK
(:meth:`Leg3DOF.ik_from_foot`), the tibia has its own radius (``tibia_radius``).

A **stance** is a ``(num_legs,)`` int array: the foothold index each leg stands on
(indices into the global foothold pool). Planted feet are treated as welded, so no
stability check yet -- validity is purely kinematic + geometric (:func:`checks`).

Almost every check is per leg (limits, ankle, leg-vs-body, leg-vs-terrain), so a
leg can be planted and validated on its own: :func:`plant_table` does that for
every (candidate foothold, leg) pair. Assembling a stance then only has to pass the
few checks that couple the legs (foot spacing, body-vs-terrain).

The moves (fixed-shape and jittable; :class:`Kit` jits them):

- :func:`sample_stance` -- body pose -> a full stance + witness posture.
  Candidates are per leg (:func:`candidates_at`: the ball each shoulder can
  reach). (:func:`propose_stances` / :func:`propose_leg` return all proposals, for the
  samplers in :mod:`.samplers` to score and select from.)
- :func:`lift_leg` -- raise one planted foot off its foothold.
- :func:`place_leg` -- put one foot at a free point (not planted).
- :func:`move_body` -- move the body with the planted feet held.
- :func:`resample_leg` -- re-plant one leg on a new foothold at the current body.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from controlkit.kinematics import candidates, collision
from controlkit.kinematics.robot import Robot
from controlkit.kinematics.types import Foothold, Posture
from controlkit.se3 import SE3

from ..config import Cfg


# --------------------------------------------------------------------------- #
# Validity                                                                    #
# --------------------------------------------------------------------------- #
def _body_local(cfg: Cfg):
    """The square trunk (the mounts' bounding box) as one box, body frame.

    Mounts sit at ``45 + 90k`` deg, radius ``mount_radius`` (:func:`radial_mounts` with 4
    legs), so the box's half-extents are ``mount_radius / sqrt(2)`` in x and y.

    Returns:
        ``(half, rot)`` of shapes (1, 3) and (1, 3, 3); the box is centred at 0.
    """
    a = cfg.mount_radius * math.cos(math.pi / 4)
    return jnp.array([[a, a, cfg.body_half_height]]), jnp.eye(3)[None]


def body_boxes(cfg: Cfg, body: SE3) -> collision.OBB:
    """The trunk as a (1,) world OBB (:func:`_body_local`)."""
    half, rot = _body_local(cfg)
    R = body.rotation().as_matrix()
    return collision.OBB(body.translation()[None], half, jnp.einsum("ij,njk->nik", R, rot))


def _link_radii(cfg: Cfg) -> jax.Array:
    """Per-link radius: coxa and femur ``link_radius``, tibia ``tibia_radius``."""
    return jnp.array([cfg.link_radius, cfg.link_radius, cfg.tibia_radius])


def _cloud_radii(leg, cfg: Cfg) -> jax.Array:
    """Clearance radius of each point of ``leg.collision_cloud``.

    Its link's radius (:func:`_link_radii`), and ``foot_radius`` at the foot (the
    last point). Repeats the per-link point counts of :meth:`Leg.collision_cloud`.
    """
    with jax.ensure_compile_time_eval():
        counts = [math.ceil(float(l) / cfg.collision_delta) + 1 for l in leg.lengths]
        r = np.repeat(np.asarray(_link_radii(cfg)), counts)
        r[-1] = max(r[-1], cfg.foot_radius)
    return jnp.asarray(r)


def foot_center(cfg: Cfg, site: Foothold) -> jax.Array:
    """Where a foot planted on ``site`` has its centre: ``foot_radius`` along the normal.

    Batched over any leading axes of ``site``.
    """
    return site.position + cfg.foot_radius * site.normal


def leg_checks(robot: Robot, scene: collision.Scene, cfg: Cfg, shoulder: SE3,
               body_box: collision.OBB, theta: jax.Array, site: Foothold,
               planted: jax.Array) -> dict:
    """Checks of a single leg, all scalar bools (True = pass).

    - ``reach``: if planted, the foot centre is at :func:`foot_center` (within 5 mm).
    - ``limits``: joint angles within limits.
    - ``ankle``: if planted, contact vector within ``ankle_limit_deg`` of the normal.
    - ``self``: no link (past the first) overlaps the trunk.
    - ``terrain``: no point of the leg penetrates the terrain; if planted, a ball of
      ``foot_clearance_radius`` around the foot is ignored (it is *meant* to touch).

    Args:
        robot, scene, cfg: the robot, terrain, and knobs.
        shoulder: the leg's world shoulder pose.
        body_box: the trunk as a (1,) world OBB (:func:`body_boxes`).
        theta: (n,) joint angles.
        site: the leg's foothold, world frame (ignored unless ``planted``).
        planted: scalar bool.

    Returns:
        dict name -> scalar bool.
    """
    leg = robot.leg
    planted = jnp.asarray(planted, bool)       # a Python bool would make ~planted == -2
    foot = shoulder.apply(leg.foot(theta))
    reach = ~planted | (jnp.linalg.norm(foot - foot_center(cfg, site)) < 5e-3)

    limits = candidates.in_limits(theta[None], leg.limits)[0]

    v = shoulder.rotation().apply(leg.contact_vector(theta))
    ankle = ~planted | (v @ site.normal >= jnp.cos(jnp.deg2rad(cfg.ankle_limit_deg)))

    b = leg.boxes(theta, radius=_link_radii(cfg))[1:]
    rot = jnp.einsum("ij,njk->nik", shoulder.rotation().as_matrix(), b.rot)
    links = collision.OBB(shoulder.apply(b.center), b.half, rot)
    hit = jax.vmap(lambda l: jax.vmap(lambda b: collision.overlap(l, b))(body_box))(links)
    self_ok = ~jnp.any(hit)

    pts = shoulder.apply(leg.collision_cloud(theta, delta=cfg.collision_delta))
    at_foot = planted & (jnp.linalg.norm(pts - site.position, axis=-1)
                         < cfg.foot_clearance_radius)
    sd = collision.sdf(scene, pts)
    terrain = jnp.all(at_foot | (sd >= _cloud_radii(leg, cfg) + cfg.terrain_margin))

    return dict(reach=reach, limits=limits, ankle=ankle, self=self_ok, terrain=terrain)


def _body_cloud(robot: Robot, cfg: Cfg, body: SE3) -> jax.Array:
    """World points filling the trunk box, ``collision_delta``-spaced."""
    with jax.ensure_compile_time_eval():
        center, half = robot.mount_box(half_height=cfg.body_half_height)
        res = tuple(max(2, math.ceil(2 * float(half[i]) / cfg.collision_delta) + 1)
                    for i in range(3))
        pts = collision.grid_box(center, half, res)
    return body.apply(pts)


def spacing_ok(cfg: Cfg, sites: Foothold, planted: jax.Array) -> jax.Array:
    """Planted feet at least ``min_foot_separation`` apart (scalar bool)."""
    d = jnp.linalg.norm(sites.position[:, None] - sites.position[None], axis=-1)
    pair = planted[:, None] & planted[None] & ~jnp.eye(planted.shape[0], dtype=bool)
    return jnp.all(~pair | (d >= cfg.min_foot_separation))


def body_terrain_ok(robot: Robot, scene, cfg: Cfg, body: SE3) -> jax.Array:
    """The body box clear of the terrain (scalar bool). Depends on the body only."""
    sd = collision.sdf(scene, _body_cloud(robot, cfg, body))
    return jnp.all(sd >= cfg.terrain_margin)


def body_checks(robot: Robot, scene, cfg: Cfg, body: SE3, sites: Foothold,
                planted: jax.Array) -> dict:
    """The checks that couple the legs: foot ``spacing`` and ``body_terrain``."""
    return dict(spacing=spacing_ok(cfg, sites, planted),
                body_terrain=body_terrain_ok(robot, scene, cfg, body))


def checks(robot: Robot, scene, cfg: Cfg, footholds: Foothold, posture: Posture,
           stance: jax.Array, planted: jax.Array) -> dict:
    """All checks of a posture on a (partial) stance.

    Legs with ``planted[i]`` must stand on ``footholds[stance[i]]``; the others are
    free (their ``stance`` entry is ignored).

    Returns:
        dict name -> bool; the per-leg checks (:func:`leg_checks`) are ``(num_legs,)``,
        the coupling ones (:func:`body_checks`) scalar.
    """
    sites = footholds[stance]
    box = body_boxes(cfg, posture.body)
    per_leg = jax.vmap(partial(leg_checks, robot, scene, cfg), (0, None, 0, 0, 0))(
        robot.shoulders(posture.body), box, posture.thetas, sites, planted)
    return per_leg | body_checks(robot, scene, cfg, posture.body, sites, planted)


def all_ok(c: dict) -> jax.Array:
    return jnp.all(jnp.stack([jnp.all(v) for v in c.values()]))


# --------------------------------------------------------------------------- #
# Candidates + plant table                                                    #
# --------------------------------------------------------------------------- #
@jax.tree_util.register_dataclass
@dataclass
class Candidates:
    """Per-leg candidate footholds: up to ``K`` pool indices per leg.

    Args:
        idx: (num_legs, K) indices into the foothold pool.
        mask: (num_legs, K) bool; False marks padding (fewer than ``K`` in range).
    """

    idx: jax.Array
    mask: jax.Array


def candidates_at(robot: Robot, cfg: Cfg, footholds: Foothold, priority: jax.Array,
                  body: SE3) -> Candidates:
    """Each leg's candidates: ``K`` footholds from the ball its shoulder can reach.

    The ball (radius = total leg length) is the exact superset of what a leg can
    reach. When it holds more than ``cfg.k_per_leg`` footholds, a uniform subset
    is kept: the ones with the highest ``priority`` (fixed iid uniforms, one per
    foothold -- so the subset is stable as the body moves, not re-drawn per call).
    """
    radius = float(sum(cfg.leg_lengths))
    sh = robot.shoulders(body).translation()                           # (L, 3)
    d2 = jnp.sum((footholds.position[None] - sh[:, None]) ** 2, -1)    # (L, M)
    score = jnp.where(d2 <= radius**2, priority[None], -jnp.inf)
    vals, idx = jax.lax.top_k(score, cfg.k_per_leg)
    return Candidates(idx, vals > -jnp.inf)

def _plant(key, robot: Robot, scene, cfg: Cfg, shoulder: SE3, body_box, site: Foothold):
    """Try ``cfg.reach_samples`` plants of one leg on ``site``; keep a valid one.

    Returns:
        ``(ok, theta)``: whether some plant passed every :func:`leg_checks`, and it.
    """
    local = Foothold(foot_center(cfg, site), site.normal).transform(shoulder.inverse())

    def one(k):
        ok, theta = robot.leg.sample_planted(k, local)
        c = leg_checks(robot, scene, cfg, shoulder, body_box, theta, site, True)
        return ok & all_ok(c), theta

    oks, thetas = jax.vmap(one)(jax.random.split(key, cfg.reach_samples))
    i = jnp.argmax(oks)
    return oks[i], thetas[i]


def plant_table(key, robot: Robot, scene, cfg: Cfg, footholds: Foothold,
                cand: Candidates, body: SE3, legs: jax.Array):
    """Valid plants of each leg in ``legs`` on each of its own candidates, at ``body``.

    Args:
        key: PRNG key.
        robot, scene, cfg: the robot, terrain, and knobs.
        footholds: (M,) the pool.
        cand: per-leg candidates (:func:`candidates_at`).
        body: the body pose.
        legs: (J,) leg indices.

    Returns:
        ``(ok, thetas)`` of shapes (J, K) and (J, K, n); padding is never ok.
    """
    idx, mask = cand.idx[legs], cand.mask[legs]                     # (J, K)
    shoulders = robot.shoulders(body)[legs]
    box = body_boxes(cfg, body)
    plant = partial(_plant, robot=robot, scene=scene, cfg=cfg, body_box=box)
    per_cand = jax.vmap(lambda k, sh, f: plant(k, shoulder=sh, site=footholds[f]),
                        (0, None, 0))
    keys = jax.random.split(key, idx.shape)
    ok, th = jax.vmap(per_cand)(keys, shoulders, idx)
    return ok & mask, th


# --------------------------------------------------------------------------- #
# Moves                                                                       #
# --------------------------------------------------------------------------- #
def propose_stances(key, robot: Robot, scene, cfg: Cfg, footholds: Foothold,
                    cand: Candidates, body: SE3):
    """``cfg.tries`` full-stance proposals (every leg planted) at ``body``.

    Builds the :func:`plant_table`, then each attempt draws one valid (foothold,
    plant) per leg from that leg's own candidates, uniformly -- the "prior" -- and
    is checked for foot spacing. Body-vs-terrain depends on the body only, so it is
    checked once for all attempts.

    Returns:
        ``(oks, postures, stances)`` of shapes (T,), (T,) and (T, num_legs).
    """
    L = robot.num_legs
    legs = jnp.arange(L)
    k_table, k_try = jax.random.split(key)
    table_ok, table_th = plant_table(k_table, robot, scene, cfg, footholds, cand,
                                     body, legs)                    # (L, K)
    body_ok = body_terrain_ok(robot, scene, cfg, body)
    planted = jnp.ones(L, bool)

    def attempt(k):
        j = jax.vmap(lambda kk, ok: candidates.sample(kk, jnp.zeros(ok.shape), ok))(
            jax.random.split(k, L), table_ok)                      # (L,) column per leg
        stance = cand.idx[legs, j]
        ok = jnp.all(table_ok[legs, j]) & spacing_ok(cfg, footholds[stance], planted)
        return ok & body_ok, Posture(body, table_th[legs, j]), stance

    return jax.vmap(attempt)(jax.random.split(k_try, cfg.tries))


def sample_stance(key, robot: Robot, scene, cfg: Cfg, footholds: Foothold,
                  cand: Candidates, body: SE3):
    """A full stance + witness posture at ``body``: the first valid proposal.

    Returns:
        ``(ok, posture, stance)``; when not ``ok`` the other two are an invalid attempt.
    """
    oks, postures, stances = propose_stances(key, robot, scene, cfg, footholds, cand, body)
    i = jnp.argmax(oks)
    return oks[i], postures[i], stances[i]


def _track_leg(robot: Robot, cfg: Cfg, shoulder: SE3, target: jax.Array,
               theta: jax.Array):
    """IK one leg to a world foot ``target``, staying close to ``theta``.

    A 3-DOF leg has finitely many IK branches (:meth:`Leg3DOF.ik_from_foot`);
    among the in-limit ones takes the one nearest ``theta`` (keeps motion
    continuous frame to frame).

    Returns:
        ``(ok, theta)``; ``ok`` false (and ``theta`` unchanged) if nothing reaches.
    """
    leg = robot.leg
    local = shoulder.inverse().apply(target)
    oks, ths = leg.ik_from_foot(local)
    oks = oks & candidates.in_limits(ths, leg.limits)
    i = candidates.best(-jnp.linalg.norm(ths - theta, axis=-1), oks)
    return oks[i], jnp.where(oks[i], ths[i], theta)


def lift_leg(robot: Robot, cfg: Cfg, footholds: Foothold, posture: Posture,
             stance: jax.Array, i: int, height: float):
    """Raise leg ``i``'s foot ``height`` off its foothold, along the normal.

    Returns:
        ``(ok, posture)``.
    """
    site = footholds[stance[i]]
    shoulder = robot.shoulders(posture.body)[i]
    ok, theta = _track_leg(robot, cfg, shoulder, foot_center(cfg, site) + height * site.normal,
                           posture.thetas[i])
    return ok, Posture(posture.body, posture.thetas.at[i].set(theta))


def place_leg(robot: Robot, cfg: Cfg, posture: Posture, i: int, point: jax.Array):
    """Put leg ``i``'s foot at the world ``point`` (free, not on a foothold).

    Returns:
        ``(ok, posture)``; ``ok`` false (and the leg unchanged) if out of reach.
    """
    shoulder = robot.shoulders(posture.body)[i]
    ok, theta = _track_leg(robot, cfg, shoulder, point, posture.thetas[i])
    return ok, Posture(posture.body, posture.thetas.at[i].set(theta))


def move_body(robot: Robot, scene, cfg: Cfg, footholds: Foothold, posture: Posture,
              stance: jax.Array, planted: jax.Array, body: SE3):
    """Move the body to ``body``, keeping the planted feet on their footholds.

    Planted legs are re-solved (:func:`_track_leg`); free legs keep their joint
    angles (so their feet ride along with the body).

    Returns:
        ``(ok, posture, checks)`` for the moved configuration.
    """
    sites = footholds[stance]
    oks, thetas = jax.vmap(partial(_track_leg, robot, cfg))(
        robot.shoulders(body), foot_center(cfg, sites), posture.thetas)
    thetas = jnp.where(planted[:, None], thetas, posture.thetas)
    new = Posture(body, thetas)
    c = checks(robot, scene, cfg, footholds, new, stance, planted)
    return all_ok(c) & jnp.all(~planted | oks), new, c


def propose_leg(key, robot: Robot, scene, cfg: Cfg, footholds: Foothold,
                cand: Candidates, posture: Posture, stance: jax.Array, i: int):
    """One proposal per candidate of leg ``i`` (row ``cand.idx[i]``): re-planted there.

    Body and other legs fixed. Each candidate gets a valid plant of leg ``i``
    (:func:`plant_table`) and the spacing check with the other legs; the current
    foothold is excluded.

    Returns:
        ``(valid, postures, stances)`` of shapes (K,), (K,) and (K, num_legs).
    """
    table_ok, table_th = plant_table(key, robot, scene, cfg, footholds, cand,
                                     posture.body, jnp.array([i]))
    row = cand.idx[i]
    planted = jnp.ones(robot.num_legs, bool)
    spaced = jax.vmap(lambda f: spacing_ok(cfg, footholds[stance.at[i].set(f)], planted))(row)
    body_ok = body_terrain_ok(robot, scene, cfg, posture.body)
    valid = table_ok[0] & spaced & (row != stance[i]) & body_ok
    K = row.shape[0]
    thetas = jnp.broadcast_to(posture.thetas, (K,) + posture.thetas.shape)
    postures = Posture(posture.body.broadcast_to((K,)), thetas.at[:, i].set(table_th[0]))
    stances = jnp.broadcast_to(stance, (K,) + stance.shape).at[:, i].set(row)
    return valid, postures, stances


def resample_leg(key, robot: Robot, scene, cfg: Cfg, footholds: Foothold,
                 cand: Candidates, posture: Posture, stance: jax.Array, i: int):
    """Plant leg ``i`` on a new foothold, drawn uniformly among the valid proposals.

    Returns:
        ``(ok, posture, stance, valid)`` -- ``valid`` (K,) marks which of leg
        ``i``'s candidates (``cand.idx[i]``) it could have been planted on.
    """
    k_prop, k_pick = jax.random.split(key)
    valid, postures, stances = propose_leg(k_prop, robot, scene, cfg, footholds, cand,
                                           posture, stance, i)
    j = candidates.sample(k_pick, jnp.zeros(valid.shape), valid)
    return valid[j], postures[j], stances[j], valid


# --------------------------------------------------------------------------- #
# Bundle                                                                      #
# --------------------------------------------------------------------------- #
class Kit:
    """Robot + terrain + foothold pool, with the moves jitted over them.

    The robot, scene and pool are closed over (constants), so the jitted functions
    take only the per-call state.
    """

    def __init__(self, robot: Robot, scene, cfg: Cfg, footholds: Foothold):
        self.robot, self.scene, self.cfg, self.footholds = robot, scene, cfg, footholds
        args = (robot, scene, cfg, footholds)
        # fixed per-foothold priorities: a stable uniform subset when a leg's ball
        # holds more than k_per_leg footholds (see candidates_at)
        priority = jax.random.uniform(jax.random.PRNGKey(0), footholds.shape)
        self.candidates = jax.jit(
            lambda body: candidates_at(robot, cfg, footholds, priority, body))
        self.sample_stance = jax.jit(
            lambda key, cand, body: sample_stance(key, *args, cand, body))
        self.lift_leg = jax.jit(
            lambda posture, stance, i, height:
                lift_leg(robot, cfg, footholds, posture, stance, i, height))
        self.place_leg = jax.jit(
            lambda posture, i, point: place_leg(robot, cfg, posture, i, point))
        self.move_body = jax.jit(
            lambda posture, stance, planted, body:
                move_body(*args, posture, stance, planted, body))
        self.resample_leg = jax.jit(
            lambda key, cand, posture, stance, i:
                resample_leg(key, *args, cand, posture, stance, i))
        self.leg_valid = jax.jit(
            lambda key, cand, posture, stance, i:
                propose_leg(key, *args, cand, posture, stance, i)[0])
        self.checks = jax.jit(
            lambda posture, stance, planted: checks(*args, posture, stance, planted))
