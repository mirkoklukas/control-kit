"""Score functions of the climb robot, per posture on a stance: :func:`make_scorer`.

A posture is scored on the feet that are down (``planted_idx``), from its statics
(:mod:`.climb_statics`) and the score LPs (:mod:`.statics`):

- ``s``: the hold margin s* of the full model (adhesion, friction pyramid, servo
  torques). >= 1: it holds under g.
- ``dist``: the disturbance margin, in units of the robot's weight: the largest push (or
  twist, as force x 0.1 m) on the body, worst of 12 directions, it still resists under g.
  0 where ``s < 1``.
- ``dist_mean``: its mean over the 12 directions (/ weight), a tie-break: many postures
  share the same worst direction (on a wall: friction against the weight).
- ``effort`` (with ``effort=True``): the least joint loading that holds the posture under g: min sum tau^2 over
  the internal forces, within the friction cones, adhesion and tau_max
  (:func:`.statics.least_torque_qp`), as RMS utilization sqrt(sum tau^2 / n) / tau_max
  (0: no load, 1: every servo at its limit). inf where the QP has no solution.
- ``viol``: the hold LP solution's constraint violation.

With ``extras``, also ``s_pm`` (s* of the point-mass model) and ``tau`` (peak |servo
torque| of the least-torque foot forces / tau_max).

:func:`make_torque_scorer`: the least-torque equilibrium within the limits (sum tau^2,
the torques) and a pass / fail disturbance check at a fixed push.

Cheap scores (geometry only, microseconds; to pre-select before the LPs):
:func:`direction_scores` (progress along a walking direction, distance off the line,
shift), :func:`foothold_scores` (distance to a target); :func:`top_k` keeps the best.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from mujoco import mjx

from .climb_statics import ClimbModel
from .statics import (cone_margin, disturbance_check, disturbance_margin, hold_margin,
                      least_torque, least_torque_fast, least_torque_qp, point_mass_statics,
                      push_check_fast)

G = jnp.array([0.0, 0.0, -9.81])


def make_scorer(cm: ClimbModel, footholds, planted_idx, adhesion, mu, g=G, extras=True,
                effort=False):
    """A jitted, batched score function for postures on stances with ``planted_idx`` down.

    Args:
        cm: the climb model.
        footholds: the foothold pool (for the surface normals).
        planted_idx: tuple of the planted legs (static).
        adhesion: A per foot (N).
        mu: friction coefficient.
        g: (3,) gravity (world).
        extras: also compute ``s_pm`` and ``tau`` (slower).
        effort: also compute ``effort`` (one QP more).

    Returns:
        ``score(postures, stances) -> dict`` of (N,) arrays (see the module docstring).
    """
    idx = np.asarray(planted_idx)
    planted = jnp.asarray([j in planted_idx for j in range(cm.mj.num_legs)])
    tau_max = float(cm.mj.forcerange[1])
    w = cm.mass * float(jnp.linalg.norm(g))

    def one(posture, stance):
        normals = footholds.normal[stance]                             # (L, 3)
        q = cm.qpos(posture, normals)
        st = cm.statics(q, g)
        s, _, viol = hold_margin(st, planted_idx, normals[idx], adhesion, mu, tau_max)
        dist, lam, _ = disturbance_margin(st, planted_idx, normals[idx], adhesion, mu, tau_max)
        holds = s >= 1.0
        out = dict(s=s, dist=jnp.where(holds, dist / w, 0.0),
                   dist_mean=jnp.where(holds, lam.mean() / w, 0.0), viol=viol)
        if effort:
            _, tau_qp, viol_qp = least_torque_qp(st, planted_idx, normals[idx], adhesion, mu,
                                                 tau_max)
            rms = jnp.sqrt(jnp.mean(tau_qp ** 2)) / tau_max
            out["effort"] = jnp.where(viol_qp < 1e-2, rms, jnp.inf)
        if extras:
            d = mjx.forward(cm.mx, mjx.make_data(cm.mx).replace(qpos=q))
            pm = point_mass_statics(d.subtree_com[1], d.xpos[cm.foot_ids], cm.mass, g)
            out["s_pm"] = hold_margin(pm, planted_idx, normals[idx], adhesion, mu, None)[0]
            _, tau = least_torque(st, planted)
            out["tau"] = jnp.abs(tau).max() / tau_max
        return out

    return jax.jit(jax.vmap(one))


def make_torque_scorer(cm: ClimbModel, footholds, planted_idx, adhesion, mu, g=G,
                       dist_min=None, method="fast", push_min=None,
                       twist_min=None):
    """A jitted, batched scorer: the least-torque equilibrium under ``g`` within the limits,
    then (optionally) the disturbance check.

    Per (posture, stance), planted feet ``planted_idx``:

    1. **torques**: the foot forces minimizing sum tau^2 subject to equilibrium, adhesion,
       the friction pyramids and tau_max. ``method="fast"``: the closed form without the
       limits (:func:`.statics.least_torque_fast`), exact where it satisfies them anyway
       (``exact``; elsewhere re-score with ``method="qp"``, see :func:`torque_scores`);
       ``"qp"``: the QP (:func:`.statics.least_torque_qp`, ~20x slower).
    2. **disturbance check** (if ``dist_min`` is given): resists a push or twist of
       ``dist_min`` x weight in each of the 12 directions (:func:`.statics.disturbance_check`,
       12 LPs: the internal forces chosen per direction).
    3. **push check, fast** (if ``push_min`` is given): pushes of ``push_min`` x weight
       along +/- each axis and twists of weight x ``twist_min`` (a lever arm, m) about +/-
       each axis, with the least-torque split of each pushed load (the servos' internal
       forces; :func:`.statics.push_check_fast`, no LP).

    The torques reported are always those of 1 (no push).

    Args:
        cm: the climb model.
        footholds: the foothold pool (for the surface normals).
        planted_idx: tuple of the planted legs (static).
        adhesion: A per foot (N).
        mu: friction coefficient.
        g: (3,) gravity (world).
        dist_min: the push to resist (/ weight) in the LP check; None: no LP check.
        method: ``"fast"`` or ``"qp"``.
        push_min: the push force to resist (/ weight) in the fast check; None: no fast check.
        twist_min: the twist to resist in the fast check, as the weight on a lever arm (m);
            None: ``0.1 x push_min`` (a push at 0.1 m, as the disturbance directions).

    Returns:
        ``score(postures, stances) -> dict`` of batched arrays:

        exact: True where 1 is the least-torque solution within the limits (``"qp"``: where
            the QP has one; ``"fast"``: where the closed form is within the limits).
        holds: True where the posture is known to hold under g (= ``exact``; for ``"fast"``,
            False only means "unknown, needs the QP").
        sum_tau2: sum of the squared servo torques of 1 (N^2 m^2); inf where not ``holds``.
        tau: (12,) the servo torques of 1 (N m).
        F: (p, 3) the planted feet's forces of 1 (N).
        cone_margin: the smallest distance of those forces to their (round, adhesion-
            shifted) friction cone's surface (N; :func:`.statics.cone_margin`): how much
            room the weakest foot has before it slips or peels.
        dist_ok: (with ``dist_min``) True where every direction passes.
        dist_dirs: (with ``dist_min``) (12,) bool, per direction.
        push_ok: (with ``push_min``) True where every direction passes the fast check.
        push_dirs: (with ``push_min``) (12,) bool, per direction.
        push_margin: (with ``push_min``) the smallest round-cone margin over the pushed
            cases (N).
    """
    idx = np.asarray(planted_idx)
    tau_max = float(cm.mj.forcerange[1])
    weight = cm.mass * float(jnp.linalg.norm(g))
    lam_min = None if dist_min is None else dist_min * weight
    if push_min is None:
        push_dirs = None
    else:
        twist = 0.1 * push_min if twist_min is None else twist_min
        f = jnp.concatenate([jnp.eye(3), jnp.zeros((3, 3))], 1) * (push_min * weight)
        m = jnp.concatenate([jnp.zeros((3, 3)), jnp.eye(3)], 1) * (twist * weight)
        push_dirs = jnp.concatenate([f, -f, m, -m])        # (12, 6): N and N m

    def one(posture, stance):
        normals = footholds.normal[stance][idx]
        st = cm.statics(cm.qpos(posture, footholds.normal[stance]), g)
        if method == "qp":
            F, tau, viol = least_torque_qp(st, planted_idx, normals, adhesion, mu, tau_max)
            exact = viol < 1e-2
        else:
            F, tau, exact = least_torque_fast(st, planted_idx, normals, adhesion, mu, tau_max)
        out = dict(exact=exact, holds=exact,
                   sum_tau2=jnp.where(exact, jnp.sum(tau ** 2), jnp.inf),
                   tau=tau, F=F, cone_margin=cone_margin(F, normals, adhesion, mu)[0])
        if lam_min is not None:
            ok, per = disturbance_check(st, planted_idx, normals, adhesion, mu, tau_max,
                                        lam_min)
            out["dist_ok"], out["dist_dirs"] = ok & exact, per
        if push_dirs is not None:
            ok, per, margin = push_check_fast(st, planted_idx, normals, adhesion, mu, tau_max,
                                              1.0, directions=push_dirs)
            out["push_ok"], out["push_dirs"], out["push_margin"] = ok, per, margin
        return out

    return jax.jit(jax.vmap(one))


def torque_scores(fast, qp, postures, stances, bucket: int = 16) -> dict:
    """Score with the fast torque scorer; re-score with the QP only where it is not exact.

    Args:
        fast, qp: :func:`make_torque_scorer` with ``method="fast"`` / ``"qp"`` (same
            settings otherwise).
        postures, stances: the batch.
        bucket: the QP batch is padded to a multiple of this (fewer compiles).

    Returns:
        The fast scorer's dict, with the QP's results where ``exact`` was False.
    """
    out = {k: np.array(v) for k, v in fast(postures, stances).items()}
    redo = np.nonzero(~out["exact"])[0]
    if len(redo):
        m = int(np.ceil(len(redo) / bucket)) * bucket
        sub = jnp.asarray(np.resize(redo, m))
        q = qp(jax.tree.map(lambda x: x[sub], postures), jnp.asarray(stances)[sub])
        for k, v in q.items():
            out[k][redo] = np.asarray(v)[: len(redo)]
    return out


def stable(sc: dict, s_min: float = 1.5, dist_min: float = 0.25) -> np.ndarray:
    """True where a posture is stable enough: hold margin >= ``s_min`` and disturbance
    margin (/ weight) >= ``dist_min``."""
    return (np.asarray(sc["s"]) >= s_min) & (np.asarray(sc["dist"]) >= dist_min)


def rank_key(sc: dict, key: str = "dist", s_min: float = 1.5,
             dist_min: float = 0.25) -> np.ndarray:
    """A sortable rank value, higher is better.

    Args:
        sc: the scores (:func:`make_scorer`).
        key: a margin (``dist``, ``s``, ...): ranked by it rounded to 0.01, ``dist_mean``
            breaks ties. ``effort``: stable postures (:func:`stable`) first, among them
            the lower ``effort``; the unstable ones after, by ``dist`` / ``dist_mean``.
        s_min, dist_min: the stability gate for ``key="effort"``.

    Returns:
        (N,) rank values.
    """
    f64 = lambda k: np.asarray(sc[k], np.float64)              # float32 loses the effort
    margin = np.round(f64("dist" if key == "effort" else key), 2) * 1e3 + f64("dist_mean")
    if key != "effort":
        return margin
    effort = np.minimum(f64("effort"), 1e3)
    return np.where(stable(sc, s_min, dist_min), 1e9 - effort, margin)


# --------------------------------------------------------------------------- #
# Cheap scores: geometry only, microseconds                                   #
# --------------------------------------------------------------------------- #
def direction_scores(bodies, start, direction, origin=None) -> dict:
    """How the body poses ``bodies`` move relative to ``start`` along ``direction``.

    Args:
        bodies: (n,) batched body poses (SE3).
        start: the reference body pose (SE3), e.g. the current body.
        direction: (3,) walking direction, world frame (normalized here).
        origin: (3,) a point on the walking line; None: ``start``'s position.

    Returns:
        dict of (n,) arrays: ``progress`` (m along the direction), ``lateral`` (m off
        the walking line, in the plane perpendicular to the direction: sideways and
        along the normal), ``shift`` (m, distance from ``start``).
    """
    d = jnp.asarray(direction, float)
    d = d / jnp.linalg.norm(d)
    p0 = start.translation()
    o = p0 if origin is None else jnp.asarray(origin, float)
    p = bodies.translation()
    rel = p - o
    return dict(progress=(p - p0) @ d,
                lateral=jnp.linalg.norm(rel - (rel @ d)[:, None] * d, axis=1),
                shift=jnp.linalg.norm(p - p0, axis=1))


def foothold_scores(footholds, stances, i, target) -> dict:
    """Leg ``i``'s new foothold against a ``target`` point.

    Returns:
        dict of (n,) arrays: ``to_target`` (m, distance of the foothold to ``target``).
    """
    pos = footholds.position[stances[:, i]]
    return dict(to_target=jnp.linalg.norm(pos - jnp.asarray(target, float), axis=1))


def top_k(ok, value, k: int) -> np.ndarray:
    """The indices of the (at most) ``k`` valid candidates with the highest ``value``,
    best first."""
    ok, value = np.asarray(ok), np.asarray(value, np.float64)
    order = np.argsort(-np.where(ok, value, -np.inf))
    return order[: min(k, int(ok.sum()))]
