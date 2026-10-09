"""One crawl step, the fast version: :func:`plan_step_fast` (``docs/planner-design.md``).

The same four phases as :func:`.step.plan_step` (B_lift, lift, B_plant, f'), with the
samplers of :mod:`.samplers` and only the fast scores, no LP / QP:

- **torques**: the least-torque foot forces from equilibrium alone (closed form), then
  the cone check (adhesion, friction pyramid) and tau_max: a candidate whose forces fail
  it is dropped (:func:`.scoring.make_torque_scorer`, ``method="fast"``);
- **push check**: the same for a push of ``push_min`` x weight along +/- each axis and a
  twist of weight x ``twist_min`` (m) about +/- each axis (12 cases), the pushed load split
  the same way (:func:`.statics.push_check_fast`).

A candidate is *good* if it passes both. Per phase:

1. **B_lift**: ``n_lift`` body poses in a disc of radius ``lift_radius`` around the body
   (four feet held: the previous B_plant), heights in ``ref_height + height_band``; good for
   the tripod T (leg i lifted) (and, if ``lift_back_max`` is set, at most that far behind
   the body along the direction); value: ``lift_progress_weight`` x progress - ``effort_weight`` x effort -
   ``shift_cost`` x shift - ``back_cost`` x distance behind - ``side_cost`` x distance off
   the walking line; the best whose leg can be raised.
2. **lift** leg i by ``lift_height`` along its normal.
3. **B_plant**: ``n_plant`` body poses in a box ahead of B_lift (body frame: x in
   [0, ``plant_reach``], y in +/- ``plant_side``; the body's x is the walking direction,
   floor and wall), the tripod held; good for T, and at ``path_points`` points along the
   straight path from B_lift (for the ``path_checked`` best); value: progress -
   ``plant_effort_weight`` x effort - ``side_cost_plant`` x distance off the line.
4. **f'**: ``n_foot`` footholds for leg i at B_plant (:func:`.samplers.sample_foothold`);
   good for the full stance S'; cost: distance to ``f_i + step_length x direction`` +
   ``ankle_cost`` x the tibia's angle to the new normal + ``foot_effort_weight`` x effort.
   If none, the next-best B_plant (up to ``plant_tries``).

``effort`` = sqrt(sum tau^2 / 12) / tau_max: RMS servo load, 1 = every servo at its limit.
A step fails when a phase has no good candidate.

**No B_plant shift** (``plant_shift=False``): leg i is planted from the lifted posture
(f' within reach of B_lift); the body moves only with all four feet down, in the next
step's B_lift (stable under the next tripod): a static crawl, whose body can cross the
diagonal between the tripods of diagonal swing legs.

**Reusing B_plant as the next B_lift** (``lift_from_plant``): the step starts by lifting
leg i right where the previous step planted (no B_lift sampling, one body shift less),
if its tripod is good there; else ``lift_fallback``: ``"sample"`` (phase 1 as above) or
``"fail"``. To make that likely, f' (given ``next_leg``) also requires the next tripod,
S' without ``next_leg``, to be good at B_plant.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from controlkit.kinematics.types import Posture
from controlkit.se3 import SE3

from .climb_statics import ClimbModel
from .samplers import Samplers, region
from .scoring import direction_scores, make_torque_scorer
from .stance import Kit, all_ok, move_body
from .step import StepResult


@dataclass
class FastStepCfg:
    """The knobs of :func:`plan_step_fast`."""
    adhesion: float = 40.0          # A per foot (N)
    mu: float = 0.5                 # friction coefficient
    push_min: float = 0.1           # push check: push force to resist (/ weight)
    twist_min: float = 0.01         # push check: twist to resist, the weight on this lever
                                    # arm (m)
    ref_height: float = 0.16        # reference body distance to the surface (m)
    height_band: tuple = (-0.04, 0.04)  # body heights: ref_height + this (m)
    n_lift: int = 1024              # B_lift candidates
    lift_radius: float = 0.08       # B_lift: offsets within this radius (m)
    lift_tries: int = 10            # B_lift: candidates tried for raising the leg, best first
    lift_back_max: float | None = None  # B_lift: a hard limit, at most this far behind the
                                    # body (m); None: none (only the soft back_cost)
    lift_progress_weight: float = 5.0  # B_lift: value per metre of progress
    effort_weight: float = 10.0     # B_lift: value per unit of effort
    shift_cost: float = 2.0         # B_lift: per metre of shift
    back_cost: float = 20.0         # B_lift: per metre behind the body along the direction
    side_cost: float = 10.0         # B_lift: per metre off the walking line
    lift_height: float = 0.03       # leg i raised this far along its normal (m)
    plant_shift: bool = True        # B_plant: shift the body with leg i lifted; False: plant
                                    # from the lifted posture, the body moves only with four
                                    # feet down (the next step's B_lift: a static crawl)
    lift_from_plant: bool = False   # B_lift = the current body (the previous B_plant), no
                                    # sampling, if the tripod is good there
    lift_fallback: str = "sample"   # lift_from_plant, tripod not good: sample (phase 1) or fail
    n_plant: int = 1024             # B_plant candidates
    plant_reach: float = 0.12       # B_plant: up to this far ahead (m)
    plant_side: float = 0.04        # B_plant: up to this far sideways (m)
    plant_effort_weight: float = 0.2  # B_plant: m of progress per unit of effort
    side_cost_plant: float = 1.0    # B_plant: m of progress per m off the line
    path_points: int = 2            # B_plant: interior points checked along the path
    path_checked: int = 20          # B_plant: candidates whose path is checked, best first
    plant_tries: int = 5            # B_plant: candidates tried for a foothold, best first
    n_foot: int = 512               # f' candidates
    step_length: float = 0.15       # f': target this far ahead of the old foothold (m)
    ankle_cost: float = 0.05        # f': per rad of tibia angle to the new normal (m)
    foot_effort_weight: float = 0.2 # f': m of distance to the target per unit of effort
    press_m: float = 0.008          # pressed: the swing foot this far past the surface (m)


class FastPlanner:
    """The jitted, batched pieces of :func:`plan_step_fast` for one robot, terrain and pool.

    Args:
        kit: the stance kit.
        cm: the climb model (statics).
        cfg: the step knobs.
    """

    def __init__(self, kit: Kit, cm: ClimbModel, cfg: FastStepCfg):
        self.kit, self.cm, self.cfg = kit, cm, cfg
        robot, fh = kit.robot, kit.footholds
        L = robot.num_legs
        self.smp = Samplers(kit)
        self.tau_max = float(cm.mj.forcerange[1])
        score = lambda idx: make_torque_scorer(cm, fh, idx, cfg.adhesion, cfg.mu,
                                               method="fast", push_min=cfg.push_min,
                                               twist_min=cfg.twist_min)
        self.tripod = {i: score(tuple(j for j in range(L) if j != i)) for i in range(L)}
        self.full = score(tuple(range(L)))
        self.move = jax.jit(jax.vmap(
            lambda p, s, pl, b: move_body(robot, kit.scene, kit.cfg, fh, p, s, pl, b)[:2],
            in_axes=(None, None, None, 0)))
        self.checks = jax.jit(lambda p, s, pl: all_ok(kit.checks(p, s, pl)))
        leg = robot.leg

        def tibia_angles(body, thetas, normals, i):
            rot = robot.shoulders(body)[i].rotation()
            v = jax.vmap(lambda th: rot.apply(leg.contact_vector(th)))(thetas)
            return jnp.arccos(jnp.clip(jnp.sum(v * normals, -1), -1.0, 1.0))

        self.tibia_angles = jax.jit(tibia_angles, static_argnums=3)

    def good(self, sc: dict, ok) -> np.ndarray:
        """Valid, within the cones and tau_max, and passing the push check."""
        return np.asarray(ok) & np.asarray(sc["exact"]) & np.asarray(sc["push_ok"])

    def effort(self, sc: dict) -> np.ndarray:
        """RMS servo load / tau_max (1: every servo at its limit)."""
        return np.sqrt(np.asarray(sc["sum_tau2"], np.float64) / 12.0) / self.tau_max


def _take(tree, j):
    return jax.tree.map(lambda x: x[j], tree)


def _path(start: SE3, ends: SE3, fracs) -> SE3:
    """Body poses at ``fracs`` of the way from ``start`` to each of ``ends``:
    (len(fracs) * n,) batched, fraction-major."""
    delta = (start.inverse() @ ends).log()                           # (n, 6)
    f = jnp.asarray(fracs)[:, None, None]
    return start @ SE3.exp((f * delta[None]).reshape(-1, 6))


def plan_step_fast(pl: FastPlanner, stance, posture: Posture, i: int, direction, key,
                   origin=None, next_leg=None) -> StepResult:
    """Plan one crawl step: move leg ``i`` ahead along ``direction``, fast scores only.

    Args:
        pl: the planner pieces (:class:`FastPlanner`).
        stance: (L,) the current stance S (foothold indices), all feet planted.
        posture: its witness posture.
        i: the swing leg.
        direction: (3,) the walking direction, world frame (unit, in the surface plane).
        key: PRNG key.
        origin: (3,) a point on the walking line (world); None: the current body position.
        next_leg: the next step's swing leg; with ``lift_from_plant``, f' also requires the
            next tripod (S' without it) to be good. None: no such requirement.

    Returns:
        The :class:`.step.StepResult`; ``scores``: ``lift_reused`` (B_lift = the current
        body), ``progress`` (m), ``ankle_deg``,
        ``lift_effort``, ``plant_effort``, ``effort_next``, ``lift_margin``,
        ``plant_margin``, ``margin_next`` (the push check's smallest cone margin, N),
        ``t_lift``, ``t_plant``, ``t_foot`` (s, per phase).
    """
    cfg, kit = pl.cfg, pl.kit
    L = kit.robot.num_legs
    d = np.asarray(direction, float)
    d /= np.linalg.norm(d)
    p0 = np.asarray(posture.body.translation())
    R0 = np.asarray(posture.body.rotation().as_matrix())
    origin = p0 if origin is None else np.asarray(origin, float)
    all_down, tripod = jnp.ones(L, bool), jnp.arange(L) != i
    heights = (cfg.ref_height + cfg.height_band[0], cfg.ref_height + cfg.height_band[1])
    k_lift, k_plant, k_foot = jax.random.split(key, 3)
    st_n = lambda n: jnp.broadcast_to(stance, (n, L))
    T = pl.tripod[i]

    # --- 1. B_lift: the current body (lift_from_plant), or sampled
    t0 = time.perf_counter()
    lifted, reused = None, False
    if cfg.lift_from_plant:
        sc = T(jax.tree.map(lambda x: x[None], posture), stance[None])
        if bool(pl.good(sc, np.ones(1, bool))[0]):
            ok_l, lp = kit.lift_leg(posture, stance, i, cfg.lift_height)
            if bool(ok_l) and bool(pl.checks(lp, stance, tripod)):
                lift_post, lifted, reused = posture, lp, True
                lift_eff = float(pl.effort(sc)[0])
                lift_margin = float(np.asarray(sc["push_margin"])[0])
        if lifted is None and cfg.lift_fallback == "fail":
            return StepResult(False, "B_lift: tripod not good at the current body", i)
    if lifted is None:
        r = cfg.lift_radius
        ok, posts = pl.smp.body(k_lift, cfg.n_lift, posture, stance, all_down,
                                region=region(x=(-r, r), y=(-r, r), height=heights))
        off = (np.asarray(posts.body.translation()) - p0) @ R0          # body frame
        ok = np.asarray(ok) & (np.linalg.norm(off[:, :2], axis=1) <= r)  # the disc in the box
        sc = T(posts, st_n(cfg.n_lift))
        c = {k: np.asarray(v) for k, v in
             direction_scores(posts.body, posture.body, d, origin).items()}
        good = pl.good(sc, ok)
        if cfg.lift_back_max is not None:
            good = good & (c["progress"] >= -cfg.lift_back_max)
        if not good.any():
            return StepResult(False, "B_lift: no good candidate", i)
        eff = pl.effort(sc)
        value = np.where(good, cfg.lift_progress_weight * c["progress"] - cfg.effort_weight * eff
                         - cfg.shift_cost * c["shift"]
                         - cfg.back_cost * np.maximum(0.0, -c["progress"])
                         - cfg.side_cost * c["lateral"], -np.inf)
        for j in np.argsort(-value)[: min(cfg.lift_tries, int(good.sum()))]:
            lift_post = _take(posts, int(j))
            ok_l, lp = kit.lift_leg(lift_post, stance, i, cfg.lift_height)
            if bool(ok_l) and bool(pl.checks(lp, stance, tripod)):
                lifted = lp
                lift_eff, lift_margin = float(eff[j]), float(np.asarray(sc["push_margin"])[j])
                break
    t_lift = time.perf_counter() - t0
    if lifted is None:
        return StepResult(False, "lift: leg cannot be raised", i)

    # --- 3. B_plant: a body shift with leg i lifted (plant_shift), or none
    t0 = time.perf_counter()
    if not cfg.plant_shift:                    # plant from the lifted posture; the body moves
        posts = jax.tree.map(lambda x: x[None], lifted)    # next with four feet down (B_lift)
        sc = T(posts, st_n(1))
        eff = pl.effort(sc)
        order = [0]
    else:
        ok, posts = pl.smp.body(k_plant, cfg.n_plant, lifted, stance, tripod,
                                region=region(x=(0.0, cfg.plant_reach),
                                              y=(-cfg.plant_side, cfg.plant_side), height=heights))
        sc = T(posts, st_n(cfg.n_plant))
        good = pl.good(sc, ok)
        if not good.any():
            return StepResult(False, "B_plant: no good candidate", i, lift=lift_post,
                              lifted=lifted)
        c = {k: np.asarray(v) for k, v in
             direction_scores(posts.body, lifted.body, d, origin).items()}
        eff = pl.effort(sc)
        value = np.where(good, c["progress"] - cfg.plant_effort_weight * eff
                         - cfg.side_cost_plant * c["lateral"], -np.inf)
        best = np.argsort(-value)[: min(cfg.path_checked, int(good.sum()))]
        fracs = np.linspace(0.0, 1.0, cfg.path_points + 2)[1:-1]
        if len(fracs):
            mid = _path(lifted.body, _take(posts.body, jnp.asarray(best)), fracs)
            ok_m, mposts = pl.move(lifted, stance, tripod, mid)
            msc = T(mposts, st_n(len(best) * len(fracs)))
            path_ok = pl.good(msc, ok_m).reshape(len(fracs), -1).all(0)
            order = [int(k) for k, g in zip(best, path_ok) if g]
        else:
            order = [int(k) for k in best]
    t_plant = time.perf_counter() - t0
    if not order:
        return StepResult(False, "B_plant: no path stays good", i, lift=lift_post, lifted=lifted)

    # --- 4. f'
    t0 = time.perf_counter()
    target = np.asarray(kit.footholds.position[stance[i]]) + cfg.step_length * d
    counts = np.zeros(4, int)                  # over the tries: valid, S' good, next good, both
    best_next = -np.inf                        # the next tripod's best cone margin (no push), N
    for k in order[: cfg.plant_tries]:
        plant = _take(posts, k)
        k_foot, sub = jax.random.split(k_foot)
        ok, cposts, cst = pl.smp.foothold(sub, cfg.n_foot, plant, stance, i)
        fsc = pl.full(cposts, cst)
        good = pl.good(fsc, ok)
        if cfg.lift_from_plant and next_leg is not None:   # the next tripod good here too
            nsc = pl.tripod[next_leg](cposts, cst)
            nxt = pl.good(nsc, ok)
            cm_next = np.where(np.asarray(ok) & good, np.asarray(nsc["cone_margin"]), -np.inf)
            best_next = max(best_next, float(cm_next.max()))
            counts += [int(np.asarray(ok).sum()), int(good.sum()), int(nxt.sum()),
                       int((good & nxt).sum())]
            good = good & nxt
        if not good.any():
            continue
        cst = np.asarray(cst)
        pos = np.asarray(kit.footholds.position[cst[:, i]])
        nrm = kit.footholds.normal[cst[:, i]]
        ang = np.asarray(pl.tibia_angles(plant.body, cposts.thetas[:, i], nrm, i))
        feff = pl.effort(fsc)
        cost = np.where(good, np.linalg.norm(pos - target, axis=1) + cfg.ankle_cost * ang
                        + cfg.foot_effort_weight * feff, np.inf)
        m = int(np.argmin(cost))
        new = _take(cposts, m)
        point = pos[m] + (kit.cfg.foot_radius - cfg.press_m) * np.asarray(nrm[m])
        ok_p, pressed = kit.place_leg(new, i, jnp.asarray(point))
        t_foot = time.perf_counter() - t0
        return StepResult(
            True, "", i, lift=lift_post, lifted=lifted, plant=plant,
            pressed=pressed if bool(ok_p) else new, posture=new, stance=jnp.asarray(cst[m]),
            scores=dict(lift_reused=reused,
                        progress=float((np.asarray(plant.body.translation()) - p0) @ d),
                        ankle_deg=float(np.degrees(ang[m])), lift_effort=lift_eff,
                        plant_effort=float(eff[k]), effort_next=float(feff[m]),
                        lift_margin=lift_margin,
                        plant_margin=float(np.asarray(sc["push_margin"])[k]),
                        margin_next=float(np.asarray(fsc["push_margin"])[m]),
                        t_lift=t_lift, t_plant=t_plant, t_foot=t_foot))
    why = "f': no good foothold"
    if cfg.lift_from_plant and next_leg is not None:
        why += (f" ({min(len(order), cfg.plant_tries)} B_plants tried; footholds valid "
                f"{counts[0]}, S' good {counts[1]}, next tripod good {counts[2]}, both {counts[3]}; "
                f"best next-tripod cone margin among S' good: {best_next:.1f} N)")
    return StepResult(False, why, i, lift=lift_post, lifted=lifted)
