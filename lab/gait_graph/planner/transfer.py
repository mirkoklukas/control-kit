"""One transfer of the tripod graph: :func:`plan_transfer` (``docs/planner-design.md``, "the
tripod view, transfers and swings").

A transfer goes from tripod ``T`` (leg i in the air, the robot at body ``B``) through the
full stance ``S`` to the next tripod ``T' = S \\ {i'}`` (the next swing leg i' lifts):

1. **B_plant** (``n_plant`` sampled around ``B``, leg i in the air, the tripod held; candidate
   0 is ``B`` itself, so there is always one): supports ``T``; the path from ``B`` too (``path_points`` interior points, for the ``path_checked``
   best); keep the best ``k_plant``. With ``node_shift=False``: ``B_plant = B``.
2. **f'** (``n_foot`` footholds of leg i per kept B_plant): reachable (the sampler's
   validity: IK, limits, collisions, spacing; ``S`` itself is not scored, its support is
   implied by T's); keep the best ``k_foot`` per B_plant.
3. **B_lift** (``n_lift`` sampled around B_plant per kept (B_plant, f'), all four feet of
   ``S`` held): supports ``T'``, and leg i' can be raised; the best per leaf. With
   ``edge_shift=False``: ``B_lift = B_plant``.

The ``k_plant x k_foot`` leaves are compared by one value, the sum of the three levels'
(the defaults keep only: B_plant effort, f' tibia angle, B_lift progress and effort; the
other terms have weight 0):

- B_plant: ``plant_progress_weight`` x progress along the direction -
  ``plant_effort_weight`` x effort - ``side_cost`` x distance off the walking line;
- f': -``target_weight`` x (distance to ``f_i + step_length x direction``) -
  ``ankle_cost`` x tibia angle;
- B_lift: ``lift_progress_weight`` x progress from B_plant - ``lift_effort_weight`` x effort
  - ``shift_cost`` x shift - ``back_cost`` x distance behind - ``side_cost`` x off the line.

Every "supports" / "good" is the fast check (:class:`.step_fast.FastPlanner.good`):
closed-form least-torque forces within the cones and tau_max, and the push check.
``effort`` = RMS servo torque / tau_max.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from controlkit.kinematics.types import Posture

from .samplers import region, sample_body
from .scoring import direction_scores
from .step_fast import FastPlanner, _path, _take


@dataclass
class TransferCfg:
    """What gets sampled, kept and how it is valued in :func:`plan_transfer`."""
    # --- level 1: B_plant (leg i in the air, tripod T held)
    node_shift: bool = True         # sample B_plant around B; False: B_plant = B
    n_plant: int = 1024             # B_plant candidates
    plant_x: tuple = (-0.04, 0.12)  # region, body frame: along the body's x (m) ...
    plant_y: tuple = (-0.08, 0.08)  # ... and y (m)
    path_points: int = 2            # node path B -> B_plant: interior points checked
    path_checked: int = 32          # ... for this many best candidates
    k_plant: int = 8                # B_plants kept
    plant_progress_weight: float = 0.0  # value per m of progress
    plant_effort_weight: float = 0.2  # value per unit of effort
    # --- level 2: f' (leg i planted: S), reachability only
    n_foot: int = 512               # footholds per kept B_plant
    k_foot: int = 8                 # footholds kept per B_plant
    step_length: float = 0.15       # target this far ahead of the old foothold (m)
    target_weight: float = 0.0      # value per m of distance to the target
    ankle_cost: float = 0.05        # value per rad of tibia angle to the new normal
    # --- level 3: B_lift (four feet of S held, tripod T' supported)
    edge_shift: bool = True         # sample B_lift around B_plant; False: B_lift = B_plant
    n_lift: int = 256               # B_lift candidates per (B_plant, f')
    lift_radius: float = 0.16       # region: offsets within this box half-width (m)
    lift_tries: int = 4             # B_lift candidates tried for raising leg i'
    lift_progress_weight: float = 1.0  # value per m of progress (from B_plant)
    lift_effort_weight: float = 0.2 # value per unit of effort
    shift_cost: float = 0.0         # per m of shift (from B_plant)
    back_cost: float = 0.0          # per m behind B_plant along the direction
    # --- shared
    side_cost: float = 0.0          # per m off the walking line (B_plant and B_lift)
    ref_height: float = 0.16        # reference body distance to the surface (m)
    height_band: tuple = (-0.04, 0.04)  # body heights: ref_height + this (m)
    lift_height: float = 0.03       # leg i' raised this far along its normal (m)


@dataclass
class TransferResult:
    """One planned transfer. Postures are full (body + 12 leg angles).

    Args:
        ok: a transfer was found.
        why: what failed, if not ``ok``.
        plant: posture at B_plant, leg i still in the air.
        planted: posture at B_plant, leg i on f' (the stance ``stance``).
        lift: posture at B_lift, four feet down.
        lifted: posture at B_lift, leg i' raised (the next node).
        stance: (L,) the full stance S.
        scores: ``progress`` (B_lift from B, m), ``plant_effort``, ``lift_effort``,
            ``plant_margin``, ``lift_margin`` (push check cone margins, N), ``to_target``
            (m), ``ankle_deg``, ``counts`` (good candidates per level), ``t`` (s per level).
    """
    ok: bool
    why: str = ""
    plant: Posture = None
    planted: Posture = None
    lift: Posture = None
    lifted: Posture = None
    stance: jax.Array = None
    scores: dict = None


def _breakdown(sc: dict, ok, n: int) -> str:
    """Where candidates drop out: kinematically valid, then within the cones and tau_max,
    then passing the push check (counts)."""
    ok = np.asarray(ok).reshape(-1)
    ex = ok & np.asarray(sc["exact"]).reshape(-1)
    pu = ex & np.asarray(sc["push_ok"]).reshape(-1)
    best = np.asarray(sc["cone_margin"]).reshape(-1)
    bm = f"{best[ok].max():.1f} N" if ok.any() else "-"
    return (f"(of {n}: kinematically valid {int(ok.sum())}, cones + tau_max ok {int(ex.sum())}, "
            f"push ok {int(pu.sum())}; best cone margin among valid {bm})")


def _lift_sampler(pl: FastPlanner, n: int):
    """``sample_body`` for many (posture, stance) at once, all feet held: jitted, vmapped over
    (key, posture, stance), cached per n on the planner."""
    cache = pl.__dict__.setdefault("_lift_samplers", {})
    if n not in cache:
        L = pl.kit.robot.num_legs
        cache[n] = jax.jit(jax.vmap(
            lambda k, p, s, reg: sample_body(pl.kit, k, n, p, s, jnp.ones(L, bool), region=reg),
            in_axes=(0, 0, 0, None)))
    return cache[n]


def plan_transfer(pl: FastPlanner, cfg: TransferCfg, stance, posture: Posture, i: int,
                  next_leg: int, direction, key, origin=None) -> TransferResult:
    """Plan one transfer: land leg ``i``, then move to where leg ``next_leg`` can lift.

    Args:
        pl: the fast planner pieces (samplers, fast scorers, moves).
        cfg: what to sample, keep and value.
        stance: (L,) foothold indices; leg ``i``'s entry is its old foothold (not planted).
        posture: the current posture: body B, leg ``i`` in the air, tripod ``T`` holding.
        i: the leg in the air.
        next_leg: the next swing leg i'.
        direction: (3,) walking direction, world frame.
        key: PRNG key.
        origin: (3,) a point on the walking line; None: the current body position.

    Returns:
        The :class:`TransferResult`.
    """
    kit = pl.kit
    L = kit.robot.num_legs
    d = np.asarray(direction, float)
    d /= np.linalg.norm(d)
    p0 = np.asarray(posture.body.translation())
    origin = p0 if origin is None else np.asarray(origin, float)
    T, Tn = pl.tripod[i], pl.tripod[next_leg]
    tripod, all_down = jnp.arange(L) != i, jnp.ones(L, bool)
    heights = (cfg.ref_height + cfg.height_band[0], cfg.ref_height + cfg.height_band[1])
    k1, k2, k3 = jax.random.split(key, 3)
    counts, t = {}, {}

    # --- 1. B_plant
    t0 = time.perf_counter()
    if cfg.node_shift:
        ok, posts = pl.smp.body(k1, cfg.n_plant, posture, stance, tripod,
                                region=region(x=cfg.plant_x, y=cfg.plant_y, height=heights))
        # candidate 0 is B itself (no motion in the node): it supports T by construction
        posts = jax.tree.map(lambda x, y: x.at[0].set(y), posts, posture)
        ok = np.asarray(ok).copy()
        ok[0] = True
        n1 = cfg.n_plant
    else:
        ok, posts, n1 = np.ones(1, bool), jax.tree.map(lambda x: x[None], posture), 1
    sc1 = T(posts, jnp.broadcast_to(stance, (n1, L)))
    good = pl.good(sc1, ok)
    counts["plant"] = int(good.sum())
    if not good.any():
        return TransferResult(False, "B_plant: no good candidate " + _breakdown(sc1, ok, n1))
    c = {k: np.asarray(v) for k, v in direction_scores(posts.body, posture.body, d, origin).items()}
    eff1 = pl.effort(sc1)
    v1 = np.where(good, cfg.plant_progress_weight * c["progress"] - cfg.plant_effort_weight * eff1
                  - cfg.side_cost * c["lateral"], -np.inf)
    best = np.argsort(-v1)[: min(cfg.path_checked, int(good.sum()))]
    fracs = np.linspace(0.0, 1.0, cfg.path_points + 2)[1:-1]
    if cfg.node_shift and len(fracs):
        padded = np.resize(best, cfg.path_checked)          # fixed size: one compile
        mid = _path(posture.body, _take(posts.body, jnp.asarray(padded)), fracs)
        ok_m, mposts = pl.move(posture, stance, tripod, mid)
        msc = T(mposts, jnp.broadcast_to(stance, (len(padded) * len(fracs), L)))
        path_ok = pl.good(msc, ok_m).reshape(len(fracs), -1).all(0)[: len(best)]
        best = best[path_ok]
    plants = [int(j) for j in best[: cfg.k_plant]]
    counts["plant_path"] = len(plants)
    t["plant"] = time.perf_counter() - t0
    if not plants:
        return TransferResult(False, "B_plant: no path stays good")

    # --- 2. f' per kept B_plant
    t0 = time.perf_counter()
    target = np.asarray(kit.footholds.position[stance[i]]) + cfg.step_length * d
    feet = []                                   # (value so far, j, posture S, stance S, info)
    counts["foot"] = 0
    for j in plants:
        plant = _take(posts, j)
        k2, sub = jax.random.split(k2)
        ok, cposts, cst = pl.smp.foothold(sub, cfg.n_foot, plant, stance, i)
        good = np.asarray(ok)                       # reachable
        counts["foot"] += int(good.sum())
        if not good.any():
            continue
        cst = np.asarray(cst)
        pos = np.asarray(kit.footholds.position[cst[:, i]])
        nrm = kit.footholds.normal[cst[:, i]]
        ang = np.asarray(pl.tibia_angles(plant.body, cposts.thetas[:, i], nrm, i))
        to_t = np.linalg.norm(pos - target, axis=1)
        v2 = np.where(good, -cfg.target_weight * to_t - cfg.ankle_cost * ang, -np.inf)
        for m in np.argsort(-v2)[: min(cfg.k_foot, int(good.sum()))]:
            feet.append((float(v1[j] + v2[m]), j, _take(cposts, int(m)), cst[m],
                         dict(to_target=float(to_t[m]), ankle_deg=float(np.degrees(ang[m])))))
    t["foot"] = time.perf_counter() - t0
    if not feet:
        return TransferResult(False, "f': no reachable foothold")

    # --- 3. B_lift per (B_plant, f'): all leaves in one batched call (padded to a fixed size)
    t0 = time.perf_counter()
    r = cfg.lift_radius
    leaves = []                                 # (value, j, S posture, S stance, B_lift, lifted, info)
    M = len(feet)
    if cfg.edge_shift:
        Mmax = cfg.k_plant * cfg.k_foot
        n3 = cfg.n_lift
        rows = np.resize(np.arange(M), Mmax)
        SP = jax.tree.map(lambda *x: jnp.stack(x), *[feet[q][2] for q in rows])
        SST = jnp.stack([jnp.asarray(feet[q][3]) for q in rows])
        ok, lposts = _lift_sampler(pl, n3)(jax.random.split(k3, Mmax), SP, SST,
                                           region(x=(-r, r), y=(-r, r), height=heights))
        flat = jax.tree.map(lambda x: x.reshape((Mmax * n3,) + x.shape[2:]), lposts)
        sc3 = Tn(flat, jnp.repeat(SST, n3, axis=0))
        good = pl.good(sc3, np.asarray(ok).reshape(-1)).reshape(Mmax, n3)
        good[M:] = False                        # the padding
        p = np.asarray(flat.body.translation()).reshape(Mmax, n3, 3)
        pp = np.asarray(SP.body.translation())[:, None]
        progress = (p - pp) @ d
        shift = np.linalg.norm(p - pp, axis=-1)
        rel = p - origin
        lateral = np.linalg.norm(rel - (rel @ d)[..., None] * d, axis=-1)
        eff3 = pl.effort(sc3).reshape(Mmax, n3)
        margin3 = np.asarray(sc3["push_margin"]).reshape(Mmax, n3)
        v3 = np.where(good, cfg.lift_progress_weight * progress - cfg.lift_effort_weight * eff3
                      - cfg.shift_cost * shift - cfg.back_cost * np.maximum(0.0, -progress)
                      - cfg.side_cost * lateral, -np.inf)
        lp_of = lambda q, m: jax.tree.map(lambda x: x[q, m], lposts)
    else:
        n3 = 1
        SP = jax.tree.map(lambda *x: jnp.stack(x), *[f_[2] for f_ in feet])
        SST = jnp.stack([jnp.asarray(f_[3]) for f_ in feet])
        sc3 = Tn(SP, SST)
        good = pl.good(sc3, np.ones(M, bool))[:, None]
        eff3 = pl.effort(sc3)[:, None]
        margin3 = np.asarray(sc3["push_margin"])[:, None]
        v3 = np.where(good, -cfg.lift_effort_weight * eff3, -np.inf)
        lp_of = lambda q, m: jax.tree.map(lambda x: x[q], SP)
    counts["lift"] = int(good.sum())
    for q in range(M):
        v12, j, sp, sst, info = feet[q]
        sst = jnp.asarray(sst)
        for m in np.argsort(-v3[q])[: min(cfg.lift_tries, int(good[q].sum()))]:
            lp = lp_of(q, int(m))
            ok_l, lifted = kit.lift_leg(lp, sst, next_leg, cfg.lift_height)
            if bool(ok_l) and bool(pl.checks(lifted, sst, jnp.arange(L) != next_leg)):
                leaves.append((v12 + float(v3[q, m]), j, sp, sst, lp, lifted,
                               info | dict(lift_effort=float(eff3[q, m]),
                                           lift_margin=float(margin3[q, m]))))
                break
    t["lift"] = time.perf_counter() - t0
    counts["leaves"] = len(leaves)
    if not leaves:
        okf = np.asarray(ok).reshape(-1)
        if cfg.edge_shift:
            keep = np.repeat(np.arange(Mmax) < M, n3)
            okf = okf & keep
        return TransferResult(False, "B_lift: no good candidate for any (B_plant, f') "
                              + _breakdown(sc3, okf, M * n3))

    v, j, sp, sst, lp, lifted, info = max(leaves, key=lambda x: x[0])
    plant = _take(posts, j)
    scores = info | dict(progress=float((np.asarray(lp.body.translation()) - p0) @ d),
                         plant_effort=float(eff1[j]),
                         plant_margin=float(np.asarray(sc1["push_margin"])[j]),
                         value=v, counts=counts, t=t)
    return TransferResult(True, "", plant=plant, planted=sp, lift=lp, lifted=lifted,
                          stance=sst, scores=scores)
