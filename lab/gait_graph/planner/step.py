"""One crawl step of the stance planner: :func:`plan_step` (``docs/planner-design.md``).

From a four-foot stance S (with its witness posture), move swing leg i to a new foothold
f' and return the next stance S'. Four phases, chosen in the order i -> B_lift -> B_plant
-> f':

1. **shift for lifting** (four feet held): sample body poses B_lift around B; keep those
   with a valid posture where the tripod T (the feet other than i) holds (s* >= 1, the
   disturbance margin >= ``dist_min``); pick the best *mean* disturbance margin (over the
   12 directions: on a wall the worst direction is capped by friction, the mean still
   sees the peel moment, so it rewards a body close to the surface), minus a cost per
   metre of shift, a larger one per metre backwards (against the walking direction), and
   one per metre the body is off the walking line (through ``origin`` along the
   direction: keeps it from drifting sideways over the steps).
2. **lift** leg i by ``lift_height`` along its surface normal (the body stays at B_lift).
3. **shift for planting** (three feet held, leg i in the air): sample body poses B_plant
   ahead of B_lift along the walking direction; keep those valid where T holds at
   B_plant and at ``path_points`` points along the straight path from B_lift; pick the
   best progress + ``plant_mean_weight`` x mean margin - ``side_cost_plant`` x distance
   off the walking line. B_plant is never behind B_lift (the grid starts at 0 ahead).
4. **plant**: among leg i's footholds within reach at B_plant (`propose_leg`), those
   making S' a valid stance that holds (four feet); pick the one closest to the target
   ``f_i + step_length * direction``, plus ``ankle_cost`` x the angle between the swing
   tibia and the new foothold's normal (the passive ankle's spring centres the pad
   perpendicular to the tibia: a small angle lands the pad nearly flat). If none, the
   next-best B_plant is tried. Also returns a ``pressed`` posture for execution: the
   swing foot ``press_m`` past the surface, to press the pad flat on touchdown.

With ``objective="effort"``, the stability margins only gate (dist >= ``dist_min``) and
the choices minimize the joint loading instead (``effort``: RMS of the least torques within
the limits / tau_max, :mod:`.scoring`): B_lift ranks by -``effort_weight`` x the tripod's
effort (instead of the mean margin), B_plant by progress - ``plant_effort_weight`` x
effort, f' adds ``foot_effort_weight`` x the effort of S'; body heights from
``heights_effort`` (also above the stand height).

Body poses are sampled as offsets in the body's own frame: xy (the surface plane) on a
grid, and z (the surface normal) at the distances ``ref_height + heights`` from the surface
(the stance's footholds; absolute, so the height cannot drift over the steps); the
orientation is kept (v0).
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from controlkit.kinematics.types import Posture
from controlkit.se3 import SE3

from .climb_statics import ClimbModel
from .scoring import make_scorer
from .stance import Kit, all_ok, lift_leg, move_body, propose_leg


@dataclass
class StepCfg:
    """The knobs of :func:`plan_step`."""
    adhesion: float = 40.0          # A per foot (N)
    mu: float = 0.5                 # friction coefficient
    dist_min: float = 0.3           # tripod disturbance margin to keep (/ weight)
    lift_radius: float = 0.08       # B_lift: offsets within this radius (m)
    lift_grid: int = 9              # B_lift: grid points per axis
    shift_cost: float = 2.0         # B_lift: margin lost per metre of shift (/ weight / m)
    back_cost: float = 20.0         # B_lift: and per metre behind B along the direction
    side_cost: float = 10.0         # B_lift: per metre the body is off the walking line
                                    # (margin units / m)
    side_cost_plant: float = 1.0    # B_plant: per metre off the walking line (m of progress
                                    # per m)
    lift_height: float = 0.03       # leg i raised this far along its normal (m)
    plant_reach: float = 0.12       # B_plant: up to this far ahead of B_lift (m)
    plant_side: float = 0.04        # B_plant: up to this far sideways (m)
    plant_grid: int = 9             # B_plant: grid points per axis
    path_points: int = 2            # B_plant: interior points checked along the path
    step_length: float = 0.15       # f': target this far ahead of the old foothold (m)
    ankle_cost: float = 0.05        # f': cost of the swing tibia's angle to the new normal,
                                    # as distance to the target (m per rad): near 0, the
                                    # spring-centred pad lands nearly flat
    press_m: float = 0.008          # pressed: the swing foot this far past the surface (m),
                                    # to push the pad flat on touchdown (executor)
    plant_tries: int = 5            # B_plant candidates tried for a foothold, best first
    ref_height: float = 0.16        # reference body distance to the surface (m): the
                                    # stand height
    heights: tuple = (0.0, -0.02, -0.04, -0.06)  # body heights sampled for B_lift and
                                    # B_plant, relative to ref_height: < 0 = closer to the
                                    # surface (m)
    plant_mean_weight: float = 0.03 # B_plant: value = progress + this x dist_mean
                                    # (m per body weight of mean margin)
    path_checked: int = 20          # B_plant: candidates whose path is checked, best first
    objective: str = "dist"         # dist: rank by the mean margin; effort: margins gate,
                                    # rank by the least joint loading (see module docstring)
    effort_weight: float = 10.0     # effort, B_lift: value lost per unit of effort
    plant_effort_weight: float = 0.2  # effort, B_plant: m of progress per unit of effort
    foot_effort_weight: float = 0.2 # effort, f': m of distance to the target per unit of effort
    heights_effort: tuple = (0.04, 0.02, 0.0, -0.02, -0.04)  # effort: body heights
                                    # relative to ref_height (m), as ``heights``


@dataclass
class StepResult:
    """One planned step. Postures are full (body + 12 leg angles).

    Args:
        ok: a step was found.
        why: what failed, if not ``ok``.
        leg: the swing leg i.
        lift: posture at B_lift, four feet down.
        lifted: posture at B_lift, leg i lifted.
        plant: posture at B_plant, leg i still lifted.
        pressed: posture at B_plant, leg i's foot ``press_m`` past the surface at f' (for
            execution; the plan's S' is ``posture``).
        posture: S' at B_plant, leg i planted on f' (the next node).
        stance: (L,) the next stance S' (foothold indices).
        scores: margins along the step: ``lift_dist`` (T at B_lift), ``plant_dist`` (T at
            B_plant), ``s_next`` / ``dist_next`` (S' with four feet), ``progress`` (body
            moved along the direction, m), ``ankle_deg`` (swing tibia vs. the new normal);
            ``lift_effort``, ``plant_effort``, ``effort_next`` (nan unless
            ``objective="effort"``).
    """
    ok: bool
    why: str = ""
    leg: int = -1
    lift: Posture = None
    lifted: Posture = None
    plant: Posture = None
    pressed: Posture = None
    posture: Posture = None
    stance: jax.Array = None
    scores: dict = None


def shifted(body: SE3, offsets: np.ndarray) -> SE3:
    """Body poses: ``body`` moved by each of ``offsets`` (K, 3), given in the body frame.

    Returns:
        A batched SE3 with K poses, the orientation of ``body``.
    """
    w = np.asarray(body.wxyz_xyz)
    R = np.asarray(body.rotation().as_matrix())
    p = w[4:] + np.asarray(offsets) @ R.T
    return SE3(jnp.asarray(np.concatenate([np.broadcast_to(w[:4], (len(p), 4)), p], 1)))


def _with_heights(off: np.ndarray, heights) -> np.ndarray:
    """Every offset (K, 3) at every height (its z replaced): (K * H, 3)."""
    h = np.asarray(heights, float)
    out = np.repeat(off[None], len(h), 0)
    out[:, :, 2] = h[:, None]
    return out.reshape(-1, 3)


def _grid(nx, ny, x0, x1, y0, y1):
    """(nx ny, 3) body-frame offsets on a grid, z = 0."""
    x, y = np.meshgrid(np.linspace(x0, x1, nx), np.linspace(y0, y1, ny), indexing="ij")
    return np.stack([x.ravel(), y.ravel(), np.zeros(x.size)], 1)


class Planner:
    """The jitted, batched pieces of :func:`plan_step` for one robot, terrain and pool.

    Args:
        kit: the stance kit (robot, scene, config, foothold pool).
        cm: the climb model (statics).
        cfg: the step knobs.
    """

    def __init__(self, kit: Kit, cm: ClimbModel, cfg: StepCfg):
        self.kit, self.cm, self.cfg = kit, cm, cfg
        robot, scene, gcfg, fh = kit.robot, kit.scene, kit.cfg, kit.footholds
        L = robot.num_legs
        args = (robot, scene, gcfg, fh)

        def move(posture, stance, planted, body):
            ok, new, _ = move_body(*args, posture, stance, planted, body)
            return ok, new

        # move to each of K bodies (batched over the bodies)
        self.move = jax.jit(jax.vmap(move, in_axes=(None, None, None, 0)))
        self.lift = jax.jit(lambda post, st, i, h: lift_leg(robot, gcfg, fh, post, st, i, h))
        self.propose = jax.jit(lambda key, cand, post, st, i:
                               propose_leg(key, *args, cand, post, st, i))
        self.checks = jax.jit(lambda post, st, planted: all_ok(kit.checks(post, st, planted)))
        self.place = kit.place_leg
        leg = robot.leg

        def tibia_angles(body, thetas, normals, i):
            """(K,) angle between leg i's tibia (back along it) and each normal (rad)."""
            rot = robot.shoulders(body)[i].rotation()
            v = jax.vmap(lambda th: rot.apply(leg.contact_vector(th)))(thetas)
            return jnp.arccos(jnp.clip(jnp.sum(v * normals, -1), -1.0, 1.0))

        self.tibia_angles = jax.jit(tibia_angles, static_argnums=3)
        # scorers: the four tripods (leg i lifted) and the full stance
        eff = cfg.objective == "effort"
        self.tripod = {i: make_scorer(cm, fh, tuple(j for j in range(L) if j != i),
                                      cfg.adhesion, cfg.mu, extras=False, effort=eff)
                       for i in range(L)}
        self.full = make_scorer(cm, fh, tuple(range(L)), cfg.adhesion, cfg.mu, extras=False,
                                effort=eff)


def _bcast(posture: Posture, n: int) -> Posture:
    return jax.tree.map(lambda x: jnp.broadcast_to(x, (n,) + x.shape), posture)


def plan_step(pl: Planner, stance, posture: Posture, i: int, direction, key,
              origin=None) -> StepResult:
    """Plan one crawl step: move leg ``i`` ahead along ``direction``.

    Args:
        pl: the planner pieces (:class:`Planner`).
        stance: (L,) the current stance S (foothold indices), all feet planted.
        posture: its witness posture.
        i: the swing leg.
        direction: (3,) the walking direction, world frame (unit, in the surface plane).
        key: PRNG key (foothold proposals).
        origin: (3,) a point on the walking line (world); the body is kept near the line
            through it along ``direction``. None: the current body position.

    Returns:
        The :class:`StepResult`.
    """
    cfg, kit = pl.cfg, pl.kit
    L = kit.robot.num_legs
    d = np.asarray(direction, float)
    d /= np.linalg.norm(d)
    all_down = jnp.ones(L, bool)
    tripod = jnp.arange(L) != i
    R = np.asarray(posture.body.rotation().as_matrix())
    p0 = np.asarray(posture.body.translation())
    origin = p0 if origin is None else np.asarray(origin, float)
    side_w = R @ np.cross([0.0, 0.0, 1.0], R.T @ d)        # sideways, in the surface plane
    side_w /= np.linalg.norm(side_w)
    by_effort = cfg.objective == "effort"
    # body z offsets that put the body at ref_height + heights from the surface (the mean
    # of the stance's footholds, along the body's z)
    feet = np.asarray(kit.footholds.position[stance])
    gap0 = float((p0 - feet.mean(0)) @ R[:, 2])
    dz = cfg.ref_height + np.asarray(cfg.heights_effort if by_effort else cfg.heights) - gap0

    def effort_ok(sc):
        """Where the least-torque QP has a solution (always, without the effort objective)."""
        return np.isfinite(sc["effort"]) if by_effort else True

    def off_line(base, off):
        """|sideways distance from the walking line| of the bodies ``base + R off`` (m)."""
        return np.abs((base + off @ R.T - origin) @ side_w)

    # --- 1. B_lift: around B, four feet held; T must hold
    r = cfg.lift_radius
    off = _grid(cfg.lift_grid, cfg.lift_grid, -r, r, -r, r)
    off = off[np.linalg.norm(off, axis=1) <= r + 1e-9]
    off = _with_heights(off, dz)
    ok, posts = pl.move(posture, stance, all_down, shifted(posture.body, off))
    ok = np.asarray(ok)
    if not ok.any():
        return StepResult(False, "B_lift: no valid body shift", i)
    sc = {k: np.asarray(v) for k, v in pl.tripod[i](posts, jnp.broadcast_to(stance, (len(off), L))).items()}
    good = ok & (sc["s"] >= 1.0) & (sc["dist"] >= cfg.dist_min) & effort_ok(sc)
    if not good.any():
        return StepResult(False, f"B_lift: tripod never holds with margin >= {cfg.dist_min} "
                                 f"(best {sc['dist'][ok].max():.2f})", i)
    back = np.maximum(0.0, -(off @ R.T) @ d)               # B_lift behind B (m)
    gain = -cfg.effort_weight * sc["effort"] if by_effort else sc["dist_mean"]
    value = np.where(good, gain - cfg.shift_cost * np.linalg.norm(off[:, :2], axis=1)
                     - cfg.back_cost * back - cfg.side_cost * off_line(p0, off), -np.inf)
    j = int(np.argmax(value))
    lift_post = jax.tree.map(lambda x: x[j], posts)
    lift_dist = float(sc["dist"][j])
    lift_effort = float(sc["effort"][j]) if by_effort else float("nan")

    # --- 2. lift leg i
    ok, lifted = pl.lift(lift_post, stance, i, cfg.lift_height)
    if not (bool(ok) and bool(pl.checks(lifted, stance, tripod))):
        return StepResult(False, "lift: leg cannot be raised", i, lift=lift_post)

    # --- 3. B_plant: ahead of B_lift, three feet held; T must hold along the way
    d_body = R.T @ d                                       # direction in the body frame
    side = np.cross([0.0, 0.0, 1.0], d_body)
    a, b = np.meshgrid(np.linspace(0.0, cfg.plant_reach, cfg.plant_grid),
                       np.linspace(-cfg.plant_side, cfg.plant_side, cfg.plant_grid), indexing="ij")
    off = a.reshape(-1, 1) * d_body + b.reshape(-1, 1) * side
    off[:, 2] = 0.0
    # the same absolute heights, from the lifted body
    lift_dz = float((np.asarray(lifted.body.translation()) - p0) @ R[:, 2])
    off = _with_heights(off, dz - lift_dz)
    n = len(off)
    # the end poses first; then the path, for the best candidates only
    ok, end_posts = pl.move(lifted, stance, tripod, shifted(lifted.body, off))
    sc = {k: np.asarray(v) for k, v in pl.tripod[i](end_posts, jnp.broadcast_to(stance, (n, L))).items()}
    end_ok = np.asarray(ok) & (sc["s"] >= 1.0) & (sc["dist"] >= cfg.dist_min) & effort_ok(sc)
    if not end_ok.any():
        return StepResult(False, "B_plant: no forward shift keeps the tripod holding", i,
                          lift=lift_post, lifted=lifted)
    end_dist = sc["dist"]
    end_effort = sc["effort"] if by_effort else np.full(n, np.nan)
    progress = (off @ R.T) @ d                             # world progress along d
    gain = (-cfg.plant_effort_weight * np.where(end_ok, sc["effort"], 0.0) if by_effort
            else cfg.plant_mean_weight * sc["dist_mean"])
    value = (progress + gain
             - cfg.side_cost_plant * off_line(np.asarray(lifted.body.translation()), off))
    best = [k for k in np.argsort(-value) if end_ok[k]][: cfg.path_checked]
    fracs = np.linspace(0.0, 1.0, cfg.path_points + 2)[1:-1]        # interior points
    if len(fracs):
        mid = (fracs[:, None, None] * off[best][None]).reshape(-1, 3)
        ok, mposts = pl.move(lifted, stance, tripod, shifted(lifted.body, mid))
        msc = pl.tripod[i](mposts, jnp.broadcast_to(stance, (len(mid), L)))
        path_ok = (np.asarray(ok) & (np.asarray(msc["s"]) >= 1.0)
                   & (np.asarray(msc["dist"]) >= cfg.dist_min)).reshape(len(fracs), -1).all(0)
        order = [k for k, good_path in zip(best, path_ok) if good_path]
    else:
        order = best
    if not order:
        return StepResult(False, "B_plant: no path keeps the tripod holding", i,
                          lift=lift_post, lifted=lifted)

    # --- 4. f': at B_plant, leg i's reachable footholds; S' valid and holding; nearest
    #     to the target
    target = np.asarray(kit.footholds.position[stance[i]]) + cfg.step_length * d
    for k in order[: cfg.plant_tries]:
        plant = jax.tree.map(lambda x: x[k], end_posts)
        key, sub = jax.random.split(key)
        valid, cands, stances = pl.propose(sub, kit.candidates(plant.body), plant, stance, i)
        valid = np.asarray(valid)
        if not valid.any():
            continue
        idx = np.nonzero(valid)[0]
        cposts = jax.tree.map(lambda x: x[idx], cands)
        cst = stances[idx]
        full = {k_: np.asarray(v) for k_, v in pl.full(cposts, cst).items()}
        holds_s = (full["s"] >= 1.0) & effort_ok(full)
        if not holds_s.any():
            continue
        pos = np.asarray(kit.footholds.position[cst[:, i]])
        nrm = kit.footholds.normal[cst[:, i]]
        ang = np.asarray(pl.tibia_angles(plant.body, cposts.thetas[:, i], nrm, i))
        cost = np.linalg.norm(pos - target, axis=1) + cfg.ankle_cost * ang
        if by_effort:
            cost = cost + cfg.foot_effort_weight * np.where(holds_s, full["effort"], 0.0)
        m = int(np.argmin(np.where(holds_s, cost, np.inf)))
        new = jax.tree.map(lambda x: x[m], cposts)
        # pressed: the foot past the surface along the normal (pivot at foot_radius - press)
        n_m = np.asarray(nrm[m])
        point = pos[m] + (kit.cfg.foot_radius - cfg.press_m) * n_m
        ok_p, pressed = pl.place(new, i, jnp.asarray(point))
        return StepResult(
            True, "", i, lift=lift_post, lifted=lifted, plant=plant,
            pressed=pressed if bool(ok_p) else new, posture=new, stance=cst[m],
            scores=dict(lift_dist=lift_dist, plant_dist=float(end_dist[k]),
                        s_next=float(full["s"][m]), dist_next=float(full["dist"][m]),
                        progress=float(np.asarray(plant.body.translation() - p0) @ d),
                        ankle_deg=float(np.degrees(ang[m])), lift_effort=lift_effort,
                        plant_effort=float(end_effort[k]),
                        effort_next=float(full["effort"][m]) if by_effort else float("nan")))
    return StepResult(False, "plant: no reachable foothold that holds", i,
                      lift=lift_post, lifted=lifted)
