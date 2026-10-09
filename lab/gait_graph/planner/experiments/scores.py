"""Score functions on sampled postures of the climb robot: floor and wall, full stance and
left front leg lifted. Look at the results in MuJoCo.

A **score function** rates a posture on its stance; here the hold margin s* (the largest
gravity factor at which the planted feet still hold it; s* >= 1: holds, see
``docs/planner-design.md``, "The hold check"). Per posture:

- ``s``: s* of the full model (adhesion, friction pyramid, servo torques);
- ``s_pm``: s* of the point-mass model (centre of mass + feet, no torques);
- ``dist``: the disturbance margin of the full model, in units of the robot's weight: the
  largest push (or twist, as force x 0.1 m) on the body, in its worst of 12 directions,
  the stance still resists under the real gravity (:func:`..statics.disturbance_margin`).
  0 where the posture does not hold at all (s < 1). Unlike s*, it depends on the posture
  even where friction caps s* (a vertical wall); but many good postures tie at that
  friction plateau (the worst direction is a load along gravity), so:
- ``dist_mean``: the mean over the 12 directions (/ weight), the tie-break among postures
  with the same ``dist``;
- ``tau``: peak |servo torque| of the least-torque foot forces, / ``tau_max``;
- ``effort``: the least joint loading under g within the limits (min sum tau^2 over the
  internal forces, :func:`..statics.least_torque_qp`), as RMS sqrt(sum tau^2 / n) / tau_max;
- ``force``: the foot forces under the real gravity: the least-torque forces within the
  limits (:func:`..statics.least_torque_qp`, unique); saved for the viewer (arrows: green
  pushes, red pulls).

Postures are ranked by ``rank_by`` (default ``dist``), rounded to 0.01, then by
``dist_mean``; the refinement accepts a change that raises that order. ``rank_by=effort``:
stable postures first (s >= ``s_min`` and dist >= ``dist_min``), among them the lowest
``effort``; the unstable ones after, by ``dist``.

Scenarios (gravity down in both):

- ``floor``: flat ground, the body around the stand pose;
- ``wall``: a vertical wall (face at x = ``wall_x``, normal -x), the body facing it at
  stand height, its forward axis pointing up the wall.

Modes: ``full`` (four feet planted, scored on all four), ``lifted`` (leg 0, the left
front, lifted ``lift_height`` along its surface normal; scored on legs 1-3).

Per (scenario, mode):

1. **Samples**: ``n_samples`` stances (`Kit.sample_stance`) at body poses jittered around
   the nominal one; the valid ones (kinematic checks: reach, limits, ankle, collisions)
   are scored.
2. **Refined**: from the ``top_k`` best, ``rounds`` rounds of local search: nudge the body
   (feet held) or re-plant one planted leg; keep a change if it stays valid and raises s.

    uv run runkit run lab.gait_graph.planner.experiments.scores
    uv run runkit run lab.gait_graph.planner.experiments.scores n_samples=128 rounds=60
    uv run runkit run lab.gait_graph.planner.experiments.scores rank_by=effort
    uv run --extra mjx python -m lab.gait_graph.planner.view experiments/runs/planner_scores/latest/out/wall_lifted_refined.npz

Prints a table per (scenario, mode). Saves to ``out/``, per (scenario, mode) and stage
(``samples`` / ``refined``), for ``ctk play``: ``{scenario}_{mode}_{stage}.npz``, one
posture per frame, best first, 1 s each (space pauses, left / right arrows step; the live
plot shows s, s_pm and tau of the frame), scene ``{scenario}_scene.xml``.
"""
import dataclasses
import math
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from controlkit.se3 import SE3
from runkit import Experiment, RunContext

from ... import terrain
from ..climb_cfg import climb_cfg
from ..climb_model.config import MjModelCfg
from ..climb_model.mjmodel import build, make_robot
from ..climb_statics import ClimbModel
from ..stance import Kit, all_ok
from ..scoring import G, make_scorer, rank_key, stable
from ..statics import least_torque_qp

REPO = Path(__file__).resolve().parents[4]
LIFTED = 0                                     # the left front leg
CHUNK = 128                                    # postures per vmapped scoring call


@dataclasses.dataclass
class ScoresCfg:
    """The climb robot, the hold parameters, the sampling and the scenes."""
    mjmodel: MjModelCfg = dataclasses.field(default_factory=MjModelCfg)
    scenarios: str = "floor,wall"   # which scenarios to run
    modes: str = "full,lifted"      # which modes to run
    n_samples: int = 64             # stance samples per (scenario, mode)
    top_k: int = 8                  # refine this many best samples (0: no refinement)
    rounds: int = 40                # local-search rounds per refined posture
    lift_height: float = 0.03       # leg 0 lifted this far, along its surface normal (m)
    adhesion: float = 40.0          # A, per foot (N)
    mu: float = 0.5                 # friction coefficient
    body_xy: float = 0.05           # sample jitter in the surface plane (m)
    body_dz: float = 0.03           # sample jitter along the surface normal (m)
    body_rot_deg: float = 10.0      # sample jitter in yaw (pitch / roll: a third) (deg)
    nudge_xy: float = 0.01          # refine: body nudge in the surface plane (m)
    nudge_rot_deg: float = 3.0      # refine: body nudge in yaw (deg)
    p_body: float = 0.5             # refine: probability of a body nudge (else a leg)
    rank_by: str = "dist"           # the score that ranks and that the refinement raises; effort
    s_min: float = 1.5              # rank_by=effort: stable if s >= s_min ...
    dist_min: float = 0.25          # ... and dist >= dist_min
    wall_x: float = 0.45            # wall face at this x (normal -x)
    wall_body_z: float = 0.6        # body height on the wall (world z)
    seed: int = 0


# ----------------------------------------------------------------------------- scenes
def _quat_axis_angle(axis, angle) -> np.ndarray:
    """wxyz quaternion of a rotation by ``angle`` (rad) about the unit ``axis``."""
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    return np.concatenate([[math.cos(angle / 2)], math.sin(angle / 2) * axis])


def _quat_mul(a, b) -> np.ndarray:
    """Hamilton product of wxyz quaternions."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def scenario(name: str, cfg: ScoresCfg):
    """A scenario's terrain boxes and nominal body pose.

    Returns:
        A tuple ``(boxes, body)``:

        boxes: ``((center, half), ...)``, the ground slab first (top face at z = 0).
        body: the nominal body pose (SE3): body z along the surface normal, at stand height.
    """
    ground = ((0.0, 0.0, -0.05), (2.0, 1.0, 0.05))
    h = cfg.mjmodel.stand_height
    if name == "floor":
        return (ground,), SE3(jnp.asarray([1.0, 0, 0, 0, 0.0, 0.0, h]))
    if name == "wall":
        wall = ((cfg.wall_x + 0.05, 0.0, 1.0), (0.05, 1.0, 1.0))      # face at x = wall_x
        # body z -> -x (the wall normal), body x -> +z (up the wall): Ry(-90 deg)
        q = _quat_axis_angle([0, 1, 0], -math.pi / 2)
        return (ground, wall), SE3(jnp.asarray([*q, cfg.wall_x - h, 0.0, cfg.wall_body_z]))
    raise ValueError(f"scenario: floor or wall, not {name!r}")


def jitter(body: SE3, rng, xy, dz, rot_deg) -> SE3:
    """``body`` moved by a random offset in its own frame: x, y ~ U[-xy, xy] (the surface
    plane), z ~ U[-dz, dz] (the normal); yaw ~ U[-rot, rot], pitch / roll a third."""
    r = math.radians(rot_deg)
    q = _quat_axis_angle([0, 0, 1], rng.uniform(-r, r))
    q = _quat_mul(q, _quat_axis_angle([0, 1, 0], rng.uniform(-r, r) / 3))
    q = _quat_mul(q, _quat_axis_angle([1, 0, 0], rng.uniform(-r, r) / 3))
    t = [rng.uniform(-xy, xy), rng.uniform(-xy, xy), rng.uniform(-dz, dz)]
    return body @ SE3(jnp.asarray([*q, *t], float))


def scene_xml(cfg: ScoresCfg, boxes) -> str:
    """The climb robot's MuJoCo scene plus the terrain boxes beyond the ground (which the
    scene's floor plane already is)."""
    spec, _ = build(cfg.mjmodel, write=False)
    for i, (c, h) in enumerate(boxes[1:]):
        spec.worldbody.add_geom(name=f"terrain{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
                                pos=list(c), size=list(h), rgba=[0.55, 0.55, 0.6, 1],
                                friction=[cfg.mjmodel.pad_friction, 0.005, 0.0001])
    return spec.to_xml()


# ----------------------------------------------------------------------------- scoring
def foot_forces(cm: ClimbModel, footholds, items, planted_idx, adhesion, mu):
    """The foot forces of each (posture, stance) under the real gravity: the least-torque
    forces within the limits (:func:`..statics.least_torque_qp`, unique). For the viewer.

    Returns:
        (N, L, 3) forces, world frame (N); 0 for the lifted leg, and where the QP has no
        solution (the posture does not hold: violation > 1e-2).
    """
    idx = np.asarray(planted_idx)
    tau_max = float(cm.mj.forcerange[1])

    @jax.jit
    @jax.vmap
    def qp(posture, stance):
        normals = footholds.normal[stance]
        st = cm.statics(cm.qpos(posture, normals), G)
        F, _, viol = least_torque_qp(st, planted_idx, normals[idx], adhesion, mu, tau_max)
        return F, viol

    parts = [qp(_stack([p for p, _ in items[i: i + CHUNK]]),
                jnp.stack([s_ for _, s_ in items[i: i + CHUNK]]))
             for i in range(0, len(items), CHUNK)]
    F = np.concatenate([np.asarray(f) for f, _ in parts])
    viol = np.concatenate([np.asarray(v) for _, v in parts])
    out = np.zeros((len(items), cm.mj.num_legs, 3))
    out[:, idx] = np.where(viol[:, None, None] < 1e-2, F, 0.0)
    return out


def _height(name: str, cfg: ScoresCfg, posture) -> float:
    """The body's (base origin's) distance to the surface along its normal (m)."""
    p = np.asarray(posture.body.translation())
    return float(cfg.wall_x - p[0]) if name == "wall" else float(p[2])


def _stack(items):
    return jax.tree.map(lambda *x: jnp.stack(x), *items)


# ----------------------------------------------------------------------------- run
def run_one(cfg: ScoresCfg, cm: ClimbModel, name: str, mode: str, ctx: RunContext,
            rng: np.random.Generator) -> dict:
    """Samples and refines them, for one (scenario, mode); prints, saves the replays."""
    boxes, nominal = scenario(name, cfg)
    gcfg = climb_cfg(cfg.mjmodel, boxes=boxes,
                     num_footholds=40_000 if name == "floor" else 80_000)
    robot = make_robot(cfg.mjmodel)
    scene = terrain.make_scene(boxes)
    fh = terrain.sample_footholds(jax.random.PRNGKey(cfg.seed), scene, gcfg.num_footholds,
                                  edge_margin=gcfg.edge_margin)
    kit = Kit(robot, scene, gcfg, fh)
    L = cfg.mjmodel.num_legs
    planted_idx = tuple(range(L)) if mode == "full" else tuple(j for j in range(L) if j != LIFTED)
    planted = jnp.asarray([j in planted_idx for j in range(L)])
    score = make_scorer(cm, fh, planted_idx, cfg.adhesion, cfg.mu, effort=True)

    def rank_key_(x):
        return rank_key(x, cfg.rank_by, cfg.s_min, cfg.dist_min)

    def score_all(items):
        """Score (posture, stance) pairs in chunks of ``CHUNK`` (one vmap each)."""
        parts = [score(_stack([p for p, _ in items[i: i + CHUNK]]),
                       jnp.stack([s_ for _, s_ in items[i: i + CHUNK]]))
                 for i in range(0, len(items), CHUNK)]
        return {k: np.concatenate([np.asarray(x[k]) for x in parts]) for k in parts[0]}

    def valid(posture, stance):
        return bool(all_ok(kit.checks(posture, stance, planted)))

    # --- 1. samples
    t0 = time.perf_counter()
    samples = []                                    # (posture, stance)
    key = jax.random.PRNGKey(cfg.seed)
    for _ in range(cfg.n_samples):
        body = jitter(nominal, rng, cfg.body_xy, cfg.body_dz, cfg.body_rot_deg)
        key, k = jax.random.split(key)
        ok, post, st = kit.sample_stance(k, kit.candidates(body), body)
        if not bool(ok):
            continue
        if mode == "lifted":
            ok, post = kit.lift_leg(post, st, LIFTED, cfg.lift_height)
            if not (bool(ok) and valid(post, st)):
                continue
        samples.append((post, st))
    if not samples:
        print(f"\n[{name} / {mode}] no valid samples")
        return {}
    sc = score_all(samples)
    t_samples = time.perf_counter() - t0

    # --- 2. refined: local search from the top k
    t0 = time.perf_counter()
    key_ = cfg.rank_by
    rk = rank_key_(sc)
    order = np.argsort(-rk)
    refined, accepted = [], 0
    for j in order[: cfg.top_k]:
        post, st = samples[j]
        best = float(rk[j])
        for _ in range(cfg.rounds):
            key, k = jax.random.split(key)
            if rng.uniform() < cfg.p_body:              # nudge the body, feet held
                body = jitter(post.body, rng, cfg.nudge_xy, cfg.nudge_xy / 2,
                              cfg.nudge_rot_deg)
                ok, new, _ = kit.move_body(post, st, planted, body)
                new_st = st
            else:                                       # re-plant one planted leg
                leg = int(rng.choice(planted_idx))
                ok, new, new_st, _ = kit.resample_leg(k, kit.candidates(post.body), post, st,
                                                      leg)
                ok = bool(ok) and valid(new, new_st)
            if not bool(ok):
                continue
            s_new = float(rank_key_(score(_stack([new]), new_st[None]))[0])
            if s_new > best:
                post, st, best = new, new_st, s_new
                accepted += 1
        refined.append((post, st))
    rc = score_all(refined) if refined else None
    t_refined = time.perf_counter() - t0

    # --- print
    print(f"\n[{name} / {mode}] {len(samples)}/{cfg.n_samples} valid samples ({t_samples:.0f} s); "
          + (f"refined top {len(refined)} x {cfg.rounds} rounds, {accepted} improvements "
             f"({t_refined:.0f} s)" if refined else "no refinement (top_k = 0)"))
    o_s = np.argsort(-rank_key_(sc))
    _table(f"samples, best {key_} first (top 5, bottom 5)", sc,
           np.concatenate([o_s[:5], o_s[-5:]]) if len(o_s) > 10 else o_s, len(samples), cfg)
    if refined:
        _table(f"refined, best {key_} first", rc, np.argsort(-rank_key_(rc)), len(refined), cfg)

    # --- save the replays
    xml = ctx.out / f"{name}_scene.xml"
    xml.write_text(scene_xml(cfg, boxes))
    stages = (("samples", samples, sc),) + ((("refined", refined, rc),) if refined else ())
    for stage, items, scores in stages:
        o = np.argsort(-rank_key_(scores))
        qpos = np.stack([np.asarray(cm.qpos(items[i][0], fh.normal[items[i][1]])) for i in o])
        height = np.array([_height(name, cfg, p) for p, _ in items])
        plot = np.stack([scores["s"][o], scores["s_pm"][o], scores["dist"][o],
                         scores["dist_mean"][o], scores["tau"][o],
                         np.minimum(scores["effort"][o], 2.0), height[o]], -1)
        path = ctx.out / f"{name}_{mode}_{stage}.npz"
        normals = np.stack([np.asarray(fh.normal[items[i][1]]) for i in o])
        forces = foot_forces(cm, fh, [items[i] for i in o], planted_idx, cfg.adhesion, cfg.mu)
        np.savez(path, qpos=qpos, qvel=np.zeros((len(o), cm.model.nv)), timestep=1.0,
                 foot_force=forces, foot_normal=normals,
                 foot_planted=np.array([j in planted_idx for j in range(L)]),
                 adhesion=cfg.adhesion, mu=cfg.mu,
                 pad_drop=cfg.mjmodel.pivot_height + cfg.mjmodel.pad_thickness,
                 model=_model_path(xml), plot=plot,
                 plot_labels=np.array(["s* full", "s* point mass", "disturbance / weight",
                                       "disturbance mean / weight", "peak tau / max",
                                       "effort (RMS tau / max)", "body height (m)"]),
                 plot_title=f"{name} / {mode} / {stage}")
    print(f"  saved {name}_{mode}_{{{','.join(x[0] for x in stages)}}}.npz (+ {name}_scene.xml)")

    def stats(x):
        st_ = stable(x, cfg.s_min, cfg.dist_min)
        return {"holds": round(float(np.mean(x["s"] >= 1)), 3),
                "stable": round(float(np.mean(st_)), 3),
                "effort_min_stable": round(float(x["effort"][st_].min()), 3) if st_.any() else None,
                "s_median": round(float(np.median(x["s"])), 3),
                "dist_median": round(float(np.median(x["dist"])), 3),
                "dist_max": round(float(x["dist"].max()), 3)}
    return {"samples": stats(sc) | {"n": len(samples)}} | ({"refined": stats(rc)} if refined else {})


def _model_path(xml: Path) -> str:
    """The scene's path for ``ctk play``: relative to the repo if inside it, else absolute
    (``ctk play`` joins it to the repo root; an absolute path stays as is)."""
    try:
        return str(xml.resolve().relative_to(REPO))
    except ValueError:
        return str(xml.resolve())


def _table(title, sc, rows, n, cfg):
    st_ = stable(sc, cfg.s_min, cfg.dist_min)
    print(f"  {title}, of {n} ({int(st_.sum())} stable: s >= {cfg.s_min}, dist >= {cfg.dist_min}):")
    print(f"  {'s* full':>8} | {'s* point mass':>13} | {'dist / weight':>13} "
          f"| {'dist mean':>9} | {'peak tau/max':>12} | {'effort':>6} | {'stable':>6} "
          f"| {'violation':>9}")
    for i in rows:
        print(f"  {sc['s'][i]:8.2f} | {sc['s_pm'][i]:13.2f} | {sc['dist'][i]:13.2f} "
              f"| {sc['dist_mean'][i]:9.2f} | {sc['tau'][i]:12.2f} | {sc['effort'][i]:6.3f} "
              f"| {'yes' if st_[i] else 'no':>6} | {sc['viol'][i]:9.1e}")


exp = Experiment("planner_scores")


@exp.run
def run(cfg: ScoresCfg, ctx: RunContext) -> dict:
    """Run every (scenario, mode); see the module docstring.

    Returns:
        Per (scenario, mode): samples (n, share holding, median / max s) and refined.
    """
    cm = ClimbModel(cfg.mjmodel)
    rng = np.random.default_rng(cfg.seed)
    print(f"planner_scores: climb robot {cm.mass:.2f} kg; A = {cfg.adhesion:.0f} N, "
          f"mu = {cfg.mu}, tau_max = {cfg.mjmodel.forcerange[1]:.0f} N m; s* >= 1: holds")
    out = {}
    for name in cfg.scenarios.split(","):
        for mode in cfg.modes.split(","):
            out[f"{name}_{mode}"] = run_one(cfg, cm, name, mode, ctx, rng)
    return out


if __name__ == "__main__":
    exp.main(sys.argv[1:])
