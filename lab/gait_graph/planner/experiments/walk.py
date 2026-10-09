"""Walk with the stance planner: chain :func:`..step.plan_step` in crawl order.

From a sampled start stance, plan ``n_steps`` crawl steps (swing order left back, left
front, right back, right front: legs 1, 0, 2, 3) along the walking direction: +x on the
floor, up (+z) on the wall. Stops at the first step it cannot plan.

    uv run runkit run lab.gait_graph.planner.experiments.walk
    uv run runkit run lab.gait_graph.planner.experiments.walk scenario=wall n_steps=12
    uv run runkit run lab.gait_graph.planner.experiments.walk step.objective=effort mjmodel.ankle_range_deg=60
    uv run runkit run lab.gait_graph.planner.experiments.walk start=stand
    uv run runkit run lab.gait_graph.planner.experiments.walk planner=fast start=stand mjmodel.ankle_range_deg=60

Prints one row per step (swing leg, body progress, the tripod's margins at B_lift and
B_plant, S' margins). Saves ``out/walk.npz`` (+ scene ``.xml``) for ``ctk play``: every
phase of every step (S, B_lift, leg lifted, B_plant, pressed, S'), interpolated linearly in
``qpos`` (``frames_per_phase`` frames each); plus the plan itself for
:mod:`.execute`: ``key_qpos`` (the key postures), ``key_magnets`` (which magnets are on
at each) and ``key_phase`` (their names).
"""
import dataclasses
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from controlkit.kinematics.types import Posture
from runkit import Experiment, RunContext

from ... import terrain
from ..climb_cfg import climb_cfg
from ..climb_model.config import MjModelCfg
from ..climb_model.mjmodel import make_robot
from ..climb_model.poses import foot_targets, leg_angles
from ..climb_statics import ClimbModel
from ..stance import Kit, all_ok
from ..step import Planner, StepCfg, plan_step
from ..samplers import region
from ..step_fast import FastPlanner, FastStepCfg, plan_step_fast
from ..transfer import TransferCfg, plan_transfer
from .scores import ScoresCfg, _model_path, scenario, scene_xml

CRAWL = (1, 0, 2, 3)                      # left back, left front, right back, right front
ROTARY = (1, 0, 3, 2)                     # left back, left front, right front, right back:
                                          # consecutive swing legs neighbours, so consecutive
                                          # tripods share a side (their support triangles
                                          # overlap), never a diagonal
GAITS = {"crawl": CRAWL, "rotary": ROTARY}


@dataclasses.dataclass
class WalkCfg:
    """The robot, the scenario and the step knobs."""
    mjmodel: MjModelCfg = dataclasses.field(default_factory=MjModelCfg)
    step: StepCfg = dataclasses.field(default_factory=StepCfg)
    planner: str = "grid"           # grid: plan_step (LP margins); fast: plan_step_fast
                                    # (samplers, closed-form torques + cone / push checks);
                                    # transfer: plan_transfer (the tripod graph: T -> S -> T')
    fast: FastStepCfg = dataclasses.field(default_factory=FastStepCfg)   # also the checks of
                                    # planner=transfer (adhesion, mu, push, twist)
    transfer: TransferCfg = dataclasses.field(default_factory=TransferCfg)
    retries: int = 2                # planner=transfer: a failed transfer is planned again (new
                                    # samples) this many times before the walk stops
    scenario: str = "floor"         # floor (walk +x) or wall (climb up, +z)
    n_steps: int = 8                # crawl steps to plan
    gait: str = "crawl"             # swing order: crawl (1, 0, 2, 3) or rotary (1, 0, 3, 2)
    start: str = "sample"           # sample: the best of start_tries sampled stances; stand:
                                    # the `stand` keyframe (floor only; feet snapped to the
                                    # nearest footholds of the pool)
    start_tries: int = 16           # start stances sampled; the best (full-stance margin) kept
    start_tripod: bool = False      # planner=fast: move the start body (feet held) to a pose
                                    # where the first swing leg's tripod is good (least effort)
    frames_per_phase: int = 10      # replay: frames between two phases
    frame_s: float = 0.05           # replay: seconds per frame
    wall_x: float = 0.45            # wall face at this x (as the scores experiment)
    wall_body_z: float = 0.6        # body height on the wall at the start
    seed: int = 0


def _qpos(cm, posture, stance, footholds):
    return np.asarray(cm.qpos(posture, footholds.normal[stance]))


def _sampled_start(cfg: WalkCfg, kit: Kit, pl: Planner, nominal, key):
    """The best of ``start_tries`` stances sampled at ``nominal``: the most stable, or with
    ``objective="effort"`` the stable one with the least effort.

    Returns:
        ``(posture, stance, message)``.
    """
    starts = []
    for _ in range(cfg.start_tries):
        key, k = jax.random.split(key)
        ok, post, st = kit.sample_stance(k, kit.candidates(nominal), nominal)
        if bool(ok):
            starts.append((post, st))
    if not starts:
        raise SystemExit("no valid start stance")
    P = jax.tree.map(lambda *x: jnp.stack(x), *[p for p, _ in starts])
    sc = {k: np.asarray(v) for k, v in pl.full(P, jnp.stack([s for _, s in starts])).items()}
    if cfg.step.objective == "effort":
        stable = (sc["dist"] >= cfg.step.dist_min) & np.isfinite(sc["effort"])
        j = int(np.argmin(np.where(stable, sc["effort"], np.inf))) if stable.any() \
            else int(np.argmax(sc["dist"]))
    else:
        j = int(np.argmax(sc["dist"]))
    msg = (f"best of {len(starts)} sampled stances (four-foot margin {sc['dist'][j]:.2f} x weight"
           + (f", effort {sc['effort'][j]:.3f}" if cfg.step.objective == "effort" else "") + ")")
    return starts[j][0], starts[j][1], msg


def _stand_start(cfg: WalkCfg, kit: Kit, pl: Planner, nominal):
    """The ``stand`` keyframe (floor): body level at ``stand_height``, each foot on the
    pool foothold nearest its keyframe target, legs re-solved onto it.

    Returns:
        ``(posture, stance, message)``.
    """
    if cfg.scenario != "floor":
        raise SystemExit("start=stand: floor only")
    robot = kit.robot
    ok, thetas = leg_angles(robot, cfg.mjmodel, cfg.mjmodel.stand_height)
    if not bool(np.all(ok)):
        raise SystemExit("start=stand: keyframe IK failed")
    targets = np.asarray(foot_targets(robot, cfg.mjmodel))
    pos = np.asarray(kit.footholds.position)
    d = np.linalg.norm(pos[None, :, :2] - targets[:, None, :2], axis=-1)    # (L, M)
    stance = jnp.asarray(d.argmin(1))
    L = robot.num_legs
    ok, posture, checks = kit.move_body(Posture(nominal, thetas), stance, jnp.ones(L, bool),
                                        nominal)
    if not bool(ok & all_ok(checks)):
        raise SystemExit(f"start=stand: the snapped stand posture is not valid ({checks})")
    sc = {k: float(np.asarray(v)[0]) for k, v in pl.full(_bcast1(posture), stance[None]).items()}
    msg = (f"stand keyframe (feet snapped by <= {1e3 * d.min(1).max():.1f} mm; four-foot "
           f"margin {sc['dist']:.2f} x weight"
           + (f", effort {sc['effort']:.3f}" if "effort" in sc else "") + ")")
    return posture, stance, msg


def _tripod_start(fpl: FastPlanner, posture, stance, i, key):
    """The start posture moved (feet held) to a body pose where the tripod without leg ``i``
    is good and leg ``i`` can be raised: the least effort among ``n_lift`` samples in the
    B_lift region (:func:`..samplers.sample_body`).

    Returns:
        ``(posture, message)``.
    """
    cfg, kit = fpl.cfg, fpl.kit
    L = kit.robot.num_legs
    r = cfg.lift_radius
    h = (cfg.ref_height + cfg.height_band[0], cfg.ref_height + cfg.height_band[1])
    ok, posts = fpl.smp.body(key, cfg.n_lift, posture, stance, jnp.ones(L, bool),
                             region=region(x=(-r, r), y=(-r, r), height=h))
    sc = fpl.tripod[i](posts, jnp.broadcast_to(stance, (cfg.n_lift, L)))
    good = fpl.good(sc, ok)
    eff = fpl.effort(sc)
    for j in np.argsort(np.where(good, eff, np.inf))[: int(good.sum())]:
        p = jax.tree.map(lambda x: x[j], posts)
        ok_l, lp = kit.lift_leg(p, stance, i, cfg.lift_height)
        if bool(ok_l) and bool(fpl.checks(lp, stance, jnp.arange(L) != i)):
            shift = np.asarray(p.body.translation() - posture.body.translation())
            return p, (f"{int(good.sum())}/{cfg.n_lift} samples good for the tripod without "
                       f"leg {i}; shift {np.round(shift, 3).tolist()} m, effort {eff[j]:.3f}, "
                       f"push margin {float(np.asarray(sc['push_margin'])[j]):.1f} N")
    raise SystemExit(f"start_tripod: no body pose where the tripod without leg {i} is good")


def _walk_transfers(cfg: WalkCfg, ctx: RunContext, kit, cm, fh, fpl, posture, stance,
                    direction, key, sc_cfg, boxes) -> dict:
    """planner=transfer: from the start stance, move the body (feet held) to support the first
    tripod and lift the first leg; then ``n_steps`` transfers (land the leg in the air, move to
    where the next leg can lift, lift it). Prints a row per transfer, saves the replay."""
    order = GAITS[cfg.gait]
    L = cfg.mjmodel.num_legs
    tc = cfg.transfer
    p_start = np.asarray(posture.body.translation())
    keyframes, magnets, phases = [_qpos(cm, posture, stance, fh)], [np.ones(L, bool)], ["start"]
    start, msg = _tripod_start(fpl, posture, stance, order[0], key)
    ok, lifted = kit.lift_leg(start, stance, order[0], fpl.cfg.lift_height)
    print(f"  transfers: n = {tc.n_plant} B_plant (keep {tc.k_plant}) x {tc.n_foot} f' (keep "
          f"{tc.k_foot}) x {tc.n_lift} B_lift; node shift {tc.node_shift}, edge shift "
          f"{tc.edge_shift}; push {cfg.fast.push_min} x weight, twist weight x {cfg.fast.twist_min} m")
    print(f"  start: body moved for the first tripod: {msg}")
    up = np.arange(L) != order[0]
    keyframes += [_qpos(cm, start, stance, fh), _qpos(cm, lifted, stance, fh)]
    magnets += [np.ones(L, bool), up]
    phases += ["B_lift", "lifted"]
    posture = lifted
    side = np.cross(direction, [0.0, 0.0, 1.0] if cfg.scenario == "floor" else [-1.0, 0, 0])
    gap = (lambda p: p[2]) if cfg.scenario == "floor" else (lambda p: cfg.wall_x - p[0])
    print(f"  {'k':>3} | {'leg':>3} | {'next':>4} | {'progress':>8} | {'side':>6} | {'gap':>5} "
          f"| {'effort plant / lift':>19} | {'margin plant / lift (N)':>23} | {'to target':>9} "
          f"| {'good: plant / path / foot / lift / leaves':>41} | time plant / foot / lift (s)")
    stopped = ""
    for k in range(cfg.n_steps):
        i, nxt = order[k % len(order)], order[(k + 1) % len(order)]
        t0 = time.perf_counter()
        for attempt in range(1 + cfg.retries):
            key, sub = jax.random.split(key)
            r = plan_transfer(fpl, tc, stance, posture, i, nxt, direction, sub, origin=p_start)
            if r.ok:
                break
            print(f"  {k:>3} | {i:>3} | {nxt:>4} | attempt {attempt + 1} failed: {r.why}")
        dt = time.perf_counter() - t0
        if not r.ok:
            stopped = f"transfer {k} (leg {i} -> {nxt}): {r.why}"
            print(f"  {k:>3} | {i:>3} | {nxt:>4} | stopped: {r.why}")
            break
        keyframes += [_qpos(cm, r.plant, stance, fh), _qpos(cm, r.planted, r.stance, fh),
                      _qpos(cm, r.lift, r.stance, fh), _qpos(cm, r.lifted, r.stance, fh)]
        magnets += [np.arange(L) != i, np.ones(L, bool), np.ones(L, bool), np.arange(L) != nxt]
        phases += ["B_plant", "planted", "B_lift", "lifted"]
        posture, stance = r.lifted, r.stance
        s, c, tt = r.scores, r.scores["counts"], r.scores["t"]
        body = np.asarray(posture.body.translation())
        print(f"  {k:>3} | {i:>3} | {nxt:>4} | {s['progress']:8.3f} | {(body - p_start) @ side:+6.3f} "
              f"| {gap(body):5.3f} | {s['plant_effort']:9.3f} {s['lift_effort']:9.3f} "
              f"| {s['plant_margin']:11.1f} {s['lift_margin']:11.1f} | {s['to_target']:9.3f} "
              f"| {c['plant']:>9} {c['plant_path']:>5} {c['foot']:>6} {c['lift']:>6} {c['leaves']:>6}    "
              f"| {tt['plant']:.2f} / {tt['foot']:.2f} / {tt['lift']:.2f} (total {dt:.2f})")
    progress = float((np.asarray(posture.body.translation()) - p_start) @ direction)
    print(f"  total body progress {progress:.3f} m" + (f"; {stopped}" if stopped else ""))
    kf = np.stack(keyframes)
    f = cfg.frames_per_phase
    frames = [kf[0]] + [kf[a] + (kf[a + 1] - kf[a]) * t
                        for a in range(len(kf) - 1) for t in np.arange(1, f + 1) / f]
    xml = ctx.out / f"{cfg.scenario}_scene.xml"
    xml.write_text(scene_xml(sc_cfg, boxes))
    np.savez(ctx.out / "walk.npz", qpos=np.stack(frames), qvel=np.zeros((len(frames), cm.model.nv)),
             timestep=cfg.frame_s, model=_model_path(xml), key_qpos=kf,
             key_magnets=np.stack(magnets), key_phase=np.array(phases))
    print(f"  saved {ctx.out}/walk.npz ({len(frames)} frames)")
    return {"transfers": (len(keyframes) - 3) // 4, "progress": round(progress, 3),
            "stopped": stopped}


def _bcast1(posture):
    return jax.tree.map(lambda x: x[None], posture)


exp = Experiment("planner_walk")


@exp.run
def run(cfg: WalkCfg, ctx: RunContext) -> dict:
    """Plan the walk; print a row per step; save the replay.

    Returns:
        ``steps`` planned, ``progress`` (body, along the direction, m), ``stopped`` (why).
    """
    sc_cfg = ScoresCfg(mjmodel=cfg.mjmodel, wall_x=cfg.wall_x, wall_body_z=cfg.wall_body_z)
    boxes, nominal = scenario(cfg.scenario, sc_cfg)
    direction = np.array([1.0, 0, 0]) if cfg.scenario == "floor" else np.array([0, 0, 1.0])
    gcfg = climb_cfg(cfg.mjmodel, boxes=boxes,
                     num_footholds=40_000 if cfg.scenario == "floor" else 80_000)
    scene = terrain.make_scene(boxes)
    fh = terrain.sample_footholds(jax.random.PRNGKey(cfg.seed), scene, gcfg.num_footholds,
                                  edge_margin=gcfg.edge_margin)
    kit = Kit(make_robot(cfg.mjmodel), scene, gcfg, fh)
    cm = ClimbModel(cfg.mjmodel)
    cfg.step.ref_height = cfg.mjmodel.stand_height            # heights around the stand height
    pl = Planner(kit, cm, cfg.step)
    fast = cfg.planner == "fast"
    if cfg.planner in ("fast", "transfer"):
        cfg.fast.ref_height = cfg.transfer.ref_height = cfg.mjmodel.stand_height
        fpl = FastPlanner(kit, cm, cfg.fast)
    key = jax.random.PRNGKey(cfg.seed)

    # start: the stand keyframe, or the best of a few sampled stances at the nominal pose
    t0 = time.perf_counter()
    if cfg.start == "stand":
        posture, stance, msg = _stand_start(cfg, kit, pl, nominal)
    else:
        posture, stance, msg = _sampled_start(cfg, kit, pl, nominal, key)
    print(f"planner_walk: {cfg.scenario}, direction {direction.tolist()}, {cfg.n_steps} steps "
          f"({cfg.gait} {GAITS[cfg.gait]}); "
          f"start: {msg}; A = {cfg.step.adhesion:.0f} N, mu = {cfg.step.mu}; "
          + (f"planner fast{' (B_lift from B_plant, fallback ' + cfg.fast.lift_fallback + ')' if cfg.fast.lift_from_plant else ''}"
             f"{' (no B_plant shift)' if not cfg.fast.plant_shift else ''}: "
             f"push {cfg.fast.push_min} x weight, twist weight x "
             f"{cfg.fast.twist_min} m, n = {cfg.fast.n_lift} / "
             f"{cfg.fast.n_plant} / {cfg.fast.n_foot}" if fast else
             f"tripod margin >= {cfg.step.dist_min}; objective {cfg.step.objective}")
          + f" ({time.perf_counter() - t0:.0f} s incl. compiling)")

    if cfg.planner == "transfer":
        return _walk_transfers(cfg, ctx, kit, cm, fh, fpl, posture, stance, direction, key,
                               sc_cfg, boxes)
    if fast and cfg.start_tripod:
        posture, msg = _tripod_start(fpl, posture, stance, GAITS[cfg.gait][0], key)
        print(f"  start body moved for the first tripod: {msg}")

    def gap(p):
        """The body's distance to the surface it walks on (m)."""
        return p[2] if cfg.scenario == "floor" else cfg.wall_x - p[0]

    side = np.cross(direction, [0.0, 0.0, 1.0] if cfg.scenario == "floor" else [-1.0, 0, 0])

    def lateral(p):
        """The body's sideways offset from the walking line (m)."""
        return float((p - p_start) @ side)

    if fast:
        print(f"  {'step':>4} | {'leg':>3} | {'progress':>8} | {'side (m)':>8} | {'gap (m)':>7} "
              f"| {'effort lift / plant / S_next':>28} | {'push margin (N) lift / plant / S_next':>37} "
              f"| {'ankle':>5} | {'B_lift':>7} | time lift / plant / f' (s)")
    else:
        print(f"  {'step':>4} | {'leg':>3} | {'progress':>8} | {'side (m)':>8} | {'gap (m)':>7} | {'body (m)':>18} | {'T at B_lift':>11} "
              f"| {'T at B_plant':>12} | {'S_next s*':>9} | {'S_next dist':>11} | {'ankle':>5} "
              f"| {'effort lift / plant / S_next':>28} | time")

    L = cfg.mjmodel.num_legs
    keyframes = [_qpos(cm, posture, stance, fh)]
    magnets = [np.ones(L, bool)]                       # per keyframe: which magnets are on
    p_start = np.asarray(posture.body.translation())
    stopped = ""
    for k in range(cfg.n_steps):
        order = GAITS[cfg.gait]
        i = order[k % len(order)]
        key, sub = jax.random.split(key)
        t0 = time.perf_counter()
        r = (plan_step_fast(fpl, stance, posture, i, direction, sub, origin=p_start,
                            next_leg=order[(k + 1) % len(order)]) if fast
             else plan_step(pl, stance, posture, i, direction, sub, origin=p_start))
        dt = time.perf_counter() - t0
        if not r.ok:
            stopped = f"step {k} (leg {i}): {r.why}"
            print(f"  {k:>4} | {i:>3} | stopped: {r.why}")
            break
        keyframes += [_qpos(cm, r.lift, stance, fh), _qpos(cm, r.lifted, stance, fh),
                      _qpos(cm, r.plant, stance, fh), _qpos(cm, r.pressed, r.stance, fh),
                      _qpos(cm, r.posture, r.stance, fh)]
        swing_off = np.arange(L) != i
        magnets += [np.ones(L, bool), swing_off, swing_off, np.ones(L, bool), np.ones(L, bool)]
        posture, stance = r.posture, r.stance
        s = r.scores
        body = np.asarray(posture.body.translation())
        if fast:
            print(f"  {k:>4} | {i:>3} | {s['progress']:8.3f} | {lateral(body):+8.3f} | {gap(body):7.3f} "
                  f"| {s['lift_effort']:8.3f} {s['plant_effort']:8.3f} {s['effort_next']:9.3f} "
                  f"| {s['lift_margin']:11.1f} {s['plant_margin']:11.1f} {s['margin_next']:12.1f} "
                  f"| {s['ankle_deg']:4.0f}d | {'reused' if s['lift_reused'] else 'sampled':>7} "
                  f"| {s['t_lift']:.2f} / {s['t_plant']:.2f} / "
                  f"{s['t_foot']:.2f} (total {dt:.2f})")
        else:
            print(f"  {k:>4} | {i:>3} | {s['progress']:8.3f} | {lateral(body):+8.3f} | {gap(body):7.3f} | {np.array2string(body, precision=3):>18} "
                  f"| {s['lift_dist']:11.2f} | {s['plant_dist']:12.2f} | {s['s_next']:9.2f} "
                  f"| {s['dist_next']:11.2f} | {s['ankle_deg']:4.0f}d "
                  f"| {s['lift_effort']:8.3f} {s['plant_effort']:8.3f} {s['effort_next']:9.3f} "
                  f"| {dt:.1f} s")

    progress = float((np.asarray(posture.body.translation()) - p_start) @ direction)
    print(f"  total body progress {progress:.3f} m" + (f"; {stopped}" if stopped else ""))

    # replay: phases interpolated linearly in qpos (orientation constant in v0)
    kf = np.stack(keyframes)
    f = cfg.frames_per_phase
    frames = [kf[0]] + [kf[a] + (kf[a + 1] - kf[a]) * t
                        for a in range(len(kf) - 1) for t in np.arange(1, f + 1) / f]
    xml = ctx.out / f"{cfg.scenario}_scene.xml"
    xml.write_text(scene_xml(sc_cfg, boxes))
    np.savez(ctx.out / "walk.npz", qpos=np.stack(frames), qvel=np.zeros((len(frames), cm.model.nv)),
             timestep=cfg.frame_s, model=_model_path(xml),
             key_qpos=kf, key_magnets=np.stack(magnets),
             key_phase=np.array(["start"] + ["B_lift", "lifted", "B_plant", "pressed", "planted"]
                                * ((len(kf) - 1) // 5)))
    print(f"  saved {ctx.out}/walk.npz ({len(frames)} frames)")
    return {"steps": (len(keyframes) - 1) // 5, "progress": round(progress, 3), "stopped": stopped}


if __name__ == "__main__":
    exp.main(sys.argv[1:])
