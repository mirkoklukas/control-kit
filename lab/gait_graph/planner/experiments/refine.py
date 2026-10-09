"""Refine one sampled transfer with the NLP (:func:`..refine.refine_transfer`) and compare.

On the floor from the ``stand`` keyframe: the body is moved (feet held) to support the
first tripod, the first leg lifted (as ``walk planner=transfer``); one transfer is sampled
(:func:`..transfer.plan_transfer`, the warm start) and refined. Before and after, checked
independently with the planner's own tools: the kinematic checks (reach, limits, ankle,
collisions; the planting foot at its refined point) and the fast scores (cones + tau_max,
push check, effort, push margin) for T at B_plant and T' at B_lift.

    uv run --extra mjx runkit run lab.gait_graph.planner.experiments.refine mjmodel.ankle_range_deg=60 fast.adhesion=0
    uv run --extra mjx ctk play experiments/runs/planner_refine/latest/out/after.npz
    uv run --extra mjx python -m lab.gait_graph.planner.view_transfer experiments/runs/planner_refine/latest/out/before.npz experiments/runs/planner_refine/latest/out/after.npz

Prints the comparison; saves ``before.npz`` / ``after.npz`` (B, B_plant, planted, B_lift,
B_lift with the lifting leg up, B') for ``ctk play`` and :mod:`..view_transfer`.
"""
import dataclasses
import sys

import jax
import jax.numpy as jnp
import numpy as np

from controlkit.kinematics.types import Foothold
from controlkit.se3 import SE3
from runkit import Experiment, RunContext

from ... import terrain
from ..climb_cfg import climb_cfg
from ..climb_model.config import MjModelCfg
from ..climb_model.mjmodel import make_robot
from ..climb_statics import ClimbModel
from ..refine import RefineCfg, refine_sequence, refine_transfer
from ..refine_simple import reach_box, refine_sequence_simple
from ..climb_model.poses import leg_angles
from ..stance import Kit, all_ok, checks
from ..step import Planner, StepCfg
from ..step_fast import FastPlanner, FastStepCfg
from ..transfer import TransferCfg, plan_transfer
from .scores import ScoresCfg, _model_path, scenario, scene_xml
from .walk import CRAWL, WalkCfg, _stand_start, _tripod_start


@dataclasses.dataclass
class RefineExpCfg:
    """The robot, the checks, the sampler and the NLP."""
    mjmodel: MjModelCfg = dataclasses.field(default_factory=MjModelCfg)
    fast: FastStepCfg = dataclasses.field(default_factory=FastStepCfg)
    transfer: TransferCfg = dataclasses.field(default_factory=TransferCfg)
    refine: RefineCfg = dataclasses.field(default_factory=RefineCfg)
    model: str = "full"             # sequence: full (refine_sequence: joints, statics, torques)
                                    # or simple (refine_sequence_simple: one rigid body, reach
                                    # boxes, forces as variables)
    transfers: int = 1              # 1: one transfer (as below); > 1: a crawl sequence of
                                    # this many, refined jointly (refine_sequence)
    retries: int = 2                # sequence: a failed sampled transfer is planned again
    advance: float = 0.06           # the end body B' = B moved this far along +x (same
                                    # orientation and height), B_lift fixed to it; < 0: free
    frames_per_phase: int = 10
    frame_s: float = 0.05
    seed: int = 0


exp = Experiment("planner_refine")


@exp.run
def run(cfg: RefineExpCfg, ctx: RunContext) -> dict:
    """Sample one transfer, refine it, compare; see the module docstring.

    Returns:
        The before / after numbers of the comparison.
    """
    sc_cfg = ScoresCfg(mjmodel=cfg.mjmodel)
    boxes, nominal = scenario("floor", sc_cfg)
    gcfg = climb_cfg(cfg.mjmodel, boxes=boxes, num_footholds=40_000)
    scene = terrain.make_scene(boxes)
    fh = terrain.sample_footholds(jax.random.PRNGKey(cfg.seed), scene, gcfg.num_footholds,
                                  edge_margin=gcfg.edge_margin)
    kit = Kit(make_robot(cfg.mjmodel), scene, gcfg, fh)
    cm = ClimbModel(cfg.mjmodel)
    cfg.fast.ref_height = cfg.transfer.ref_height = cfg.mjmodel.stand_height
    fpl = FastPlanner(kit, cm, cfg.fast)
    L = cfg.mjmodel.num_legs
    i, nxt = CRAWL[0], CRAWL[1]
    key = jax.random.PRNGKey(cfg.seed)
    wcfg = WalkCfg(mjmodel=cfg.mjmodel, step=StepCfg(adhesion=cfg.fast.adhesion, mu=cfg.fast.mu))
    posture, stance, _ = _stand_start(wcfg, kit, Planner(kit, cm, wcfg.step), nominal)
    start, msg = _tripod_start(fpl, posture, stance, i, key)
    _, node = kit.lift_leg(start, stance, i, cfg.fast.lift_height)
    print(f"planner_refine: floor, A = {cfg.fast.adhesion:.0f} N, push {cfg.fast.push_min} x "
          f"weight, twist weight x {cfg.fast.twist_min} m; transfer leg {i} -> {nxt}")
    print(f"  start: {msg}")
    if cfg.transfers > 1:
        return _sequence(cfg, ctx, kit, cm, fh, fpl, scene, node, stance, key, sc_cfg, boxes)
    warm = plan_transfer(fpl, cfg.transfer, stance, node, i, nxt, np.array([1.0, 0, 0]), key)
    if not warm.ok:
        raise SystemExit(f"no sampled transfer: {warm.why}")
    goal = None
    if cfg.advance >= 0:
        w = np.asarray(node.body.wxyz_xyz)
        goal = SE3(jnp.asarray(np.concatenate([w[:4], w[4:] + np.array([cfg.advance, 0.0, 0.0])])))
        print(f"  B' = B + {cfg.advance:.3f} m along +x (B_lift fixed to it); the sampled "
              f"B_lift is {np.asarray(warm.lift.body.translation() - node.body.translation()).round(3).tolist()} m from B")
    ref = refine_transfer(fpl, cfg.refine, warm, i, nxt, goal=goal)
    print(f"  NLP (SLSQP): {ref['message']} after {ref['nit']} iterations, {ref['t']:.1f} s; "
          f"objective {ref['cost0']:.3f} -> {ref['cost']:.3f}; largest violation: equalities "
          f"{ref['eq_viol']:.1e}, inequalities {ref['ineq_viol']:.1e}")

    S = jnp.asarray(warm.stance)

    def evaluate(planted, lift, foot):
        """Independent checks of one transfer's two key postures."""
        pool = Foothold(fh.position.at[S[i]].set(jnp.asarray(foot)), fh.normal)
        kin = lambda p, mask: {k: bool(np.all(np.asarray(v)))
                               for k, v in checks(kit.robot, scene, kit.cfg, pool, p, S, mask).items()}
        one = lambda p: jax.tree.map(lambda x: x[None], p)
        sp = fpl.tripod[i](one(planted), S[None])
        sl = fpl.tripod[nxt](one(lift), S[None])
        val = lambda sc, k: float(np.asarray(sc[k])[0])
        return dict(kin_plant=kin(planted, jnp.ones(L, bool)), kin_lift=kin(lift, jnp.ones(L, bool)),
                    plant_good=bool(fpl.good(sp, np.ones(1, bool))[0]),
                    lift_good=bool(fpl.good(sl, np.ones(1, bool))[0]),
                    plant_effort=float(fpl.effort(sp)[0]), lift_effort=float(fpl.effort(sl)[0]),
                    plant_margin=val(sp, "push_margin"), lift_margin=val(sl, "push_margin"),
                    sum_tau2=val(sp, "sum_tau2") + val(sl, "sum_tau2"))

    before = evaluate(warm.planted, warm.lift, np.asarray(fh.position[S[i]]))
    after = evaluate(ref["planted"], ref["lift"], ref["foot"])
    print(f"  {'':>8} | {'kinematics ok (plant / lift)':>28} | {'good (plant / lift)':>19} "
          f"| {'effort (plant / lift)':>21} | {'push margin N (plant / lift)':>28} | sum tau^2")
    for name, e in (("before", before), ("after", after)):
        bad = [f"{w}:{k}" for w, d_ in (("plant", e["kin_plant"]), ("lift", e["kin_lift"]))
               for k, v in d_.items() if not v]
        kin = "yes / yes" if not bad else " ".join(bad)
        print(f"  {name:>8} | {kin:>28} | {str(e['plant_good']):>8} / {str(e['lift_good']):<8} "
              f"| {e['plant_effort']:9.4f} / {e['lift_effort']:<9.4f} "
              f"| {e['plant_margin']:12.2f} / {e['lift_margin']:<12.2f} | {e['sum_tau2']:.3f}")
    move = lambda a, b: float(np.linalg.norm(np.asarray(a.body.translation() - b.body.translation())))
    print(f"  moved: B_plant {1e3 * move(ref['planted'], warm.planted):.1f} mm, B_lift "
          f"{1e3 * move(ref['lift'], warm.lift):.1f} mm, planting foot "
          f"{1e3 * float(np.linalg.norm(ref['foot'] - np.asarray(fh.position[S[i]]))):.1f} mm")

    # replays
    xml = ctx.out / "floor_scene.xml"
    xml.write_text(scene_xml(sc_cfg, boxes))
    nrm = lambda st: fh.normal[st]
    for name, (plant, planted, lift) in (("before", (warm.plant, warm.planted, warm.lift)),
                                         ("after", (ref["plant"], ref["planted"], ref["lift"]))):
        _, lifted = kit.lift_leg(lift, S, nxt, cfg.fast.lift_height)
        kf = np.stack([np.asarray(cm.qpos(p, nrm(st))) for p, st in
                       ((node, stance), (plant, S), (planted, S), (lift, S), (lifted, S),
                        (lifted, S))])
        f = cfg.frames_per_phase
        frames = [kf[0]] + [kf[a] + (kf[a + 1] - kf[a]) * t
                            for a in range(len(kf) - 1) for t in np.arange(1, f + 1) / f]
        T_, Tn_ = np.arange(L) != i, np.arange(L) != nxt
        np.savez(ctx.out / f"{name}.npz", qpos=np.stack(frames),
                 qvel=np.zeros((len(frames), cm.model.nv)), timestep=cfg.frame_s,
                 model=_model_path(xml), key_qpos=kf,
                 key_phase=np.array(["B", "B_plant", "B_plant planted", "B_lift",
                                     "B_lift, lifting leg up", "B'"]),
                 key_support=np.stack([T_, T_, np.ones(L, bool), Tn_, Tn_, Tn_]),
                 pad_drop=cfg.mjmodel.pivot_height + cfg.mjmodel.pad_thickness)
    print(f"  saved before.npz / after.npz")
    strip = lambda e: {k: v for k, v in e.items() if not isinstance(v, dict)}
    return {"before": strip(before), "after": strip(after), "nit": ref["nit"],
            "success": ref["success"], "eq_viol": ref["eq_viol"], "ineq_viol": ref["ineq_viol"]}


def _sequence(cfg, ctx, kit, cm, fh, fpl, scene, node, stance, key, sc_cfg, boxes) -> dict:
    """``transfers`` crawl transfers sampled (:func:`..transfer.plan_transfer`), refined jointly
    (:func:`..refine.refine_sequence`); per transfer the independent checks before / after;
    ``before.npz`` / ``after.npz`` with every transfer's key postures for
    :mod:`..view_transfer`."""
    L = cfg.mjmodel.num_legs
    d = np.array([1.0, 0, 0])
    legs, warms, nodes, st = [], [], [node], jnp.asarray(stance)
    for k in range(cfg.transfers):
        i, nxt = CRAWL[k % 4], CRAWL[(k + 1) % 4]
        for _ in range(1 + cfg.retries):
            key, sub = jax.random.split(key)
            w = plan_transfer(fpl, cfg.transfer, st, nodes[-1], i, nxt, d, sub,
                              origin=np.asarray(node.body.translation()))
            if w.ok:
                break
        if not w.ok:
            raise SystemExit(f"sampled transfer {k} ({i} -> {nxt}) failed: {w.why}")
        legs.append((i, nxt))
        warms.append(w)
        nodes.append(w.lifted)
        st = jnp.asarray(w.stance)
    print(f"  sampled {len(warms)} transfers (crawl), progress "
          f"{float((np.asarray(warms[-1].lift.body.translation()) - np.asarray(node.body.translation())) @ d):.3f} m")
    if cfg.model == "simple":
        box = reach_box(kit, leg_angles(kit.robot, cfg.mjmodel, cfg.mjmodel.stand_height)[1][0])
        print(f"  simple model: reach box in the shoulder frame {np.round(box[0], 3).tolist()} .. "
              f"{np.round(box[1], 3).tolist()} m")
        ref = refine_sequence_simple(fpl, cfg.refine, stance, warms, legs, box)
        print(f"  joint angles by IK: valid (all checks) for {sum(ref['ik_ok'])}/{len(warms)} transfers")
    else:
        ref = refine_sequence(fpl, cfg.refine, stance, warms, legs)
    nvar = (33 if cfg.model == "simple" else 39) * len(warms)
    print(f"  NLP (SLSQP), {nvar} variables ({cfg.model}): {ref['message']} after {ref['nit']} "
          f"iterations, {ref['t']:.1f} s; objective {ref['cost0']:.3f} -> {ref['cost']:.3f}; "
          f"largest violation: equalities {ref['eq_viol']:.1e}, inequalities {ref['ineq_viol']:.1e}")
    print(f"  time: {'part':>26} | {'calls':>5} | {'total s':>7} | {'ms per call':>11}")
    for name, (n_, t_) in ref["prof"].items():
        print(f"        {name:>26} | {n_:>5} | {t_:7.1f} | {1e3 * t_ / max(n_, 1):11.1f}")

    # stance positions per transfer, refined
    pos = np.asarray(fh.position[jnp.asarray(stance)]).copy()
    S_pos_after = []
    for k, (i, _) in enumerate(legs):
        pos[i] = ref["feet"][k]
        S_pos_after.append(pos.copy())

    def evaluate(k, planted, lift, S_pos):
        S = jnp.asarray(warms[k].stance)
        pool = Foothold(fh.position.at[S].set(jnp.asarray(S_pos)), fh.normal)
        kin = lambda p_: all(bool(np.all(np.asarray(v))) for v in
                             checks(kit.robot, scene, kit.cfg, pool, p_, S, jnp.ones(L, bool)).values())
        one = lambda p_: jax.tree.map(lambda x: x[None], p_)
        i, nxt = legs[k]
        sp, sl = fpl.tripod[i](one(planted), S[None]), fpl.tripod[nxt](one(lift), S[None])
        v = lambda sc, key_: float(np.asarray(sc[key_])[0])
        return dict(kin=kin(planted) and kin(lift),
                    good=bool(fpl.good(sp, np.ones(1, bool))[0]) and bool(fpl.good(sl, np.ones(1, bool))[0]),
                    sum_tau2=v(sp, "sum_tau2") + v(sl, "sum_tau2"),
                    margin=min(v(sp, "push_margin"), v(sl, "push_margin")))

    print(f"  {'k':>2} | {'legs':>6} | {'kinematics before / after':>25} | {'good before / after':>19} "
          f"| {'sum tau^2 before / after':>24} | {'push margin N before / after':>28} | moved B_plant / B_lift / foot (mm)")
    rows = []
    for k in range(len(warms)):
        w = warms[k]
        b = evaluate(k, w.planted, w.lift, np.asarray(fh.position[jnp.asarray(w.stance)]))
        a = evaluate(k, ref["planted"][k], ref["lift"][k], S_pos_after[k])
        mv = lambda p_, q_: 1e3 * float(np.linalg.norm(np.asarray(p_.body.translation() - q_.body.translation())))
        df = 1e3 * float(np.linalg.norm(ref["feet"][k] - np.asarray(fh.position[jnp.asarray(w.stance)[legs[k][0]]])))
        rows.append(dict(k=k, before=b, after=a))
        print(f"  {k:>2} | {legs[k][0]} -> {legs[k][1]} | {str(b['kin']):>11} / {str(a['kin']):<11} "
              f"| {str(b['good']):>8} / {str(a['good']):<8} | {b['sum_tau2']:10.3f} / {a['sum_tau2']:<11.3f} "
              f"| {b['margin']:12.2f} / {a['margin']:<13.2f} | {mv(ref['planted'][k], w.planted):6.1f} / "
              f"{mv(ref['lift'][k], w.lift):6.1f} / {df:6.1f}")

    # replays: per transfer B, B_plant (leg up), planted, B_lift, B_lift lifting leg up
    xml = ctx.out / "floor_scene.xml"
    xml.write_text(scene_xml(sc_cfg, boxes))
    r, h = kit.cfg.foot_radius, cfg.fast.lift_height
    for name in ("before", "after"):
        kf, phases, support = [], [], []
        prev = node
        for k, (i, nxt) in enumerate(legs):
            w = warms[k]
            S = jnp.asarray(w.stance)
            n = fh.normal[S]
            if name == "before":
                plant, planted, lift, lifted = w.plant, w.planted, w.lift, w.lifted
            else:
                Sp = S_pos_after[k]
                planted, lift = ref["planted"][k], ref["lift"][k]
                _, plant = kit.place_leg(planted, i, jnp.asarray(Sp[i] + (r + h) * np.asarray(n[i])))
                _, lifted = kit.place_leg(lift, nxt, jnp.asarray(Sp[nxt] + (r + h) * np.asarray(n[nxt])))
            T_, Tn_ = np.arange(L) != i, np.arange(L) != nxt
            for p_, ph, sup in ((prev, f"{k}: B", T_), (plant, f"{k}: B_plant", T_),
                                (planted, f"{k}: B_plant planted", np.ones(L, bool)),
                                (lift, f"{k}: B_lift", Tn_), (lifted, f"{k}: B_lift, leg {nxt} up", Tn_)):
                kf.append(np.asarray(cm.qpos(p_, n)))
                phases.append(ph)
                support.append(sup)
            prev = lifted
        kf = np.stack(kf)
        np.savez(ctx.out / f"{name}.npz", qpos=kf, qvel=np.zeros((len(kf), cm.model.nv)),
                 timestep=cfg.frame_s, model=_model_path(xml), key_qpos=kf,
                 key_phase=np.array(phases), key_support=np.stack(support),
                 pad_drop=cfg.mjmodel.pivot_height + cfg.mjmodel.pad_thickness)
    print(f"  saved before.npz / after.npz ({len(legs)} transfers x 5 key postures)")
    tot = lambda key_: sum(r_[key_]["sum_tau2"] for r_ in rows)
    return {"transfers": len(legs), "nit": ref["nit"], "success": ref["success"],
            "eq_viol": ref["eq_viol"], "ineq_viol": ref["ineq_viol"],
            "sum_tau2_before": tot("before"), "sum_tau2_after": tot("after"),
            "all_good_after": all(r_["after"]["good"] and r_["after"]["kin"] for r_ in rows)}


if __name__ == "__main__":
    exp.main(sys.argv[1:])
