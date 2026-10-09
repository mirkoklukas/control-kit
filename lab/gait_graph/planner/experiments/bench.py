"""Benchmark the step samplers (:mod:`..samplers`) and the scores, separately, per
candidate, over batch sizes.

On the floor from the ``stand`` keyframe, or on the wall (``scenario=wall``) from the best of
``start_tries`` sampled stances (as :mod:`.walk`), swing leg ``leg``. One chain of inputs is built
first (B_lift: the first valid candidate whose leg can be raised; B_plant: the first valid
candidate), then:

1. **Sampling**: each phase's sampler (B_lift, B_plant: :func:`..samplers.sample_body`;
   f': :func:`..samplers.sample_foothold`) at each batch size of ``NS``: ms per candidate
   (compiled) and the valid share.
2. **Scoring**: on batches of valid candidates of each size of ``KS`` (valid ones of the
   largest sampling batch, repeated as needed), ms per candidate:

   - ``statics``: one ``mjx.forward`` + Jacobians (inside every score);
   - ``torque fast``: :func:`..scoring.make_torque_scorer` ``method="fast"``: closed-form
     least torque, limits check, cone margin; and the share not exact (would need the QP);
   - ``torque QP``: the same with ``method="qp"``;
   - ``dist check``: :func:`..statics.disturbance_check` at ``dist_min`` (12 capped LPs;
     batch sizes up to ``dist_max_batch``);
   - ``push fast``: :func:`..statics.push_check_fast` at the same push (no LP);
   - ``push score``: :func:`..statics.push_score`, the ball radius (``docs/push-score.md``);
     the share with ``r* >= `` the push.

3. **Push fast vs. the LP check**: on the unique valid candidates (up to
   ``dist_max_batch``): the pass rates, and how often the fast check fails where the LP
   passes (needs tuned internal forces) or passes where the LP fails (should not happen:
   the fast check is sufficient).

   And the push score against both: ``r* >= push`` must imply that push fast passes (its 12
   directions lie in the ball), and ``r*`` must not exceed the min over the 12 directions
   of :func:`..statics.push_lambda`.

   Tripod scores (leg ``leg`` lifted) for B_lift and B_plant, full stance for f'.

All times after compilation (the mean of ``repeats`` calls).

    uv run --extra mjx runkit run lab.gait_graph.planner.experiments.bench
    uv run --extra mjx runkit run lab.gait_graph.planner.experiments.bench mjmodel.ankle_range_deg=60
    uv run --extra mjx runkit run lab.gait_graph.planner.experiments.bench scenario=wall

Prints two tables; returns their rows.
"""
import dataclasses
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from runkit import Experiment, RunContext

from ... import terrain
from ..climb_cfg import climb_cfg
from ..climb_model.config import MjModelCfg
from ..climb_model.mjmodel import make_robot
from ..climb_statics import ClimbModel
from ..samplers import Samplers, region
from ..scoring import G, make_torque_scorer
from ..stance import Kit
from ..statics import (disturbance_check, disturbance_directions, push_check_fast,
                       push_lambda, push_score, push_slacks)
from ..step import Planner, StepCfg
from .scores import ScoresCfg, scenario
from .walk import WalkCfg, _sampled_start, _stand_start

NS = (64, 256, 1024, 4096)          # sampling batch sizes
KS = (1, 64, 256, 1024, 4096, 16384)   # scoring batch sizes


@dataclasses.dataclass
class BenchCfg:
    """The robot, the regions and the checks."""
    mjmodel: MjModelCfg = dataclasses.field(default_factory=MjModelCfg)
    scenario: str = "floor"         # floor (start: the stand keyframe) or wall (a sampled start)
    start_tries: int = 16           # wall: start stances sampled, the most stable kept
    wall_x: float = 0.45            # wall face at this x (as the scores experiment)
    wall_body_z: float = 0.6        # body height on the wall at the start
    leg: int = 1                    # the swing leg
    lift_radius: float = 0.08       # B_lift region: x, y in +/- this (m)
    plant_reach: float = 0.12       # B_plant region: x, y in +/- this (m)
    height_band: float = 0.04       # heights: stand height +/- this (m)
    dist_min: float = 0.3           # disturbance check: push to resist (/ weight)
    dist_max_batch: int = 1024      # the dist check only up to this batch size (slow)
    repeats: int = 3                # timed calls per measurement (after the compile)
    adhesion: float = 40.0
    mu: float = 0.5
    seed: int = 0


def timed(f, *args, repeats=3):
    """``(compile_s, call_s, out)``: the first call's time minus a call, the mean of
    ``repeats`` more calls."""
    t = time.perf_counter()
    out = jax.block_until_ready(f(*args))
    first = time.perf_counter() - t
    t = time.perf_counter()
    for _ in range(repeats):
        out = jax.block_until_ready(f(*args))
    call = (time.perf_counter() - t) / repeats
    return max(first - call, 0.0), call, out


def score_fns(cm: ClimbModel, fh, planted_idx, cfg: BenchCfg) -> dict:
    """The scores, each jitted and vmapped over (postures, stances)."""
    idx = np.asarray(planted_idx)
    tau_max = float(cm.mj.forcerange[1])
    lam = cfg.dist_min * cm.mass * 9.81

    def st(p, s):
        n = fh.normal[s]
        return cm.statics(cm.qpos(p, n), G), n[idx]

    def statics_only(p, s):
        a, _ = st(p, s)
        return a.bias_joint

    def dist(p, s):
        a, n = st(p, s)
        return disturbance_check(a, planted_idx, n, cfg.adhesion, cfg.mu, tau_max, lam)[0]

    def push(p, s):
        a, n = st(p, s)
        return push_check_fast(a, planted_idx, n, cfg.adhesion, cfg.mu, tau_max, lam)[0]

    def score(p, s):
        a, n = st(p, s)
        return push_score(a, planted_idx, n, cfg.adhesion, cfg.mu, tau_max)[0]

    def lam12(p, s):
        a, n = st(p, s)
        sl, h = push_slacks(a, planted_idx, n, cfg.adhesion, cfg.mu, tau_max)
        return push_lambda(sl, h, disturbance_directions(1.0)).min()   # unit directions in w

    return {
        "statics": jax.jit(jax.vmap(statics_only)),
        "torque fast": make_torque_scorer(cm, fh, planted_idx, cfg.adhesion, cfg.mu,
                                          method="fast"),
        "torque QP": make_torque_scorer(cm, fh, planted_idx, cfg.adhesion, cfg.mu,
                                        method="qp"),
        "dist check": jax.jit(jax.vmap(dist)),
        "push fast": jax.jit(jax.vmap(push)),
        "push score": jax.jit(jax.vmap(score)),
        "_lam12": jax.jit(jax.vmap(lam12)),             # not timed: the sanity check
    }


def _take(tree, idx):
    return jax.tree.map(lambda x: x[jnp.asarray(idx)], tree)


exp = Experiment("planner_bench")


@exp.run
def run(cfg: BenchCfg, ctx: RunContext) -> dict:
    """Time the samplers and the scores; see the module docstring.

    Returns:
        ``sampling``: one dict per (phase, n); ``scoring``: one dict per (phase, k).
    """
    sc_cfg = ScoresCfg(mjmodel=cfg.mjmodel, wall_x=cfg.wall_x, wall_body_z=cfg.wall_body_z)
    boxes, nominal = scenario(cfg.scenario, sc_cfg)
    gcfg = climb_cfg(cfg.mjmodel, boxes=boxes,
                     num_footholds=40_000 if cfg.scenario == "floor" else 80_000)
    scene = terrain.make_scene(boxes)
    fh = terrain.sample_footholds(jax.random.PRNGKey(cfg.seed), scene, gcfg.num_footholds,
                                  edge_margin=gcfg.edge_margin)
    kit = Kit(make_robot(cfg.mjmodel), scene, gcfg, fh)
    cm = ClimbModel(cfg.mjmodel)
    L = cfg.mjmodel.num_legs
    i = cfg.leg
    smp = Samplers(kit)
    wcfg = WalkCfg(mjmodel=cfg.mjmodel, step=StepCfg(adhesion=cfg.adhesion, mu=cfg.mu),
                   scenario=cfg.scenario, start_tries=cfg.start_tries, wall_x=cfg.wall_x,
                   wall_body_z=cfg.wall_body_z)
    key = jax.random.PRNGKey(cfg.seed)
    pl = Planner(kit, cm, wcfg.step)
    if cfg.scenario == "floor":
        posture, stance, msg = _stand_start(wcfg, kit, pl, nominal)
    else:
        posture, stance, msg = _sampled_start(wcfg, kit, pl, nominal, key)
    all_down, tripod = jnp.ones(L, bool), jnp.arange(L) != i
    h0, hb = cfg.mjmodel.stand_height, cfg.height_band
    r_lift = region(x=(-cfg.lift_radius, cfg.lift_radius), y=(-cfg.lift_radius, cfg.lift_radius),
                    height=(h0 - hb, h0 + hb))
    r_plant = region(x=(-cfg.plant_reach, cfg.plant_reach), y=(-cfg.plant_reach, cfg.plant_reach),
                     height=(h0 - hb, h0 + hb))
    print(f"planner_bench: {cfg.scenario}, start {msg}; swing leg {i}; {jax.devices()[0].platform}, "
          f"{jax.devices()[0].device_kind}")

    # --- the chain of inputs
    ok, posts = smp.body(key, 64, posture, stance, all_down, region=r_lift)
    lifted = None
    for j in np.nonzero(np.asarray(ok))[0]:
        ok_l, lp = kit.lift_leg(_take(posts, j), stance, i, 0.03)
        if bool(ok_l):
            lifted = lp
            break
    if lifted is None:
        raise SystemExit("no B_lift candidate whose leg can be raised")
    ok, posts = smp.body(key, 64, lifted, stance, tripod, region=r_plant)
    if not np.asarray(ok).any():
        raise SystemExit("no valid B_plant candidate")
    plant = _take(posts, int(np.argmax(np.asarray(ok))))

    phases = {
        "B_lift": (lambda k, n: smp.body(k, n, posture, stance, all_down, region=r_lift), "tripod"),
        "B_plant": (lambda k, n: smp.body(k, n, lifted, stance, tripod, region=r_plant), "tripod"),
        "f'": (lambda k, n: smp.foothold(k, n, plant, stance, i), "full"),
    }

    # --- 1. sampling
    print(f"\n  sampling: ms per candidate (valid share)")
    print(f"  {'phase':>7} | " + " | ".join(f"{'n = ' + str(n):>16}" for n in NS))
    sampling, pools = [], {}
    for name, (sample, _) in phases.items():
        cells = []
        for n in NS:
            _, call, out = timed(sample, key, n, repeats=cfg.repeats)
            v = float(np.mean(np.asarray(out[0])))
            sampling.append(dict(phase=name, n=n, ms=1e3 * call / n, valid=v))
            cells.append(f"{1e3 * call / n:7.4f} ({v:4.2f})")
        pools[name] = out                                  # the largest batch
        print(f"  {name:>7} | " + " | ".join(f"{c:>16}" for c in cells))

    # --- 2. scoring
    fns = {"tripod": score_fns(cm, fh, tuple(j for j in range(L) if j != i), cfg),
           "full": score_fns(cm, fh, tuple(range(L)), cfg)}
    names = [nm for nm in fns["tripod"] if not nm.startswith("_")]
    lam = cfg.dist_min * cm.mass * 9.81
    print(f"\n  scoring: ms per candidate (torque fast: share not exact, i.e. needing the QP; "
          f"push fast: share passing); "
          f"dist check at {cfg.dist_min} x weight, batches <= {cfg.dist_max_batch}")
    print(f"  {'phase':>7} | {'k':>5} | " + " | ".join(f"{nm:>18}" for nm in names))
    agree = {}
    scoring = []
    for name, (_, which) in phases.items():
        out = pools[name]
        keep = np.nonzero(np.asarray(out[0]))[0]
        for k in KS:
            idx = np.resize(keep, k)
            P = _take(out[1], idx)
            S = (jnp.asarray(out[2])[jnp.asarray(idx)] if len(out) > 2
                 else jnp.broadcast_to(stance, (k, L)))
            row, cells = dict(phase=name, k=k), []
            for nm in names:
                f = fns[which][nm]
                if nm == "dist check" and k > cfg.dist_max_batch:
                    cells.append(f"{'-':>18}")
                    continue
                _, call, res = timed(f, P, S, repeats=cfg.repeats)
                row[nm] = 1e3 * call / k
                if nm == "torque fast":
                    row["not_exact"] = float(np.mean(~np.asarray(res["exact"])))
                    cells.append(f"{row[nm]:8.4f} ({row['not_exact']:4.2f})")
                elif nm == "push fast":
                    row["push_pass"] = float(np.mean(np.asarray(res)))
                    cells.append(f"{row[nm]:8.4f} ({row['push_pass']:4.2f})")
                elif nm == "push score":
                    row["score_pass"] = float(np.mean(np.asarray(res) >= lam))
                    cells.append(f"{row[nm]:8.4f} ({row['score_pass']:4.2f})")
                else:
                    cells.append(f"{row[nm]:18.4f}")
            scoring.append(row)
            print(f"  {name:>7} | {k:>5} | " + " | ".join(f"{c:>18}" for c in cells))
        # agreement, on the unique valid candidates
        u = np.unique(keep)[: cfg.dist_max_batch]
        P = _take(out[1], u)
        S = (jnp.asarray(out[2])[jnp.asarray(u)] if len(out) > 2
             else jnp.broadcast_to(stance, (len(u), L)))
        lp = np.asarray(fns[which]["dist check"](P, S))
        fast = np.asarray(fns[which]["push fast"](P, S))
        r = np.asarray(fns[which]["push score"](P, S))
        l12 = np.asarray(fns[which]["_lam12"](P, S))
        agree[name] = dict(n=len(u), lp_pass=float(lp.mean()), fast_pass=float(fast.mean()),
                           fast_fail_lp_pass=int((~fast & lp).sum()),
                           fast_pass_lp_fail=int((fast & ~lp).sum()),
                           score_pass=float((r >= lam).mean()),
                           score_pass_fast_fail=int(((r >= lam) & ~fast).sum()),
                           score_above_lam12=int((r > l12 + 1e-6).sum()),
                           r_median=float(np.median(r)), lam12_median=float(np.median(l12)))
    print(f"\n  push fast vs. the LP check (both at {cfg.dist_min} x weight), unique valid candidates")
    print(f"  {'phase':>7} | {'n':>5} | {'LP pass':>7} | {'fast pass':>9} | {'fast fail, LP pass':>18} "
          f"| {'fast pass, LP fail':>18}")
    for name, a in agree.items():
        print(f"  {name:>7} | {a['n']:>5} | {a['lp_pass']:7.2f} | {a['fast_pass']:9.2f} "
              f"| {a['fast_fail_lp_pass']:>18} | {a['fast_pass_lp_fail']:>18}")
    print(f"\n  push score (r*, N) vs. push fast and the 12 directions, unique valid candidates")
    print(f"  {'phase':>7} | {'n':>5} | {'r* >= push':>10} | {'r* >= push, fast fail':>21} "
          f"| {'r* > min lam12':>14} | {'median r*':>9} | {'median lam12':>12}")
    for name, a in agree.items():
        print(f"  {name:>7} | {a['n']:>5} | {a['score_pass']:10.2f} | {a['score_pass_fast_fail']:>21} "
              f"| {a['score_above_lam12']:>14} | {a['r_median']:9.2f} | {a['lam12_median']:12.2f}")
    return {"sampling": sampling, "scoring": scoring, "agreement": agree}


if __name__ == "__main__":
    exp.main(sys.argv[1:])
