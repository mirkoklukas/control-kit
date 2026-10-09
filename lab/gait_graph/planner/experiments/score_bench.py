"""Benchmark posture sampling (reach grid) and stance scoring per batch size, to compare a
laptop CPU with a GPU.

Climb robot (4 legs, the config's joint limits, the body box as keep-out), a flat floor
with ``num_footholds`` footholds, nominal body at stand height. A 1 cm reach grid
(``lab/kinematics/reach_sdf.py``), sampling from ``lab/kinematics/sampling.py``. Stages, per
batch size n:

- ``leg, fixed body``: n footholds + angles for leg ``leg`` at the nominal body
  (:func:`fixed_body_sampler`: one lookup per foothold, then draws).
- ``body + leg``: n body poses in the region, each with a foothold + angles for leg ``leg``
  (:func:`joint_sampler`, ``tries`` per leg).
- ``body + posture``: the same for all four legs.
- ``push score``: :func:`..statics.push_score` on the n postures of ``body + posture``
  (MuJoCo statics, least torque, all limits).
- ``push min-norm``: :func:`..statics.push_score_min_norm` (kinematic centre of mass,
  minimum-norm forces, adhesion and friction).
- ``hold min-norm``: :func:`..statics.hold_min_norm` (the same forces, no pushes: the
  equilibrium check).

Scores run in chunks of ``score_chunk`` (``jax.lax.map``) to bound memory. Times after
compilation (mean of ``repeats`` calls); compile times separately.

    uv run --extra mjx runkit run lab.gait_graph.planner.experiments.score_bench

Prints the device, the per-sample times and the compile times; returns the rows.
"""
import dataclasses
import sys
import time

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from runkit import Experiment, RunContext

from controlkit.kinematics.types import Posture
from controlkit.se3 import SE3

from ... import terrain
from ....kinematics.reach_sdf import Keepout, branches_fn, make_reach_grid
from ....kinematics.sampling import fixed_body_sampler, joint_sampler
from ..climb_model.config import MjModelCfg
from ..climb_model.mjmodel import make_robot
from ..climb_statics import ClimbModel
from ..scoring import G
from ..statics import hold_min_norm, kinematic_com, push_score, push_score_min_norm


@dataclasses.dataclass
class ScoreBenchCfg:
    """The robot, the scene, the batch sizes."""
    mjmodel: MjModelCfg = dataclasses.field(default_factory=MjModelCfg)
    h: float = 0.01                 # reach grid voxel size (m)
    num_footholds: int = 40_000     # drawn on the floor slab (about half survive)
    leg: int = 0                    # the single leg
    tries: int = 8                  # footholds tried per leg and sample (joint sampling)
    batches: tuple = (1, 1024, 16384, 262144)
    score_chunk: int = 16384        # scoring batch per lax.map step (bounds memory)
    repeats: int = 3                # timed calls per measurement (after the compile)
    adhesion: float = 40.0
    mu: float = 0.5
    region: tuple = (0.06, 0.06, 0.03, 5.0, 5.0, 10.0)   # +- x, y, z (m), roll, pitch, yaw (deg)
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


exp = Experiment("score_bench")


@exp.run
def run(cfg: ScoreBenchCfg, ctx: RunContext) -> dict:
    """Time sampling and scoring per batch size; see the module docstring.

    Returns:
        ``rows``: one dict per (stage, n) with ``ms`` per call, ``us`` per sample,
        ``compile_s`` and, for the samplers, ``valid``; ``device``: platform and versions.
    """
    mj = cfg.mjmodel
    batches = cfg.batches                  # a CLI override arrives as a string: "[1,1024]"
    if isinstance(batches, str):
        batches = tuple(int(b) for b in batches.strip("[]() ").split(",") if b.strip())
    robot = make_robot(mj)
    cm = ClimbModel(mj)
    mm = cm.mass_model()
    L = mj.num_legs
    dev = jax.devices()[0]
    device = dict(platform=dev.platform, kind=dev.device_kind, jax=jax.__version__,
                  mujoco=mujoco.__version__)
    print(f"score_bench: {device}")

    # --- reach grid, floor, footholds
    body_center, body_half = robot.mount_box(half_height=mj.body_half_height)
    keepout = Keepout(robot.mounts[0].inverse(), body_center, body_half)
    t0 = time.perf_counter()
    grid = make_reach_grid(robot.leg, cfg.h, keepout=keepout)
    print(f"reach grid {grid.shape} at {grid.h * 1e3:.0f} mm: {time.perf_counter() - t0:.2f} s")
    g = grid.device()
    branches = branches_fn(robot.leg, keepout=keepout)
    scene = terrain.make_scene((((0.0, 0.0, -0.05), (2.0, 1.0, 0.05)),))
    fh = terrain.sample_footholds(jax.random.PRNGKey(cfg.seed), scene, cfg.num_footholds,
                                  edge_margin=0.02)
    foot_r = mj.pivot_height + mj.pad_thickness
    targets = fh.position + foot_r * fh.normal
    body0 = SE3(jnp.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, mj.stand_height]))
    hi = jnp.asarray(cfg.region, float)
    lo = -hi
    print(f"{targets.shape[0]:,} footholds")

    # --- scores on (bodies, thetas, footholds) of full postures, chunked
    planted = tuple(range(L))
    tau_max = float(mj.forcerange[1])

    def one_push(b, th, f):
        n = fh.normal[f]
        st = cm.statics(cm.qpos(Posture(b, th), n), G)
        return push_score(st, planted, n, cfg.adhesion, cfg.mu, tau_max)[0]

    def feet_com(b, th):
        p = Posture(b, th)
        sh = robot.shoulders(b)
        feet = jax.vmap(lambda i: sh[i].apply(robot.leg.forward(th[i]).translation()[-1]))(
            jnp.arange(L))
        return feet, kinematic_com(robot, p, mm)

    def one_mn_push(b, th, f):
        feet, com = feet_com(b, th)
        return push_score_min_norm(com, feet, fh.normal[f], cm.mass, G, cfg.adhesion, cfg.mu)[0]

    def one_mn_hold(b, th, f):
        feet, com = feet_com(b, th)
        return hold_min_norm(com, feet, fh.normal[f], cm.mass, G, cfg.adhesion, cfg.mu)[0]

    def chunked(one):
        return jax.jit(lambda b, th, f: jax.lax.map(lambda x: one(*x), (b, th, f),
                                                    batch_size=cfg.score_chunk))

    scorers = {"push score": chunked(one_push), "push min-norm": chunked(one_mn_push),
               "hold min-norm": chunked(one_mn_hold)}

    # --- per batch size
    rows = []
    key = jax.random.PRNGKey(cfg.seed + 1)
    for n in batches:
        stages = {
            "leg, fixed body": (fixed_body_sampler(robot, g, targets, cfg.leg, n), (key, body0)),
            "body + leg": (joint_sampler(robot, g, targets, (cfg.leg,), n, tries=cfg.tries),
                           (key, body0, lo, hi)),
            "body + posture": (joint_sampler(robot, g, targets, planted, n, tries=cfg.tries),
                               (key, body0, lo, hi)),
        }
        for name, (f, args) in stages.items():
            comp, call, out = timed(f, *args, repeats=cfg.repeats)
            rows.append(dict(stage=name, n=n, ms=1e3 * call, us=1e6 * call / n, compile_s=comp,
                             valid=float(np.mean(np.asarray(out[0])))))
        valid, bodies, foot_idx, thetas, _, _ = out                     # body + posture
        for name, f in scorers.items():
            comp, call, res = timed(f, bodies, thetas, foot_idx, repeats=cfg.repeats)
            rows.append(dict(stage=name, n=n, ms=1e3 * call, us=1e6 * call / n, compile_s=comp,
                             valid=float(np.mean(np.asarray(res) > 0))))
        print(f"  n = {n:>7,} done")

    names = list(dict.fromkeys(r["stage"] for r in rows))
    by = {(r["stage"], r["n"]): r for r in rows}
    print(f"\n  us per sample (ms per call) on {device['platform']} / {device['kind']}")
    print(f"  {'stage':>15} | " + " | ".join(f"{'n = ' + format(n, ','):>20}" for n in batches))
    for nm in names:
        print(f"  {nm:>15} | " + " | ".join(
            f"{by[nm, n]['us']:9.4f} ({by[nm, n]['ms']:7.2f})" for n in batches))
    print(f"\n  compile s")
    for nm in names:
        print(f"  {nm:>15} | " + " | ".join(f"{by[nm, n]['compile_s']:20.2f}" for n in batches))
    print(f"\n  valid share (samplers) / score > 0 or holds (scores), largest n")
    for nm in names:
        print(f"  {nm:>15} | {by[nm, batches[-1]]['valid']:.2f}")
    return {"rows": rows, "device": device}


if __name__ == "__main__":
    exp.main(sys.argv[1:])
