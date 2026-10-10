"""Benchmark terrain collision checks: JAX (exact box-box for the body, point clouds for the
legs) vs. MuJoCo (MJX kinematics + collision), for scenes of 1 to 10 boxes.

Postures: full postures (body + four legs) sampled with the reach grid around the nominal
body (:func:`...kinematics.sampling.joint_sampler`), feet on the floor. Scenes: the floor
slab plus ``K - 1`` random boxes on the floor around the robot (so some postures collide).

Per scene and batch size, ms per call / us per posture after compilation (compile times
separately), and the share of postures that pass:

- ``body points``: :func:`..stance.body_terrain_ok_points` (the body box filled with points,
  ``collision_delta`` apart, vs. the terrain's SDF), the old check;
- ``body box-box``: :func:`..stance.body_terrain_ok` (exact separating-axis test, one
  per terrain box);
- ``legs points``: :func:`..stance.leg_terrain_ok` for all four legs (the planted feet's
  contact ignored within ``foot_clearance_radius``);
- ``mujoco``: the climb robot + the boxes as geoms, robot-vs-terrain contacts only (pad
  cells excluded: planted feet are meant to touch), ``mjx.kinematics`` + ``mjx.collision``;
  body clear iff no contact of the base geom has ``dist < 0``, likewise the leg capsules.

Agreement: body box-box vs. MuJoCo's base, legs points vs. MuJoCo's leg capsules.

    uv run --extra mjx runkit run lab.gait_graph.planner.experiments.collision_bench
"""
import dataclasses
import sys
import time

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from runkit import Experiment, RunContext

from controlkit.kinematics import collision
from controlkit.kinematics.types import Foothold, Posture
from controlkit.se3 import SE3

from ... import terrain
from ....kinematics.reach_sdf import Keepout, make_reach_grid
from ....kinematics.sampling import joint_sampler
from ..climb_cfg import climb_cfg
from ..climb_model.config import MjModelCfg
from ..climb_model.mjmodel import make_robot, robot_spec
from ..climb_statics import ClimbModel
from ..stance import body_terrain_ok, body_terrain_ok_points, leg_terrain_ok


@dataclasses.dataclass
class CollisionBenchCfg:
    """The robot, the scenes, the batch sizes."""
    mjmodel: MjModelCfg = dataclasses.field(default_factory=MjModelCfg)
    num_boxes: tuple = (1, 3, 10)       # terrain boxes per scene (the floor slab included)
    batches: tuple = (1024, 16384)
    chunk: int = 16384                  # lax.map batch size (bounds memory)
    repeats: int = 3
    h: float = 0.01                     # reach grid voxel size (m)
    num_footholds: int = 40_000
    region: tuple = (0.06, 0.06, 0.03, 5.0, 5.0, 10.0)
    seed: int = 0


def timed(f, *args, repeats=3):
    """``(compile_s, call_s, out)``: first call minus a call; mean of ``repeats`` calls."""
    t = time.perf_counter()
    out = jax.block_until_ready(f(*args))
    first = time.perf_counter() - t
    t = time.perf_counter()
    for _ in range(repeats):
        out = jax.block_until_ready(f(*args))
    call = (time.perf_counter() - t) / repeats
    return max(first - call, 0.0), call, out


def _batches(s):
    """A CLI override arrives as a string ("[1,1024]")."""
    if isinstance(s, str):
        return tuple(int(b) for b in s.strip("[]() ").split(",") if b.strip())
    return tuple(s)


def make_scene_boxes(key, k):
    """The floor slab (top face at z = 0) plus ``k - 1`` boxes resting on the floor within
    0.4 m of the origin: ``(centers, halves)``, (k, 3) each."""
    c = [np.array([0.0, 0.0, -0.05])]
    h = [np.array([2.0, 1.0, 0.05])]
    rng = np.random.default_rng(int(jax.random.randint(key, (), 0, 2**31 - 1)))
    for _ in range(k - 1):
        hh = np.array([rng.uniform(0.02, 0.08), rng.uniform(0.02, 0.08), rng.uniform(0.02, 0.10)])
        xy = rng.uniform(-0.4, 0.4, 2)
        c.append(np.array([xy[0], xy[1], hh[2]]))
        h.append(hh)
    return np.stack(c), np.stack(h)


def mujoco_checker(mj: MjModelCfg, cm: ClimbModel, centers, halves):
    """MJX model of the robot + the terrain boxes, robot-vs-terrain contacts only. Returns
    a function ``(body SE3, thetas (L, 3)) -> (body_ok, legs_ok)``."""
    spec = robot_spec(mj)
    for a in [a for a in spec.actuators if a.trntype == mujoco.mjtTrn.mjTRN_BODY]:
        spec.delete(a)                                     # MJX has no adhesion actuators
    for g in spec.geoms:
        if g.contype == 0 and g.conaffinity == 0:
            continue
        body = g.parent.name
        if body.startswith("pad"):                         # planted feet are meant to touch
            g.contype, g.conaffinity = 0, 0
        else:
            g.contype, g.conaffinity = 2, 1                # robot: collides with terrain only
    for k, (c, h) in enumerate(zip(centers, halves)):
        spec.worldbody.add_geom(name=f"terrain{k}", type=mujoco.mjtGeom.mjGEOM_BOX,
                                pos=c.tolist(), size=h.tolist(), contype=1, conaffinity=2)
    m = spec.compile()
    mx = mjx.put_model(m)
    base_geoms = [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name == "base"]
    leg_geoms = [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.startswith("leg")]
    qadr = np.asarray(cm.leg_qadr)
    d0 = mjx.collision(mx, mjx.kinematics(mx, mjx.make_data(mx)))
    pairs = np.asarray(d0.contact.geom)                    # (ncon, 2), static per model
    is_body = jnp.asarray(np.isin(pairs, base_geoms).any(1))
    is_leg = jnp.asarray(np.isin(pairs, leg_geoms).any(1))

    def check(body: SE3, thetas):
        w = body.wxyz_xyz
        q = jnp.zeros(m.nq).at[:7].set(jnp.concatenate([w[4:], w[:4]]))
        q = q.at[qadr.reshape(-1)].set(thetas.reshape(-1))
        d = mjx.collision(mx, mjx.kinematics(mx, mjx.make_data(mx).replace(qpos=q)))
        pen = d.contact.dist < 0.0
        return ~jnp.any(pen & is_body), ~jnp.any(pen & is_leg)

    return check, int(pairs.shape[0])


exp = Experiment("collision_bench")


@exp.run
def run(cfg: CollisionBenchCfg, ctx: RunContext) -> dict:
    """Time and compare the terrain checks; see the module docstring."""
    mj = cfg.mjmodel
    robot = make_robot(mj)
    cm = ClimbModel(mj)
    ccfg = climb_cfg(mj)
    L = mj.num_legs
    dev = jax.devices()[0]
    print(f"collision_bench: {dev.platform} / {dev.device_kind}, jax {jax.__version__}, mujoco {mujoco.__version__}")

    body_center, body_half = robot.mount_box(half_height=mj.body_half_height)
    grid = make_reach_grid(robot.leg, cfg.h, keepout=Keepout(robot.mounts[0].inverse(), body_center, body_half))
    floor = terrain.make_scene((((0.0, 0.0, -0.05), (2.0, 1.0, 0.05)),))
    fh = terrain.sample_footholds(jax.random.PRNGKey(cfg.seed), floor, cfg.num_footholds, edge_margin=0.02)
    targets = fh.position + (mj.pivot_height + mj.pad_thickness) * fh.normal
    body0 = SE3(jnp.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, mj.stand_height]))
    hi = jnp.asarray(cfg.region, float)

    batches = _batches(cfg.batches)
    nmax = max(batches)
    valid, bodies, foot_idx, thetas, _, _ = joint_sampler(robot, grid.device(), targets, tuple(range(L)), nmax)(
        jax.random.PRNGKey(cfg.seed + 1), body0, -hi, hi)
    print(f"{nmax:,} sampled postures ({float(valid.mean()):.2f} valid); all checked")
    planted = jnp.ones(L, bool)

    rows, agree = [], {}
    for k in _batches(cfg.num_boxes):
        centers, halves = make_scene_boxes(jax.random.PRNGKey(cfg.seed + 10 + k), k)
        scene = collision.Scene(box_center=jnp.asarray(centers, jnp.float32),
                                box_half=jnp.asarray(halves, jnp.float32))
        mj_check, ncon = mujoco_checker(mj, cm, centers, halves)

        def legs_points(b, th, f):
            sh = robot.shoulders(b)
            sites = Foothold(fh.position[f], fh.normal[f])
            return jax.vmap(lambda i, site: leg_terrain_ok(robot, scene, ccfg, sh[i], th[i], site, True))(
                jnp.arange(L), sites).all()

        checks = {
            "body points": lambda b, th, f: body_terrain_ok_points(robot, scene, ccfg, b),
            "body box-box": lambda b, th, f: body_terrain_ok(robot, scene, ccfg, b),
            "legs points": legs_points,
            "mujoco": lambda b, th, f: mj_check(b, th),
        }
        outs = {}
        for name, one in checks.items():
            f = jax.jit(lambda b, th, ff, one=one: jax.lax.map(lambda x: one(*x), (b, th, ff),
                                                               batch_size=cfg.chunk))
            for n in batches:
                sl = lambda a: jax.tree.map(lambda x: x[:n], a)
                comp, call, out = timed(f, sl(bodies), sl(thetas), sl(foot_idx), repeats=cfg.repeats)
                outs[name, n] = out
                res = out if name != "mujoco" else (out[0] & out[1])
                rows.append(dict(boxes=k, check=name, n=n, ms=1e3 * call, us=1e6 * call / n,
                                 compile_s=comp, pass_share=float(np.mean(np.asarray(res)))))
        mb, ml = (np.asarray(a) for a in outs["mujoco", nmax])
        bb, lp, bp = (np.asarray(outs[nm, nmax]) for nm in ("body box-box", "legs points", "body points"))
        agree[k] = dict(ncon=ncon, body_bb_vs_mj=float(np.mean(bb == mb)), body_pts_vs_mj=float(np.mean(bp == mb)),
                        legs_vs_mj=float(np.mean(lp == ml)),
                        body_mj_hit_bb_ok=int((~mb & bb).sum()), body_bb_hit_mj_ok=int((mb & ~bb).sum()),
                        legs_mj_hit_pts_ok=int((~ml & lp).sum()), legs_pts_hit_mj_ok=int((ml & ~lp).sum()))
        print(f"  {k} boxes done ({ncon} MJX contact slots)")

    print(f"\n  us per posture (pass share) at n = {nmax:,}; ms per call in brackets")
    names = ["body points", "body box-box", "legs points", "mujoco"]
    print(f"  {'boxes':>5} | " + " | ".join(f"{nm:>24}" for nm in names))
    by = {(r["boxes"], r["check"], r["n"]): r for r in rows}
    for k in _batches(cfg.num_boxes):
        print(f"  {k:>5} | " + " | ".join(
            f"{by[k, nm, nmax]['us']:7.3f} ({by[k, nm, nmax]['pass_share']:.2f}) [{by[k, nm, nmax]['ms']:7.1f}]"
            for nm in names))
    print(f"\n  compile s (n = {nmax:,})")
    for k in _batches(cfg.num_boxes):
        print(f"  {k:>5} | " + " | ".join(f"{by[k, nm, nmax]['compile_s']:24.2f}" for nm in names))
    print(f"\n  agreement with MuJoCo (n = {nmax:,})")
    for k, a in agree.items():
        print(f"  {k:>2} boxes: body box-box {a['body_bb_vs_mj']:.4f} (MJ hit / box-box clear "
              f"{a['body_mj_hit_bb_ok']}, box-box hit / MJ clear {a['body_bb_hit_mj_ok']}), body points "
              f"{a['body_pts_vs_mj']:.4f}; legs {a['legs_vs_mj']:.4f} (MJ hit / points clear "
              f"{a['legs_mj_hit_pts_ok']}, points hit / MJ clear {a['legs_pts_hit_mj_ok']})")
    return dict(rows=rows, agreement=agree)


if __name__ == "__main__":
    exp.main(sys.argv[1:])
