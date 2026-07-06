"""Benchmark: parallel MJX throughput for welded-stance settling (runs on A100).

Self-contained runkit experiment. It

  1. samples a batch of valid hexapod postures on the climb terrain
     (body pose + joint angles + planted feet), as in notebooks/07_sample_postures;
  2. computes each posture's static joint torques + foot reaction forces
     (``mjx_stance``, no stepping) and saves them with the postures;
  3. sweeps batch size x settle steps, timing the parallel weld-and-settle
     (``mjx_weld``) to measure env-steps/s -- the parallel-throughput question.

Everything the experiment needs lives here; the physics primitives come from the
sibling ``mjx_*`` modules. Outputs land in the runkit run dir (``ctx.out``).

Run (on the A100):
    uv run --extra mjx python -m lab.stance_graph.benchmark \
        batch_sizes='[256,1024,4096,16384]' step_counts='[50,200,800]' --tag=a100
"""
import time
from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp

import mujoco
from jaxlie import SE3

from runkit import experiment, RunContext, main

from lab.stance_graph.kinematics import from_te, infer_posture, SHOULDERS
from lab.stance_graph.sample import body_sampler
from lab.stance_graph.mjx_sample import _leg_reach_mask
from lab.stance_graph.mjx_stance import MODEL, load_model, to_qpos, stance_forces_batch
from lab.stance_graph import mjx_terrain as terrain
from lab.stance_graph.mjx_weld import weld_in_place


# --------------------------------------------------------------------------- #
# Config                                                                      #
# --------------------------------------------------------------------------- #
@dataclass
class Cfg:
    seed: int = 0

    # posture sampling (see notebooks/07_sample_postures.ipynb)
    pool_size: int = 100_000                       # terrain surface points
    n_bodies: int = 16                             # body poses sampled around the base
    n_stances: int = 4_000                         # foot layouts per body
    body_xyz: tuple = (0.8, 0.0, 0.35)             # base body position (a climb pose)
    body_rpy_deg: tuple = (0.0, -40.0, 0.0)        # base body orientation
    stance: tuple = (0, 2, 3, 5)                   # planted leg ids
    xyz_delta: float = 0.02                        # body-position jitter (m)
    rpy_delta_deg: float = 3.0                     # body-orientation jitter (deg)

    # throughput sweep
    batch_sizes: tuple = (256, 1024, 4096)
    step_counts: tuple = (50, 200)
    repeats: int = 3                               # timed reps (min is reported)


# --------------------------------------------------------------------------- #
# Posture sampling                                                            #
# --------------------------------------------------------------------------- #
def sample_postures(cfg: Cfg, key):
    """Sample valid hexapod postures on the climb terrain.

    Args:
        cfg: the experiment config.
        key: PRNG key.

    Returns:
        bodies: (K,) SE3 base poses.
        theta: (K, 6, 3) joint angles.
        feet: (K, 6, 3) foot world positions.
        stance_mask: (6,) bool -- planted legs (shared by all K).
        mesh: ``(V, F, FN)`` climb-surface mesh (for logging/replay).
    """
    V, F, FN = terrain.climb_surface()
    xs, _ = terrain.sample_mesh(V, F, M=cfg.pool_size)
    xs = jnp.asarray(xs)

    body = from_te(jnp.asarray(cfg.body_xyz), jnp.deg2rad(jnp.asarray(cfg.body_rpy_deg)))
    stance = jnp.asarray(cfg.stance)

    # per-leg reach mask over the pool, then sample S footholds per leg from it.
    shoulders = body @ SHOULDERS
    mask = jax.vmap(_leg_reach_mask, (0, None))(shoulders, xs)              # (6, M)

    k_body, k_feet = jax.random.split(key)
    bodies = body_sampler(k_body, body, N=cfg.n_bodies,
                          xyz_delta=cfg.xyz_delta,
                          rpy_delta=jnp.deg2rad(cfg.rpy_delta_deg))
    keys = jax.random.split(k_feet, 6)
    inds = jax.vmap(lambda kk, m: jax.random.choice(
        kk, xs.shape[0], (cfg.n_stances,), p=m / m.sum()))(keys, mask)      # (6, S)
    feet = xs[inds.T]                                                       # (S, 6, 3)

    valid, theta, feet_ = infer_posture(bodies, feet, stance)              # (N,S), (N,S,6,3)...
    bodies = bodies.reshape((cfg.n_bodies, 1)).broadcast_to((cfg.n_bodies, cfg.n_stances))
    bodies = bodies[valid]
    stance_mask = jnp.zeros(6, bool).at[stance].set(True)
    return bodies, theta[valid], feet_[valid], stance_mask, (V, F, FN)


# --------------------------------------------------------------------------- #
# Throughput sweep                                                            #
# --------------------------------------------------------------------------- #
def _ints(x):
    """Coerce a sweep field to a tuple of ints (tuple/list, or a CLI string like
    ``"[256,1024]"`` / ``"256,1024"``)."""
    if isinstance(x, str):
        x = [p for p in x.strip().strip("[]()").replace(",", " ").split() if p]
    return tuple(int(v) for v in x)


def _tile_to(x, n):
    """Repeat ``x`` along axis 0 up to length ``n`` (postures are placeholders here)."""
    reps = -(-n // x.shape[0])
    return jnp.concatenate([x] * reps, axis=0)[:n]


def sweep_throughput(cfg, mjx_model, foot_ids, qpos, stances):
    """Time the parallel weld-and-settle over the batch-size x settle-step grid.

    Args:
        cfg: config (``batch_sizes``, ``step_counts``, ``repeats``).
        mjx_model, foot_ids: from ``load_model``.
        qpos: (K, nq) available postures (tiled to reach each batch size).
        stances: (K, 6) planted masks.

    Returns:
        dict of equal-length arrays: batch, steps, cold_s, warm_s, env_steps_per_s.
        ``cold_s`` is the first run (includes JIT compile); ``warm_s`` is the fastest
        of the ``repeats`` runs right after (compiled). Throughput uses ``warm_s``.
    """
    rows = {k: [] for k in ("batch", "steps", "cold_s", "warm_s", "env_steps_per_s")}
    for B in _ints(cfg.batch_sizes):
        qb, sb = _tile_to(qpos, B), _tile_to(stances, B)
        for T in _ints(cfg.step_counts):
            fn = jax.jit(lambda q, s, T=T: weld_in_place(mjx_model, foot_ids, q, s, n_steps=T)[0].qpos)
            t0 = time.perf_counter()
            jax.block_until_ready(fn(qb, sb))                      # cold: JIT compile + first run
            cold_s = time.perf_counter() - t0
            warm_s = np.inf
            for _ in range(cfg.repeats):                           # warm: compiled, right after
                t0 = time.perf_counter()
                jax.block_until_ready(fn(qb, sb))
                warm_s = min(warm_s, time.perf_counter() - t0)
            sps = B * T / warm_s
            for k, v in zip(rows, (B, T, cold_s, warm_s, sps)):
                rows[k].append(v)
            print(f"  B={B:>6} steps={T:>5}  cold={cold_s:6.2f}s "
                  f"warm={warm_s*1e3:8.1f}ms  {sps/1e6:7.2f} M env-steps/s")
    return {k: np.asarray(v) for k, v in rows.items()}


# --------------------------------------------------------------------------- #
# Experiment                                                                  #
# --------------------------------------------------------------------------- #
@experiment(name="mjx_stance_bench")
def run(cfg: Cfg, ctx: RunContext):
    print(f"[bench] device: {jax.devices()[0].platform.upper()}  ->  {ctx.out}")
    key = jax.random.PRNGKey(cfg.seed)
    mjx_model, foot_ids = load_model()

    # 1. postures
    bodies, theta, feet, stance_mask, mesh = sample_postures(cfg, key)
    K = int(theta.shape[0])
    qpos = to_qpos(bodies, theta)
    stances = jnp.broadcast_to(stance_mask, (K, 6))
    print(f"[bench] sampled {K} valid postures  (stance {np.asarray(cfg.stance)})")

    # 2. static forces (this is what we save with each posture)
    foot_forces, joint_torques = jax.jit(stance_forces_batch)(mjx_model, foot_ids, qpos, stances)
    jax.block_until_ready(foot_forces)

    # 3. throughput sweep
    print("[bench] weld-and-settle throughput:")
    sweep = sweep_throughput(cfg, mjx_model, foot_ids, qpos, stances)

    # save everything
    results = ctx.out / "results"
    results.mkdir(parents=True, exist_ok=True)
    V, F, FN = mesh
    postures = {
        "body_wxyz_xyz": np.asarray(bodies.wxyz_xyz),   # (K, 7) qw qx qy qz x y z
        "qpos": np.asarray(qpos),                        # (K, nq)
        "theta": np.asarray(theta),                      # (K, 6, 3)
        "feet": np.asarray(feet),                        # (K, 6, 3)
        "foot_forces": np.asarray(foot_forces),          # (K, 6, 3) world
        "joint_torques": np.asarray(joint_torques),      # (K, 6, 3) coxa/femur/tibia
        "stance_mask": np.asarray(stance_mask),          # (6,) bool
    }
    np.savez(results / "postures.npz", model=str(MODEL.name),
             mesh_verts=np.asarray(V), mesh_faces=np.asarray(F), mesh_normals=np.asarray(FN),
             **postures)
    np.savez(results / "throughput.npz", **sweep)

    peak = float(sweep["env_steps_per_s"].max())
    max_tau = float(np.abs(np.asarray(joint_torques)).max())
    _write_summary(results / "summary.txt", cfg, ctx, postures, sweep, peak, max_tau)

    print(f"[bench] peak {peak/1e6:.2f} M env-steps/s | {K} postures | max |tau| {max_tau:.2f} Nm")
    return {"n_postures": K, "peak_env_steps_per_s": peak, "max_abs_torque": max_tau}


def _write_summary(path, cfg, ctx, postures, sweep, peak, max_tau):
    """Write a human-readable summary of the saved array shapes + sweep runtimes."""
    K = postures["theta"].shape[0]
    lines = [
        f"mjx_stance_bench   device={jax.devices()[0].platform.upper()}   id={ctx.id}",
        "=" * 68,
        f"postures: {K}   stance(planted legs): {np.flatnonzero(postures['stance_mask']).tolist()}",
        f"max |joint torque|: {max_tau:.2f} Nm",
        "",
        "arrays (postures.npz):",
    ]
    lines += [f"  {name:<15} {str(tuple(a.shape)):<14} {a.dtype}"
              for name, a in postures.items()]
    lines += [
        "",
        "throughput (weld-and-settle)   cold = first run incl. JIT compile; warm = compiled:",
        f"  {'batch':>8} {'steps':>7} {'cold[s]':>10} {'warm[ms]':>10} {'M env-steps/s':>15}",
    ]
    for b, s, c, r, sps in zip(sweep["batch"], sweep["steps"], sweep["cold_s"],
                               sweep["warm_s"], sweep["env_steps_per_s"]):
        lines.append(f"  {int(b):>8} {int(s):>7} {c:>10.2f} {r*1e3:>10.1f} {sps/1e6:>15.2f}")
    lines += ["", f"peak (warm): {peak/1e6:.2f} M env-steps/s", ""]
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main(run)
