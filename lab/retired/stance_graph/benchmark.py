"""Benchmark: parallel MJX throughput for welded-stance settling (runs on A100).

Self-contained runkit experiment. It

  1. samples a batch of valid hexapod postures on the climb terrain
     (body pose + joint angles + planted feet), as in notebooks/07_sample_postures;
  2. computes each posture's static joint torques + foot reaction forces
     (``mjx_stance``, no stepping) and saves them with the postures;
  3. sweeps batch size x settle steps, timing the parallel weld-and-settle
     (``mjx_weld``) to measure env-steps/s -- the parallel-throughput question.

Results are written *incrementally*: postures.npz and the summary header are saved
up front, and each sweep cell is appended to summary.txt (and a CSV) as it
finishes -- so a killed run keeps everything completed so far.

Everything the experiment needs lives here; the physics primitives come from the
sibling ``mjx_*`` modules. Outputs land in the runkit run dir (``ctx.out``).

Run (on the A100):
    uv run --extra mjx python -m lab.stance_graph.benchmark \
        batch_sizes='[256,1024,4096,16384]' step_counts='[50,200,800]' --tag=a100
"""
import csv as _csv
import time
from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp

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
    k_mesh, k_body, k_feet = jax.random.split(key, 3)
    V, F, FN = terrain.climb_surface()
    xs, _ = terrain.sample_mesh(k_mesh, V, F, M=cfg.pool_size)

    body = from_te(jnp.asarray(cfg.body_xyz), jnp.deg2rad(jnp.asarray(cfg.body_rpy_deg)))
    stance = jnp.asarray(cfg.stance)

    # per-leg reach mask over the pool, then sample S footholds per leg from it.
    shoulders = body @ SHOULDERS
    mask = jax.vmap(_leg_reach_mask, (0, None))(shoulders, xs)              # (6, M)
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
# Small IO helpers (incremental writing)                                      #
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


def _append(path, *lines):
    """Append ``lines`` (newline-terminated) to a text file."""
    with open(path, "a") as f:
        f.write("\n".join(str(ln) for ln in lines) + "\n")


def _append_csv(path, row):
    """Append a dict ``row`` to a CSV, writing the header on first use."""
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = _csv.writer(f)
        if new:
            w.writerow(list(row))
        w.writerow(list(row.values()))


def summary_header(path, name, ctx, dt, postures, max_tau):
    """Create summary.txt with the header + posture-array shapes. Sweep rows are
    appended incrementally (see the sweep functions), so a killed run keeps them."""
    K = postures["theta"].shape[0]
    lines = [
        f"{name}   device={jax.devices()[0].platform.upper()}   id={ctx.id}",
        "=" * 72,
        f"sim dt: {dt * 1e3:.3f} ms  ({1.0 / dt:.0f} Hz)",
        f"postures: {K}   stance(planted legs): {np.flatnonzero(postures['stance_mask']).tolist()}",
        f"max |joint torque|: {max_tau:.2f} Nm",
        "",
        "arrays (postures.npz):",
    ]
    lines += [f"  {n:<15} {str(tuple(a.shape)):<14} {a.dtype}" for n, a in postures.items()]
    path.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# Sweeps (each appends its section + rows as they complete)                    #
# --------------------------------------------------------------------------- #
def sweep_scoring(cfg, mjx_model, foot_ids, qpos, stances, summary=None, csv=None):
    """Time the static force 'scoring' (``stance_forces_batch``) across batch sizes.

    A proxy for how long it takes to *score* a posture (body + theta): one
    ``mjx.forward`` + solve per posture, no stepping. Each row is appended to
    ``summary``/``csv`` (if given) as it finishes.

    Returns:
        dict of equal-length arrays: batch, cold_s, warm_s, postures_per_s,
        us_per_posture. ``cold_s`` includes JIT compile; ``warm_s`` is the fastest
        of the ``repeats`` compiled runs.
    """
    if summary is not None:
        _append(summary, "",
                "force scoring (stance_forces_batch, statics -- 0 steps)   cold incl. JIT compile:",
                f"  {'batch':>8} {'cold[s]':>10} {'warm[ms]':>10} {'M postures/s':>14} {'us/posture':>12}")
    fn = jax.jit(lambda q, s: stance_forces_batch(mjx_model, foot_ids, q, s))
    rows = {k: [] for k in ("batch", "cold_s", "warm_s", "postures_per_s", "us_per_posture")}
    for B in _ints(cfg.batch_sizes):
        qb, sb = _tile_to(qpos, B), _tile_to(stances, B)
        t0 = time.perf_counter()
        jax.block_until_ready(fn(qb, sb))                          # cold: JIT compile + first run
        cold_s = time.perf_counter() - t0
        warm_s = np.inf
        for _ in range(cfg.repeats):                               # warm: compiled
            t0 = time.perf_counter()
            jax.block_until_ready(fn(qb, sb))
            warm_s = min(warm_s, time.perf_counter() - t0)
        pps, ups = B / warm_s, warm_s / B * 1e6
        for k, v in zip(rows, (B, cold_s, warm_s, pps, ups)):
            rows[k].append(v)
        line = f"  {B:>8} {cold_s:>10.2f} {warm_s*1e3:>10.2f} {pps/1e6:>14.3f} {ups:>12.2f}"
        print("[score]" + line)
        if summary is not None:
            _append(summary, line)
        if csv is not None:
            _append_csv(csv, dict(batch=B, cold_s=cold_s, warm_s=warm_s,
                                  postures_per_s=pps, us_per_posture=ups))
    return {k: np.asarray(v) for k, v in rows.items()}


def sweep_throughput(cfg, mjx_model, foot_ids, qpos, stances, dt=1.0, summary=None, csv=None):
    """Time the parallel weld-and-settle over the batch-size x settle-step grid.

    Each cell is appended to ``summary``/``csv`` (if given) as it finishes.

    Returns:
        dict of equal-length arrays: batch, steps, cold_s, warm_s, env_steps_per_s.
        ``cold_s`` is the first run (includes JIT compile); ``warm_s`` is the fastest
        of the ``repeats`` runs right after. Throughput uses ``warm_s``.
    """
    if summary is not None:
        _append(summary, "",
                "weld-and-settle throughput   cold = first run incl. JIT compile; warm = compiled:",
                f"  {'batch':>8} {'steps':>7} {'sim[ms]':>9} {'cold[s]':>10} {'warm[ms]':>10} {'M env-steps/s':>15}")
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
            line = (f"  {B:>8} {T:>7} {T*dt*1e3:>9.1f} {cold_s:>10.2f} "
                    f"{warm_s*1e3:>10.1f} {sps/1e6:>15.2f}")
            print("[settle]" + line)
            if summary is not None:
                _append(summary, line)
            if csv is not None:
                _append_csv(csv, dict(batch=B, steps=T, cold_s=cold_s, warm_s=warm_s,
                                      env_steps_per_s=sps))
    return {k: np.asarray(v) for k, v in rows.items()}


# --------------------------------------------------------------------------- #
# Experiment                                                                  #
# --------------------------------------------------------------------------- #
@experiment(name="mjx_stance_bench")
def run(cfg: Cfg, ctx: RunContext):
    print(f"[bench] device: {jax.devices()[0].platform.upper()}  ->  {ctx.out}")
    key = jax.random.PRNGKey(cfg.seed)
    mjx_model, foot_ids = load_model()
    dt = float(mjx_model.opt.timestep)
    print(f"[bench] sim dt = {dt * 1e3:.3f} ms  ({1.0 / dt:.0f} Hz)")

    # postures + static forces
    bodies, theta, feet, stance_mask, mesh = sample_postures(cfg, key)
    K = int(theta.shape[0])
    qpos = to_qpos(bodies, theta)
    stances = jnp.broadcast_to(stance_mask, (K, 6))
    print(f"[bench] sampled {K} valid postures  (stance {np.asarray(cfg.stance)})")
    foot_forces, joint_torques = jax.jit(stance_forces_batch)(mjx_model, foot_ids, qpos, stances)
    jax.block_until_ready(foot_forces)
    max_tau = float(np.abs(np.asarray(joint_torques)).max())

    # persist postures + summary header UP FRONT (before the sweeps)
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
    summary = results / "summary.txt"
    summary_header(summary, "mjx_stance_bench", ctx, dt, postures, max_tau)

    # sweeps -- append each row as it completes; save each npz right after
    print("[bench] force scoring throughput (stance_forces_batch):")
    scoring = sweep_scoring(cfg, mjx_model, foot_ids, qpos, stances,
                            summary=summary, csv=results / "scoring.csv")
    np.savez(results / "scoring.npz", **scoring)

    print("[bench] weld-and-settle throughput:")
    sweep = sweep_throughput(cfg, mjx_model, foot_ids, qpos, stances, dt=dt,
                             summary=summary, csv=results / "throughput.csv")
    np.savez(results / "throughput.npz", **sweep)

    peak = float(sweep["env_steps_per_s"].max())
    score_pps = float(scoring["postures_per_s"].max())
    _append(summary, "",
            f"peak scoring (warm): {score_pps/1e6:.3f} M postures/s",
            f"peak settle (warm):  {peak/1e6:.2f} M env-steps/s")

    print(f"[bench] peak {peak/1e6:.2f} M env-steps/s | scoring {score_pps/1e6:.3f} M postures/s "
          f"| {K} postures | max |tau| {max_tau:.2f} Nm")
    return {"n_postures": K, "sim_dt": dt, "peak_env_steps_per_s": peak,
            "peak_score_postures_per_s": score_pps, "max_abs_torque": max_tau}


if __name__ == "__main__":
    main(run)
