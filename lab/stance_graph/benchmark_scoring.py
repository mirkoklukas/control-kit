"""Benchmark: force-scoring throughput only (fast -- no weld-and-settle sweep).

A trimmed sibling of ``benchmark.py``: samples valid climb-terrain postures and
times ``stance_forces_batch`` (statics, 0 steps) across batch sizes -- a proxy for
how long *scoring* a posture (body + theta) costs. Saves the postures + their
forces + the scoring timing. Use this when you only care about scoring cost and
don't want to pay for the settle-dynamics sweep.

The posture pipeline and the timing loop are shared with ``benchmark.py``
(``sample_postures`` / ``sweep_scoring``); only the experiment wiring differs.

Run (on the A100):
    uv run --extra mjx python -m lab.stance_graph.benchmark_scoring \
        batch_sizes='[256,1024,4096,16384,65536]' n_stances=8000 --tag=a100
"""
from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp

from runkit import experiment, RunContext, main

from lab.stance_graph.mjx_stance import MODEL, load_model, to_qpos, stance_forces_batch
from lab.stance_graph.benchmark import sample_postures, sweep_scoring


@dataclass
class Cfg:
    seed: int = 0

    # posture sampling (see notebooks/07_sample_postures.ipynb)
    pool_size: int = 100_000
    n_bodies: int = 16
    n_stances: int = 4_000
    body_xyz: tuple = (0.8, 0.0, 0.35)
    body_rpy_deg: tuple = (0.0, -40.0, 0.0)
    stance: tuple = (0, 2, 3, 5)
    xyz_delta: float = 0.02
    rpy_delta_deg: float = 3.0

    # scoring sweep
    batch_sizes: tuple = (256, 1024, 4096, 16384)
    repeats: int = 3


def _write_summary(path, ctx, postures, scoring, max_tau):
    """Write a human-readable summary of the postures + scoring runtimes."""
    K = postures["theta"].shape[0]
    lines = [
        f"mjx_scoring_bench   device={jax.devices()[0].platform.upper()}   id={ctx.id}",
        "=" * 72,
        f"postures: {K}   stance(planted legs): {np.flatnonzero(postures['stance_mask']).tolist()}",
        f"max |joint torque|: {max_tau:.2f} Nm",
        "",
        "arrays (postures.npz):",
    ]
    lines += [f"  {name:<15} {str(tuple(a.shape)):<14} {a.dtype}"
              for name, a in postures.items()]
    lines += [
        "",
        "force scoring (stance_forces_batch, statics -- 0 steps)   cold incl. JIT compile:",
        f"  {'batch':>8} {'cold[s]':>10} {'warm[ms]':>10} {'M postures/s':>14} {'us/posture':>12}",
    ]
    for b, c, r, pps, ups in zip(scoring["batch"], scoring["cold_s"], scoring["warm_s"],
                                 scoring["postures_per_s"], scoring["us_per_posture"]):
        lines.append(f"  {int(b):>8} {c:>10.2f} {r*1e3:>10.2f} {pps/1e6:>14.3f} {ups:>12.2f}")
    lines += ["", f"peak scoring (warm): {scoring['postures_per_s'].max()/1e6:.3f} M postures/s", ""]
    path.write_text("\n".join(lines))


@experiment(name="mjx_scoring_bench")
def run(cfg: Cfg, ctx: RunContext):
    print(f"[score] device: {jax.devices()[0].platform.upper()}  ->  {ctx.out}")
    key = jax.random.PRNGKey(cfg.seed)
    mjx_model, foot_ids = load_model()

    bodies, theta, feet, stance_mask, _ = sample_postures(cfg, key)
    K = int(theta.shape[0])
    qpos = to_qpos(bodies, theta)
    stances = jnp.broadcast_to(stance_mask, (K, 6))
    print(f"[score] sampled {K} valid postures  (stance {np.asarray(cfg.stance)})")

    # compute the forces (to save) then time the scoring across batch sizes
    foot_forces, joint_torques = jax.jit(stance_forces_batch)(mjx_model, foot_ids, qpos, stances)
    jax.block_until_ready(foot_forces)
    print("[score] force scoring throughput (stance_forces_batch):")
    scoring = sweep_scoring(cfg, mjx_model, foot_ids, qpos, stances)

    results = ctx.out / "results"
    results.mkdir(parents=True, exist_ok=True)
    postures = {
        "body_wxyz_xyz": np.asarray(bodies.wxyz_xyz),
        "qpos": np.asarray(qpos),
        "theta": np.asarray(theta),
        "feet": np.asarray(feet),
        "foot_forces": np.asarray(foot_forces),
        "joint_torques": np.asarray(joint_torques),
        "stance_mask": np.asarray(stance_mask),
    }
    np.savez(results / "postures.npz", model=str(MODEL.name), **postures)
    np.savez(results / "scoring.npz", **scoring)

    score_pps = float(scoring["postures_per_s"].max())
    max_tau = float(np.abs(np.asarray(joint_torques)).max())
    _write_summary(results / "summary.txt", ctx, postures, scoring, max_tau)

    print(f"[score] peak {score_pps/1e6:.3f} M postures/s | {K} postures | max |tau| {max_tau:.2f} Nm")
    return {"n_postures": K, "peak_score_postures_per_s": score_pps, "max_abs_torque": max_tau}


if __name__ == "__main__":
    main(run)
