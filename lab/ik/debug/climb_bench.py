"""GPU benchmark: batched rigid-stance stability settle in MJX.

Mirrors the CPU ``stability_test`` (rigid legs, gravity-comp, position servos,
settle under gravity, read drift/tilt), but vmapped over a batch so we can
measure how many poses/sec we can screen on the GPU.

Adhesion caveat: MJX's JAX backend does NOT support adhesion actuators
(``mjTRN_BODY``). Two paths:

  --impl jax   strip the 6 adhesion actuators and emulate the magnetic hold as a
               constant normal press (``xfrc_applied``, +x into the wall) on the
               planted feet. Runs anywhere (CPU now, CUDA later). Default.
  --impl warp  keep the real <adhesion> actuators and drive them via ctrl.
               Needs the ``mujoco_warp`` package + a CUDA GPU.

Run (jax):   uv run --extra mjx python -m lab.ik.debug.climb_bench --sizes 64,256,1024
Run (warp):  uv run --extra mjx python -m lab.ik.debug.climb_bench --impl warp
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[3]
CLIMB = ROOT / "models" / "climb0.xml"

PLANTED = (0, 2, 4)      # wall tripod used for the benchmark pose
PRESS_N = 150.0          # xfrc surrogate for adhesion hold force (jax path)


def build_model(impl, rigid_damping=1e4):
    """Load climb0, freeze the legs, tune the solver. For the jax backend, strip
    the adhesion actuators (unsupported); the warp backend keeps them."""
    xml = CLIMB.read_text()
    if impl == "jax":
        xml = re.sub(r"\s*<adhesion[^>]*/>", "", xml)     # drop mjTRN_BODY actuators
    m = mujoco.MjModel.from_xml_string(xml)
    m.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    m.opt.iterations, m.opt.ls_iterations = 100, 50
    m.dof_damping[6:] = rigid_damping                     # rigid legs
    return m


def batch_qpos(m, B, jitter=1e-3, seed=0):
    """B copies of the wall-tripod pose with tiny base-position jitter."""
    from lab.ik.sticky_test import wall_tripod_qpos
    q = wall_tripod_qpos(m, PLANTED, into_wall=-0.005)
    Q = np.tile(q, (B, 1))
    rng = np.random.default_rng(seed)
    Q[:, :3] += rng.normal(0, jitter, (B, 3))             # perturb base xyz
    return Q


def make_rollout(m, mx, impl, steps):
    """jit(vmap) rollout: (B, nq) qpos -> (B, 7) final base pos+quat."""
    import jax, jax.numpy as jnp
    import mujoco.mjx as mjx

    nv, nu = m.nv, m.nu
    theta_n = nv - 6                                       # 18 leg joints

    # jax path: constant normal press on the planted feet (adhesion surrogate)
    press = np.zeros((m.nbody, 6))
    for i in PLANTED:
        press[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"foot{i}"), 0] = PRESS_N
    press = jnp.asarray(press)

    # warp path: real adhesion ctrl (1 on planted feet, appended after the 18 servos)
    adh = np.zeros(nu - theta_n)
    for k, i in enumerate(range(6)):
        if i in PLANTED and (theta_n + i) < nu:
            adh[i] = 1.0
    adh = jnp.asarray(adh)

    d0 = mjx.make_data(mx)

    def ctrl_of(theta):
        return theta if impl == "jax" else jnp.concatenate([theta, adh])

    def single(q):
        d = d0.replace(qpos=q)
        if impl == "jax":
            d = d.replace(xfrc_applied=press)
        d = mjx.forward(mx, d)                             # populate qfrc_bias
        theta = q[7:7 + theta_n]

        def body(d, _):
            qfrc = d.qfrc_applied.at[6:].set(d.qfrc_bias[6:])   # gravity-comp rigid legs
            d = d.replace(qfrc_applied=qfrc, ctrl=ctrl_of(theta))
            if impl == "jax":
                d = d.replace(xfrc_applied=press)
            return mjx.step(mx, d), None

        d, _ = jax.lax.scan(body, d, None, length=steps)
        return d.qpos[:7]

    return jax.jit(jax.vmap(single))


def verdict(out, Q0):
    """out (B,7) final base, Q0 (B,nq) start -> drift_mm, tilt_deg arrays."""
    import jax.numpy as jnp
    drift = jnp.linalg.norm(out[:, :3] - jnp.asarray(Q0[:, :3]), axis=-1) * 1000
    dot = jnp.abs(jnp.sum(out[:, 3:7] * jnp.asarray(Q0[:, 3:7]), axis=-1))
    tilt = jnp.degrees(2 * jnp.arccos(jnp.clip(dot, 0.0, 1.0)))
    return np.asarray(drift), np.asarray(tilt)


def run(impl, sizes, steps, reps):
    import jax
    print(f"backend: {jax.default_backend()}  devices: {jax.devices()}")
    import mujoco.mjx as mjx

    m = build_model(impl)
    try:
        mx = mjx.put_model(m, impl=impl)
    except Exception as e:
        print(f"\nput_model(impl={impl!r}) failed: {type(e).__name__}: {e}")
        if impl == "warp":
            print("-> the warp backend needs the `mujoco_warp` package + a CUDA GPU.")
        return

    rollout = make_rollout(m, mx, impl, steps)
    print(f"impl={impl}  steps={steps}  press={PRESS_N}N (jax only)  planted={PLANTED}\n")
    print(f"{'batch':>7} {'compile(s)':>11} {'run(s)':>9} {'poses/s':>10} "
          f"{'env-steps/s':>13}  {'stable%':>8}")
    for B in sizes:
        Q = batch_qpos(m, B)
        Qj = jax.numpy.asarray(Q)

        t = time.perf_counter()
        out = rollout(Qj); out.block_until_ready()          # includes JIT compile
        t_compile = time.perf_counter() - t

        t = time.perf_counter()
        for _ in range(reps):
            out = rollout(Qj); out.block_until_ready()
        t_run = (time.perf_counter() - t) / reps

        drift, tilt = verdict(out, Q)
        stable = float(np.mean((drift <= 30) & (tilt <= 8)) * 100)
        print(f"{B:>7} {t_compile:>11.2f} {t_run:>9.3f} {B/t_run:>10.0f} "
              f"{B*steps/t_run:>13.0f}  {stable:>7.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["jax", "warp"], default="jax")
    ap.add_argument("--sizes", default="256,1024,4096,16384",
                    help="comma-separated batch sizes")
    ap.add_argument("--steps", type=int, default=375)       # 1.5 s @ dt=0.004
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    run(a.impl, [int(s) for s in a.sizes.split(",")], a.steps, a.reps)


if __name__ == "__main__":
    main()
