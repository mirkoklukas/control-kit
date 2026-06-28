"""Hexapod forward locomotion with MPPI -- record on a GPU box, replay locally.

The abstract MPPI pieces live in controlkit.mpc; everything system-specific is
in env.py -- the MJX dynamics step, the trunk observation, the exploration
proposal, and the cost. The tunable knobs are in config.py (Cfg). The task:
make the radial hexapod walk *stably in +x* from a standing start.

Key design points (see docs/mpc.md):

  * The 18 actuators are POSITION servos, so `ctrl` is a vector of target joint
    angles (radians), not a force. The proposal builds each step's target by
    integrating noise into the *previous* target -- a correlated (Brownian)
    random walk -- clipped to each joint's range. This is the key to getting
    motion: a zero-mean proposal around the rest pose is held there by the stiff
    servo and its candidates average out to "stand still". Integrating into the
    target lets candidate rollouts drift through large, sustained arcs (~8x the
    limb excursion of the zero-mean / achieved-angle variants). The executed
    motion is still throttled, because cold MPPI averages the candidates' first
    step back toward zero -- warm-start (carrying the plan forward) is the fix.
  * The cost is on the free-floating trunk: reward forward speed while keeping
    the body upright, level, on its line, and not spinning. Stability is
    weighted above progress on purpose -- a hexapod that face-plants scores far
    worse than one that creeps.
  * Control runs at ~31 Hz, not the 250 Hz sim rate: each planned target is held
    for `decimation` sim steps. Decimating lets each decision drive real motion,
    spans a gait cycle in the horizon, and gives the executed step enough credit
    that it stops averaging back to "stand still".

There's still no cross-tick warm-start (the plan isn't carried forward) -- that
is the planned next version (v1). MJX has no viewer and only flies on CUDA/TPU
(CPU works but is slow; the Apple GPU can't run MJX at all -- see
docs/gotchas.md), so the loop is split in two:

  record  (headless)  -- run the controller, save the state trajectory to .npz.
      uv run --extra mjx python -m lab.hexapod_mpc_exp record
      uv run --extra mjx python -m lab.hexapod_mpc_exp record --out runs/hexapod.npz

  play    (local, needs a display)  -- replay a saved trajectory in the viewer.
      uv run mjpython -m lab.hexapod_mpc_exp play runs/hexapod.npz

We save qpos/qvel per step, not frames: mj_forward reconstructs every geom/site
pose from the state at playback, so the .npz is tiny and replays at any
camera/res, decoupled from the slow per-tick planning.
"""
import time
from pathlib import Path
from typing import Annotated

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import typer
from mujoco import mjx

from .config import Cfg
from .env import ctrl_limits, get_obs, make_controller, upright_z

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models" / "hexapod.xml"
DEFAULT_OUT = ROOT / "runs" / "hexapod.npz"

app = typer.Typer(
    add_completion=False, no_args_is_help=True,
    help="Hexapod forward walking with MPPI: 'record' a trajectory on a GPU box, "
         "then 'play' it back locally in the viewer. See module docstring / docs/mpc.md.",
)


@app.command()
def record(
    out: Annotated[Path, typer.Option(help="where to save the recorded trajectory")] = DEFAULT_OUT,
    steps: Annotated[int, typer.Option(help="real-system steps to record")] = Cfg.steps,
    samples: Annotated[int, typer.Option(help="MPPI candidate rollouts per tick (N)")] = Cfg.samples,
    horizon: Annotated[int, typer.Option(help="MPPI planning horizon (T)")] = Cfg.horizon,
) -> None:
    """Run the MPPI controller headless, log the state trajectory, save to an .npz.

    GPU-friendly: no viewer is imported, so this runs on the headless cloud box.
    The --steps/--samples/--horizon options exist mainly for a quick local smoke
    test; the defaults are the GPU-shaped values.
    """
    cfg = Cfg(steps=steps, samples=samples, horizon=horizon)

    mj_model = mujoco.MjModel.from_xml_path(str(MODEL))
    mjx_model = mjx.put_model(mj_model)
    ctrl_lo, ctrl_hi = ctrl_limits(mj_model)
    policy, step, cost = make_controller(mjx_model, ctrl_lo, ctrl_hi, cfg)

    # initial state: rest pose, then settle onto the feet with zero control
    s = mjx.make_data(mjx_model)
    s = mjx.forward(mjx_model, s)
    zero = jnp.zeros(mjx_model.nu)
    for _ in range(cfg.settle):
        s = step(s, zero)

    key = jax.random.PRNGKey(0)

    # frame 0 is the settled state; then one frame per executed step
    qpos = [np.asarray(s.qpos)]
    qvel = [np.asarray(s.qvel)]
    ctrl = []

    x0 = float(s.qpos[0])
    print(f"recording {cfg.steps} steps  (H={cfg.horizon}, N={cfg.samples}, lam={cfg.lam}, "
          f"backend={jax.default_backend()}); step 1 includes the JIT compile")
    print(f"{'step':>5}{'elapsed':>9}{'x':>8}{'vx':>8}{'z':>7}{'up':>6}{'cost':>8}{'rate':>9}")

    t0 = time.time()
    t_mark, mark_step = t0, 0
    for i in range(cfg.steps):
        key, sub = jax.random.split(key)
        u0 = policy(sub, s)
        s = step(s, u0)
        qpos.append(np.asarray(s.qpos))
        qvel.append(np.asarray(s.qvel))
        ctrl.append(np.asarray(u0))
        if i == 0 or (i + 1) % cfg.log_every == 0:
            now = time.time()
            rate = (i + 1 - mark_step) / (now - t_mark)
            y = get_obs(None, s)
            print(f"{i+1:>5}{now - t0:>8.1f}s{float(s.qpos[0]):>8.3f}"
                  f"{float(s.qvel[0]):>8.3f}{float(s.qpos[2]):>7.3f}"
                  f"{float(upright_z(s.qpos[3:7])):>6.2f}{float(cost(y)):>8.2f}{rate:>7.1f}/s")
            t_mark, mark_step = now, i + 1

    dist = float(s.qpos[0]) - x0
    up = float(upright_z(s.qpos[3:7]))
    fell = (float(s.qpos[2]) < 0.5 * cfg.z_nominal) or (up < 0.5)
    print(f"\n{cfg.steps} steps in {time.time() - t0:.1f}s")
    print(f"forward distance: {dist:+.3f} m   final z: {float(s.qpos[2]):.3f}   upright: {up:.2f}")
    print("FELL OVER (lower stability weight, or warm-start)" if fell else
          (f"walked forward {dist:.2f} m, stayed up" if dist > 0.05 else
           "stayed up but barely moved (try more samples/horizon, or warm-start)"))

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        qpos=np.stack(qpos), qvel=np.stack(qvel), ctrl=np.stack(ctrl),
        timestep=mj_model.opt.timestep * cfg.decimation,   # frame spacing for real-time replay
        model=str(MODEL.relative_to(ROOT)),
    )
    print(f"saved {len(qpos)} frames -> {out}")


@app.command()
def play(
    file: Annotated[Path, typer.Argument(help="saved .npz trajectory to replay")],
) -> None:
    """Replay a saved trajectory in the passive viewer (local; needs mjpython on macOS)."""
    from controlkit.viz import play as play_npz

    play_npz(file)


if __name__ == "__main__":
    app()
