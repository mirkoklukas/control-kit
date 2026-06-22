"""Hexapod forward locomotion with MPPI -- record on a GPU box, replay locally.

Same skeleton as examples/01_mpc_cartpole.py: the abstract MPPI pieces live in
controlkit.mpc (`make_rollout_sampler` + `make_mppi_planner`); everything
system-specific is here -- the MJX dynamics step, the trunk observation, the
exploration proposal, and the cost. The task: make the radial hexapod walk
*stably in +x* from a standing start.

What's different from the cartpole:

  * The 18 actuators are POSITION servos, so `ctrl` is a vector of target joint
    angles (radians), not a force. The proposal builds each step's target by
    integrating noise into the *previous* target -- a correlated (Brownian)
    random walk -- clipped to each joint's range. This is the key to getting
    motion: a zero-mean proposal around the rest pose is held there by the stiff
    servo and its candidates average out to "stand still"; even anchoring the
    noise to the current *achieved* angle mean-reverts (the servo pulls it back).
    Integrating into the target lets candidate rollouts drift through large,
    sustained arcs (~8x the limb excursion of the zero-mean / achieved-angle
    variants). That widens *exploration*; the executed motion is still throttled,
    because cold MPPI averages the candidates' first step back toward zero --
    warm-start (carrying the plan forward as the nominal) is the fix for that.
    See docs/mpc.md.
  * The cost is on the free-floating trunk: reward forward speed while keeping
    the body upright, level, on its line, and not spinning. Stability is
    weighted above progress on purpose -- a hexapod that face-plants scores far
    worse than one that creeps.
  * Control runs at ~31 Hz, not the 250 Hz sim rate: each planned target is held
    for DECIMATION sim steps. At 250 Hz a receding-horizon plan only ever
    executes one 4 ms step and the legs barely move before re-planning (and 39/40
    of each plan is thrown away). Decimating lets each decision drive real motion,
    spans a gait cycle in the horizon, and gives the executed step enough credit
    that it stops averaging back to "stand still".

There's still no cross-tick warm-start (the plan isn't carried forward), so this
stays close to the cold cartpole; but the residual (around-current) proposal
gives the per-tick exploration the temporal structure a zero-mean proposal
lacks. A previous-plan nominal is the next step (docs/mpc.md).

MJX has no viewer and only flies on CUDA/TPU (CPU works but is slow; the Apple
GPU can't run MJX at all -- see docs/gotchas.md), so the loop is split in two:

  record  (headless)  -- run the controller, save the state trajectory to .npz.
      uv run python examples/02_mpc_hexapod.py record
      uv run python examples/02_mpc_hexapod.py record --out runs/hexapod.npz

  play    (local, needs a display)  -- replay a saved trajectory in the viewer.
      uv run mjpython examples/02_mpc_hexapod.py play runs/hexapod.npz

We save qpos/qvel per step, not frames: mj_forward reconstructs every geom/site
pose from the state at playback, so the .npz is tiny and replays at any
camera/res, decoupled from the slow per-tick planning.
"""

import sys
import time
from pathlib import Path
from typing import Annotated

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import typer
from mujoco import mjx

# make controlkit importable from src/ without an editable install
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from controlkit.mpc import make_mppi_planner, make_rollout_sampler  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "hexapod.xml"
DEFAULT_OUT = ROOT / "runs" / "hexapod.npz"

# MPPI hyperparameters (sim dt = 0.004 s; we control every DECIMATION sim steps)
DECIMATION = 8        # hold each planned target for this many sim steps (~31 Hz control)
HORIZON = 20          # T: planning steps -> 0.64 s lookahead at DECIMATION=8 (a gait cycle)
SAMPLES = 128         # N: candidate rollouts per tick
LAMBDA = 0.2          # temperature: lower -> greedier (commit harder to the best candidate)
NOISE_SIGMA = 0.1     # per-control-step increment integrated into the target (rad)
STEPS = 150           # control decisions to record (150 * 8 * 0.004 = 4.8 s)
LOG_EVERY = 10        # print a progress line every this many control steps

# Task targets
VX_TARGET = 0.3       # desired forward speed (m/s)
Z_NOMINAL = 0.20      # standing trunk height (m); the freejoint starts here
SETTLE = 25           # zero-control steps to settle onto the feet before recording

app = typer.Typer(
    add_completion=False, no_args_is_help=True,
    help="Hexapod forward walking with MPPI: 'record' a trajectory on a GPU box, "
         "then 'play' it back locally in the viewer. See module docstring / docs/mpc.md.",
)


def get_obs(key, s):
    """Observation: trunk pose + twist, then the previous target (s.ctrl).

    Layout (31): [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz | u0..u17].
    The cost reads only the leading trunk block [0:13]. The trailing 18 entries
    are the *previous* position-servo targets -- MJX keeps the ctrl we set in the
    state -- and the proposal integrates noise into them, a random walk on the
    target itself, which is what gives the limbs large, sustained motion. (Anchor
    the noise to the achieved angle qpos[7:] instead and the stiff servo just
    pulls it back to rest: ~8x less limb travel, measured.)
    For the freejoint, qpos[0:3] is world position, qpos[3:7] the orientation
    quaternion, qvel[0:3] world-frame linear velocity, qvel[3:6] body-frame
    angular velocity.
    """
    return jnp.concatenate([s.qpos[0:3], s.qpos[3:7], s.qvel[0:6], s.ctrl])


def cost(y):
    """Per-step trunk cost: forward progress, weighted under stability.

    0 is unreachable (the speed term alone is VX_TARGET^2 at standstill); what
    matters is the shape -- tilting, sinking, drifting, or spinning all cost more
    than missing the target speed, so candidates that stay a level, on-course
    stance win.
    """
    _x, y_lat, z = y[0], y[1], y[2]
    _qw, qx, qy, qz = y[3], y[4], y[5], y[6]
    vx, vy, vz = y[7], y[8], y[9]
    wx, wy, wz = y[10], y[11], y[12]

    forward = (vx - VX_TARGET) ** 2          # cruise at the target speed
    upright = qx**2 + qy**2                   # roll + pitch tilt (yaw left freer)
    height = (z - Z_NOMINAL) ** 2             # don't sink or pronk
    lateral = y_lat**2 + 0.3 * vy**2          # hold the +x line
    spin = wx**2 + wy**2 + wz**2              # damp body rotation

    return (2.0 * forward + 8.0 * upright + 8.0 * height
            + 1.0 * lateral + 0.5 * qz**2 + 0.05 * spin + 0.2 * vz**2)


def ctrl_limits(mj_model):
    """Per-actuator (lo, hi) target bounds, in radians, from each joint's range."""
    jid = mj_model.actuator_trnid[:, 0]            # joint driven by each actuator
    lo = mj_model.jnt_range[jid, 0]
    hi = mj_model.jnt_range[jid, 1]
    return jnp.asarray(lo), jnp.asarray(hi)


def make_controller(mjx_model, ctrl_lo, ctrl_hi, horizon, samples, lam):
    """Wire the abstract mpc pieces to this system; return (policy, step), jitted.

    policy(key, s) -> u0   receding-horizon target angles (head of the MPPI plan).
    step(s, u)     -> s'   one real-system dynamics step.
    """
    nu = mjx_model.nu

    def env_step(s, u):                        # hold the target for DECIMATION sim steps
        def sub(s, _):
            return mjx.step(mjx_model, s.replace(ctrl=u)), None
        s, _ = jax.lax.scan(sub, s, None, length=DECIMATION)
        return s

    def explore(key, y):                       # proposal: integrate noise into the
        prev = y[13:13 + nu]                   # previous target (s.ctrl) -> a Brownian
        u = prev + NOISE_SIGMA * jax.random.normal(key, (nu,))   # random walk on the
        return jnp.clip(u, ctrl_lo, ctrl_hi)   # target; clipped to joint ranges

    sampler = make_rollout_sampler(env_step, explore, get_obs)
    planner = make_mppi_planner(sampler, cost, horizon, samples, lam)

    policy = jax.jit(lambda key, s: planner(key, s)[0])
    step = jax.jit(env_step)
    return policy, step


def upright_z(quat):
    """World z-component of the trunk's local up-axis (1 = level, <0 = flipped)."""
    w, x, y, z = quat
    return 1.0 - 2.0 * (x * x + y * y)


@app.command()
def record(
    out: Annotated[Path, typer.Option(help="where to save the recorded trajectory")] = DEFAULT_OUT,
    steps: Annotated[int, typer.Option(help="real-system steps to record")] = STEPS,
    samples: Annotated[int, typer.Option(help="MPPI candidate rollouts per tick (N)")] = SAMPLES,
    horizon: Annotated[int, typer.Option(help="MPPI planning horizon (T)")] = HORIZON,
) -> None:
    """Run the MPPI controller headless, log the state trajectory, save to an .npz.

    GPU-friendly: no viewer is imported, so this runs on the headless cloud box.
    The --steps/--samples/--horizon options exist mainly for a quick local smoke
    test; the defaults are the GPU-shaped values.
    """
    mj_model = mujoco.MjModel.from_xml_path(str(MODEL))
    mjx_model = mjx.put_model(mj_model)
    ctrl_lo, ctrl_hi = ctrl_limits(mj_model)
    policy, step = make_controller(mjx_model, ctrl_lo, ctrl_hi, horizon, samples, LAMBDA)

    # initial state: rest pose, then settle onto the feet with zero control
    s = mjx.make_data(mjx_model)
    s = mjx.forward(mjx_model, s)
    zero = jnp.zeros(mjx_model.nu)
    for _ in range(SETTLE):
        s = step(s, zero)

    key = jax.random.PRNGKey(0)

    # frame 0 is the settled state; then one frame per executed step
    qpos = [np.asarray(s.qpos)]
    qvel = [np.asarray(s.qvel)]
    ctrl = []

    x0 = float(s.qpos[0])
    print(f"recording {steps} steps  (H={horizon}, N={samples}, lam={LAMBDA}, "
          f"backend={jax.default_backend()}); step 1 includes the JIT compile")
    print(f"{'step':>5}{'elapsed':>9}{'x':>8}{'vx':>8}{'z':>7}{'up':>6}{'cost':>8}{'rate':>9}")

    t0 = time.time()
    t_mark, mark_step = t0, 0
    for i in range(steps):
        key, sub = jax.random.split(key)
        u0 = policy(sub, s)
        s = step(s, u0)
        qpos.append(np.asarray(s.qpos))
        qvel.append(np.asarray(s.qvel))
        ctrl.append(np.asarray(u0))
        if i == 0 or (i + 1) % LOG_EVERY == 0:
            now = time.time()
            rate = (i + 1 - mark_step) / (now - t_mark)
            y = get_obs(None, s)
            print(f"{i+1:>5}{now - t0:>8.1f}s{float(s.qpos[0]):>8.3f}"
                  f"{float(s.qvel[0]):>8.3f}{float(s.qpos[2]):>7.3f}"
                  f"{float(upright_z(s.qpos[3:7])):>6.2f}{float(cost(y)):>8.2f}{rate:>7.1f}/s")
            t_mark, mark_step = now, i + 1

    dist = float(s.qpos[0]) - x0
    up = float(upright_z(s.qpos[3:7]))
    fell = (float(s.qpos[2]) < 0.5 * Z_NOMINAL) or (up < 0.5)
    print(f"\n{steps} steps in {time.time() - t0:.1f}s")
    print(f"forward distance: {dist:+.3f} m   final z: {float(s.qpos[2]):.3f}   upright: {up:.2f}")
    print("FELL OVER (lower stability weight, or warm-start)" if fell else
          (f"walked forward {dist:.2f} m, stayed up" if dist > 0.05 else
           "stayed up but barely moved (try more SAMPLES/HORIZON, or warm-start)"))

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        qpos=np.stack(qpos), qvel=np.stack(qvel), ctrl=np.stack(ctrl),
        timestep=mj_model.opt.timestep * DECIMATION,   # frame spacing for real-time replay
        model=str(MODEL.relative_to(ROOT)),
    )
    print(f"saved {len(qpos)} frames -> {out}")


@app.command()
def play(
    file: Annotated[Path, typer.Argument(help="saved .npz trajectory to replay")],
) -> None:
    """Replay a saved trajectory in the passive viewer (local; needs mjpython on macOS)."""
    from mujoco import viewer as mj_viewer

    npz = np.load(file)
    qpos, qvel, dt = npz["qpos"], npz["qvel"], float(npz["timestep"])
    print(f"loaded {len(qpos)} frames from {file} (dt={dt}s)")

    mj_model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(mj_model)
    with mj_viewer.launch_passive(mj_model, data) as viewer:
        while viewer.is_running():
            for q, v in zip(qpos, qvel):
                if not viewer.is_running():
                    break
                data.qpos[:] = q
                data.qvel[:] = v
                mujoco.mj_forward(mj_model, data)   # reconstruct poses for rendering
                viewer.sync()
                time.sleep(dt)                       # real-time pacing
            time.sleep(0.5)                          # pause, then loop the replay


if __name__ == "__main__":
    app()
