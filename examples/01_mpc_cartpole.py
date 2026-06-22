"""Cartpole swing-up with MPPI -- record on a GPU box, replay locally.

Built from the abstract pieces in controlkit.mpc: the framework-agnostic
`make_rollout_sampler` + `make_mppi_planner`. Everything system-specific lives
here -- the MJX dynamics step, the cartpole observation, the exploration
proposal, and the cost. The task: pole hanging down -> pump energy in -> catch
and balance upright.

This is the *cold* version -- the proposal is zero-mean every tick (no
warm-start yet). Warm-starting (reference-plan conditioning) is the next step;
see docs/mpc.md.

MJX has no viewer and only flies on CUDA/TPU (CPU works but is slow; the Apple
GPU can't run MJX at all -- see docs/gotchas.md). So the loop is split in two:

  record  (headless)  -- run the controller, save the state trajectory to an
                         .npz. This is what runs on the cloud box.
      uv run python examples/01_mpc_cartpole.py record
      uv run python examples/01_mpc_cartpole.py record --out runs/cartpole.npz

  play    (local, needs a display)  -- replay a saved trajectory in the viewer.
      uv run mjpython examples/01_mpc_cartpole.py play runs/cartpole.npz

We save qpos/qvel per step, not rendered frames: the trajectory is a few KB, and
mj_forward reconstructs everything the viewer needs (geom/site poses) from the
state at playback time. So you scp the .npz down and replay at any camera/res,
fully decoupled from the slow per-tick planning.
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
MODEL = ROOT / "models" / "cartpole.xml"
DEFAULT_OUT = ROOT / "runs" / "cartpole.npz"

# MPPI hyperparameters
HORIZON = 30          # T: planning steps (0.30 s at dt=0.01)
SAMPLES = 100         # N: candidate rollouts per tick
LAMBDA = 1.0          # temperature: lower -> greedier
NOISE_SIGMA = 6.0     # exploration std on control (N)
CTRL_MIN, CTRL_MAX = -20.0, 20.0
STEPS = 250           # real-system steps (2.5 s)
LOG_EVERY = 10        # print a progress line every this many steps

app = typer.Typer(
    add_completion=False, no_args_is_help=True,
    help="Cartpole swing-up with MPPI: 'record' a trajectory on a GPU box, "
         "then 'play' it back locally in the viewer. See module docstring / docs/mpc.md.",
)

def get_obs(key, s):
    """Slim observation [cart_x, pole_angle, cart_vel, pole_angvel]."""
    return jnp.concatenate([s.qpos, s.qvel])


def cost(y):
    """Per-observation cost: 0 upright & centered, ~4 on the angle term hanging down."""
    x, theta, _, theta_dot = y
    upright = (jnp.cos(theta) - 1.0) ** 2
    return 5.0 * upright + 0.5 * x**2 + 0.02 * theta_dot**2


def wrap(theta):
    """Wrap an angle to [-pi, pi] and take |.| (0 = upright)."""
    return jnp.abs(jnp.arctan2(jnp.sin(theta), jnp.cos(theta)))


def make_controller(mjx_model):
    """Wire the abstract mpc pieces to this system; return (policy, step), jitted.

    policy(key, s) -> u0   the receding-horizon action (head of the MPPI plan).
    step(s, u)     -> s'   one real-system dynamics step.
    """
    nu = mjx_model.nu

    def env_step(s, u):
        return mjx.step(mjx_model, s.replace(ctrl=u))

    def explore(key, y):                       # cold proposal: zero-mean noise
        u = NOISE_SIGMA * jax.random.normal(key, (nu,))
        return jnp.clip(u, CTRL_MIN, CTRL_MAX)

    sampler = make_rollout_sampler(env_step, explore, get_obs)
    planner = make_mppi_planner(sampler, cost, HORIZON, SAMPLES, LAMBDA)

    # The receding-horizon controller is the head of the plan. (We could instead
    # drive the real loop with make_rollout_sampler(env_step, policy, identity)
    # -- the same primitive nested -- but an explicit loop keeps compile cheap
    # and lets us watch progress.)
    policy = jax.jit(lambda key, s: planner(key, s)[0])
    step = jax.jit(env_step)
    return policy, step


@app.command()
def record(
    out: Annotated[Path, typer.Option(help="where to save the recorded trajectory")] = DEFAULT_OUT
) -> None:
    """Run the MPPI controller headless, log the state trajectory, save to an .npz.

    GPU-friendly: no viewer is imported, so this runs on the headless cloud box.
    """
    mj_model = mujoco.MjModel.from_xml_path(str(MODEL))
    mjx_model = mjx.put_model(mj_model)
    policy, step = make_controller(mjx_model)

    # initial state: pole hanging straight down
    s = mjx.make_data(mjx_model)
    s = s.replace(qpos=s.qpos.at[1].set(jnp.pi))
    s = mjx.forward(mjx_model, s)
    key = jax.random.PRNGKey(0)

    # frame 0 is the initial state; then one frame per executed step
    qpos = [np.asarray(s.qpos)]
    qvel = [np.asarray(s.qvel)]
    ctrl = []

    print(f"recording {STEPS} steps  (H={HORIZON}, N={SAMPLES}, lam={LAMBDA}, "
          f"backend={jax.default_backend()}); step 1 includes the JIT compile")
    print(f"{'step':>5}{'elapsed':>9}{'angle':>8}{'cost':>8}{'best':>8}{'rate':>9}")

    t0 = time.time()
    t_mark, mark_step = t0, 0
    best = float(wrap(s.qpos[1]))
    for i in range(STEPS):
        key, sub = jax.random.split(key)
        u0 = policy(sub, s)
        s = step(s, u0)
        qpos.append(np.asarray(s.qpos))
        qvel.append(np.asarray(s.qvel))
        ctrl.append(np.asarray(u0))
        ang = float(wrap(s.qpos[1]))
        best = min(best, ang)
        if i == 0 or (i + 1) % LOG_EVERY == 0:        # first step, then every LOG_EVERY
            now = time.time()
            rate = (i + 1 - mark_step) / (now - t_mark)
            c = float(cost(get_obs(None, s)))
            print(f"{i+1:>5}{now - t0:>8.1f}s{ang:>8.3f}{c:>8.3f}{best:>8.3f}{rate:>7.1f}/s")
            t_mark, mark_step = now, i + 1
    final = float(wrap(s.qpos[1]))

    print(f"\n{STEPS} steps in {time.time() - t0:.1f}s")
    print(f"closest to upright: {best:.3f} rad, final: {final:.3f} rad")
    print("SWUNG UP" if best < 0.3 else
          "did not reach upright (try more SAMPLES/HORIZON, or warm-start)")

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        qpos=np.stack(qpos), qvel=np.stack(qvel), ctrl=np.stack(ctrl),
        timestep=mj_model.opt.timestep, model=str(MODEL.relative_to(ROOT)),
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
