"""Cartpole swing-up with MPPI, built from the abstract pieces in controlkit.mpc.

Same task as 03_mpc_cartpole.py (pole hanging down -> pump up -> balance), but
the planner is the framework-agnostic `make_rollout_sampler` +
`make_mppi_planner`. Everything system-specific lives here: the MJX dynamics
step, the cartpole observation, the exploration proposal, and the cost.

This is the *cold* version -- the proposal is zero-mean every tick (no
warm-start yet). Warm-starting (reference-plan conditioning) is the next step;
see docs/mpc.md.

Headless check (works in a sandbox):
    uv run python examples/03b_mpc_cartpole.py

With viewer (run locally):
    uv run python examples/03b_mpc_cartpole.py --render
"""

import argparse
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

# make controlkit importable from src/ without an editable install
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from controlkit.mpc import make_mppi_planner, make_rollout_sampler  # noqa: E402

MODEL = Path(__file__).resolve().parent.parent / "models" / "cartpole.xml"

# MPPI hyperparameters
HORIZON = 30          # T: planning steps (0.30 s at dt=0.01)
SAMPLES = 100         # N: candidate rollouts per tick
LAMBDA = 1.0          # temperature: lower -> greedier
NOISE_SIGMA = 6.0     # exploration std on control (N)
CTRL_MIN, CTRL_MAX = -20.0, 20.0
STEPS = 250           # real-system steps (2.5 s)


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


def main(render: bool = False) -> None:
    mj_model = mujoco.MjModel.from_xml_path(str(MODEL))
    mjx_model = mjx.put_model(mj_model)
    nu = mj_model.nu

    # --- system-specific wiring for the abstract mpc pieces ---
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

    # initial state: pole hanging straight down
    s = mjx.make_data(mjx_model)
    s = s.replace(qpos=s.qpos.at[1].set(jnp.pi))
    s = mjx.forward(mjx_model, s)
    key = jax.random.PRNGKey(0)

    if render:
        import numpy as np
        from mujoco import viewer as mj_viewer

        data = mujoco.MjData(mj_model)
        with mj_viewer.launch_passive(mj_model, data) as viewer:
            while viewer.is_running():
                key, sub = jax.random.split(key)
                u0 = policy(sub, s)
                s = step(s, u0)
                data.qpos[:] = np.asarray(s.qpos)   # mirror mjx state for rendering
                data.qvel[:] = np.asarray(s.qvel)
                mujoco.mj_forward(mj_model, data)
                viewer.sync()
    else:
        t0 = time.time()
        best = float(wrap(s.qpos[1]))
        for _ in range(STEPS):
            key, sub = jax.random.split(key)
            u0 = policy(sub, s)
            s = step(s, u0)
            best = min(best, float(wrap(s.qpos[1])))
        final = float(wrap(s.qpos[1]))
        print(f"{STEPS} steps in {time.time() - t0:.1f}s "
              f"(H={HORIZON}, N={SAMPLES}, lam={LAMBDA})")
        print(f"closest to upright: {best:.3f} rad, final: {final:.3f} rad")
        print("SWUNG UP" if best < 0.3 else
              "did not reach upright (try more SAMPLES/HORIZON, or warm-start)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--render", action="store_true")
    main(**vars(p.parse_args()))
