"""System-specific wiring for the hexapod forward-walking MPPI experiment.

The abstract MPPI pieces live in controlkit.mpc (`make_rollout_sampler` +
`make_mppi_planner`); everything system-specific is here -- the MJX dynamics
step, the trunk observation, the exploration proposal, and the cost. The task:
make the radial hexapod walk *stably in +x* from a standing start. See the
__main__.py module docstring and docs/mpc.md for the why behind the choices.
"""
import jax
import jax.numpy as jnp
from mujoco import mjx

from controlkit.mpc import make_mppi_planner, make_rollout_sampler

from .config import Cfg


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


def upright_z(quat):
    """World z-component of the trunk's local up-axis (1 = level, <0 = flipped)."""
    w, x, y, z = quat
    return 1.0 - 2.0 * (x * x + y * y)


def ctrl_limits(mj_model):
    """Per-actuator (lo, hi) target bounds, in radians, from each joint's range."""
    jid = mj_model.actuator_trnid[:, 0]            # joint driven by each actuator
    lo = mj_model.jnt_range[jid, 0]
    hi = mj_model.jnt_range[jid, 1]
    return jnp.asarray(lo), jnp.asarray(hi)


def make_cost(cfg: Cfg):
    """Build the per-step trunk cost, closing over the task targets in cfg.

    0 is unreachable (the speed term alone is vx_target^2 at standstill); what
    matters is the shape -- tilting, sinking, drifting, or spinning all cost more
    than missing the target speed, so candidates that stay a level, on-course
    stance win.
    """
    def cost(y):
        _x, y_lat, z = y[0], y[1], y[2]
        _qw, qx, qy, qz = y[3], y[4], y[5], y[6]
        vx, vy, vz = y[7], y[8], y[9]
        wx, wy, wz = y[10], y[11], y[12]

        forward = (vx - cfg.vx_target) ** 2       # cruise at the target speed
        upright = qx**2 + qy**2                    # roll + pitch tilt (yaw left freer)
        height = (z - cfg.z_nominal) ** 2          # don't sink or pronk
        lateral = y_lat**2 + 0.3 * vy**2           # hold the +x line
        spin = wx**2 + wy**2 + wz**2               # damp body rotation

        return (2.0 * forward + 8.0 * upright + 8.0 * height
                + 1.0 * lateral + 0.5 * qz**2 + 0.05 * spin + 0.2 * vz**2)
    return cost


def make_controller(mjx_model, ctrl_lo, ctrl_hi, cfg: Cfg):
    """Wire the abstract mpc pieces to this system; return (policy, step, cost).

    policy(key, s) -> u0   receding-horizon target angles (head of the MPPI plan).
    step(s, u)     -> s'   one real-system dynamics step (held for cfg.decimation).
    cost(y)        -> c    the same per-step cost the planner uses (for logging).
    policy and step are jitted.
    """
    nu = mjx_model.nu
    cost = make_cost(cfg)

    def env_step(s, u):                        # hold the target for `decimation` sim steps
        def sub(s, _):
            return mjx.step(mjx_model, s.replace(ctrl=u)), None
        s, _ = jax.lax.scan(sub, s, None, length=cfg.decimation)
        return s

    def explore(key, y):                       # proposal: integrate noise into the
        prev = y[13:13 + nu]                   # previous target (s.ctrl) -> a Brownian
        u = prev + cfg.noise_sigma * jax.random.normal(key, (nu,))   # random walk on the
        return jnp.clip(u, ctrl_lo, ctrl_hi)   # target; clipped to joint ranges

    sampler = make_rollout_sampler(env_step, explore, get_obs)
    planner = make_mppi_planner(sampler, cost, cfg.horizon, cfg.samples, cfg.lam)

    policy = jax.jit(lambda key, s: planner(key, s)[0])
    step = jax.jit(env_step)
    return policy, step, cost
