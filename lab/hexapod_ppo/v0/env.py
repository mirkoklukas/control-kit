"""MJX environment for hexapod_ppo v0 -- a Brax ``Env`` around the MJX dynamics.

System-specific wiring: maps the policy's normalized action to position-servo
targets, steps MJX ``decimation`` times per control step, and scores the result
with this experiment's ``reward.compute_reward``. The ids, command, kinematics,
and general reward terms come from the stable core (``controlkit.*``); this file
only glues them to the Brax training API.

Cross-step reward state (``foot_state``, ``last_action``) is threaded through
``State.info``. The command is a fixed forward twist (``Command.straight(vx)``),
so it is closed over rather than carried -- which also sidesteps Brax's default
AutoResetWrapper not resetting ``info`` (only ``foot_state``/``last_action`` go
stale for a single step after an auto-reset, a negligible transient).
"""
import dataclasses

import jax
import jax.numpy as jnp
import mujoco
from brax.envs.base import Env, State
from mujoco import mjx

from controlkit.utils import (
    base_ang_vel,
    base_lin_vel,
    foot_contacts,
    keyframe,
    projected_gravity,
)
from controlkit.hexapod import Command, FootState, HexapodIds

from .config import Cfg
from .reward import compute_reward

_WORLD_UP = (0.0, 0.0, 1.0)
TERM_NAMES = (
    "lin_vel", "ang_vel", "base_height", "lin_vel_z", "ang_vel_xy", "orientation",
    "action_rate", "torques", "dof_acc", "collision", "feet_air_time",
)


class HexapodEnv(Env):
    """Walk-straight hexapod environment (Brax ``Env``, MJX backend)."""

    def __init__(self, mj_model: mujoco.MjModel, cfg: Cfg):
        self.cfg = cfg
        self._mjx = mjx.put_model(mj_model)
        self.ids = HexapodIds(mj_model)
        # Coerce weights to float: YAML loads bare exponents like "1e-4" (no dot)
        # as strings, which would blow up deep in the reward as str*array.
        self.weights = dataclasses.replace(
            cfg.reward_weights,
            **{f.name: float(getattr(cfg.reward_weights, f.name))
               for f in dataclasses.fields(cfg.reward_weights)},
        )
        self.command = Command.straight(cfg.vx)  # fixed forward command
        self.sim_dt = float(mj_model.opt.timestep)
        self.control_dt = self.sim_dt * cfg.decimation

        # ctrl clip range = each actuated joint's range (position-servo targets).
        jid = mj_model.actuator_trnid[:, 0]
        self._ctrl_lo = jnp.asarray(mj_model.jnt_range[jid, 0])
        self._ctrl_hi = jnp.asarray(mj_model.jnt_range[jid, 1])
        self._nu = int(mj_model.nu)
        self._feet = jnp.asarray(self.ids.feet())

        # Stance pose from the <keyframe name="home">: the env resets here, and
        # actions are commanded as offsets around it (see step). Falls back to
        # qpos0 if the model has no such keyframe.
        try:
            self._home_qpos, self._home_qvel = keyframe(mj_model, "home")
        except KeyError:
            self._home_qpos = jnp.asarray(mj_model.qpos0)
            self._home_qvel = jnp.zeros(int(mj_model.nv))
        self._rest = self._home_qpos[7:]  # per-joint stance angles (action center)
        self.z_nominal = float(self._home_qpos[2])  # standing trunk height, from the model

        # observation size (host-side, once).
        d = mjx.forward(self._mjx, mjx.make_data(self._mjx))
        self._obs_size = int(self._obs(d, jnp.zeros(self._nu)).shape[0])

    # --- Brax Env API ---
    @property
    def observation_size(self) -> int:
        return self._obs_size

    @property
    def action_size(self) -> int:
        return self._nu

    @property
    def backend(self) -> str:
        return "mjx"

    @property
    def dt(self) -> float:
        return self.control_dt

    def reset(self, rng: jax.Array) -> State:
        rng, key = jax.random.split(rng)
        n = self.cfg.reset_joint_noise
        qpos = self._home_qpos.at[7:].add(
            jax.random.uniform(key, (self._nu,), minval=-n, maxval=n)
        )
        data = mjx.forward(
            self._mjx,
            mjx.make_data(self._mjx).replace(qpos=qpos, qvel=self._home_qvel),
        )

        # seed last_contact from the actual stance (avoids a spurious step-0 touchdown).
        contact = foot_contacts(data, self._feet, self.cfg.contact_force_thresh)
        foot_state = FootState(air_time=jnp.zeros(6), last_contact=contact)
        last_action = jnp.zeros(self._nu)

        obs = self._obs(data, last_action)
        metrics = {f"reward/{k}": jnp.zeros(()) for k in TERM_NAMES}
        info = {"foot_state": foot_state, "last_action": last_action, "rng": rng}
        return State(data, obs, jnp.zeros(()), jnp.zeros(()), metrics, info)

    def step(self, state: State, action: jax.Array) -> State:
        # Actions are offsets around the stance pose, so a zero/neutral action
        # holds the home keyframe instead of driving the joints to 0 (legs flat).
        ctrl = jnp.clip(
            self._rest + self.cfg.action_scale * action, self._ctrl_lo, self._ctrl_hi
        )

        def sim(d, _):
            return mjx.step(self._mjx, d.replace(ctrl=ctrl)), None

        data, _ = jax.lax.scan(sim, state.pipeline_state, None, length=self.cfg.decimation)

        total, terms, new_foot = compute_reward(
            data,
            cmd=self.command,
            ids=self.ids,
            w=self.weights,
            world_up=_WORLD_UP,
            foot_state=state.info["foot_state"],
            last_action=state.info["last_action"],
            action=action,
            air_time_target=self.cfg.air_time_target,
            contact_force_thresh=self.cfg.contact_force_thresh,
            height_target=self.z_nominal,
            dt=self.control_dt,
        )

        up = projected_gravity(data, self.ids.base, jnp.asarray(_WORLD_UP))[2]
        fell = (data.qpos[2] < self.cfg.z_min_frac * self.z_nominal) | (
            up < self.cfg.up_min
        )

        obs = self._obs(data, action)
        # Merge, don't replace: brax wrappers inject keys (e.g. EvalWrapper's
        # 'reward'), and the scan carry must keep a stable pytree structure.
        metrics = dict(state.metrics)
        metrics.update({f"reward/{k}": terms[k] for k in TERM_NAMES})
        info = {**state.info, "foot_state": new_foot, "last_action": action}
        return state.replace(
            pipeline_state=data,
            obs=obs,
            reward=total,
            done=fell.astype(jnp.float32),
            metrics=metrics,
            info=info,
        )

    def _obs(self, data: mjx.Data, last_action: jax.Array) -> jax.Array:
        """Observation (66): base twist + tilt + command + joint state + last action."""
        base = self.ids.base
        return jnp.concatenate([
            base_lin_vel(data, base),                 # 3
            base_ang_vel(data, base),                 # 3
            projected_gravity(data, base, jnp.asarray(_WORLD_UP)),  # 3 (tilt)
            jnp.array([self.command.v[0], self.command.v[1], self.command.omega[2]]),  # 3
            data.qpos[7:],                            # 18 joint angles
            data.qvel[6:],                            # 18 joint velocities
            last_action,                              # 18
        ])
