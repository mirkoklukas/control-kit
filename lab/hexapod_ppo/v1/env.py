"""MJX environment for hexapod_ppo v1 -- a Brax ``Env`` around the MJX dynamics.

System-specific wiring: maps the policy's normalized action to position-servo
targets, steps MJX ``decimation`` times per control step, and scores the result
with this experiment's reward. The ids, command, kinematics, and general reward
terms come from the stable core (``controlkit.*``); this file only glues them to
the Brax training API.

``reset``/``step`` stay thin and delegate the two system-specific pieces to
module-level helpers: ``_observe(env, state)`` builds the observation and
``_reward(env, state, action, state_next)`` scores the step (after the physics
step, against the resulting state).

State layout follows Brax's intent:
- ``pipeline_state`` -- physics only, the ``mjx.Data`` (steppable by ``mjx.step``).
- ``obs`` -- a Brax ``Observation`` (here a flat ``jax.Array``).
- ``info`` -- cross-step bookkeeping: ``last_action`` and per-foot ``contact``/``load``.

The command is a fixed forward twist (``Command.straight(vx)``), so it is closed
over rather than carried. ``AutoResetWrapper`` resets ``pipeline_state``/``obs``
but not ``info``, so the ``info`` bookkeeping is stale for one step after an
auto-reset (a negligible transient).
"""
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
from brax.envs.base import Env, State
from brax.training.types import Observation
from mujoco import mjx

from controlkit.utils import (
    base_ang_vel,
    base_lin_vel,
    to_base_frame,
    foot_velocities,
    contact_normal_forces,
    keyframe,
    projected_gravity,
)
from controlkit.hexapod import Command, HexapodIds

from .config import Cfg


def get_contact_loads(data, geoms):
    # (ncon,); 
    normal, g1, g2 = contact_normal_forces(data)

    # Strong contacts: normal force above threshold (inactive contacts have normal 0)
    # (ncon,); inactive contacts have normal 0
    strong = normal > 0.01  

    # Contact detection: is each geom in active contact with normal force above threshold?
    # (n_feet, ncon) -> # (n_feet,)
    involves = (g1[None, :] == geoms[:, None]) | (g2[None, :] == geoms[:, None])  
    contact = jnp.any(involves & strong[None, :], axis=1) 

    load = jnp.sum(jnp.where(involves & strong[None, :], normal[None, :], 0.0), axis=1)

    return contact, load 


# Note `brax.envs.base.Env`:
# > https://github.com/google/brax/blob/main/brax/envs/base.py
class HexapodEnv(Env):
    """Walk-straight hexapod environment (Brax ``Env``, MJX backend)."""

    def __init__(self, mj_model: mujoco.MjModel, cfg: Cfg):
        self.cfg = cfg
        self.model = mjx.put_model(mj_model) 
        self.ids = HexapodIds(mj_model)
        self.command = Command.straight(cfg.vx)  # fixed forward command
        self.sim_dt = float(mj_model.opt.timestep)
        self.control_dt = self.sim_dt * cfg.decimation
        self.world_up = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)
        self.feet = jnp.asarray(self.ids.feet())
        self.feet_bodies = jnp.asarray(self.ids.feet_bodies())

        # # Coerce weights to float: YAML loads bare exponents like "1e-4" (no dot)
        # # as strings, which would blow up deep in the reward as str*array.
        # self.weights = dataclasses.replace(
        #     cfg.reward_weights,
        #     **{f.name: float(getattr(cfg.reward_weights, f.name))
        #        for f in dataclasses.fields(cfg.reward_weights)},
        # )

        # TODO: should that go on HexapodIds? It's a model property, not a system property.
        # ctrl clip range = each actuated joint's range (position-servo targets).
        jid = mj_model.actuator_trnid[:, 0]
        self._ctrl_lo = jnp.asarray(mj_model.jnt_range[jid, 0])
        self._ctrl_hi = jnp.asarray(mj_model.jnt_range[jid, 1])

        # Action size = number of controls (u)
        self._nu = int(mj_model.nu)

        # Standing pose -- reset target, nominal trunk height, and the
        # action-offset center all reference the model <keyframe> named by
        # cfg.keyframe (default "home"); fall back to qpos0 if it doesn't exist.
        try:
            self.initial_qpos, self.initial_qvel = keyframe(mj_model, cfg.keyframe)
        except KeyError:
            self.initial_qpos = jnp.asarray(mj_model.qpos0)
            self.initial_qvel = jnp.zeros(int(mj_model.nv))

        self.initial_joint_angles = self.initial_qpos[7:]  # per-joint stance angles (action-offset center)
        self.stand_height = float(self.initial_qpos[2])  # standing trunk height, from the model

        # observation size: derived from a probe observation, host-side, once.
        # Mapping[str, int] keyed the same way as the observation.
        # observation size (host-side, once).
        self._obs_size = None
        _d = mjx.forward(self.model, mjx.make_data(self.model))
        _obs = _observe(self, State(_d, None, jnp.zeros(()), jnp.zeros(()), {}, {"last_action": jnp.zeros(self._nu), "contact": jnp.zeros(len(self.feet)), "load": jnp.zeros(len(self.feet))}))
        self._obs_size = int(_obs.shape[0])

    # --- Brax Env API ---
    @property
    def observation_size(self):
        return self._obs_size

    # --- Brax Env API ---
    @property
    def action_size(self) -> int:
        return self._nu

    # --- Brax Env API ---
    @property
    def backend(self) -> str:
        return "mjx"

    @property
    def dt(self) -> float:
        return self.control_dt

    # --- Brax Env API ---
    def reset(self, rng: jax.Array) -> State:
        rng, key = jax.random.split(rng)
        n = self.cfg.reset_joint_noise
        qpos = self.initial_qpos.at[7:].add(
            jax.random.uniform(key, (self._nu,), minval=-n, maxval=n)
        )
        data = mjx.forward(
            self.model,
            mjx.make_data(self.model).replace(qpos=qpos, qvel=self.initial_qvel),
        )

        contact, load = get_contact_loads(data, self.feet)
        info = {
            "last_action": jnp.zeros(self._nu),
            "contact": contact,
            "load": load,}

        state = State(data, None, 0.0, 0.0, {}, info)
        obs = _observe(self, state)
        state = state.replace(obs=obs)

        return state

    # --- Brax Env API ---
    def step(self, state: State, action: jax.Array) -> State:
        # Note: 
        # State(pipeline_state, obs, reward, done, metrics, info)
        # > https://github.com/google/brax/blob/main/brax/envs/base.py

        #
        # Pysics step --> New Data:
        #
        # Actions are offsets around the stance pose, so a zero/neutral action
        # holds the home keyframe instead of driving the joints to 0 (legs flat).
        ctrl = jnp.clip(
                    self.initial_joint_angles + self.cfg.action_scale * action, self._ctrl_lo, self._ctrl_hi
                )
        def sim(d, _):
            return mjx.step(self.model, d.replace(ctrl=ctrl)), None
        data, _ = jax.lax.scan(sim, state.pipeline_state, None, length=self.cfg.decimation)
        contact, load = get_contact_loads(data, self.feet)
        up = projected_gravity(data, self.ids.base, self.world_up)[2]
        done = (data.qpos[2] < self.cfg.z_min_frac * self.stand_height) | (
            up < self.cfg.up_min
        )

        info_next = {
            **state.info, 
            "contact": contact, 
            "load": load,         
            "last_action": action  # the action that got me here
        }

        state_next = State(data, None, None, done.astype(jnp.float32), {}, info_next)

        obs = _observe(self, state_next)
        reward, (terms, ) = _reward(self, state, action, state_next)

        metrics_next = {f"reward/{k}": v for k, v in terms.items() if k != "total"}

        state_next = state_next.replace(obs=obs, reward=reward, metrics=metrics_next)

        return state_next



# Note `brax.training.types.Observation`:
# > https://github.com/google/brax/blob/main/brax/training/types.py
# > `Observation = Union[jnp.ndarray, Mapping[str, jnp.ndarray]]`
# > `ObservationSize = Union[int, Mapping[str, Union[Tuple[int, ...], int]]]`
def _observe(env: HexapodEnv, state: State) -> Observation:
    """Observation: base twist + tilt + command + z + joint state + last action."""
    data = state.pipeline_state
    last_action = state.info["last_action"] # the action that got me here
    base = env.ids.base
    tilt = projected_gravity(data, base, env.world_up) # world-up in base frame (tilt)
    return jnp.concatenate([
        base_lin_vel(data, base),                 # 3
        base_ang_vel(data, base),                 # 3
        jnp.array([env.command.v[0], env.command.v[1], env.command.omega[2]]),  # 3
        tilt,  # 3 (tilt)
        data.qpos[[2]],                            # 1 height
        data.qpos[7:],                            # 18 joint angles
        data.qvel[6:],                            # 18 joint velocities
        last_action,                              # 18
    ])


def _reward(env: HexapodEnv, state: State, action: jax.Array, state_next: State) -> tuple[jax.Array, Any]:
    """Score one control step against the post-step ``state_next`` (its physics +
    per-foot ``contact``/``load``), using ``state`` for the *previous* step's
    ``last_action``/``contact`` and ``action`` for the action just applied.
    Returns ``(total, (terms,))`` where ``terms`` is the weighted per-term dict
    (plus ``"total"``) summing to ``total``.
    """
    ids = env.ids
    cmd = env.command
    cfg = env.cfg
    ws = cfg.reward_weights
    last_action = state.info["last_action"]
    world_up = env.world_up
    data = state_next.pipeline_state
    contact = state_next.info["contact"]
    last_contact = state.info["contact"]
    load = state_next.info["load"]

    # These are quantities in the base frame 
    # (heading-invariant) that the reward terms need.
    omega = base_ang_vel(data, ids.base)
    v     = base_lin_vel(data, ids.base)
    up    = to_base_frame(data, ids.base, world_up)

    COMMAD_SIGMA = 0.5

    rs = {}

    # TERMs: Velocity Command (xy-plane)
    # (base frame, heading-invariant)
    v_cmd      = cmd.v[:2]
    dotyaw_cmd = cmd.omega[2] 
    dotyaw = omega[2]
    lin_vel = jnp.exp(-jnp.sum((v_cmd - v[:2]) ** 2) / COMMAD_SIGMA**2)
    ang_vel = jnp.exp(-((dotyaw_cmd - dotyaw) ** 2) / COMMAD_SIGMA**2)
    rs["lin_vel"] = lin_vel
    rs["ang_vel"] = ang_vel

    # TERM: Base Height Keeping
    # --- base height keeping: dense "alive" bonus, 1.0 at the target trunk height,
    # falling off to 0 as it sinks. Disabled (0) when no height_target is given. ---
    z_target = env.stand_height
    z = data.qpos[2]
    val = jnp.clip(z - z_target, min=-jnp.inf, max=0.0)
    base_height = jnp.exp(-(val ** 2) / (0.25*z_target)**2)
    rs["base_height"] = base_height

    # TERMs: Stability Penalties
    lin_vel_z = v[2] ** 2
    ang_vel_xy = omega[0] ** 2 + omega[1] ** 2
    orientation = jnp.sum(up[:2] ** 2)
    rs["lin_vel_z"] = - lin_vel_z
    rs["ang_vel_xy"] = - ang_vel_xy
    rs["orientation"] = - orientation

    # TERM: Action Smoothness
    # --- action smoothness ---
    action_rate = jnp.mean((action - last_action) ** 2)
    rs["action_rate"] = - action_rate

    # TERMS: Effort / Regularization over the Leg DOFs
    leg_dofs = ids.leg_dofs()
    torques = jnp.mean(data.qfrc_actuator[leg_dofs] ** 2)
    dof_acc = jnp.mean(data.qacc[leg_dofs] ** 2)
    rs["torques"] = - torques
    rs["dof_acc"] = - dof_acc

    # TERM: Load distribution
    rs["load_var"] = - jnp.var(load, where=(contact & last_contact))

    # TERM: Slip
    feet_vel  = foot_velocities(data, env.feet, env.feet_bodies, ids.base)
    feet_slip = jnp.sum(jnp.linalg.norm(feet_vel, axis=-1) * load)
    rs["feet_slip"] = - feet_slip

    # Weigh and Assemble
    rs = {k: v * getattr(ws, k, 1.0) for k, v in rs.items()}
    total = sum(rs.values())
    rs["total"] = total


    # TODO: We should add some stability terms.

    return total, (rs,)

