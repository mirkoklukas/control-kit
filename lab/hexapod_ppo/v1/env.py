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
import dataclasses
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

        # Position-servo gains from config: override the model's kp/kv (set in the
        # <position> default class) so they're a run knob. MuJoCo encodes a position
        # actuator as gainprm[0]=kp, biasprm[1]=-kp, biasprm[2]=-kv (per actuator).
        # Mutates mj_model in place, before put_model bakes it into the MJX model.
        mj_model.actuator_gainprm[:, 0] = cfg.kp
        mj_model.actuator_biasprm[:, 1] = -cfg.kp
        mj_model.actuator_biasprm[:, 2] = -cfg.kv

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

        _empty_state = State(_d, None, jnp.zeros(()), jnp.zeros(()), {}, {"last_action": jnp.zeros(self._nu), "contact": jnp.zeros(len(self.feet), dtype=bool), "load": jnp.zeros(len(self.feet)), "air_time": jnp.zeros(len(self.feet)), "ground_time": jnp.zeros(len(self.feet)), "reward_weights": self._reward_weights()})
        _obs = _observe(self, _empty_state)
        self._obs_size = int(_obs.shape[0])
        _,(_rs, *_) = _reward(self, _empty_state, jnp.zeros(self._nu), _empty_state)
        self._reward_terms = list(_rs.keys())  # the reward term names, for logging

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

    def _reward_weights(self):
        """Per-term reward weights as a dict of jax scalars, seeded from cfg.

        Carried in ``state.info`` so ``_reward`` reads them as *traced* inputs:
        changing a weight value then does not recompile the training step (the
        env_state is a jit input there). NB: built from cfg floats, so reset and
        the eval unroll still recompile per config -- brax calls reset inside the
        eval jit, where the literals bake in.
        """
        return {f.name: jnp.asarray(getattr(self.cfg.reward_weights, f.name), jnp.float32)
                for f in dataclasses.fields(self.cfg.reward_weights)}

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
            "load": load,
            "air_time": jnp.zeros(len(self.feet)),  # per-foot seconds airborne (0 while in contact)
            "ground_time": jnp.zeros(len(self.feet)),  # per-foot seconds planted (0 while airborne)
            "reward_weights": self._reward_weights(),}

        # metrics must carry the same keys step() writes (one per reward term),
        # zeroed, so the scan carry pytree is invariant and brax's EvalWrapper can
        # accumulate them. Keyed off the actual reward terms (self._reward_terms);
        # step() update()s these in place (brax convention) rather than rebuilding.
        metrics = {f"reward/{k}": jnp.zeros(()) for k in self._reward_terms}
        state = State(data, None, 0.0, 0.0, metrics, info)
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
        # per-foot dwell timers: air_time accumulates control_dt while off the
        # ground (0 on contact), ground_time accumulates while planted (0 airborne).
        # The [t] value in state.info is the pre-switch duration; _reward reads it
        # + contact[t+1] to detect touch-down (air->ground) / lift-off (ground->air).
        air_time = jnp.where(contact, 0.0, state.info["air_time"] + self.control_dt)
        ground_time = jnp.where(contact, state.info["ground_time"] + self.control_dt, 0.0)
        up = projected_gravity(data, self.ids.base, self.world_up)[2]
        done = (data.qpos[2] < self.cfg.z_min_frac * self.stand_height) | (
            up < self.cfg.up_min
        )

        info_next = {
            **state.info,
            "contact": contact,
            "load": load,
            "air_time": air_time,
            "ground_time": ground_time,
            "last_action": action  # the action that got me here
        }

        # Carry the incoming metrics dict through and update our term values in
        # place (brax convention, see ant.py): this preserves the key set across
        # the scan carry, including keys the wrappers inject (e.g. EvalWrapper's
        # "reward"), instead of rebuilding the dict and dropping them.
        state_next = State(data, None, None, done.astype(jnp.float32), dict(state.metrics), info_next)

        obs = _observe(self, state_next)
        reward, (terms, ) = _reward(self, state, action, state_next)

        state_next.metrics.update(
            {f"reward/{k}": v for k, v in terms.items()})
        state_next = state_next.replace(obs=obs, reward=reward)

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
        data.qpos[2:3],                            # 1 height
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
    ws = state.info["reward_weights"]  # traced weights carried in state (not env.cfg)
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
    tracking_lin_vel = jnp.exp(-jnp.sum((v_cmd - v[:2]) ** 2) / COMMAD_SIGMA**2)
    tracking_ang_vel = jnp.exp(-((dotyaw_cmd - dotyaw) ** 2) / COMMAD_SIGMA**2)
    rs["tracking_lin_vel"] = tracking_lin_vel
    rs["tracking_ang_vel"] = tracking_ang_vel

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
    action_rate = jnp.sum((action - last_action) ** 2)
    rs["action_rate"] = - action_rate

    # TERMS: Effort / Regularization over the Leg DOFs
    leg_dofs = ids.leg_dofs()
    torques = jnp.sum(data.qfrc_actuator[leg_dofs] ** 2)
    dof_acc = jnp.sum(data.qacc[leg_dofs] ** 2)
    rs["torques"] = - torques
    rs["dof_acc"] = - dof_acc

    # TERM: Load distribution
    # Guard the empty mask: with <2 feet in sustained contact (e.g. a flight
    # phase), jnp.var divides by 0 -> NaN. Variance needs >=2 samples anyway.
    sustained = contact & last_contact
    load_normalized = jnp.where(sustained, load / jnp.sum(load), 0.0)
    load_var = jnp.where(jnp.sum(sustained) > 1, jnp.var(load_normalized, where=sustained), 0.0)
    rs["load_var"] = - load_var

    # TERM: Slip
    feet_vel  = foot_velocities(data, env.feet, env.feet_bodies, ids.base)
    feet_slip = jnp.sum(jnp.linalg.norm(feet_vel, axis=-1) * load)
    rs["feet_slip"] = - feet_slip

    # TERM: Feet air time
    # Reward deliberate steps: at each touch-down, credit the swing duration above
    # air_time_target (encourages real strides, discourages shuffling). Nonzero only
    # on the touch-down step; gated off when no motion is commanded. air_time[t] is
    # the pre-touchdown swing (state.info); +control_dt makes it the full swing at
    # landing. contact = contact[t+1].
    air_time = state.info["air_time"]
    first_contact = (air_time > 0.0) & contact
    cmd_active = jnp.linalg.norm(cmd.v[:2]) > 0.05
    feet_air_time = jnp.sum((air_time + env.control_dt - env.cfg.air_time_target) * first_contact)
    rs["feet_air_time"] = feet_air_time * cmd_active

    # Weigh and Assemble
    rs = {k: v * ws[k] for k, v in rs.items()}
    total = sum(rs.values())
    rs["total"] = total


    # TODO: We should add some stability terms.

    return total, (rs,)


def _make_episode_sampler(key, env: HexapodEnv, make_policy, n_steps, n_stochastic):
    """Build a jitted sampler: ``params -> (det, sto)`` rollouts of ``n_steps``.

    Returns ``(sample, term_names)``. ``sample`` gives one **deterministic** episode
    from a fixed reset (same initial state every call, so it is comparable across
    evals) plus ``n_stochastic`` episodes from the *sampling* policy with independent
    reset + action noise (so they show the exploration spread). All run without
    auto-reset: a fall is just recorded. Each rollout yields
    ``(qpos, qvel, done, reward, terms)`` stacked along the time axis (``terms`` is
    ``[n, len(term_names)]``, the weighted per-term contributions that sum to
    ``reward``, in ``term_names`` order); the stochastic batch is ``vmap``-ed, so its
    leaves carry a leading ``[K]`` axis.

    ``term_names`` is the per-term label order, aligned with the ``terms`` columns.
    Rollout stacks ``state.metrics`` leaves in sorted-key order, so the labels are
    just the env's reward-term names sorted the same way.
    """
    base = key
    term_names = tuple(sorted(env._reward_terms))  # sorted-key order, matches metrics leaves

    def rollout(key, policy):
        reset_key, act_key = jax.random.split(key, 2)
        state = env.reset(reset_key)

        def body(carry, _):
            st, k = carry
            k, ak = jax.random.split(k)
            action, _ = policy(st.obs, ak)
            st = env.step(st, action)
            ps = st.pipeline_state
            terms = jnp.stack(jax.tree_util.tree_leaves(st.metrics))  # sorted-key order
            return (st, k), (ps.qpos, ps.qvel, st.done, st.reward, terms)

        _, out = jax.lax.scan(body, (state, act_key), None, length=n_steps)
        return out

    @jax.jit
    def sample(params):
        det_policy = make_policy(params, deterministic=True)
        sto_policy = make_policy(params, deterministic=False)
        det = rollout(base, det_policy)  # fixed reset
        keys = jax.random.split(jax.random.fold_in(base, 1), n_stochastic)
        sto = jax.vmap(rollout, in_axes=(0, None))(keys, sto_policy)  # leaves: [K, n, ...]
        return det, sto

    return sample, term_names

