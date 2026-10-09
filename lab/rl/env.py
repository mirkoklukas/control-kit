"""Gymnasium environment for the policy test: the spider walking on the floor.

Used by :mod:`.test_policy` (PPO training / eval). Model and config come from
:mod:`.model` / :mod:`.config`; everything here is plain MuJoCo on CPU.

Task: track a fixed forward velocity ``cmd_vx`` (body x) on the flat floor, heading
held (yaw rate 0). Magnets stay off (adhesion ctrl 0) and are not in the action.
Training default ``pad_cells=1``: without adhesion the cells only cost contacts.

Environment (Gymnasium):
- Control at ``policy_hz`` (default 50 Hz): each action is held for
  ``1 / (policy_hz * timestep)`` physics steps. Episodes last ``episode_s``.
- Reset: ``stand`` keyframe + uniform joint noise ``reset_noise``.
- Action (12,) in [-1, 1]: joint position targets = ``stand`` + ``action_scale`` *
  action, clipped to the joint limits, sent to the leg servos.
- Observation, chosen by ``cfg.obs`` (fixed at construction: it sets the network's
  input size):

  - ``"v1"`` (101,), default: after Hwangbo et al. 2019, *Learning Agile and Dynamic
    Motor Skills for Legged Robots* (arXiv:1901.08652). Gravity in the body frame (3),
    base height (1), base linear and angular velocity in the body frame (3 + 3), joint
    positions relative to ``stand`` (12) and velocities (12), joint history (48):
    position error (servo target - joint angle) and velocity 1 and 2 policy steps
    back (the paper: t - 0.01 s and t - 0.02 s; here t - 0.02 s and t - 0.04 s, one
    and two steps at 50 Hz), previous action (12), command (3). New, not in the
    paper: pad face heights above the floor (4).
  - ``"v2"`` (76,): gravity in the body frame (3), base linear velocity (3) and
    angular velocity (3) in the body frame, joint positions relative to ``stand``
    (12) and velocities (12), ankle angles (8), previous action (12), command (3),
    foot (ankle pivot) positions relative to the base in the body frame (12), pad
    face heights above the floor (4), planted flags (4).

Reward (docs/policy-and-reward-terms.md; the terms are pure functions in
:mod:`.rewards`, composed in :meth:`WalkEnv.step`), every term times the policy dt:

    r = w_lin K(lin_sharpness |v_xy - v_hat|) + w_yaw K(yaw_sharpness |w_z|)
        + w_air sum_touchdown (min(t_air, air_max) - air_target)
        - w_torque |tau|^2 - w_action_rate |a_t - a_t-1|^2 - w_joint_speed |qdot|^2
        - w_orient |z_world - z_body| - w_height (d_hat - d)^2
        - w_clear sum_air (h_hat - h_i)^2 |v_t,i|
        - w_drag sum_{touching, not planted} |v_t,i|
        - w_hard_clear sum_{h_i < hard_clear_height} max(0, |v_t,i| - hard_clear_vtol)
        - w_slip sum_planted (|v_t,i| + r_pad |w_n,i|)
        - w_ankle sum_planted sum_ab max(0, |q| - q_safe)^2
        - w_support max(0, min_support - n_planted)
    K(x) = 4 / (e^x + 2 + e^-x)   (1 at x = 0)

The weights are used as given. Changing them over training (a curriculum) is the
caller's job: it pushes an adjusted config into ``env.cfg`` -- e.g. ``test_policy``
evaluates a :class:`.scheduled_config.ScheduledConfig` and sets ``cfg`` via
``set_wrapper_attr``. Setting ``cfg`` re-derives the constants the env computes from
it (command vector, angle thresholds).

Contact history: the env keeps the last ``contact_history`` policy steps of each
foot's pad normal force (``force_hist``), planted state (``planted_hist``) and world
position (``foot_pos_hist``, ankle pivot), and per-foot timers (``air_time``,
``planted_time``); a foot is planted when it carries more than ``contact_force_min``
*and* its pad lies flat (tilt within ``pad_tilt_max_deg``), and touching above
``contact_touch_min``. Touching but not planted (light, or on a tilted pad): the drag
term. The air-time and
support terms read from it.

Termination (cost ``w_term``): trunk touches the floor, or the body tips more than
``tip_deg`` from vertical. Truncation at ``episode_s``.

Frames used below (MuJoCo conventions):
- ``R = xmat[base].reshape(3, 3)`` maps body to world coordinates: its *columns*
  are the body's x/y/z axes in world coordinates; ``R.T`` maps world -> body.
- The base's free joint: ``qvel[0:3]`` is the linear velocity of the body origin in
  *world* coordinates, ``qvel[3:6]`` the angular velocity in the *body* frame.
- Hinge joints (legs, ankles) have one scalar entry in ``qpos`` / ``qvel``; the
  ``*_qadr`` / ``*_vadr`` index arrays below say where.
"""
import math

import gymnasium as gym
import mujoco
import numpy as np

from . import rewards as rw
from .config import MjModelCfg, WalkEnvCfg
from .mjmodel import adhesion_actuators, build, pad_cell_bodies

class WalkEnv(gym.Env):
    """The spider on the floor, tracking a forward velocity. See the module docstring.

    One instance = one simulated robot. ``test_policy`` runs several in parallel
    (one per subprocess) and pushes an adjusted ``cfg`` into each once per PPO iteration
    (its reward schedule; see the module docstring).

    Args:
        mjcfg: the robot model (:class:`.config.MjModelCfg`).
        cfg: task and reward (:class:`.config.WalkEnvCfg`).
        seed: seed for the first ``reset()`` if that call gives none.
    """

    metadata = {"render_modes": []}

    def __init__(self, mjcfg: MjModelCfg, cfg: WalkEnvCfg, seed: int = 0):
        self.mjcfg = mjcfg                                # the robot model
        self.cfg = cfg                                    # task + reward (property: derives constants)

        # --- model and timing ---
        _, self.model = build(mjcfg, write=False)         # robot + floor + keyframes
        m = self.model
        self.data = mujoco.MjData(m)
        # physics steps per policy step (e.g. 10 x 2 ms = 20 ms at 50 Hz)
        # §`policy_hz * timestep` means policy_cycles-per-timestep
        self.n_sub = max(1, round(1.0 / (cfg.policy_hz * mjcfg.timestep)))
        self.dt = self.n_sub * m.opt.timestep            # policy step (s)
        self.max_steps = int(cfg.episode_s / self.dt)    # episode length (policy steps)

        # --- the `stand` pose: reset state and the action's zero point ---
        key = m.key("stand")
        self.qpos0 = key.qpos.copy()                     # full qpos of `stand`
        self.servo0 = key.ctrl[:12].copy()               # leg servo targets of `stand`

        # --- where things are in qpos / qvel / ctrl (index arrays) ---
        # 12 leg joints, leg-major (leg0 j0..j2, leg1 j0..j2, ...): qpos and qvel indices
        self.jnt_qadr = np.array([m.joint(f"leg{i}_j{k}").qposadr[0]
                                  for i in range(4) for k in range(3)])
        self.jnt_vadr = np.array([m.joint(f"leg{i}_j{k}").dofadr[0]
                                  for i in range(4) for k in range(3)])
        # 8 ankle hinges (per foot: a, b): qpos indices
        self.ank_qadr = np.array([m.joint(f"ankle{i}_{s}").qposadr[0]
                                  for i in range(4) for s in "ab"])
        # servo target limits (= joint limits); the first 12 actuators are the servos
        self.ctrl_lo, self.ctrl_hi = m.actuator_ctrlrange[:12, 0], m.actuator_ctrlrange[:12, 1]
        # all adhesion actuators (pad cells of all feet); kept at 0 (magnets off)
        self.adh = np.concatenate([adhesion_actuators(m, i) for i in range(4)])

        # --- bodies and geoms ---
        self.base = m.body("base").id                    # the trunk body (free joint)
        self.trunk = m.geom("base").id                   # the trunk's box geom
        self.floor = m.geom("floor").id                  # (not used yet)
        self.feet = np.array([m.body(f"foot{i}").id for i in range(4)])  # ankle pivots
        self.pads = np.array([m.body(f"pad{i}").id for i in range(4)])   # pad bodies (tilt)
        # body id -> leg index for pad-cell bodies, -1 for every other body. Maps a
        # contact's geom to the foot it belongs to.
        self.body_leg = np.full(m.nbody, -1)
        for i in range(4):
            self.body_leg[pad_cell_bodies(m, i)] = i

        # --- constants for the reward and termination ---
        # height of the ankle pivot above a flat pad's contact face: foot height = pivot
        # height - this
        self.pad_drop = mjcfg.pivot_height + mjcfg.pad_thickness
        self.r_pad = 0.5 * mjcfg.pad_size                  # turns pad yaw rate into edge speed
        # (constants from `cfg` -- q_safe, cos_pad_flat, cos_tip, cmd -- see the cfg setter)

        # --- gym spaces ---
        self.action_space = gym.spaces.Box(-1.0, 1.0, (12,), np.float32)
        # the observation: v1 (after Hwangbo et al. 2019) or v2; see the module docstring
        self._obs = {"v1": self._obs_v1, "v2": self._obs_v2}[cfg.obs]
        n_obs = {"v1": 101, "v2": 76}[cfg.obs]
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (n_obs,), np.float32)

        # --- per-episode state (set properly in reset) ---
        self._seed = seed                  # used for the first reset if none is given
        self.prev_action = np.zeros(12)    # last action, for the action-rate cost and obs
        self.steps = 0                     # policy steps in this episode
        # contact history: the last `contact_history` policy steps of per-foot pad normal
        # force (N, oldest first), planted state and position, and per-foot timers
        self.force_hist = np.zeros((cfg.contact_history, 4))
        self.planted_hist = np.zeros((cfg.contact_history, 4), bool)
        self.foot_pos_hist = np.zeros((cfg.contact_history, 4, 3))  # world, ankle pivots
        self.planted = np.zeros(4, bool)   # foot planted at the latest step
        self.air_time = np.zeros(4)        # time since lift-off (0 while planted), s
        self.planted_time = np.zeros(4)    # time since touchdown (0 while in the air), s
        # joint history (obs v1): [position error, velocity] (24,) at the end of the
        # previous policy step (row 0) and the one before (row 1)
        self.joint_hist = np.zeros((2, 24))

        # scratch buffers for MuJoCo calls that write into an array
        self._vel, self._cf = np.zeros(6), np.zeros(6)

    # ------------------------------------------------------------------ config
    @property
    def cfg(self) -> WalkEnvCfg:
        """Task + reward config. Setting it (e.g. a scheduled, adjusted config pushed by
        the training loop) re-derives the constants computed from it."""
        return self._cfg

    @cfg.setter
    def cfg(self, cfg: WalkEnvCfg):
        self._cfg = cfg
        self.q_safe = math.radians(cfg.ankle_safe_deg)                    # ankle angle without cost
        self.cos_pad_flat = math.cos(math.radians(cfg.pad_tilt_max_deg))  # planted: pad flat
        self.cos_tip = math.cos(math.radians(cfg.tip_deg))
        self.cmd = np.array([cfg.cmd_vx, 0.0, 0.0])   # command: (vx, vy, yaw rate), body frame
        # not re-derived (fixed at construction): policy_hz / episode_s (timing),
        # contact_history (buffer size) and obs (observation layout)

    # ------------------------------------------------------------------ helpers
    def _base_frame(self):
        """The trunk's orientation and velocities.

        Returns:
            ``(R, v_body, w_body)``: rotation matrix (3, 3), body -> world (columns =
            body axes in world coordinates); linear and angular velocity (3,) each,
            in the body frame.
        """
        R = self.data.xmat[self.base].reshape(3, 3)
        v_body = R.T @ self.data.qvel[0:3]   # free joint: linear velocity is in world -> rotate
        w_body = self.data.qvel[3:6]         # free joint: angular velocity already in body
        return R, v_body, w_body

    def _joint_state(self):
        """Leg joint position error (servo target - joint angle) and velocity, (24,)."""
        d = self.data
        q = d.qpos[self.jnt_qadr]
        return np.concatenate([d.ctrl[:12] - q, d.qvel[self.jnt_vadr]])

    def _obs_v1(self):
        """The observation v1 (101,), float32: after Hwangbo et al. 2019
        (arXiv:1901.08652), plus foot heights. Layout: see the module docstring."""
        R, v, w = self._base_frame()
        d = self.data
        return np.concatenate([
            R.T @ np.array([0.0, 0.0, -1.0]),                     # gravity direction, body (3)
            [d.qpos[2]],                                          # base height (1)
            v, w,                                                 # base velocities, body (3 + 3)
            d.qpos[self.jnt_qadr] - self.qpos0[self.jnt_qadr],    # leg joints rel. `stand` (12)
            d.qvel[self.jnt_vadr],                                # leg joint velocities (12)
            self.joint_hist.ravel(),                              # joint history, t-1, t-2 (48)
            self.prev_action,                                     # previous action (12)
            self.cmd,                                             # command (3)
            # new, not in the paper: pad face heights above the floor (4)
            d.xpos[self.feet, 2] - self.pad_drop,
        ]).astype(np.float32)

    def _obs_v2(self):
        """The observation v2 (76,), float32. Layout: see the module docstring."""
        R, v, w = self._base_frame()
        d = self.data
        foot = d.xpos[self.feet]                                  # ankle pivots, world (4, 3)
        return np.concatenate([
            R.T @ np.array([0.0, 0.0, -1.0]),                     # gravity direction, body (3)
            v, w,                                                 # base velocities, body (3 + 3)
            d.qpos[self.jnt_qadr] - self.qpos0[self.jnt_qadr],    # leg joints rel. `stand` (12)
            d.qvel[self.jnt_vadr],                                # leg joint velocities (12)
            d.qpos[self.ank_qadr],                                # ankle angles (8)
            self.prev_action,                                     # previous action (12)
            self.cmd,                                             # command (3)
            ((foot - d.xpos[self.base]) @ R).ravel(),             # feet rel. base, body (12)
            foot[:, 2] - self.pad_drop,                           # pad face above floor (4)
            self.planted,                                         # planted flags (4)
        ]).astype(np.float32)

    def _contact_state(self):
        """Pad contact forces per foot, and whether the trunk touches the floor.

        Loops over MuJoCo's contacts. Only *active* contacts count: with the pads'
        margin, contacts are generated slightly before touching, and those carry no
        force yet (``efc_address < 0``).

        Returns:
            ``(force, trunk)``: per-foot pad normal force (4,), N, summed over that
            foot's contacts; and a bool, true if the trunk geom is in contact.
        """
        m, d = self.model, self.data
        force, trunk, f = np.zeros(4), False, self._cf
        for j in range(d.ncon):
            c = d.contact[j]
            if c.efc_address < 0:          # generated within the margin, not active
                continue
            for b in (m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]):
                if self.body_leg[b] >= 0:  # this side of the contact is a pad cell
                    mujoco.mj_contactForce(m, d, j, f)   # f[0] = normal force
                    force[self.body_leg[b]] += f[0]
            if self.trunk in (c.geom1, c.geom2):
                trunk = True
        return force, trunk

    def _pad_flat(self):
        """(4,) bool: each pad lies flat on the floor -- its face normal within
        ``pad_tilt_max_deg`` of the floor normal.

        The face normal is the pad body's +x axis pointing *out* of the face, i.e. -z
        when the pad lies flat: ``cos(tilt) = -(pad x-axis) . z`` = ``-R_pad[2, 0]``.
        This is the pad's own tilt against the floor, not the ankle angle (which is
        the pad relative to the tibia).
        """
        cos_tilt = -self.data.xmat[self.pads][:, 6]      # xmat row-major: [6] = R[2, 0]
        return cos_tilt > self.cos_pad_flat

    def _planted(self, force):
        """(4,) bool: planted = carrying load (normal force above ``contact_force_min``)
        and the pad flat (:meth:`_pad_flat`)."""
        return (force > self.cfg.contact_force_min) & self._pad_flat()

    def _update_contact_history(self, force):
        """Push this step's foot forces, planted states and positions into the history;
        update the per-foot timers.

        Args:
            force: (4,) per-foot pad normal force this step (N).

        Returns:
            ``(planted, touchdown, air_at_touchdown)``, each (4,): planted = loaded and
            flat (:meth:`_planted`); touchdown = planted now, not at the previous step;
            air_at_touchdown = the air time that just ended (0 where no touchdown).
            "Air" time is the time not planted.
        """
        # shift the history by one row (oldest row out), newest force in the last row
        self.force_hist = np.roll(self.force_hist, -1, axis=0)
        self.force_hist[-1] = force

        planted = self._planted(force)
        self.planted_hist = np.roll(self.planted_hist, -1, axis=0)
        self.planted_hist[-1] = planted
        self.foot_pos_hist = np.roll(self.foot_pos_hist, -1, axis=0)
        self.foot_pos_hist[-1] = self.data.xpos[self.feet]
        touchdown = planted & ~self.planted                       # was in the air, now planted
        air_td = np.where(touchdown, self.air_time, 0.0)          # air time read before reset
        self.air_time = np.where(planted, 0.0, self.air_time + self.dt)
        self.planted_time = np.where(planted, self.planted_time + self.dt, 0.0)
        self.planted = planted
        return planted, touchdown, air_td

    def _foot_vel(self, i):
        """Velocity of foot ``i``'s ankle pivot.

        Args:
            i: leg index (0..3).

        Returns:
            ``(angular, linear)``, (3,) each, in world coordinates.
        """
        # flg_local=0: world frame. MuJoCo returns [angular, linear].
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
                                 self.feet[i], self._vel, 0)
        return self._vel[:3].copy(), self._vel[3:].copy()

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        """Start an episode: `stand` pose with joint noise, magnets off, history cleared.

        Args:
            seed: RNG seed; if None, the constructor's seed is used on the first reset.
            options: unused (Gymnasium API).

        Returns:
            ``(observation, info)`` with an empty ``info``.
        """
        if seed is None and self._seed is not None:
            seed, self._seed = self._seed, None
        super().reset(seed=seed)           # seeds self.np_random
        m, d = self.model, self.data

        # `stand` keyframe, with uniform noise on the 12 leg joints
        mujoco.mj_resetDataKeyframe(m, d, m.key("stand").id)
        d.qpos[self.jnt_qadr] += self.np_random.uniform(-1, 1, 12) * self.cfg.reset_noise
        d.ctrl[:12], d.ctrl[self.adh] = self.servo0, 0.0   # servos hold `stand`, magnets off
        mujoco.mj_forward(m, d)                              # positions, contacts for the state

        self.prev_action = np.zeros(12)
        self.steps = 0
        # contact history: cleared to zeros. `planted` comes from the reset state,
        # so feet already on the ground don't count as a touchdown on the first step.
        force, _ = self._contact_state()
        self.force_hist = np.zeros((self.cfg.contact_history, 4))
        self.planted_hist = np.zeros((self.cfg.contact_history, 4), bool)
        self.foot_pos_hist = np.zeros((self.cfg.contact_history, 4, 3))
        self.planted = self._planted(force)
        self.air_time, self.planted_time = np.zeros(4), np.zeros(4)
        # joint history: both rows = the reset state (no fake jump in the first steps)
        self.joint_hist = np.tile(self._joint_state(), (2, 1))
        return self._obs(), {}

    def step(self, action):
        """Apply one action for one control step, then score the result.

        One control step = the action's servo targets held for ``n_sub`` sim steps
        (``dt`` seconds in total).

        Args:
            action: (12,) in [-1, 1] (clipped): offsets from the `stand` servo targets,
                in units of ``action_scale`` rad.

        Returns:
            A tuple ``(observation, reward, terminated, truncated, info)``, the
            Gymnasium API (>= 0.26):

            observation: the next observation (see :meth:`_obs`).
            reward: the sum of the reward terms for this step (each already times dt).
            terminated: the episode ended by the task itself -- a true terminal
                state. Here: the trunk touches the floor, or the body tips more
                than ``tip_deg``. Nothing comes after it, so the learner uses no
                value beyond this step (the value target is just the reward).
            truncated: the episode was cut off from outside the task -- here the
                time limit, ``episode_s``. The robot could have gone on, so the
                learner bootstraps from the value of the last observation (SB3 does
                this via ``TimeLimit.truncated`` / ``terminal_observation``). Reported
                as ``terminated`` instead, every episode would end in a fake "nothing
                after this" state the policy cannot see coming (it has no clock).
            info: per-step values for logging:
                terms: each reward term, already times dt (dict).
                vx: forward speed, body frame (m/s).
                force: per-foot pad normal force (4,), N.
                planted: per-foot planted, loaded and pad flat (4,) bool.
                foot_pos: world positions of the ankle pivots (4, 3), m.
                air_td: air times of this step's touchdowns (list, s).

        Note:
            ``truncated`` could instead be left to Gymnasium's ``TimeLimit`` wrapper
            (``max_episode_steps``); keeping it here keeps ``episode_s`` in the config.
        """
        cfg, m, d = self.cfg, self.model, self.data

        # -- Joint history ---
        # Current state becomes history and is then updated
        # push the state at the end of the previous step (its targets
        # still in ctrl), before the new action overwrites them. 
        self.joint_hist = np.stack([self._joint_state(), self.joint_hist[0]])

        # --- Act ---
        # Translate action to control (here already done)
        # Set the servo targets, hold them for n_sub physics steps
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        d.ctrl[:12] = np.clip(self.servo0 + cfg.action_scale * a, self.ctrl_lo, self.ctrl_hi)
        # `n_sub = "mj_steps_per_gym_step" = gym_dt/mj_dt `
        for _ in range(self.n_sub): 
            mujoco.mj_step(m, d)
        self.steps += 1

        # --- Observe and Asses the Result ---
        # I.e. body state, contacts, termination
        observation = self._obs()
        R, v, w = self._base_frame()
        force, trunk = self._contact_state()
        feet, touchdown, air_td = self._update_contact_history(force)   # feet = planted
        # foot states: in the air | touching but not planted -- light, or loaded on a
        # tilted pad (drag) | planted: loaded and flat (slip, ankle)
        touching = force > cfg.contact_touch_min
        tipped = R[2, 2] < self.cos_tip        # R[2, 2] = cos(body z-axis, world z)
        terminated = bool(trunk or tipped)
        truncated = self.steps >= self.max_steps

        # --- Reward ---
        # Gather the inputs of the reward terms (the MuJoCo side) ---
        tau = d.actuator_force[:12]                           # servo torques (N m)
        v_t, w_n, h = np.zeros(4), np.zeros(4), np.zeros(4)   # per foot
        for i in range(4):
            wv, lv = self._foot_vel(i)
            v_t[i] = math.hypot(lv[0], lv[1])                 # speed along the floor
            w_n[i] = wv[2]                                    # twist about the floor normal
            h[i] = d.xpos[self.feet[i], 2] - self.pad_drop    # pad face above the floor
        q_ankle = d.qpos[self.ank_qadr].reshape(4, 2)         # (a, b) per foot

        # The reward terms (rewards.py): +w * reward, -w * cost
        terms = {
            # tracking (rewards): forward velocity and yaw rate vs. the command
            "lin": cfg.w_lin * rw.lin_vel_tracking(v[:2], self.cmd[:2], cfg.lin_sharpness),
            "yaw": cfg.w_yaw * rw.yaw_rate_tracking(w[2], self.cmd[2], cfg.yaw_sharpness),
            # regularization (costs): effort, smoothness, posture
            "torque": -cfg.w_torque * rw.torque(tau),
            "action_rate": -cfg.w_action_rate * rw.action_rate(a, self.prev_action),
            "joint_speed": -cfg.w_joint_speed * rw.joint_speed(d.qvel[self.jnt_vadr]),
            "orient": -cfg.w_orient * rw.orientation(R[:, 2]),
            "height": -cfg.w_height * rw.body_height(d.qpos[2], self.mjcfg.stand_height),
            # feet (costs): clearance in the air, drag when touching without load,
            # slip and ankle range when planted
            "clear": -cfg.w_clear * rw.foot_clearance(h, v_t, touching, cfg.clear_height),
            "drag": -cfg.w_drag * rw.foot_drag(v_t, touching, feet),
            "hard_clear": -cfg.w_hard_clear * rw.hard_clearance(
                h, v_t, cfg.hard_clear_height, cfg.hard_clear_vtol),
            "slip": -cfg.w_slip * rw.foot_slip(v_t, w_n, feet, self.r_pad),
            "ankle": -cfg.w_ankle * rw.ankle_range(q_ankle, feet, self.q_safe),
            # gait: feet short of min_support planted (graded); air time per touchdown
            # (can be negative: a step shorter than air_target costs)
            "support": -cfg.w_support * rw.min_support(feet, cfg.min_support),
            "air": cfg.w_air * rw.air_time(touchdown, air_td, cfg.air_target, cfg.air_max),
        }
        # per step -> times dt; plain floats from here on (for logging / the return value)
        terms = {k: float(v_ * self.dt) for k, v_ in terms.items()}
        terms["term"] = -cfg.w_term if terminated else 0.0        # one-time, not times dt
        reward = float(sum(terms.values()))

        self.prev_action = a
        
        info = {"terms": terms, "vx": float(v[0]), "force": force.copy(),
                "planted": feet.copy(), "foot_pos": d.xpos[self.feet].copy(),
                "air_td": air_td[touchdown].tolist()}
        return observation, reward, terminated, truncated, info
