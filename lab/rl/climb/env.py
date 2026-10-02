"""Gymnasium environment for the climb experiments: the spider with switchable magnetic
feet, walking on the floor under (optionally) tilted gravity.

Copied from ``lab/rl/env.py`` (``WalkEnv``, obs v1) on 2026-10-01 and extended; see
``docs/climb-env-design.md`` (``lab/rl/docs``). Used by :mod:`.train`. Plain MuJoCo on CPU.

Task: track a fixed forward velocity ``cmd_vx`` (body x) along the floor, heading held
(yaw rate 0). The floor stays the world xy-plane; with ``gravity_random`` the gravity
direction is redrawn at every reset (a tilted floor, a wall, a ceiling).

Environment (Gymnasium):
- Control at ``policy_hz`` (default 50 Hz): each action is held for
  ``1 / (policy_hz * timestep)`` physics steps. Episodes last ``episode_s``.
- Reset: gravity drawn (if ``gravity_random``: tilt from the floor normal ~ U[0,
  ``gravity_tilt_max_deg``], azimuth ~ U[0, 2 pi)), ``stand`` keyframe + uniform joint
  noise ``reset_noise``, magnets on if ``magnet_start_on``, then ``settle_s`` of sim
  with the servos holding ``stand`` under normal gravity (the noise can lift pads out of
  the contact margin), then the drawn gravity is switched on. The phase clock starts at
  a random phase.
- Action (16,) in [-1, 1]:
  - [0:12] joint position targets = ``stand`` + ``action_scale`` * action, clipped to
    the joint limits, sent to the leg servos.
  - [12:16] magnet commands, one per foot: on iff > 0. All pad cells of a foot switch
    together (the cells are a sim device; the real foot has one magnet). The actual
    state follows the command after ``magnet_delay_s`` (rounded to control steps); a
    command reverted before then never switches.
  - ``magnet_mode="clock"``: the magnet outputs are ignored; the magnets follow the
    phase clock instead (on in the clock's stance, off in its swing), and the policy
    only learns the legs. The action keeps 16 entries, so a run can later branch with
    ``magnet_mode="policy"`` and hand the magnets over.
- Observation ``"v3"`` (122,): v1 of ``lab/rl/env.py`` (after Hwangbo et al. 2019,
  arXiv:1901.08652) with the hard-coded gravity term replaced and magnet / clock terms
  added. In order: surface normal in the body frame (3), gravity direction in the body
  frame (3), base height (1), base linear and angular velocity in the body frame
  (3 + 3), joint positions relative to ``stand`` (12) and velocities (12), joint
  history (48): position error (servo target - joint angle) and velocity 1 and 2 policy
  steps back, previous action (16), command (3), pad face heights above the floor (4),
  magnet commands (4), magnets actually on (4), attached flags (4), sin / cos of the
  clock phase (2).

Attached (replaces the walk env's *planted*): the magnet is actually on, at least
``attach_cells_min`` pad cells (0: all N^2) have an active contact, and the pad lies flat
(tilt within ``pad_tilt_max_deg``). Not a force threshold: with the magnet on, the pad
normal force includes the adhesion pull (``adhesion_gain / N^2`` per cell), so it no
longer says "loaded". Touching = any pad cell in active contact.

Reward (docs/policy-and-reward-terms.md; pure functions in :mod:`.rewards`), every term
times the policy dt:

    r = w_lin K(lin_sharpness |v_xy - v_hat|) + w_yaw K(yaw_sharpness |w_z|)
        + w_air sum_touchdown (min(t_air, air_max) - air_target)
        + w_phase mean_i 1[foot i matches the clock]
        - w_torque |tau|^2 - w_action_rate |a_t - a_t-1|^2 - w_joint_speed |qdot|^2
        - w_orient |z_world - z_body| - w_height (d_hat - d)^2
        - w_clear sum_air (h_hat - h_i)^2 |v_t,i|
        - w_drag sum_{touching, not attached} |v_t,i|
        - w_hard_clear sum_{h_i < hard_clear_height} max(0, |v_t,i| - hard_clear_vtol)
        - w_slip sum_attached (|v_t,i| + r_pad |w_n,i|)
        - w_ankle sum_attached sum_ab max(0, |q| - q_safe)^2
        - w_support max(0, min_support - n_attached)
        - w_switch sum_i 1[magnet command toggled]
    K(x) = 4 / (e^x + 2 + e^-x)   (1 at x = 0)

"Touchdown" and "air time" are about *attached*: a touchdown is a foot becoming
attached, air time the time it was not. ``z_world`` in the orientation term is the floor
normal, not gravity. Phase clock: leg i's phase is ``(phase + phase_offsets[i]) % 1``,
desired stance while it is below ``duty``.

The weights are used as given; a training schedule pushes an adjusted config into
``env.cfg`` (see ``train.py``). Setting ``cfg`` re-derives the constants the env
computes from it.

Contact history: the last ``contact_history`` policy steps of each foot's pad normal
force (``force_hist``), attached state (``attached_hist``) and world position
(``foot_pos_hist``, ankle pivot), and per-foot timers (``air_time``, ``attached_time``).

Termination (cost ``w_term``): trunk touches the floor, the body tips more than
``tip_deg`` from the floor normal, or fewer than ``min_attached`` feet are attached
(0: off). Truncation at ``episode_s``.

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
from .config import ClimbEnvCfg, MjModelCfg
from .mjmodel import adhesion_actuators, build, gravity_dir, pad_cell_bodies

N_ACT = 16                          # 12 servo targets + 4 magnet commands
N_OBS = {"v3": 122}                 # observation sizes


class ClimbEnv(gym.Env):
    """The spider with magnetic feet, tracking a forward velocity. See the module
    docstring.

    One instance = one simulated robot. ``train`` runs several in parallel (one per
    subprocess) and pushes an adjusted ``cfg`` into each once per PPO iteration.

    Args:
        mjcfg: the robot model (:class:`.config.MjModelCfg`).
        cfg: task and reward (:class:`.config.ClimbEnvCfg`).
        seed: seed for the first ``reset()`` if that call gives none.
    """

    metadata = {"render_modes": []}

    def __init__(self, mjcfg: MjModelCfg, cfg: ClimbEnvCfg, seed: int = 0):
        self.mjcfg = mjcfg                                # the robot model
        self.cfg = cfg                                    # task + reward (property: derives constants)

        # --- model and timing ---
        _, self.model = build(mjcfg, write=False)         # robot + floor + keyframes
        m = self.model
        self.data = mujoco.MjData(m)
        # physics steps per policy step (e.g. 10 x 2 ms = 20 ms at 50 Hz)
        self.n_sub = max(1, round(1.0 / (cfg.policy_hz * mjcfg.timestep)))
        self.dt = self.n_sub * m.opt.timestep            # policy step (s)
        self.max_steps = int(cfg.episode_s / self.dt)    # episode length (policy steps)
        self.n_settle = round(cfg.settle_s / m.opt.timestep)  # sim steps of settling at reset
        # magnet switch delay in control steps: a command takes effect this many steps late
        self.mag_delay = round(cfg.magnet_delay_s / self.dt)

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
        # adhesion actuators per foot (4, N^2): one magnet drives all cells of its foot
        self.adh = np.stack([adhesion_actuators(m, i) for i in range(4)])
        self.n_cells_all = self.adh.shape[1]             # N^2
        self.cells_min = cfg.attach_cells_min or self.n_cells_all

        # --- bodies and geoms ---
        self.base = m.body("base").id                    # the trunk body (free joint)
        self.trunk = m.geom("base").id                   # the trunk's box geom
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
        self.normal = np.array([0.0, 0.0, 1.0])            # floor normal (world)

        # --- gym spaces ---
        self.action_space = gym.spaces.Box(-1.0, 1.0, (N_ACT,), np.float32)
        self._obs = {"v3": self._obs_v3}[cfg.obs]
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (N_OBS[cfg.obs],), np.float32)

        # --- per-episode state (set properly in reset) ---
        self._seed = seed                  # used for the first reset if none is given
        self.g_dir = np.array([0.0, 0.0, -1.0])  # gravity direction (world, unit)
        self.prev_action = np.zeros(N_ACT) # last action, for the action-rate cost and obs
        self.steps = 0                     # policy steps in this episode
        self.phase0 = 0.0                  # clock phase at reset, in [0, 1)
        # magnets: command, actual state, and how many steps the command has differed
        # from the actual state (switches once that exceeds `mag_delay`)
        self.mag_cmd = np.zeros(4, bool)
        self.mag_on = np.zeros(4, bool)
        self.mag_pending = np.zeros(4, int)
        # contact history: the last `contact_history` policy steps of per-foot pad normal
        # force (N, oldest first), attached state and position, and per-foot timers
        self.force_hist = np.zeros((cfg.contact_history, 4))
        self.attached_hist = np.zeros((cfg.contact_history, 4), bool)
        self.foot_pos_hist = np.zeros((cfg.contact_history, 4, 3))  # world, ankle pivots
        self.attached = np.zeros(4, bool)  # foot attached at the latest step
        self.air_time = np.zeros(4)        # time since detaching (0 while attached), s
        self.attached_time = np.zeros(4)   # time since attaching (0 while not), s
        # joint history: [position error, velocity] (24,) at the end of the previous
        # policy step (row 0) and the one before (row 1)
        self.joint_hist = np.zeros((2, 24))

        # scratch buffers for MuJoCo calls that write into an array
        self._vel, self._cf = np.zeros(6), np.zeros(6)

    # ------------------------------------------------------------------ config
    @property
    def cfg(self) -> ClimbEnvCfg:
        """Task + reward config. Setting it (e.g. a scheduled, adjusted config pushed by
        the training loop) re-derives the constants computed from it."""
        return self._cfg

    @cfg.setter
    def cfg(self, cfg: ClimbEnvCfg):
        self._cfg = cfg
        self.q_safe = math.radians(cfg.ankle_safe_deg)                    # ankle angle without cost
        self.cos_pad_flat = math.cos(math.radians(cfg.pad_tilt_max_deg))  # attached: pad flat
        self.cos_tip = math.cos(math.radians(cfg.tip_deg))
        self.cmd = np.array([cfg.cmd_vx, 0.0, 0.0])   # command: (vx, vy, yaw rate), body frame
        self.phase_offsets = np.asarray(cfg.phase_offsets, float)
        # not re-derived (fixed at construction): policy_hz / episode_s / settle_s /
        # magnet_delay_s (timing), contact_history (buffer size), attach_cells_min and
        # obs (observation layout)

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

    def _phase(self):
        """The clock phase, in [0, 1)."""
        return (self.phase0 + self.steps * self.dt / self.cfg.phase_period_s) % 1.0

    def _clock_stance(self):
        """(4,) bool: the clock wants foot i in stance (leg phase below ``duty``).
        Leg i's phase is the clock phase shifted by ``phase_offsets[i]``."""
        return (self._phase() + self.phase_offsets) % 1.0 < self.cfg.duty

    def _obs_v3(self):
        """The observation v3 (122,), float32. Layout: see the module docstring."""
        R, v, w = self._base_frame()
        d = self.data
        ph = 2 * math.pi * self._phase()
        return np.concatenate([
            R.T @ self.normal,                                    # surface normal, body (3)
            R.T @ self.g_dir,                                     # gravity direction, body (3)
            [d.qpos[2]],                                          # base height (1)
            v, w,                                                 # base velocities, body (3 + 3)
            d.qpos[self.jnt_qadr] - self.qpos0[self.jnt_qadr],    # leg joints rel. `stand` (12)
            d.qvel[self.jnt_vadr],                                # leg joint velocities (12)
            self.joint_hist.ravel(),                              # joint history, t-1, t-2 (48)
            self.prev_action,                                     # previous action (16)
            self.cmd,                                             # command (3)
            d.xpos[self.feet, 2] - self.pad_drop,                 # pad face above floor (4)
            self.mag_cmd, self.mag_on,                            # magnets: command, actual (4 + 4)
            self.attached,                                        # attached flags (4)
            [math.sin(ph), math.cos(ph)],                         # clock phase (2)
        ]).astype(np.float32)

    def _contact_state(self):
        """Pad contact forces and touching cells per foot, and whether the trunk touches
        the floor.

        Loops over MuJoCo's contacts. Only *active* contacts count: with the pads'
        margin, contacts are generated slightly before touching, and those carry no
        force yet (``efc_address < 0``).

        Returns:
            ``(force, n_cells, trunk)``: per-foot pad normal force (4,), N, summed over
            that foot's contacts (includes the adhesion pull when the magnet is on);
            per-foot number of distinct pad cells in contact (4,); and a bool, true if
            the trunk geom is in contact.
        """
        m, d = self.model, self.data
        force, trunk, f = np.zeros(4), False, self._cf
        cells = set()                      # pad-cell bodies with an active contact
        for j in range(d.ncon):
            c = d.contact[j]
            if c.efc_address < 0:          # generated within the margin, not active
                continue
            for b in (m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]):
                if self.body_leg[b] >= 0:  # this side of the contact is a pad cell
                    mujoco.mj_contactForce(m, d, j, f)   # f[0] = normal force
                    force[self.body_leg[b]] += f[0]
                    cells.add(b)
            if self.trunk in (c.geom1, c.geom2):
                trunk = True
        n_cells = np.bincount(self.body_leg[list(cells)], minlength=4) if cells else np.zeros(4, int)
        return force, n_cells, trunk

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

    def _attached(self, n_cells):
        """(4,) bool: attached = magnet actually on, at least ``cells_min`` pad cells in
        contact, and the pad flat (:meth:`_pad_flat`)."""
        return self.mag_on & (n_cells >= self.cells_min) & self._pad_flat()

    def _set_magnets(self, cmd):
        """New magnet commands -> actual states (after the delay) -> adhesion ctrl.

        Args:
            cmd: (4,) bool, the magnet commands of this control step.
        """
        self.mag_cmd = cmd
        differs = cmd != self.mag_on
        self.mag_pending = np.where(differs, self.mag_pending + 1, 0)
        switch = differs & (self.mag_pending > self.mag_delay)
        self.mag_on = np.where(switch, cmd, self.mag_on)
        self.mag_pending[switch] = 0
        self.data.ctrl[self.adh] = self.mag_on[:, None].astype(float)

    def _update_contact_history(self, force, attached):
        """Push this step's foot forces, attached states and positions into the history;
        update the per-foot timers.

        Args:
            force: (4,) per-foot pad normal force this step (N).
            attached: (4,) bool, per-foot attached this step (:meth:`_attached`).

        Returns:
            ``(touchdown, air_at_touchdown)``, each (4,): touchdown = attached now, not
            at the previous step; air_at_touchdown = the air time that just ended (0
            where no touchdown). "Air" time is the time not attached.
        """
        # shift the history by one row (oldest row out), newest in the last row
        self.force_hist = np.roll(self.force_hist, -1, axis=0)
        self.force_hist[-1] = force
        self.attached_hist = np.roll(self.attached_hist, -1, axis=0)
        self.attached_hist[-1] = attached
        self.foot_pos_hist = np.roll(self.foot_pos_hist, -1, axis=0)
        self.foot_pos_hist[-1] = self.data.xpos[self.feet]
        touchdown = attached & ~self.attached                     # was not attached, now is
        air_td = np.where(touchdown, self.air_time, 0.0)          # air time read before reset
        self.air_time = np.where(attached, 0.0, self.air_time + self.dt)
        self.attached_time = np.where(attached, self.attached_time + self.dt, 0.0)
        self.attached = attached
        return touchdown, air_td

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
        """Start an episode: gravity drawn, `stand` pose with joint noise, magnets set,
        settled, history cleared.

        Args:
            seed: RNG seed; if None, the constructor's seed is used on the first reset.
            options: optional ``{"tilt_deg": ..., "azimuth_deg": ...}``: this episode's
                gravity, instead of drawing it (for checks and evals).

        Returns:
            ``(observation, info)``; ``info["gravity"]`` is this episode's gravity
            vector (world, m/s^2).
        """
        if seed is None and self._seed is not None:
            seed, self._seed = self._seed, None
        super().reset(seed=seed)           # seeds self.np_random
        m, d, cfg, rng = self.model, self.data, self.cfg, self.np_random

        # gravity: tilt from the floor normal ~ U[0, max], azimuth ~ U[0, 360) deg
        tilt, azim = 0.0, 0.0
        if cfg.gravity_random:
            tilt = rng.uniform(0.0, cfg.gravity_tilt_max_deg)
            azim = rng.uniform(0.0, 360.0)
        if options:
            tilt = options.get("tilt_deg", tilt)
            azim = options.get("azimuth_deg", azim)
        # `stand` keyframe, with uniform noise on the 12 leg joints
        mujoco.mj_resetDataKeyframe(m, d, m.key("stand").id)
        d.qpos[self.jnt_qadr] += rng.uniform(-1, 1, 12) * cfg.reset_noise
        d.ctrl[:12] = self.servo0                           # servos hold `stand`
        on = np.full(4, bool(cfg.magnet_start_on))
        self.mag_cmd, self.mag_on, self.mag_pending = on.copy(), on.copy(), np.zeros(4, int)
        d.ctrl[self.adh] = on[:, None].astype(float)
        # settle under normal gravity (servos on `stand`, magnets as set): the joint noise
        # can lift pads out of the contact margin, where they don't attach. Then tilt.
        m.opt.gravity[:] = gravity_dir(0.0)
        for _ in range(self.n_settle):
            mujoco.mj_step(m, d)
        m.opt.gravity[:] = gravity_dir(tilt, azim)
        self.g_dir = m.opt.gravity / np.linalg.norm(m.opt.gravity)
        mujoco.mj_forward(m, d)                              # positions, contacts for the state

        self.prev_action = np.zeros(N_ACT)
        self.prev_action[12:] = np.where(on, 1.0, -1.0)      # the magnet command held so far
        self.steps = 0
        self.phase0 = rng.uniform()
        # contact history: cleared to zeros. `attached` comes from the reset state,
        # so feet already attached don't count as a touchdown on the first step.
        _, n_cells, _ = self._contact_state()
        self.force_hist = np.zeros((cfg.contact_history, 4))
        self.attached_hist = np.zeros((cfg.contact_history, 4), bool)
        self.foot_pos_hist = np.zeros((cfg.contact_history, 4, 3))
        self.attached = self._attached(n_cells)
        self.air_time, self.attached_time = np.zeros(4), np.zeros(4)
        # joint history: both rows = the reset state (no fake jump in the first steps)
        self.joint_hist = np.tile(self._joint_state(), (2, 1))
        return self._obs(), {"gravity": m.opt.gravity.copy()}

    def step(self, action):
        """Apply one action for one control step, then score the result.

        One control step = the action's servo targets and magnet states held for
        ``n_sub`` sim steps (``dt`` seconds in total).

        Args:
            action: (16,) in [-1, 1] (clipped): [0:12] offsets from the `stand` servo
                targets, in units of ``action_scale`` rad; [12:16] magnet commands (on
                iff > 0).

        Returns:
            ``(observation, reward, terminated, truncated, info)``, the Gymnasium API
            (>= 0.26); see ``lab/rl/env.py`` for terminated vs. truncated.

            info: per-step values for logging:
                terms: each reward term, already times dt (dict).
                vx: forward speed, body frame (m/s).
                force: per-foot pad normal force (4,), N (includes the adhesion pull).
                n_cells: per-foot pad cells in contact (4,).
                attached: per-foot attached (4,) bool.
                magnet: per-foot magnet actually on (4,) bool.
                foot_pos: world positions of the ankle pivots (4, 3), m.
                air_td: air times of this step's touchdowns (list, s).
        """
        cfg, m, d = self.cfg, self.model, self.data

        # --- joint history: the state at the end of the previous step (its targets
        # still in ctrl), before the new action overwrites them
        self.joint_hist = np.stack([self._joint_state(), self.joint_hist[0]])

        # --- act: servo targets and magnets, held for n_sub physics steps
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        d.ctrl[:12] = np.clip(self.servo0 + cfg.action_scale * a[:12], self.ctrl_lo, self.ctrl_hi)
        # the clock for this step (before the step counter moves on)
        stance = self._clock_stance()
        if cfg.magnet_mode == "clock":
            mag_cmd, n_switch = stance, 0.0                   # scripted: toggles cost nothing
        else:
            mag_cmd = a[12:] > 0
            n_switch = rw.magnet_switch(mag_cmd, self.mag_cmd)  # toggles, before the update
        self._set_magnets(mag_cmd)
        for _ in range(self.n_sub):
            mujoco.mj_step(m, d)
        self.steps += 1

        # --- observe and assess: contacts, attached, termination
        R, v, w = self._base_frame()
        force, n_cells, trunk = self._contact_state()
        attached = self._attached(n_cells)
        touchdown, air_td = self._update_contact_history(force, attached)
        observation = self._obs()
        touching = n_cells > 0
        tipped = R[2, 2] < self.cos_tip        # R[2, 2] = cos(body z-axis, floor normal)
        detached = attached.sum() < cfg.min_attached          # min_attached 0: never
        terminated = bool(trunk or tipped or detached)
        truncated = self.steps >= self.max_steps

        # --- reward inputs (the MuJoCo side)
        tau = d.actuator_force[:12]                           # servo torques (N m)
        v_t, w_n, h = np.zeros(4), np.zeros(4), np.zeros(4)   # per foot
        for i in range(4):
            wv, lv = self._foot_vel(i)
            v_t[i] = math.hypot(lv[0], lv[1])                 # speed along the floor
            w_n[i] = wv[2]                                    # twist about the floor normal
            h[i] = d.xpos[self.feet[i], 2] - self.pad_drop    # pad face above the floor
        q_ankle = d.qpos[self.ank_qadr].reshape(4, 2)         # (a, b) per foot

        # the reward terms (rewards.py): +w * reward, -w * cost
        terms = {
            # tracking (rewards): forward velocity and yaw rate vs. the command
            "lin": cfg.w_lin * rw.lin_vel_tracking(v[:2], self.cmd[:2], cfg.lin_sharpness),
            "yaw": cfg.w_yaw * rw.yaw_rate_tracking(w[2], self.cmd[2], cfg.yaw_sharpness),
            # regularization (costs): effort, smoothness, posture
            "torque": -cfg.w_torque * rw.torque(tau),
            # servo actions only: magnet toggles have their own cost (switch)
            "action_rate": -cfg.w_action_rate * rw.action_rate(a[:12], self.prev_action[:12]),
            "joint_speed": -cfg.w_joint_speed * rw.joint_speed(d.qvel[self.jnt_vadr]),
            "orient": -cfg.w_orient * rw.orientation(R[:, 2]),
            "height": -cfg.w_height * rw.body_height(d.qpos[2], self.mjcfg.stand_height),
            # feet (costs): clearance in the air, drag when touching but not attached,
            # slip and ankle range when attached
            "clear": -cfg.w_clear * rw.foot_clearance(h, v_t, touching, cfg.clear_height),
            "drag": -cfg.w_drag * rw.foot_drag(v_t, touching, attached),
            "hard_clear": -cfg.w_hard_clear * rw.hard_clearance(
                h, v_t, cfg.hard_clear_height, cfg.hard_clear_vtol),
            "slip": -cfg.w_slip * rw.foot_slip(v_t, w_n, attached, self.r_pad),
            "ankle": -cfg.w_ankle * rw.ankle_range(q_ankle, attached, self.q_safe),
            # gait: feet short of min_support attached (graded); air time per touchdown;
            # the clock's contact pattern
            "support": -cfg.w_support * rw.min_support(attached, cfg.min_support),
            "air": cfg.w_air * rw.air_time(touchdown, air_td, cfg.air_target, cfg.air_max),
            "phase": cfg.w_phase * rw.phase_match(stance, attached, self.mag_on, touching),
            # magnets: toggles cost (EPM switching energy)
            "switch": -cfg.w_switch * n_switch,
        }
        # per step -> times dt; plain floats from here on (for logging / the return value)
        terms = {k: float(v_ * self.dt) for k, v_ in terms.items()}
        terms["term"] = -cfg.w_term if terminated else 0.0        # one-time, not times dt
        reward = float(sum(terms.values()))

        self.prev_action = a

        info = {"terms": terms, "vx": float(v[0]), "force": force.copy(),
                "n_cells": n_cells.copy(), "attached": attached.copy(),
                "magnet": self.mag_on.copy(), "foot_pos": d.xpos[self.feet].copy(),
                "air_td": air_td[touchdown].tolist()}
        return observation, reward, terminated, truncated, info
