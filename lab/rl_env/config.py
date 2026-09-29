"""Shared configs for the rl_env experiments: the spider model, and the parts several
modules read.

A config lives with its only reader; configs read by several modules live here:

- :class:`ModelCfg` -- the robot model (geometry, masses, foot, adhesion, servos, sim,
  keyframes). Read by ``model.py``, ``poses.py``, ``scripted.py``, ``env.py`` and the
  tests.
- :class:`ScriptedCfg` -- timing of the scripted stand-up / hold motions
  (``scripted.py``, used by ``test_tilt``).
- :class:`WalkEnvCfg` -- task, reward and curriculum of the walking env (``env.py``,
  used by ``test_policy``).

Each experiment's own config sits next to its run function and composes these
(``TiltCfg`` in ``test_tilt.py``, ``FootTestCfg`` in ``test_foot.py``, ``PolicyCfg`` in
``test_policy.py``). Nested fields are set with dotted keys: ``model.kp=12``.

All dimensions are placeholders to be tuned after looking at the model.
"""
import ast
from dataclasses import dataclass


@dataclass
class ModelCfg:
    """The robot model: everything :func:`.model.build` and :mod:`.poses` need."""
    # --- robot geometry (spider: 4 radial legs, hip-yaw / hip-pitch / knee-pitch) ---
    num_legs: int = 4
    mount_radius: float = 0.10                    # body centre -> hip-yaw axis (m)
    leg_lengths: tuple = (0.05, 0.15, 0.20)       # coxa, femur, tibia (m)
    joint_limits_deg: tuple = ((-60, 60), (-100, 100), (-160, 160))
    body_half_height: float = 0.03                # trunk box z half-extent (m)
    link_radius: float = 0.012                    # leg capsule radius (m)
    nose_offset: float = 0.005                    # front marker cube (side R/3): top/front this far past the trunk (m)

    # --- masses (kg); total ~ body + num_legs * (coxa + femur + tibia + pad) ---
    mass_body: float = 1.8
    mass_links: tuple = (0.04, 0.08, 0.08)        # coxa, femur, tibia
    mass_pad: float = 0.05

    # --- foot: passive Cardan ankle + square pad (neutral: pad face _|_ tibia) ---
    pad_size: float = 0.04          # pad side length (m)
    pad_cells: int = 3              # pad = N x N cell bodies, each with its own adhesion
    pad_thickness: float = 0.006    # (m)
    pivot_height: float = 0.005     # ankle pivot above the pad's back face (m)
    ankle_range_deg: float = 45.0   # +/- hard stop on both ankle axes
    ankle_stiffness: float = 0.05   # centering spring (N m / rad)
    ankle_damping: float = 0.005    # (N m s / rad)
    ankle_armature: float = 1e-4    # regularises the tiny pad inertia
    pad_friction: float = 0.5       # mu (painted steel ~0.3-0.5)

    # --- adhesion (MuJoCo `adhesion` actuator on each pad cell; ctrl in [0, 1]) ---
    adhesion_gain: float = 30.0     # max attraction force per foot (N), split over cells
    adhesion_margin: float = 0.002  # pad geom margin: adhesion acts within this gap (m)

    # --- leg servos ---
    kp: float = 10.0                # position gain (N m / rad); was 20 -- slower, softer servo
    kv: float = 1.0                 # velocity gain (N m s / rad); was 0.5 -- kv/kp ~ 0.1 s
    forcerange: tuple = (-5.0, 5.0) # torque limit (N m)
    joint_damping: float = 0.05
    armature: float = 0.005

    # --- sim ---
    timestep: float = 0.002
    integrator: str = "implicitfast"

    # --- keyframes: feet planted at stand_foot_radius; body raised from rest to stand ---
    stand_foot_radius: float = 0.28 # body centre -> foot, horizontal (m)
    stand_height: float = 0.16      # base (mount plane) height in `stand` (m)
    rest_clearance: float = 0.002   # trunk-bottom gap above ground in `rest` (m)


@dataclass
class ScriptedCfg:
    """Timing of the scripted motions in :mod:`.scripted`."""
    # --- rest -> stand -> rest ---
    transition_time: float = 2.0    # per transition (s)
    hold_time: float = 1.0          # pause at each end (s)
    # --- hold / sweep (stand under tilted gravity, adhesion on) ---
    hold_duration: float = 3.0      # hold at the final tilt (s)
    sweep_time: float = 6.0         # sweep: ramp 0 -> tilt over this (s)


@dataclass
class WalkEnvCfg:
    """Task, reward and curriculum of :class:`.env.WalkEnv` (walk forward on the floor)."""
    # --- task / control ---
    cmd_vx: float = 0.15            # forward speed command (m/s), body x
    policy_hz: float = 50.0         # control rate; physics substeps = 1 / (policy_hz * timestep)
    episode_s: float = 10.0         # episode length (s)
    action_scale: float = 0.3       # action in [-1, 1] -> joint target offset from `stand` (rad)
    reset_noise: float = 0.05       # uniform joint-angle noise at reset (rad)

    # --- reward weights (docs/policy-and-reward-terms.md); every term is multiplied by
    # the policy dt. Values at full strength; a training schedule (``_schedule`` in the
    # experiment config, see scheduled_config.py) may scale them over training. ---
    w_lin: float = 5.0              # R_lin  forward-velocity tracking (reward); kernel peaks
                                    # at 1 (the paper's 10 is for its 0.25-peak kernel). 2.5
                                    # was below the gait costs: runs settled on standing
    lin_sharpness: float = 40.0     # R_lin = K(lin_sharpness * err). Standing still at
                                    # cmd 0.1 m/s earns 42% of the max at 20, 7% at 40
    w_yaw: float = 3.0              # R_yaw  yaw-rate tracking (reward); paper's 6 / 4 = 1.5,
                                    # doubled (2026-09-27) to hold the heading better
    yaw_sharpness: float = 1.0      # R_yaw = K(yaw_sharpness * err), err in rad/s
    w_torque: float = 0.005         # C_tau  ||tau||^2
    w_action_rate: float = 0.5      # C_smooth, on actions: ||a_t - a_t-1||^2
    w_joint_speed: float = 0.03     # C_phidot ||qdot||^2
    w_orient: float = 0.4           # C_orient ||n - z_body||
    w_height: float = 50.0          # C_dist (d_hat - d)^2; doc's 1.0 is negligible in m^2
    w_clear: float = 100.0          # C_clear sum_swing (h_hat - h)^2 |v_t|; doc's 0.1 negligible
    w_slip: float = 5.0             # C_slip sum_planted |v_t| + r_pad |w_n|; paper 2, raised
                                    # (2026-09-27): planted pads were skating ~0.1 m/s
    w_drag: float = 2.0             # C_drag sum_{touching, not planted} |v_t| (new, see doc)
    w_hard_clear: float = 0.0       # C_hard_clear sum_{h < hard_clear_height}
                                    # max(0, |v_t| - hard_clear_vtol): low feet stay put
    w_ankle: float = 5.0            # C_ankle sum_contact max(0, |q| - q_safe)^2
    w_air: float = 20.0             # R_air: per touchdown, (min(t_air, air_max) - air_target);
                                    # times dt like all terms
    w_support: float = 5.0          # C_support: per foot short of min_support planted feet
                                    # (graded); doc: 20, on the wall
    w_stall: float = 0.0            # C_stall ramp(t_behind): t_behind = distance behind the
                                    # minimum pace stall_frac |cmd_xy|, in seconds at that pace
    w_term: float = 50.0            # termination cost (trunk contact / tipped over). The
                                    # paper's 1 would make sitting down (ending the episode)
                                    # cheaper than an episode of early exploration costs

    # --- reward parameters ---
    clear_height: float = 0.06      # h_hat (m)
    hard_clear_height: float = 0.01 # h_min: feet below this should not move along the floor (m)
    hard_clear_vtol: float = 0.01   # speed allowed below h_min (m/s)
    ankle_safe_deg: float = 30.0    # q_safe
    air_target: float = 0.25        # t_hat: desired air time per step (s)
    air_max: float = 0.5            # air time counted at most this (no lingering legs)
    min_support: int = 3            # crawl: at most one foot in the air
    stall_frac: float = 0.5         # minimum pace, as a fraction of |cmd_xy|
    stall_grace: float = 0.5        # time behind the pace without cost (s)
    stall_ramp: float = 1.0         # then the cost rises to w_stall over this (s)

    # --- contact history ---
    contact_force_min: float = 1.0  # a foot is planted (loaded) above this normal force (N)
    contact_touch_min: float = 0.05 # a foot is touching above this normal force (N)
    pad_tilt_max_deg: float = 3.0   # planted also needs the pad flat: its face normal within
                                    # this of the surface normal. 3 deg lifts the far edge of a
                                    # 40 mm pad ~2 mm (the contact margin); loaded pads sit at
                                    # ~0.3 deg (median), 99% < 6 deg (measured 2026-09-27).
                                    # Touching but not planted (light or tilted) -> drag
    contact_history: int = 100      # contact-force history length (policy steps; 2 s at 50 Hz)

    # --- termination ---
    tip_deg: float = 60.0           # terminate when body z is further than this from vertical


def parse_overrides(cls, argv: list[str], extra: dict) -> tuple[object, dict]:
    """``key=val`` args -> (config of class ``cls``, extras).

    For the tests that are not runkit experiments (``test_tilt``, ``test_foot``). Keys
    are fields of ``cls`` -- dotted for nested configs (``model.kp=12``) -- or keys of
    ``extra``. Built with runkit's parser, so the rules match runkit experiments; tuple
    values are written as python literals (``model.forcerange=(-3,3)``).

    Args:
        cls: the config dataclass to build.
        argv: ``key=val`` strings.
        extra: non-config keys with their defaults; values are cast to the default's
            type (``None`` defaults are parsed as float).

    Returns:
        ``(cfg, extras)`` with ``extras`` a copy of ``extra``, updated.
    """
    from runkit.config import build_cfg, parse_overrides as runkit_parse

    out, tokens = dict(extra), []
    for a in argv:
        k, v = a.split("=", 1)
        if k in out:
            d = out[k]
            out[k] = v if isinstance(d, str) else float(v) if d is None else type(d)(v)
        else:
            tokens.append(a)
    overrides = runkit_parse(tokens)
    _literal_tuples(overrides)
    try:
        return build_cfg(cls, overrides), out
    except (ValueError, TypeError) as e:
        raise SystemExit(str(e)) from None


def _literal_tuples(d: dict) -> None:
    """In place: string values written as tuples (``"(-3,3)"``) -> tuples."""
    for k, v in d.items():
        if isinstance(v, dict):
            _literal_tuples(v)
        elif isinstance(v, str) and v.startswith("("):
            d[k] = ast.literal_eval(v)

