"""Config for the interactive controller experiment.

One dataclass of knobs. The controller-feel section (edit rates + limits) is what
this first slice is for tuning; the robot / model section defines the welded
actuated robot the later MPC slice drives.
"""
from dataclasses import dataclass, field
import jax.numpy as jnp

@dataclass
class Cfg:
    # --- robot geometry (the draft's 4x4-DOF radial robot) ---
    num_legs: int = 4
    mount_radius: float = 0.20
    leg_lengths: tuple = (0.05, 0.05, 0.2, 0.3)
    joint_limits_deg: tuple = ((-90, 90), (-180, 180), (-150, 150), (-150, 150))

    # --- rest posture (feet planted radially, body level) ---
    body_height: float = 0.20     # nominal trunk height (m)
    foot_radius: float = 0.40     # radial distance of the planted feet (m)

    # --- actuated model (a proper damped, force-limited servo) ---
    # Coupled to noise_sigma: the MPPI's target exploration must not saturate
    # forcerange, or the servo is bang-bang and the planner can't rank candidates.
    # Keep kp * noise_sigma well under forcerange (here 200 * 0.01 = 2 << 8).
    kp: float = 300.0            # position-servo gain
    kv: float = 8.0              # servo velocity gain (damps the servo, not the joint)
    forcerange: tuple = (-26.0, 26.0)  # actuator force cap (rad-> N*m); also limits foot drag
    joint_damping: float = 0.4   # light passive damping (kv does the heavy lifting)
    armature: float = 0.008      # reflected rotor inertia (stabilises the stiff servo)
    frictionloss: float = 0.02   # joint dry friction
    body_half_height: float = 0.05  # trunk box half-thickness (matches the target ghost)
    body_density: float = 300.0  # kg/m^3 (default 1000 makes the ~8 L trunk far too heavy)
    leg_density: float = 400.0
    # A near-massless foot welded to the world conditions the constraint badly;
    # this makes the foot mass comparable to the shank (~0.17 kg), which helps the
    # weld hold more than solver tuning does.
    foot_density: float = 1000.0
    integrator: str = "implicitfast"  # stiff servos are unstable under the default Euler
    timestep: float = 0.004      # sim dt (250 Hz)
    # timestep: float = 0.008      # sim dt (250 Hz)

    # --- solver (a stiff closed kinematic loop wants a tight Newton solve) ---
    solver: str = "Newton"
    iterations: int = 100
    tolerance: float = 1e-10

    # --- foot welds (feet pinned to the ground; the whole point of M1b) ---
    weld_feet: bool = True
    # torquescale scales the weld's rotational constraint: 0 = a point/position pin
    # (feet can still pivot), 1 = a rigid 6-DOF weld.
    weld_torquescale: float = 0.0
    # solref[0] must stay >= 2*timestep or the constraint goes unstable (lower the
    # timestep for more stiffness). solimp: high dmax for a tight hold.
    weld_solref: tuple = (0.008, 1.0)
    weld_solimp: tuple = (0.9, 0.999, 0.001, 0.5, 2.0)

    # --- controller feel: edit rates at full stick deflection (TUNE HERE) ---
    v_xy: float = 0.25            # body xy speed (m/s)
    v_z: float = 0.15             # body z speed (m/s), on the d-pad
    w_pitch: float = 1.2          # pitch rate (rad/s)
    w_yaw: float = 1.2            # yaw rate (rad/s)

    # --- target limits (clamp the box to a sane working envelope) ---
    xy_range: float = 0.50        # |x|,|y| max offset from rest (m)
    z_range: tuple = (0.05, 0.5)  # absolute body-z bounds (m)
    pitch_range: float = jnp.pi/2      # |pitch| max (rad); yaw is unbounded

    # --- loop ---
    control_hz: float = 60.0

    # --- MPPI regulator (M1b): chase the body target through the leg servos ---
    horizon: int = 4          # T control steps planned ahead
    decimation: int = 8        # sim steps per control step (dt 0.004 -> ~62 Hz control)
    samples: int = 512         # N candidate rollouts per tick
    lam: float = 0.3           # softmax temperature (lower = greedier; too low commits to noise)
    noise_sigma: float = 0.01  # exploration std on servo targets (rad); kp*this << forcerange
    w_pos: float = 300.0       # body position error weight (error is in metres^2)
    w_ori: float = 200.0        # body orientation error weight (quaternion, in [0,1])
    w_vel: float = 0.2         # velocity penalty (damps thrashing; over all dofs)
    w_reg: float = 0.0         # pull servo targets toward rest (regularizer, graceful reach)
    w_ctrl: float = 0.0        # control-smoothness penalty (per-step target change)
