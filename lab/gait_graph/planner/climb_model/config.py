"""The climb robot's model config: :class:`MjModelCfg`.

Copied from ``lab/rl/climb/config.py`` (2026-10-05; the planner keeps its own copy, as
``lab/rl/climb`` does). The robot: 4 radial legs (hip yaw / hip pitch / knee), passive
Cardan ankles with square magnetic pads of N x N sphere cells.
"""
from dataclasses import dataclass


@dataclass
class MjModelCfg:
    """The robot model: everything :func:`.mjmodel.build` and :mod:`.poses` need."""
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
    pad_cell_shape: str = "sphere"  # "sphere" (default since 2026-10-02): N x N spheres, 1
                                    # contact each, ~3x faster, same pull-off / shear as boxes
                                    # (test_foot); "box": N x N boxes tiling the pad (4 each)
    pad_sphere_radius: float = 0.0  # sphere cells: radius (m); 0 -> pad_thickness / 2
    pad_thickness: float = 0.006    # (m)
    pivot_height: float = 0.005     # ankle pivot above the pad's back face (m)
    ankle_range_deg: float = 45.0   # +/- hard stop on both ankle axes
    ankle_stiffness: float = 0.05   # centering spring (N m / rad)
    ankle_damping: float = 0.005    # (N m s / rad)
    ankle_armature: float = 1e-4    # regularises the tiny pad inertia
    pad_friction: float = 0.5       # mu (painted steel ~0.3-0.5)

    # --- adhesion (MuJoCo `adhesion` actuator on each pad cell; ctrl in [0, 1]) ---
    adhesion_gain: float = 40.0     # max attraction force per foot (N), split over cells.
                                    # 40 since 2026-10-02: magnet_test holds one leg lifted
                                    # at every gravity direction (30 failed at 90-150 deg)
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
