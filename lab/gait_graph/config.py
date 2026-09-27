"""Knobs for the gait-graph playground: robot, terrain, validity, sampling, feel."""
from dataclasses import dataclass

import jax.numpy as jnp


@dataclass
class Cfg:
    # --- robot geometry (same 4x4-DOF radial robot as lab.controller) ---
    num_legs: int = 4
    mount_radius: float = 0.20
    leg_lengths: tuple = (0.05, 0.05, 0.2, 0.3)
    joint_limits_deg: tuple = ((-90, 90), (-180, 180), (-150, 150), (-150, 150))
    body_half_height: float = 0.05
    link_radius: float = 0.02
    foot_radius: float = 0.02

    # --- initial body pose ---
    body_height: float = 0.25

    # --- terrain: axis-aligned boxes, (center, half). The first is the ground slab,
    # its top face at z = 0. ---
    boxes: tuple = (
        ((0.0, 0.0, -0.05), (3.0, 3.0, 0.05)),     # ground
        ((0.9, 0.0, 0.10), (0.25, 0.6, 0.10)),     # step, 20 cm
        ((1.6, 0.0, 0.25), (0.25, 0.6, 0.25)),     # block, 50 cm
        ((0.2, 0.9, 0.35), (0.6, 0.08, 0.35)),     # wall (side faces are footholds too)
    )

    # --- footholds ---
    num_footholds: int = 60_000   # area-uniform samples over all box faces (about half survive filtering)
    edge_margin: float = 0.03     # drop samples closer than this to a face edge
    k_per_leg: int = 1024         # candidate footholds per leg (from its shoulder's reach ball)

    # --- validity ---
    ankle_limit_deg: float = 45.0      # max angle between contact vector and normal
    min_foot_separation: float = 0.08  # planted feet must be at least this far apart
    collision_delta: float = 0.02      # spacing of the robot's collision cloud
    foot_clearance_radius: float = 0.06  # ignore terrain contact this close to a planted foot
    terrain_margin: float = 0.0        # extra keep-out from the terrain (on top of link radius)

    # --- mass model (densities as lab.controller; used by the force scoring) ---
    body_density: float = 300.0
    leg_density: float = 400.0
    foot_density: float = 1000.0

    # --- scoring (see scoring.py): score = wf*exp(-|f|/W) + wt*exp(-|tau|/(W*l)) + ws*sigma_min
    # with W the robot's weight and l its total leg length ---
    w_force: float = 1.0
    w_torque: float = 1.0
    w_sigma: float = 1.0

    # --- sampling ---
    reach_samples: int = 16       # plant attempts per (foothold, leg) in the reach test
    tries: int = 10_000           # parallel full-stance attempts per sample
    track_yaws: int = 9           # hip-yaw candidates when re-solving a planted leg
    track_yaw_span_deg: float = 20.0

    # --- aiming a leg (sampling weight exp(-d^2 / sigma^2) to a movable sphere) ---
    aim_sigma: float = 0.05       # initial sigma (m)
    aim_sigma_step: float = 1.25  # L1 / R1 divide / multiply sigma by this
    aim_sigma_range: tuple = (0.01, 0.5)
    aim_marker_max: float = 0.02  # marker radius of the most likely foothold (m)
    aim_marker_min: float = 0.004 # marker radii are clipped below at this (m)

    # --- controller feel (as lab.controller) ---
    v_xy: float = 0.25
    v_z: float = 0.15
    w_pitch: float = 1.2
    w_yaw: float = 1.2
    xy_range: float = 3.0
    z_range: tuple = (0.05, 1.0)
    pitch_range: float = float(jnp.pi / 2)
    control_hz: float = 60.0
