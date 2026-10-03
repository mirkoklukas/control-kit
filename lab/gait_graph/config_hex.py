"""Knobs for the hexapod (6x3-DOF radial) variant of the gait-graph playground.

Same as :class:`.config.Cfg` except the robot: six legs (shoulders at 30, 90, ...,
330 deg) of hip-yaw / hip-pitch / knee-pitch. ``track_yaws`` /
``track_yaw_span_deg`` are unused (a 3-DOF leg has no yaw to search over).
"""
from dataclasses import dataclass

from .config import Cfg


@dataclass
class HexCfg(Cfg):
    num_legs: int = 6
    mount_radius: float = 0.14
    leg_lengths: tuple = (0.05, 0.2, 0.3)
    joint_limits_deg: tuple = ((-90, 90), (-150, 150), (-150, 150))
    body_half_height: float = 0.035
    link_radius: float = 0.02     # coxa and femur
    tibia_radius: float = 0.015

    # The force model (controlkit.forces.model_from_robot) builds the body as the
    # mounts' bounding box; the hexagon has 0.75 of its area. Statics only see the
    # body's mass and centre of mass, so scaling the density gives the hexagon's
    # mass at 300 kg/m^3 exactly.
    body_density: float = 0.75 * 300.0

    # terrain as Cfg on a 10 x 10 m ground (Cfg: 6 x 6), plus a 2 m wall flush
    # behind the 50 cm block (its front face at the block's back face, x = 1.85)
    boxes: tuple = (
        ((0.0, 0.0, -0.05), (5.0, 5.0, 0.05)),     # ground
        *Cfg.boxes[1:],
        ((1.93, 0.0, 1.0), (0.08, 0.6, 1.0)),      # back wall, 2 m
    )
    num_footholds: int = 150_000  # Cfg's density (60k on its terrain) on the bigger ground

    # the body target: x/y over the whole ground, z unconstrained
    xy_range: float = 5.0
    z_range: tuple = (float("-inf"), float("inf"))
