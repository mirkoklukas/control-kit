"""The target box: a body pose the controller sculpts, rate-based.

Stick deflection is a *velocity* on the target, integrated each tick, so holding
a stick slews the target and releasing holds it. State is a minimal
``[x, y, z, pitch, yaw]`` (roll parked at 0 for now); :meth:`TargetBox.pose`
hands MuJoCo the mocap ``pos`` + ``quat``.

Mapping (raw sticks: +x right, +y down):
- left stick  -> body x / y along the current heading (up = forward, right = right)
- right stick -> pitch / yaw  (up = pitch up, right = yaw right)
- R2 / L2 triggers -> +z / -z  (proportional)
"""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation

from .config import Cfg
from .input import ControllerState


class TargetBox:
    """Mutable body-pose target, integrated from controller input.

    Args:
        cfg: edit rates + limits + the rest height (the initial z).
    """

    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self.reset()

    def reset(self):
        """Snap the target back to the rest pose (origin xy, level, rest height)."""
        c = self.cfg
        self.x = 0.0
        self.y = 0.0
        self.z = c.body_height
        self.pitch = 0.0
        self.yaw = 0.0

    def update(self, s: ControllerState, dt: float):
        """Integrate one tick of controller input into the target, then clamp.

        Args:
            s: current controller state.
            dt: seconds since the last update.
        """
        c = self.cfg
        # Update yaw first, so the left stick drives along the *current* heading.
        # Right stick: right -> yaw right (unbounded), up -> pitch up.
        self.yaw += -s.rx * c.w_yaw * dt
        self.pitch += -s.ry * c.w_pitch * dt

        # Left stick, resolved into the yawed heading frame: up -> forward, right
        # -> the robot's right. heading = (cos yaw, sin yaw), right = (sin yaw, -cos yaw).
        fwd = -s.ly * c.v_xy * dt
        side = s.lx * c.v_xy * dt
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        self.x += fwd * cy + side * sy
        self.y += fwd * sy - side * cy

        # Triggers set z: R2 -> +z, L2 -> -z (proportional to squeeze).
        self.z += (s.r2 - s.l2) * c.v_z * dt

        self.x = float(np.clip(self.x, -c.xy_range, c.xy_range))
        self.y = float(np.clip(self.y, -c.xy_range, c.xy_range))
        self.z = float(np.clip(self.z, c.z_range[0], c.z_range[1]))
        self.pitch = float(np.clip(self.pitch, -c.pitch_range, c.pitch_range))

    def pose(self):
        """Current target as MuJoCo mocap arrays.

        Returns:
            ``(pos, quat)`` -- ``pos`` (3,), ``quat`` (4,) wxyz.
        """
        pos = np.array([self.x, self.y, self.z])
        xyzw = Rotation.from_euler("xyz", [0.0, self.pitch, self.yaw]).as_quat()
        # scipy gives xyzw; MuJoCo mocap wants wxyz
        quat = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])
        return pos, quat
