"""The body target the gamepad flies (rate-based), as in ``lab.controller.target``.

State is ``[x, y, z, pitch, yaw]`` (roll parked at 0). Stick deflection is a
velocity on the target; releasing holds it.

Mapping (raw sticks: +x right, +y down):
- left stick  -> x / y along the current heading
- right stick -> pitch / yaw
- R2 / L2     -> +z / -z
"""
from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np

from controlkit.se3 import SE3

from .config import Cfg
from .input import ControllerState


class Target:
    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self.x, self.y, self.z = 0.0, 0.0, cfg.body_height
        self.pitch = self.yaw = 0.0

    def update(self, s: ControllerState, dt: float) -> bool:
        """Integrate one tick of input. Returns whether anything moved."""
        c = self.cfg
        before = self.state()
        self.yaw += -s.rx * c.w_yaw * dt
        self.pitch += -s.ry * c.w_pitch * dt
        fwd, side = -s.ly * c.v_xy * dt, s.lx * c.v_xy * dt
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        self.x += fwd * cy + side * sy
        self.y += fwd * sy - side * cy
        self.z += (s.r2 - s.l2) * c.v_z * dt

        self.x = float(np.clip(self.x, -c.xy_range, c.xy_range))
        self.y = float(np.clip(self.y, -c.xy_range, c.xy_range))
        self.z = float(np.clip(self.z, c.z_range[0], c.z_range[1]))
        self.pitch = float(np.clip(self.pitch, -c.pitch_range, c.pitch_range))
        return self.state() != before

    def state(self) -> tuple:
        return (self.x, self.y, self.z, self.pitch, self.yaw)

    def set_state(self, st: tuple):
        self.x, self.y, self.z, self.pitch, self.yaw = st

    def quat(self) -> np.ndarray:
        """Orientation ``Rz(yaw) @ Ry(pitch)`` as a wxyz quaternion (plain numpy).

        Same rotation as ``SE3.from_te(t, [0, pitch, yaw])``, but without eager JAX
        ops, which cost milliseconds each -- this runs every frame.
        """
        cz, sz = math.cos(self.yaw / 2), math.sin(self.yaw / 2)
        cp, sp = math.cos(self.pitch / 2), math.sin(self.pitch / 2)
        return np.array([cz * cp, -sz * sp, cz * sp, sz * cp])

    def body(self) -> SE3:
        """The target as an SE3 body pose."""
        return SE3(jnp.asarray(np.concatenate([self.quat(), [self.x, self.y, self.z]])))

    def mocap(self):
        """``(pos, quat wxyz)`` for the ghost mocap body."""
        return np.array([self.x, self.y, self.z]), self.quat()


class Point:
    """A free 3D point the sticks move in the camera frame (the leg-aiming sphere).

    Left stick moves it in the screen plane (right / up), right stick up/down
    along the viewing direction (depth), right stick left/right like the left
    stick. R2 / L2 move it up / down in the world.
    """

    def __init__(self, cfg: Cfg, pos=(0.0, 0.0, 0.0)):
        self.cfg = cfg
        self.pos = np.asarray(pos, float).copy()

    def update(self, s: ControllerState, dt: float, azimuth: float,
               elevation: float) -> bool:
        """Integrate one tick of input. Returns whether it moved.

        Args:
            s: controller state (raw sticks: +x right, +y down).
            dt: seconds since the last update.
            azimuth, elevation: the viewer camera's angles, radians (MuJoCo's
                convention: viewing direction ``(ce*ca, ce*sa, se)``).
        """
        ca, sa = math.cos(azimuth), math.sin(azimuth)
        ce, se = math.cos(elevation), math.sin(elevation)
        forward = np.array([ce * ca, ce * sa, se])
        right = np.array([sa, -ca, 0.0])
        up = np.cross(right, forward)
        v = self.cfg.v_xy * dt
        delta = (v * (s.lx + s.rx) * right + v * -s.ly * up + v * -s.ry * forward
                 + np.array([0.0, 0.0, (s.r2 - s.l2) * self.cfg.v_z * dt]))
        self.pos = self.pos + delta
        return bool(np.any(delta))
