"""The hexapod build (6x3-DOF radial) and its MuJoCo scene.

As :mod:`.robot`, with a :class:`Leg3DOF`, one material per link, and a
hexagonal body (corners at the mounts) instead of the mounts' bounding box.
"""
from __future__ import annotations

import math
import re

import jax.numpy as jnp
import numpy as np

from controlkit.kinematics.leg3dof import Leg3DOF
from controlkit.kinematics.robot import Robot, radial_mounts

from . import terrain
from .config_hex import HexCfg
from .robot import _ASSETS, GHOST_SCALE, _vec

LINK_MATERIALS = ["coxa", "femur", "tibia"]


def make_robot(cfg: HexCfg) -> Robot:
    """The radial Nx3-DOF robot from ``cfg`` dimensions."""
    leg = Leg3DOF.from_lengths(
        jnp.asarray(cfg.leg_lengths),
        jnp.deg2rad(jnp.asarray(cfg.joint_limits_deg, dtype=float)),
    )
    return Robot(mounts=radial_mounts(cfg.num_legs, radius=cfg.mount_radius), leg=leg)


def _hex_mesh(name: str, cfg: HexCfg, scale: float = 1.0) -> str:
    """A ``<mesh>`` asset: the hexagonal prism with corners at the 6 mounts."""
    R, h = cfg.mount_radius * scale, cfg.body_half_height * scale
    angles = [math.radians(30 + 60 * k) for k in range(6)]
    verts = [(R * math.cos(a), R * math.sin(a), z) for z in (-h, h) for a in angles]
    return f'    <mesh name="{name}" vertex="{_vec(np.array(verts))}"/>\n'


def scene_xml(robot: Robot, cfg: HexCfg) -> str:
    """Robot + terrain + lights + a solid mocap ``target`` ghost of the body.

    Copy of :func:`.robot.scene_xml` with this module's ``LINK_MATERIALS``, the
    body box (on robot and ghost) swapped for a hexagonal prism mesh, and the
    tibia capsules at ``tibia_radius``.
    """
    xml = robot.to_mjcf(weld_feet=False, link_radius=cfg.link_radius,
                        foot_radius=cfg.foot_radius,
                        body_half_height=cfg.body_half_height, base_material="base",
                        foot_material="foot", link_materials=LINK_MATERIALS)
    meshes = _hex_mesh("hex", cfg) + _hex_mesh("hex_ghost", cfg, GHOST_SCALE)
    # checker repeat scaled with the ground (robot._ASSETS: 10 on a 6 m ground)
    rep = 10 * cfg.boxes[0][1][0] / 3.0
    assets = _ASSETS.replace('texrepeat="10 10"', f'texrepeat="{rep:.6g} {rep:.6g}"')
    xml = xml.replace("  <worldbody>\n",
                      assets.replace("  </asset>\n", meshes + "  </asset>\n")
                      + "  <worldbody>\n")
    xml, n = re.subn(r'<geom class="base" name="base" [^>]*/>',
                     '<geom class="base" name="base" type="mesh" mesh="hex"/>', xml)
    assert n == 1, "base geom not found"
    xml = xml.replace('material="tibia"/>', f'material="tibia" size="{cfg.tibia_radius:.6g}"/>')
    xml = xml.replace('    <joint type="hinge"',
                      '    <geom contype="0" conaffinity="0"/>\n    <joint type="hinge"')

    # dark nose plate on top, shifted forward (+x): the heading, on body and ghost
    center, half = robot.mount_box(half_height=cfg.body_half_height)
    cx, cy, cz = (float(v) for v in np.asarray(center))
    hx, hy, hz = (float(v) for v in np.asarray(half))
    nose_pos = f"{cx + 0.35 * hx:.6g} {cy:.6g} {cz + hz + 0.006:.6g}"
    nose_size = f"{0.55 * hx:.6g} {0.25 * hy:.6g} 0.006"
    xml = xml.replace(
        '      <freejoint name="root"/>\n',
        '      <freejoint name="root"/>\n'
        f'      <geom name="body_nose" type="box" pos="{nose_pos}" size="{nose_size}" '
        'rgba="0.2 0.2 0.2 1"/>\n')

    g = GHOST_SCALE
    ghost_nose = f"{(cx + 0.35 * hx) * g:.6g} {cy * g:.6g} {(cz + hz) * g + 0.006:.6g}"
    ghost_nose_size = f"{0.55 * hx * g:.6g} {0.25 * hy * g:.6g} 0.006"

    ground, *rest = cfg.boxes
    extras = (
        terrain.mjcf_geoms([ground], material="grid", prefix="ground")
        + terrain.mjcf_geoms(rest, material="terrain")
        + '    <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>\n'
        # above the back wall, a bit in front, aimed at its front face (a straight-down
        # light would leave the face dark)
        + '    <light pos="1.2 0 3.5" dir="0.65 0 -2.5" diffuse="0.6 0.6 0.6"/>\n'
        + f'    <body name="target" mocap="true" pos="0 0 {cfg.body_height:.6g}">\n'
        + '      <geom type="mesh" mesh="hex_ghost" material="base"/>\n'
        + f'      <geom type="box" pos="{ghost_nose}" size="{ghost_nose_size}" rgba="0.2 0.2 0.2 1"/>\n'
        + "    </body>\n"
    )
    return xml.replace("  </worldbody>\n", extras + "  </worldbody>\n")
