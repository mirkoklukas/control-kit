"""The robot build and the (kinematic, visualization-only) MuJoCo scene."""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from controlkit.kinematics.leg4dof import Leg4DOF
from controlkit.kinematics.robot import Robot, radial_mounts

from . import terrain
from .config import Cfg

# Colour scheme from lab.controller.
LINK_MATERIALS = ["coxa", "coxa", "femur", "tibia"]

_ASSETS = (
    "  <asset>\n"
    '    <material name="base"  rgba="1 1 1 1"/>\n'
    '    <material name="coxa"  rgba="0.42 0.42 0.42 1"/>\n'
    '    <material name="femur" rgba="0.32 0.32 0.32 1"/>\n'
    '    <material name="tibia" rgba="0.039216 0.760784 1 1"/>\n'
    '    <material name="foot"  rgba="1 0.258824 0.976471 1"/>\n'
    '    <material name="terrain" rgba="0.2 0.28 0.36 1"/>\n'
    '    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" '
    'width="512" height="512"/>\n'
    '    <texture name="grid" type="2d" builtin="checker" rgb1="0.2 0.3 0.4" '
    'rgb2="0.1 0.15 0.2" width="512" height="512"/>\n'
    '    <material name="grid" texture="grid" texrepeat="10 10"/>\n'
    "  </asset>\n"
)

# A geom rgba other than MuJoCo's default overrides its material; setting it back
# to the default hands the colour back to the material.
RGBA_MATERIAL = (0.5, 0.5, 0.5, 1.0)
RGBA_BAD = (1.0, 0.25, 0.25, 1.0)
RGBA_SELECTED = (1.0, 0.6, 0.1, 1.0)
GHOST_SCALE = 0.97


def make_robot(cfg: Cfg) -> Robot:
    """The radial 4x4-DOF robot from ``cfg`` dimensions."""
    leg = Leg4DOF.from_lengths(
        jnp.asarray(cfg.leg_lengths),
        jnp.deg2rad(jnp.asarray(cfg.joint_limits_deg, dtype=float)),
    )
    return Robot(mounts=radial_mounts(cfg.num_legs, radius=cfg.mount_radius), leg=leg)


def _vec(a):
    return " ".join(f"{float(v):.6g}" for v in np.asarray(a).ravel())


def scene_xml(robot: Robot, cfg: Cfg) -> str:
    """Robot + terrain + light + a solid mocap ``target`` ghost of the body.

    Nothing is ever stepped: the playground writes ``qpos`` and calls
    ``mj_forward``, so collisions are disabled everywhere (``contype=0``). The
    ground slab (first box) gets the checker ``grid``, the other boxes ``terrain``.
    """
    xml = robot.to_mjcf(weld_feet=False, link_radius=cfg.link_radius,
                        foot_radius=cfg.foot_radius,
                        body_half_height=cfg.body_half_height, base_material="base",
                        foot_material="foot", link_materials=LINK_MATERIALS)
    xml = xml.replace("  <worldbody>\n", _ASSETS + "  <worldbody>\n")
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

    # The ghost is shrunk a little so its faces never coincide with the body's
    # (coplanar faces z-fight when the two overlap).
    g = GHOST_SCALE
    ghost_nose = f"{(cx + 0.35 * hx) * g:.6g} {cy * g:.6g} {(cz + hz) * g + 0.006:.6g}"
    ghost_nose_size = f"{0.55 * hx * g:.6g} {0.25 * hy * g:.6g} 0.006"

    ground, *rest = cfg.boxes
    extras = (
        terrain.mjcf_geoms([ground], material="grid", prefix="ground")
        + terrain.mjcf_geoms(rest, material="terrain")
        + '    <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>\n'
        + f'    <body name="target" mocap="true" pos="0 0 {cfg.body_height:.6g}">\n'
        + f'      <geom type="box" pos="{_vec(center * g)}" size="{_vec(half * g)}" material="base"/>\n'
        + f'      <geom type="box" pos="{ghost_nose}" size="{ghost_nose_size}" rgba="0.2 0.2 0.2 1"/>\n'
        + "    </body>\n"
    )
    return xml.replace("  </worldbody>\n", extras + "  </worldbody>\n")


def leg_geoms(model, num_legs: int) -> list[np.ndarray]:
    """Geom ids of each leg (its link capsules and its foot)."""
    out = [[] for _ in range(num_legs)]
    for g in range(model.ngeom):
        name = model.body(model.geom_bodyid[g]).name
        for i in range(num_legs):
            if name.startswith(f"leg{i}_") or name == f"foot{i}":
                out[i].append(g)
    return [np.array(ids) for ids in out]
