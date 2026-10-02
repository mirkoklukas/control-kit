"""Build the MuJoCo model: robot (from ``controlkit``) + feet (here) + scene.

Copied from ``lab/rl/mjmodel.py`` (2026-10-01), plus :func:`gravity_dir`.

Three separate steps, so robot and environment stay decoupled:

- :func:`robot_spec` -- ``Robot.to_mjcf`` (legs, servos) loaded into an ``MjSpec``,
  then edited: explicit masses, and each sphere foot replaced by a passive Cardan
  ankle carrying a square pad of N x N cells, each with an ``adhesion`` actuator.
- :func:`add_flat_scene` -- floor, light, sim options, and gravity tilted by
  ``tilt_deg`` (rotating gravity instead of the ground: 90 deg is a wall).
- :func:`build` -- both, plus the ``rest`` / ``stand`` keyframes.

Naming: leg joints/servos ``leg{i}_j{k}``, ankle hinges ``ankle{i}_a`` (about the
foot y-axis) / ``ankle{i}_b`` (foot z), pad body ``pad{i}`` with cell bodies
``pad{i}_c{k}``, adhesion actuators ``adhere{i}_{k}``. The foot frame's +x runs
along the tibia, so the pad face (normal +x) is perpendicular to the tibia at ankle
neutral.

Parameters: every number is a field of :class:`.config.MjModelCfg` (``config.py``),
except the scene's gravity tilt, which is an argument of :func:`build`. The climb env
builds with no tilt and sets ``model.opt.gravity`` per episode (:func:`gravity_dir`). None is hardcoded here. Where each one lands in the model:

- Geometry: ``mount_radius``, ``leg_lengths`` (coxa/femur/tibia),
  ``joint_limits_deg`` (also the servo ctrlrange), ``body_half_height``,
  ``link_radius`` -> passed to ``Robot.to_mjcf``.
- Masses: ``mass_body``, ``mass_links`` -> set on the trunk/link geoms in
  :func:`robot_spec`; ``mass_pad`` -> split over the pad cells in :func:`_add_foot`.
- Leg servos: ``kp``, ``kv``, ``forcerange``, ``joint_damping``, ``armature`` ->
  ``Robot.to_mjcf`` (the ``<position>`` / ``<joint>`` defaults).
- Ankle (:func:`_add_foot`): ``ankle_range_deg`` (+/- hard stop, both axes),
  ``ankle_stiffness``, ``ankle_damping``, ``ankle_armature``.
- Pad (:func:`_add_foot`): ``pad_size``, ``pad_cells`` (N x N), ``pad_thickness``,
  ``pivot_height``, ``pad_friction`` (also the floor's).
- Adhesion (:func:`_add_foot`): ``adhesion_gain`` (N per foot at ctrl 1, split over
  the cells), ``adhesion_margin`` (pad geom margin = gap).
- Sim / scene (:func:`add_flat_scene`): ``timestep``, ``integrator``; gravity tilt
  from ``build(..., tilt_deg=...)``.
- Keyframes (``poses.py``): ``stand_foot_radius``, ``stand_height``,
  ``rest_clearance``.

Override any of them per run: ``python -m lab.rl.climb.train mjmodel.ankle_range_deg=45``.
"""
from __future__ import annotations

import math
from pathlib import Path

import jax.numpy as jnp
import mujoco
import numpy as np

from controlkit.kinematics import Leg3DOF, Robot, radial_mounts

from .config import MjModelCfg

MODELS_DIR = Path(__file__).parent / "models"

_RGBA = {
    "body": (0.9, 0.9, 0.9, 1), "coxa": (0.42, 0.42, 0.42, 1),
    "femur": (0.32, 0.32, 0.32, 1), "tibia": (0.04, 0.76, 1.0, 1),
    "pad": (1.0, 0.26, 0.98, 1), "ankle": (0.2, 0.2, 0.2, 1),
    "nose": (0.05, 0.05, 0.05, 1),
}


def make_robot(cfg: MjModelCfg) -> Robot:
    """The 4x3-DOF spider as a kinematic :class:`Robot`."""
    leg = Leg3DOF.from_lengths(
        jnp.asarray(cfg.leg_lengths),
        jnp.deg2rad(jnp.asarray(cfg.joint_limits_deg, dtype=float)),
    )
    return Robot(mounts=radial_mounts(cfg.num_legs, radius=cfg.mount_radius), leg=leg)


def _add_foot(spec: mujoco.MjSpec, cfg: MjModelCfg, i: int) -> None:
    """Replace leg ``i``'s sphere foot with a Cardan ankle + pad + adhesion."""
    foot = spec.body(f"foot{i}")
    for g in list(foot.geoms):
        spec.delete(g)

    # The tibia capsule's rounded end would stick out past the ankle pivot and hit
    # the ground / pad when the ankle flexes: end the collision capsule 2r short and
    # bridge the gap with a thin visual-only shank.
    r, L = cfg.link_radius, cfg.leg_lengths[-1]
    tibia = spec.body(f"leg{i}_2").geoms[0]
    tibia.fromto = [0, 0, 0, L - 2 * r, 0, 0]
    spec.body(f"leg{i}_2").add_geom(
        type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[L - 2 * r, 0, 0, L, 0, 0],
        size=[0.4 * r, 0, 0], contype=0, conaffinity=0, mass=0, rgba=_RGBA["tibia"])

    pad = foot.add_body(name=f"pad{i}")
    lim = cfg.ankle_range_deg
    for suffix, axis in (("a", (0, 1, 0)), ("b", (0, 0, 1))):
        pad.add_joint(
            name=f"ankle{i}_{suffix}", type=mujoco.mjtJoint.mjJNT_HINGE, axis=axis,
            range=[-math.radians(lim), math.radians(lim)], limited=mujoco.mjtLimited.mjLIMITED_TRUE,
            stiffness=cfg.ankle_stiffness, damping=cfg.ankle_damping,
            armature=cfg.ankle_armature)

    # visual pivot marker
    pad.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.6 * r, 0, 0],
                 contype=0, conaffinity=0, mass=0, rgba=_RGBA["ankle"])
    # N x N grid of cells, each its own (joint-less) child body with a box geom and
    # its own adhesion actuator of gain A / N^2. MuJoCo's adhesion acts on a body's
    # contacts with the full gain regardless of how many there are, so splitting the
    # pad makes the force scale with the touching area: an edge gets ~1/N, a corner
    # ~1/N^2 (a single-body pad held the full force on a corner -- see docs/notes.md,
    # "foot peeling problem"). margin == gap is what we intended (contacts seen within
    # the margin, inactive until touching); in 3.9.0 gap has no effect, so the pad
    # rests ~margin above the surface (accepted).
    n, s, t = cfg.pad_cells, cfg.pad_size, cfg.pad_thickness
    x, w = cfg.pivot_height + 0.5 * t, s / n
    for k in range(n * n):
        dy, dz = (k // n + 0.5) * w - 0.5 * s, (k % n + 0.5) * w - 0.5 * s
        cell = pad.add_body(name=f"pad{i}_c{k}", pos=[x, dy, dz])
        cell.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.5 * t, 0.5 * w, 0.5 * w],
            mass=cfg.mass_pad / n**2, friction=[cfg.pad_friction, 0.005, 0.0001],
            margin=cfg.adhesion_margin, gap=cfg.adhesion_margin, rgba=_RGBA["pad"])
        spec.add_exclude(bodyname1=f"leg{i}_2", bodyname2=f"pad{i}_c{k}")
        act = spec.add_actuator(name=f"adhere{i}_{k}", target=f"pad{i}_c{k}",
                                trntype=mujoco.mjtTrn.mjTRN_BODY)
        act.set_to_adhesion(gain=cfg.adhesion_gain / n**2)
        act.ctrlrange = [0.0, 1.0]
        # the robot's default <position> class carries the servo forcerange; don't inherit it
        act.forcelimited = mujoco.mjtLimited.mjLIMITED_FALSE


def pad_cell_bodies(model: mujoco.MjModel, i: int) -> np.ndarray:
    """Body ids of foot ``i``'s pad cells (``pad{i}_c{k}``)."""
    return np.array([b for b in range(model.nbody) if model.body(b).name.startswith(f"pad{i}_c")])


def adhesion_actuators(model: mujoco.MjModel, i: int) -> np.ndarray:
    """Actuator ids of foot ``i``'s per-cell adhesion (``adhere{i}_{k}``)."""
    return np.array([a for a in range(model.nu) if model.actuator(a).name.startswith(f"adhere{i}_")])


def robot_spec(cfg: MjModelCfg) -> mujoco.MjSpec:
    """The robot alone (no floor), with ankles, pads and adhesion actuators.

    Actuator order: the ``num_legs * 3`` leg servos (leg-major), then the adhesion
    actuators, foot-major: ``pad_cells**2`` per foot.
    """
    robot = make_robot(cfg)
    xml = robot.to_mjcf(
        name="spider", link_radius=cfg.link_radius, body_half_height=cfg.body_half_height,
        base_pos=(0.0, 0.0, cfg.stand_height), weld_feet=False, actuators=True,
        kp=cfg.kp, kv=cfg.kv, forcerange=cfg.forcerange,
        joint_damping=cfg.joint_damping, armature=cfg.armature)
    spec = mujoco.MjSpec.from_string(xml)

    base = spec.body("base").geoms[0]
    base.mass, base.rgba = cfg.mass_body, _RGBA["body"]
    # front marker (+x): a black cube of side mount_radius / 3 whose top sits
    # `nose_offset` above the trunk top and whose front sits `nose_offset` in front
    # of the trunk's front face. Massless and non-colliding -- a pure heading marker.
    hx, hy, hz = base.size
    h, d = cfg.mount_radius / 6, cfg.nose_offset
    spec.body("base").add_geom(
        name="body_nose", type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[base.pos[0] + hx + d - h, base.pos[1], base.pos[2] + hz + d - h],
        size=[h, h, h], rgba=_RGBA["nose"], mass=0, contype=0, conaffinity=0)
    for i in range(cfg.num_legs):
        for k, part in enumerate(("coxa", "femur", "tibia")):
            g = spec.body(f"leg{i}_{k}").geoms[0]
            g.mass, g.rgba = cfg.mass_links[k], _RGBA[part]
        _add_foot(spec, cfg, i)
    return spec


def gravity(tilt_deg: float) -> list[float]:
    """Gravity tilted by ``tilt_deg`` about world y: 0 floor, 90 wall (+x), 180 ceiling."""
    th = math.radians(tilt_deg)
    return [9.81 * math.sin(th), 0.0, -9.81 * math.cos(th)]


def gravity_dir(tilt_deg: float, azimuth_deg: float = 0.0) -> np.ndarray:
    """Gravity tilted by ``tilt_deg`` from -z, towards azimuth ``azimuth_deg`` (about z,
    from +x). ``gravity_dir(t, 0) == gravity(t)``.

    Returns:
        (3,) gravity vector (m/s^2), world frame.
    """
    th, ph = math.radians(tilt_deg), math.radians(azimuth_deg)
    return 9.81 * np.array([math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph),
                            -math.cos(th)])


def add_flat_scene(spec: mujoco.MjSpec, cfg: MjModelCfg, tilt_deg: float = 0.0) -> mujoco.MjSpec:
    """Add floor, light, sim options and gravity to ``spec``, in place.

    Args:
        spec: the robot spec.
        cfg: model config (timestep, integrator, pad friction for the floor).
        tilt_deg: gravity tilt about world y (0 floor, 90 wall, 180 ceiling).
    """
    opt = spec.option
    opt.timestep = cfg.timestep
    opt.integrator = getattr(mujoco.mjtIntegrator, f"mjINT_{cfg.integrator.upper()}")
    opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC      # recommended with adhesion
    opt.impratio = 10.0
    opt.gravity = gravity(tilt_deg)

    spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.2, 0.3, 0.4], rgb2=[0.1, 0.15, 0.2], width=512, height=512)
    mat = spec.add_material(name="grid", texrepeat=[10, 10], reflectance=0.0)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                            size=[0, 0, 0.05], material="grid",
                            friction=[cfg.pad_friction, 0.005, 0.0001])
    spec.worldbody.add_light(pos=[0, 0, 2], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8])
    return spec


def build(cfg: MjModelCfg, *, tilt_deg: float = 0.0, write: bool = True):
    """Robot + flat scene + ``rest``/``stand`` keyframes.

    Args:
        cfg: model config.
        tilt_deg: gravity tilt about world y (0 floor, 90 wall, 180 ceiling).
        write: also write the model XML to ``models/scene_flat.xml`` (for
            inspection and ``ctk play``).

    Returns:
        ``(spec, model)``.
    """
    from .poses import keyframe

    spec = add_flat_scene(robot_spec(cfg), cfg, tilt_deg)
    model = spec.compile()
    robot = make_robot(cfg)
    for name in ("rest", "stand"):
        qpos, ctrl = keyframe(model, robot, cfg, name)
        spec.add_key(name=name, qpos=qpos, ctrl=ctrl)
    model = spec.compile()
    if write:
        MODELS_DIR.mkdir(exist_ok=True)
        (MODELS_DIR / "scene_flat.xml").write_text(spec.to_xml())
    return spec, model
