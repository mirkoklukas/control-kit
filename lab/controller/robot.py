"""The concrete robot build, its rest posture, and the scene it lives in.

`make_robot` / `rest_posture` are pure kinematics (the draft's 4x4-DOF radial
robot); `scene_model` compiles the *actuated, welded* MJCF and injects a ground
plane, a light, and the mocap **target box** the controller flies around.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from controlkit.kinematics.robot import Robot, radial_mounts
from controlkit.kinematics.leg4dof import Leg4DOF
from controlkit.kinematics.types import Posture
from controlkit.se3 import SE3

from .config import Cfg

# Per-link materials for the 4-DOF leg (link lengths 0.05, 0.05, 0.2, 0.3): the two
# short proximal links read as the coxa, the 0.2 link the femur, the 0.3 link the tibia.
LINK_MATERIALS = ["coxa", "coxa", "femur", "tibia"]

# Scene assets: robot part colours, a gradient skybox, and the checker ground.
_ASSETS = (
    "  <asset>\n"
    '    <material name="base"  rgba="1 1 1 1"/>\n'
    '    <material name="coxa"  rgba="0.42 0.42 0.42 1"/>\n'
    '    <material name="femur" rgba="0.32 0.32 0.32 1"/>\n'
    '    <material name="tibia" rgba="0.039216 0.760784 1 1"/>\n'
    '    <material name="foot"  rgba="1 0.258824 0.976471 1"/>\n'
    '    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" '
    'width="512" height="512"/>\n'
    '    <texture name="grid" type="2d" builtin="checker" rgb1="0.2 0.3 0.4" '
    'rgb2="0.1 0.15 0.2" width="512" height="512"/>\n'
    '    <material name="grid" texture="grid" texrepeat="10 10" reflectance="0.2"/>\n'
    "  </asset>\n"
)


def make_robot(cfg: Cfg) -> Robot:
    """The draft's radial 4x4-DOF robot from ``cfg`` dimensions."""
    leg = Leg4DOF.from_lengths(
        jnp.asarray(cfg.leg_lengths),
        jnp.deg2rad(jnp.asarray(cfg.joint_limits_deg, dtype=float)),
    )
    return Robot(mounts=radial_mounts(cfg.num_legs, radius=cfg.mount_radius), leg=leg)


def rest_posture(robot: Robot, cfg: Cfg) -> Posture:
    """A level stance: body at ``body_height``, feet planted radially at ``foot_radius``.

    Each foot's world target is straight out along its mount, on the ground; the
    leg's config is solved by IK (``ik_from_foot_and_yaw`` at yaw 0, elbow-down
    branch). Used only to *initialize* the pose -- control needs no IK.

    Args:
        robot: the robot.
        cfg: geometry + rest knobs.

    Returns:
        The rest :class:`Posture`.

    Raises:
        RuntimeError: if a foot target is unreachable (adjust ``foot_radius`` /
            ``body_height``).
    """
    body = SE3.from_te(jnp.array([0.0, 0.0, cfg.body_height]), jnp.zeros(3))
    shoulders = robot.shoulders(body)                          # (num_legs,) world

    # foot straight out along each mount's world direction, on the ground (z=0)
    ang = np.deg2rad(np.arange(360.0 / cfg.num_legs / 2, 360.0, 360.0 / cfg.num_legs))
    feet_world = jnp.array([[cfg.foot_radius * np.cos(a),
                             cfg.foot_radius * np.sin(a), 0.0] for a in ang])

    thetas = []
    for i in range(cfg.num_legs):
        foot_local = shoulders[i].inverse().apply(feet_world[i])
        ok, branches = robot.leg.ik_from_foot_and_yaw(foot_local, jnp.array(0.0))
        idx = int(jnp.argmax(ok))                             # first reachable (elbow-down first)
        if not bool(ok[idx]):
            raise RuntimeError(
                f"leg {i}: rest foot unreachable at radius={cfg.foot_radius}, "
                f"height={cfg.body_height}")
        thetas.append(branches[idx])
    return Posture(body, jnp.stack(thetas))


def _vec(a):
    return " ".join(f"{float(v):.9g}" for v in np.asarray(a).ravel())


def pin_all_feet(model, data, num_legs: int, qpos):
    """Activate the foot ``connect`` pins, anchored at each foot's pose at ``qpos``.

    Places the model at ``qpos``, reads each foot's world position, writes it as
    the pin's world anchor (``eq_data[3:6]``), and turns the pin on -- both
    ``model.eq_active0`` (so ``mujoco.rollout`` inherits it) and this ``data``'s
    ``eq_active``. The anchor lives on the shared model, so the planner's rollout
    pool needs no per-data patching.

    Args:
        model, data: compiled model and the data to activate on.
        num_legs: leg count.
        qpos: (nq,) posture whose foot positions to pin at.
    """
    import mujoco

    # place the model at qpos so mj_forward gives the foot world positions
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    for i in range(num_legs):
        eid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, f"pin_foot{i}")
        fid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"foot{i}")
        if eid < 0:
            continue
        # pin the foot at its current world position
        model.eq_data[eid, 3:6] = data.xpos[fid]
        # activate on the model default too, so mujoco.rollout inherits the pin
        model.eq_active0[eid] = 1
        data.eq_active[eid] = 1


def scene_xml(robot: Robot, cfg: Cfg) -> str:
    """The full scene MJCF as a string: actuated robot + ground + target box + foot pins.

    Starts from :meth:`Robot.to_mjcf` (actuated, densities; no welds) and injects
    the sim ``<option>``, a checker-textured ground, a light, and a **mocap**
    ``target`` body (a translucent trunk-sized ghost box with a flat plate on top
    marking the front, +x).

    When ``cfg.weld_feet`` the feet are held with ``<connect>`` equalities -- 3-DOF
    ball pins, which (unlike a weld) actually hold position under load. They are
    emitted **inactive**; :func:`pin_all_feet` anchors them at the rest pose and
    turns them on at runtime.

    Returns:
        The MJCF string (compile with :func:`scene_model`).
    """
    xml = robot.to_mjcf(actuators=True, link_radius=0.02, kp=cfg.kp, kv=cfg.kv,
                        forcerange=cfg.forcerange, joint_damping=cfg.joint_damping,
                        armature=cfg.armature, frictionloss=cfg.frictionloss,
                        weld_feet=False, body_half_height=cfg.body_half_height,
                        base_pos=(0.0, 0.0, cfg.body_height),
                        body_density=cfg.body_density, leg_density=cfg.leg_density,
                        foot_density=cfg.foot_density, base_material="base",
                        foot_material="foot", link_materials=LINK_MATERIALS)
    xml = xml.replace(
        '  <compiler angle="radian" autolimits="true"/>\n',
        '  <compiler angle="radian" autolimits="true"/>\n'
        f'  <option timestep="{cfg.timestep:.9g}" integrator="{cfg.integrator}" '
        f'solver="{cfg.solver}" iterations="{cfg.iterations}" '
        f'tolerance="{cfg.tolerance:.9g}"/>\n')

    xml = xml.replace("  <worldbody>\n", _ASSETS + "  <worldbody>\n")

    center, half = robot.mount_box(half_height=cfg.body_half_height)
    cx, cy, cz = (float(v) for v in np.asarray(center))
    hx, hy, hz = (float(v) for v in np.asarray(half))
    # flat plate on top, elongated in x and shifted forward -> reads as the nose
    nose_pos = f"{cx + 0.35 * hx:.9g} {cy:.9g} {cz + hz + 0.006:.9g}"
    nose_size = f"{0.55 * hx:.9g} {0.25 * hy:.9g} 0.006"

    # the same nose on the robot's own body, so its heading is visible too. Opaque,
    # massless and non-colliding -> a pure marker that doesn't perturb the dynamics.
    xml = xml.replace(
        '      <freejoint name="root"/>\n',
        '      <freejoint name="root"/>\n'
        f'      <geom name="body_nose" type="box" pos="{nose_pos}" size="{nose_size}" '
        'rgba="0.2 0.2 0.2 1" mass="0" contype="0" conaffinity="0"/>\n')

    extras = (
        '    <geom name="ground" type="plane" size="0 0 0.05" material="grid"/>\n'
        '    <light pos="0 0 2" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>\n'
        f'    <body name="target" mocap="true" pos="0 0 {cfg.body_height:.9g}">\n'
        f'      <geom type="box" pos="{_vec(center)}" size="{_vec(half)}" '
        'rgba="1.0 1.0 1.0 1.0" contype="0" conaffinity="0"/>\n'
        f'      <geom type="box" pos="{nose_pos}" size="{nose_size}" '
        'rgba="0.2 0.2 0.2 1.0" contype="0" conaffinity="0"/>\n'
        '    </body>\n'
    )
    xml = xml.replace("  </worldbody>\n", extras + "  </worldbody>\n")

    # Foot pins: 3-DOF <connect> per foot, emitted inactive (anchor at the foot
    # origin). pin_all_feet sets each world anchor at the rest pose and activates.
    if cfg.weld_feet:
        sr, si = _vec(cfg.weld_solref), _vec(cfg.weld_solimp)
        pins = "".join(
            f'    <connect name="pin_foot{i}" body1="foot{i}" anchor="0 0 0" '
            f'solref="{sr}" solimp="{si}" active="false"/>\n'
            for i in range(cfg.num_legs))
        xml = xml.replace("</mujoco>", f"  <equality>\n{pins}  </equality>\n</mujoco>")
    return xml


def scene_model(robot: Robot, cfg: Cfg):
    """Compile :func:`scene_xml` into a ``mujoco.MjModel``."""
    import mujoco
    return mujoco.MjModel.from_xml_string(scene_xml(robot, cfg))


def main():
    """Dump the scene MJCF next to this module (``lab/controller/scene.xml``)."""
    from pathlib import Path

    cfg = Cfg()
    xml = scene_xml(make_robot(cfg), cfg)
    out = Path(__file__).with_name("scene.xml")
    out.write_text(xml)
    print(f"wrote {out} ({len(xml)} bytes)")


if __name__ == "__main__":
    main()
