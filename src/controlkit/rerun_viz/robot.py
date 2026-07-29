"""Drawing a robot: body, legs, whole postures.

The layer that knows there is a robot. Everything takes a
:class:`..kinematics.Robot` explicitly rather than closing over one, so a
quadruped and a hexapod are the same call with a different argument.

Every helper takes the rerun entity ``path`` first, as ``rr.log(path, ...)`` does.
"""

import jax.numpy as jnp
import numpy as np
import rerun as rr
from jaxlie import SE3

from controlkit.kinematics import Leg, Robot
from controlkit.rerun_viz import shapes
from controlkit.rerun_viz.colors import (
    BODY_COLOR,
    CONTACT_COLOR,
    EDGE_COLOR,
    MOMENT_COLOR,
    SEGMENT_COLOR,
    colors_from_values,
)
from controlkit.rerun_viz.draw import log_box, log_joint, log_mesh, log_spheres


def body_radius(robot: Robot) -> float:
    """Circumradius of the body polygon, read off the mounts.

    The shoulders sit on it by construction, so this cannot drift out of sync
    with the robot the way a hard-coded default would.

    Args:
        robot: the robot.

    Returns:
        Distance from the body origin to a mount, in the xy-plane.
    """
    t = np.asarray(robot.mounts.translation())[0]
    return float(np.linalg.norm(t[:2]))


def log_body(path, robot: Robot, body: SE3, *, color=BODY_COLOR,
             thickness=0.05, edge_color=EDGE_COLOR):
    """Draw the body as a prism with one corner per leg.

    A hexapod gets a hexagon, a quadruped a square -- the corner angles match
    :func:`..kinematics.robot.radial_mounts`, so each corner lands on a shoulder.

    Args:
        path: rerun entity path (body under ``{path}/body``).
        robot: the robot; supplies the corner count and the radius.
        body: SE3 world pose of the base.
        color: RGB(A) uint8 body color.
        thickness: prism thickness in z (metres).
        edge_color: wireframe edge color; None to skip.
    """
    verts, faces = shapes.polygon_prism(robot.num_legs, body_radius(robot), thickness)
    normals = shapes.vertex_normals(verts, faces)              # so it shades like the legs
    t = np.asarray(body.translation())
    q_xyzw = np.asarray(body.rotation().as_quaternion_xyzw())
    rr.log(f"{path}/body", rr.Transform3D(translation=t, rotation=rr.Quaternion(xyzw=q_xyzw)))
    log_mesh(f"{path}/body/mesh", verts, faces, color=color, normals=normals,
             edge_color=edge_color)


def log_leg(path, leg: Leg, shoulder: SE3, theta, *,
            foot_radius=0.02, foot_color=SEGMENT_COLOR, disk_colors=None,
            joint_values=None, joint_cmap="viridis", joint_min=None, joint_max=None,
            **joint_kw) -> SE3:
    """Draw one leg -- joints, links, foot -- at a world-frame shoulder pose.

    The chain comes from ``leg.forward`` in the leg's local frame, lifted to the
    world by ``shoulder``. Generic in the joint count.

    Args:
        path: rerun entity path (joint ``k`` at ``{path}/j{k}``, foot at
            ``{path}/foot``).
        leg: the leg to draw (a :class:`..kinematics.Leg`).
        shoulder: SE3 world pose of the shoulder (leg-base) frame.
        theta: (num_joints,) joint angles (radians).
        foot_radius: foot-sphere radius (m).
        foot_color: RGB(A) of the foot sphere.
        disk_colors: explicit (num_joints, 3/4) per-joint disk colors; takes
            precedence over ``joint_values``.
        joint_values: (num_joints,) scalars mapped through ``joint_cmap`` to color
            the disks (e.g. torque); ignored if ``disk_colors`` is given.
        joint_cmap: matplotlib colormap for ``joint_values``.
        joint_min: lower color bound for ``joint_values`` (default data min).
        joint_max: upper color bound for ``joint_values`` (default data max).
        **joint_kw: forwarded to :func:`..draw.log_joint` (``disk_radius``,
            ``link_width``, ``link_color``, ...).

    Returns:
        (num_joints + 1,) SE3 world frames -- joints, then the foot.
    """
    frames = shoulder @ leg.forward(theta)
    njoint = leg.num_joints
    # Disks are flattened along each joint's rotation axis, boxes span each link;
    # `axes` and `lengths` supply those. `axes` is public on the leg for exactly
    # this (drawing) -- the leg's full structure still lives in `forward`.
    lengths, axes = leg.lengths, leg.axes

    if disk_colors is None and joint_values is not None:       # scalars -> per-joint colors
        disk_colors = colors_from_values(joint_values, cmap=joint_cmap,
                                         vmin=joint_min, vmax=joint_max)
    for k in range(njoint):                                    # disk + link per joint
        jk = dict(joint_kw)
        if disk_colors is not None:
            jk["disk_color"] = disk_colors[k]
        log_joint(f"{path}/j{k}", frames[k], axes[k], ("x", float(lengths[k])), **jk)

    foot = np.asarray(frames[njoint].translation())            # end-effector = foot
    log_spheres(f"{path}/foot", foot[None], foot_radius, colors=foot_color)
    return frames


def log_posture(path, robot: Robot, posture, *, support=None, hide_free=False,
                show_bounding_box=False, bbox_margin=0.0, bbox_half_height=0.05,
                bbox_color=(140, 140, 140),
                frc=None, mom=None, tau=None, tmin=None, tmax=None, cmap="bwr",
                body_size=(0.2, 0.15, 0.025), body_color=BODY_COLOR,
                foot_radius=0.02, foot_color=SEGMENT_COLOR,
                force_scale=0.005, moment_scale=0.02, arrow_radius=0.006,
                force_color=CONTACT_COLOR, moment_color=MOMENT_COLOR,
                edge_color=EDGE_COLOR,
                **joint_kw):
    """Draw a whole posture: the body plus every leg, with optional forces.

    Args:
        path: rerun entity path.
        robot: the robot; supplies mounts, lengths and joint axes.
        posture: :class:`..kinematics.Posture` (``body`` SE3 + ``(num_legs,
            num_joints)`` ``thetas``).
        support: :class:`..kinematics.Support`; needed only for ``hide_free``, to
            know which legs are planted.
        hide_free: draw only the supported legs (those in ``support.ids``); skip
            the free ones, feet and forces included. Requires ``support``.
        show_bounding_box: draw the robot's ``mount_box`` (the self-collision
            keep-out region) as a wireframe, oriented with the body.
        bbox_margin: x/y half-width added to the mount box.
        bbox_half_height: z half-extent of the mount box.
        bbox_color: RGB(A) of the wireframe.
        frc: (num_legs, 3) world-frame foot forces; drawn as arrows under
            ``{path}/forces``.
        mom: (num_legs, 3) world-frame foot moments; drawn under ``{path}/moments``.
        tau: (num_legs, num_joints) joint torques; each joint's disk is colored by
            its torque through ``cmap``. Bounds are shared across all legs, so
            colors are comparable leg to leg.
        tmin: lower torque color bound (default -max|tau|).
        tmax: upper torque color bound (default +max|tau|).
        cmap: matplotlib colormap for torque (diverging, e.g. "bwr").
        body_size: (x, y, z) sides of the flat body cube -- only used by the
            commented-out cube below.
        body_color: RGB(A) of the body.
        foot_radius: foot-sphere radius (m).
        foot_color: RGB(A) of the foot spheres.
        force_scale: force-arrow length per newton (m/N).
        moment_scale: moment-arrow length per newton-metre (m/Nm).
        arrow_radius: force/moment-arrow shaft radius (m).
        force_color: RGB(A) of the force arrows.
        moment_color: RGB(A) of the moment arrows.
        **joint_kw: forwarded to :func:`..draw.log_joint`.
    """
    body = posture.body
    thetas = jnp.asarray(posture.thetas)
    nleg, njoint = thetas.shape[0], robot.leg.num_joints
    shown = np.ones(nleg, bool)
    if hide_free:                                              # draw only planted legs
        shown = np.zeros(nleg, bool)
        shown[np.asarray(support.ids)] = True

    # Per-joint torque colors, on bounds shared across every leg.
    jcolors = None
    if tau is not None:
        tvals = np.asarray(tau).reshape(-1)
        hi = float(np.abs(tvals).max()) if tmax is None else tmax
        lo = -hi if tmin is None else tmin
        jcolors = colors_from_values(tvals, cmap=cmap, vmin=lo, vmax=hi).reshape(nleg, njoint, 3)

    log_body(path, robot, body, color=body_color, edge_color=edge_color)

    if show_bounding_box:                                     # self-collision keep-out
        center, half = robot.mount_box(margin=bbox_margin, half_height=bbox_half_height)
        log_box(f"{path}/bbox", np.asarray(body.apply(center)), np.asarray(half),
                quaternion=np.asarray(body.rotation().as_quaternion_xyzw()),
                color=bbox_color)

    # flat cube body (kept in case we switch back)
    # t = np.asarray(body.translation(), np.float32)
    # q_xyzw = np.asarray(body.rotation().as_quaternion_xyzw())
    # rr.log(f"{path}/body", rr.Boxes3D(
    #     centers=[t], half_sizes=[np.asarray(body_size, np.float32) / 2],
    #     quaternions=[rr.Quaternion(xyzw=q_xyzw)], colors=[body_color], fill_mode="solid"))

    shoulders = body @ robot.mounts                             # (num_legs,) world poses
    feet, foot_legs = [], []
    for i in range(nleg):
        if not shown[i]:                                       # hidden free leg
            continue
        frames = log_leg(f"{path}/leg{i}", robot.leg, shoulders[i], thetas[i],
                         foot_radius=foot_radius, foot_color=foot_color,
                         disk_colors=None if jcolors is None else jcolors[i],
                         **joint_kw)
        feet.append(np.asarray(frames[njoint].translation()))
        foot_legs.append(i)

    if feet:
        feet = np.stack(feet)
        if frc is not None:                                    # reaction force arrows at feet
            vecs = np.asarray(frc).reshape(nleg, 3)[foot_legs] * force_scale
            rr.log(f"{path}/forces", rr.Arrows3D(
                origins=feet, vectors=vecs, radii=arrow_radius, colors=force_color))
        if mom is not None:                                    # reaction moment arrows at feet
            mvecs = np.asarray(mom).reshape(nleg, 3)[foot_legs] * moment_scale
            rr.log(f"{path}/moments", rr.Arrows3D(
                origins=feet, vectors=mvecs, radii=arrow_radius, colors=moment_color))
