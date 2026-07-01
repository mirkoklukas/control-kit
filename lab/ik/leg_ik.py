"""Closed-form inverse kinematics for a single hexapod leg (analytic 3R).

Given a shoulder pose, a world-frame foot point, and the three segment lengths,
return the coxa/femur/tibia joint angles directly -- no numerical solver, no
MuJoCo at solve time (pure numpy). This is the analytic counterpart to the
Jacobian-based :mod:`lab.ik.planted` solver.

Geometry. The leg is a 3R chain: coxa (yaw about local ``+z``) -> femur (lift
about local ``+y``) -> tibia (knee about local ``+y``). The *shoulder frame* is
the frame in which, at zero joint angles, the leg extends along ``+x`` with the
axes above; ``(x, q)`` is its world position and orientation (quaternion
``wxyz``). Segment lengths are the pure local ``+x`` lengths at the zero pose:
``Lc`` coxa, ``Lf`` femur, ``Lt`` tibia (0.025 / 0.2 / 0.206 for this model).

Two solutions. The coxa yaw is fixed by the foot's azimuth, but the femur+tibia
subchain is a planar 2R, so the knee has two mirror configurations (elbow "up" /
"down"), returned as ``theta3 = +/- arccos(...)``. (A third fold, ``theta1 + pi``
reaching backward, is not returned -- it needs negative radial reach and is
normally out of joint range.)

Derivation (foot expressed in the shoulder frame as ``p``):

    theta1 = atan2(p_y, p_x)                        # yaw to the foot's azimuth
    u  = hypot(p_x, p_y) - Lc                       # radial reach past the coxa
    xi = -p_z                                       # +rot about y tilts +x -> -z
    cos(theta3) = (u^2 + xi^2 - Lf^2 - Lt^2) / (2 Lf Lt)
    theta3 = +/- arccos(...)                        # elbow up / down
    theta2 = atan2(xi, u) - atan2(Lt sin theta3, Lf + Lt cos theta3)

Run ``uv run python -m lab.ik.leg_ik`` to verify against the MuJoCo model.
"""
from __future__ import annotations

import numpy as np


def _rotmat_wxyz(q):
    """Rotation matrix (3, 3) from a ``wxyz`` quaternion (normalized here)."""
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


def leg_ik(x, q, y, lengths):
    """Analytic 3R IK for one leg.

    Parameters
    ----------
    x : (3,) array
        Shoulder (coxa joint) position in world coordinates.
    q : (4,) array
        Shoulder orientation, quaternion ``wxyz``. In this frame the leg extends
        along ``+x`` at zero angles, coxa spins about ``+z``, femur/tibia about
        ``+y``.
    y : (3,) array
        Foot point in world coordinates.
    lengths : (3,) array
        Segment lengths ``(Lc, Lf, Lt)`` for coxa / femur / tibia.

    Returns
    -------
    sols : (2, 3) ndarray
        The two joint-angle solutions ``[theta_coxa, theta_femur, theta_tibia]``
        (radians), elbow-"down" (``+arccos``) first then elbow-"up" (``-arccos``).
        Joint limits are *not* applied.
    reachable : bool
        ``True`` if the foot lies within the annulus the leg can span. When
        ``False`` the angles place the foot as close as possible (knee straight
        or fully folded) and both solutions coincide.
    """
    Lc, Lf, Lt = np.asarray(lengths, float)

    # Foot in the shoulder frame: undo the shoulder pose (world -> local).
    p = _rotmat_wxyz(q).T @ (np.asarray(y, float) - np.asarray(x, float))

    # Coxa yaw: point the leg's vertical plane at the foot's azimuth.
    theta1 = np.arctan2(p[1], p[0])

    # In-plane target for the femur+tibia 2R, measured from the femur joint
    # (which sits Lc along the radial direction from the shoulder).
    u = np.hypot(p[0], p[1]) - Lc  # horizontal reach past the coxa
    xi = -p[2]                     # vertical (see module docstring for the sign)

    # Knee angle by the law of cosines; clamp handles unreachable targets.
    D2 = u * u + xi * xi
    c3 = (D2 - Lf * Lf - Lt * Lt) / (2 * Lf * Lt)
    reachable = -1.0 <= c3 <= 1.0
    c3 = np.clip(c3, -1.0, 1.0)

    sols = []
    for sign in (+1.0, -1.0):  # elbow down, elbow up
        theta3 = sign * np.arccos(c3)
        # Femur angle: aim atan2(xi, u) minus the tibia's contribution.
        theta2 = np.arctan2(xi, u) - np.arctan2(Lt * np.sin(theta3), Lf + Lt * np.cos(theta3))
        sols.append([theta1, theta2, theta3])

    return np.array(sols), reachable


# --------------------------------------------------------------------------- #
# Verification against the MuJoCo model                                       #
# --------------------------------------------------------------------------- #
def _verify():
    """Round-trip check: for random joint angles, forward the model to get the
    shoulder pose + foot point, then confirm ``leg_ik`` recovers those angles.

    Shows how a caller derives the shoulder pose from the base pose: it is the
    coxa body's fixed local offset (``body_pos`` / ``body_quat``) mapped through
    the base frame, i.e. independent of the joint values.
    """
    from pathlib import Path
    import mujoco

    root = Path(__file__).resolve().parents[2]
    model = mujoco.MjModel.from_xml_path(str(root / "models" / "hexapod.xml"))
    data = mujoco.MjData(model)

    free = int(np.argmax(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE))
    base = int(model.jnt_bodyid[free])
    rng = np.random.default_rng(0)

    def bid(name):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)

    def sid(name):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)

    print(f"{'leg':>3} {'set (coxa,femur,tibia)':>28}   {'recovered':>28}  {'err (rad)':>10}")
    max_err = 0.0
    for i in range(6):
        coxa_b, femur_b, tibia_b = bid(f"coxa{i}"), bid(f"femur{i}"), bid(f"tibia{i}")
        # Segment lengths straight from the model geometry (local +x offsets).
        Lc = model.body_pos[femur_b][0]      # coxa joint -> femur joint
        Lf = model.body_pos[tibia_b][0]      # femur joint -> tibia joint
        Lt = model.site_pos[sid(f"foot{i}")][0]  # tibia joint -> foot
        lengths = np.array([Lc, Lf, Lt])

        # Random joint angles within range; random base pose.
        jadr = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{s}{i}")]
                for s in ("coxa", "femur", "tibia")]
        lims = np.array([model.jnt_range[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{s}{i}")]
                         for s in ("coxa", "femur", "tibia")])
        angles = rng.uniform(lims[:, 0], lims[:, 1])

        mujoco.mj_resetDataKeyframe(model, data, 0)
        data.qpos[:3] = rng.uniform(-0.1, 0.1, 3)
        bq = rng.normal(size=4); data.qpos[3:7] = bq / np.linalg.norm(bq)
        for adr, a in zip(jadr, angles):
            data.qpos[adr] = a
        mujoco.mj_forward(model, data)

        # Shoulder pose from the base pose (independent of joint values).
        R_base = data.xmat[base].reshape(3, 3)
        x_sh = data.xpos[base] + R_base @ model.body_pos[coxa_b]
        R_sh = R_base @ _rotmat_wxyz(model.body_quat[coxa_b])
        q_sh = np.zeros(4); mujoco.mju_mat2Quat(q_sh, R_sh.flatten())
        y_foot = data.site_xpos[sid(f"foot{i}")]

        sols, reachable = leg_ik(x_sh, q_sh, y_foot, lengths)
        # Match against whichever of the two branches is closest.
        err = np.min(np.abs((sols - angles + np.pi) % (2 * np.pi) - np.pi).max(axis=1))
        max_err = max(max_err, err)
        best = sols[np.argmin(np.abs((sols - angles + np.pi) % (2 * np.pi) - np.pi).max(axis=1))]
        print(f"{i:>3} {np.array2string(angles, precision=3, floatmode='fixed'):>28}   "
              f"{np.array2string(best, precision=3, floatmode='fixed'):>28}  {err:10.2e}")

    print(f"\nmax angle error over 6 legs: {max_err:.2e} rad")


if __name__ == "__main__":
    _verify()
