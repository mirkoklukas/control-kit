"""Reachable body poses for a planted-foot set, plus a static stability test.

Step one of the "alphabet of stable stances" work. Given a set of planted feet
(fixed at world targets) this module:

1. generates **reachable body poses** -- 6-DOF body poses ``(x, y, z, roll,
   pitch, yaw)`` for which every planted leg can reach its target within joint
   limits (analytic ``leg_ik``, knee-"down" branch). Swing (non-planted) legs are
   ignored for reachability and left at their home rest angles.
2. provides a **static stability test** -- the body CoM projected onto the ground
   must lie inside the support polygon (convex hull of the planted feet). Returns
   a signed ``margin`` (distance to the nearest edge, >0 inside).

The CoM here is a simple, MuJoCo-free proxy: a point rigidly attached to the base
(``com_offset`` in the base frame, default the base origin). Swap in the exact
``subtree_com`` later if you want leg mass accounted for -- see ``com_mujoco``.

Model geometry (shoulder offsets, segment lengths, joint limits, home targets) is
read once from the ``MjModel`` at construction; the hot path is pure numpy.

Run ``uv run python -m lab.ik.stance`` for a demo + end-to-end check.
"""
from __future__ import annotations

import dataclasses
import itertools

import numpy as np

from .leg_ik import leg_ik, _rotmat_wxyz

N_LEGS = 6


# --------------------------------------------------------------------------- #
# Small rotation / geometry helpers (pure numpy)                              #
# --------------------------------------------------------------------------- #
def quat_from_rpy(roll, pitch, yaw):
    """``wxyz`` quaternion for R = Rz(yaw) Ry(pitch) Rx(roll) (ZYX intrinsic)."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def _quat_from_mat(R):
    """``wxyz`` quaternion from a rotation matrix (Shepperd's stable branch)."""
    m00, m11, m22 = R[0, 0], R[1, 1], R[2, 2]
    tr = m00 + m11 + m22
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        w, x, y, z = 0.25 / s, (R[2, 1] - R[1, 2]) * s, (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif m11 > m22:
        s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    return np.array([w, x, y, z])


def _convex_hull_ccw(pts):
    """Counter-clockwise convex hull of 2D points (Andrew's monotone chain).

    Returns an ``(m, 2)`` array (>= 3 points), or the input if it is degenerate
    (fewer than 3 distinct / non-collinear points).
    """
    pts = np.unique(np.asarray(pts, float), axis=0)
    if len(pts) < 3:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def polygon_margin(p, poly):
    """Signed distance from point ``p`` to a CCW convex polygon.

    Positive inside (distance to the nearest edge), negative outside. Returns
    ``-inf`` for a degenerate polygon (fewer than 3 vertices).
    """
    poly = np.asarray(poly, float)
    if len(poly) < 3:
        return -np.inf
    p = np.asarray(p, float)
    dists = []
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        e = b - a
        # Signed distance to the line through edge (a, b); >0 to the left, i.e.
        # inside for a CCW polygon.
        dists.append((e[0] * (p[1] - a[1]) - e[1] * (p[0] - a[0])) / np.linalg.norm(e))
    return float(min(dists))


# --------------------------------------------------------------------------- #
# Stance model                                                                #
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class Stance:
    """Model-derived geometry for reachability + stability queries.

    Build with :meth:`from_model`. All arrays are indexed by leg 0..5. A body
    pose is a 6-vector ``(x, y, z, roll, pitch, yaw)``.
    """

    shoulder_pos: np.ndarray   # (6, 3) coxa-joint offset in the base frame
    shoulder_quat: np.ndarray  # (6, 4) coxa-frame orientation in the base frame (wxyz)
    lengths: np.ndarray        # (6, 3) (Lc, Lf, Lt) per leg
    jnt_lo: np.ndarray         # (6, 3) lower joint limits (rad)
    jnt_hi: np.ndarray         # (6, 3) upper joint limits (rad)
    foot_home: np.ndarray      # (6, 3) foot world positions at the home pose (default targets)
    qadr: np.ndarray           # (6, 3) qpos addresses of each leg's joints
    home_qpos: np.ndarray      # (nq,) home keyframe (seeds swing legs + root)
    com_offset: np.ndarray     # (3,) CoM proxy in the base frame (default 0 = base origin)

    @classmethod
    def from_model(cls, model, seed_key: str = "home", com_offset=None) -> "Stance":
        import mujoco

        def jid(name):
            return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)

        def bid(name):
            return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)

        def sid(name):
            return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)

        shoulder_pos = np.zeros((N_LEGS, 3))
        shoulder_quat = np.zeros((N_LEGS, 4))
        lengths = np.zeros((N_LEGS, 3))
        jnt_lo = np.zeros((N_LEGS, 3))
        jnt_hi = np.zeros((N_LEGS, 3))
        qadr = np.zeros((N_LEGS, 3), dtype=int)
        for i in range(N_LEGS):
            coxa_b, femur_b, tibia_b = bid(f"coxa{i}"), bid(f"femur{i}"), bid(f"tibia{i}")
            shoulder_pos[i] = model.body_pos[coxa_b]
            shoulder_quat[i] = model.body_quat[coxa_b]
            # Segment lengths = local +x offsets down the chain.
            lengths[i] = [model.body_pos[femur_b][0], model.body_pos[tibia_b][0],
                          model.site_pos[sid(f"foot{i}")][0]]
            js = [jid(f"coxa{i}"), jid(f"femur{i}"), jid(f"tibia{i}")]
            jnt_lo[i] = [model.jnt_range[j][0] for j in js]
            jnt_hi[i] = [model.jnt_range[j][1] for j in js]
            qadr[i] = [model.jnt_qposadr[j] for j in js]

        key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, seed_key)
        home_qpos = model.key_qpos[key].copy()

        # Home foot world positions = default plant targets.
        data = mujoco.MjData(model)
        data.qpos[:] = home_qpos
        mujoco.mj_forward(model, data)
        foot_home = np.array([data.site_xpos[sid(f"foot{i}")] for i in range(N_LEGS)])

        return cls(
            shoulder_pos=shoulder_pos, shoulder_quat=shoulder_quat, lengths=lengths,
            jnt_lo=jnt_lo, jnt_hi=jnt_hi, foot_home=foot_home, qadr=qadr,
            home_qpos=home_qpos,
            com_offset=np.zeros(3) if com_offset is None else np.asarray(com_offset, float),
        )

    # --- reachability -------------------------------------------------------- #
    def _pose_frames(self, pose6):
        """Return ``(base_pos, R_base)`` for a 6-vector body pose."""
        base_pos = np.asarray(pose6[:3], float)
        R_base = _rotmat_wxyz(quat_from_rpy(*pose6[3:6]))
        return base_pos, R_base

    def leg_angles(self, pose6, leg, target):
        """Knee-"down" joint angles for one leg to hold ``target`` at ``pose6``.

        Returns ``(angles (3,), ok)``. ``ok`` is False if the target is out of
        the leg's reach or the solution violates a joint limit.
        """
        base_pos, R_base = self._pose_frames(pose6)
        x_sh = base_pos + R_base @ self.shoulder_pos[leg]
        R_sh = R_base @ _rotmat_wxyz(self.shoulder_quat[leg])
        sols, reachable = leg_ik(x_sh, _quat_from_mat(R_sh), target, self.lengths[leg])
        angles = sols[0]  # elbow-"down" branch (+arccos), matches home stance
        ok = bool(reachable) and bool(
            np.all(angles >= self.jnt_lo[leg] - 1e-9) and np.all(angles <= self.jnt_hi[leg] + 1e-9)
        )
        return angles, ok

    def reachable(self, pose6, planted, targets=None):
        """Is ``pose6`` reachable with the given planted feet?

        Returns ``(ok, qpos)``. ``qpos`` is the full configuration (planted legs
        solved, swing legs at home rest) when ``ok``, else ``None``.
        """
        targets = self._targets(planted, targets)
        q = self.home_qpos.copy()
        q[:3] = pose6[:3]
        q[3:7] = quat_from_rpy(*pose6[3:6])
        for leg in planted:
            angles, ok = self.leg_angles(pose6, leg, targets[leg])
            if not ok:
                return False, None
            q[self.qadr[leg]] = angles
        return True, q

    # --- stability (pure numpy CoM proxy) ----------------------------------- #
    def com(self, pose6):
        """World-frame CoM proxy: a base-attached point (default the base origin)."""
        base_pos, R_base = self._pose_frames(pose6)
        return base_pos + R_base @ self.com_offset

    def support_polygon(self, planted, targets=None):
        """CCW convex hull (xy) of the planted feet -- the support polygon."""
        targets = self._targets(planted, targets)
        return _convex_hull_ccw(np.array([targets[i][:2] for i in planted]))

    def stability(self, pose6, planted, targets=None):
        """Static stability: ``(stable, margin)`` from the CoM-in-polygon test."""
        margin = polygon_margin(self.com(pose6)[:2], self.support_polygon(planted, targets))
        return margin > 0.0, margin

    # --- helpers ------------------------------------------------------------- #
    def _targets(self, planted, targets):
        if targets is None:
            return {i: self.foot_home[i] for i in planted}
        return {i: np.asarray(targets[i], float) for i in planted}


# --------------------------------------------------------------------------- #
# Body-pose generation                                                        #
# --------------------------------------------------------------------------- #
def sample_reachable(stance, planted, lo, hi, n=2000, mode="random", per_axis=None,
                     targets=None, seed=0):
    """Generate reachable body poses in the box ``[lo, hi]`` (6-vectors).

    ``mode="random"`` draws ``n`` uniform samples. ``mode="grid"`` sweeps a
    Cartesian grid with ``per_axis`` points on each of the 6 axes (a length-6
    list; use 1 to freeze an axis) -- coarseness is per-axis here since a uniform
    6-D grid explodes.

    Returns ``(poses (m, 6), qpos (m, nq))`` for the reachable subset.
    """
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    if mode == "random":
        rng = np.random.default_rng(seed)
        cand = rng.uniform(lo, hi, size=(n, 6))
    elif mode == "grid":
        if per_axis is None:
            raise ValueError("grid mode needs per_axis (length-6 list of counts)")
        axes = [np.linspace(lo[i], hi[i], per_axis[i]) for i in range(6)]
        cand = np.array(list(itertools.product(*axes)))
    else:
        raise ValueError(f"unknown mode {mode!r}")

    poses, qposes = [], []
    for pose6 in cand:
        ok, q = stance.reachable(pose6, planted, targets)
        if ok:
            poses.append(pose6)
            qposes.append(q)
    return (np.array(poses).reshape(-1, 6),
            np.array(qposes).reshape(-1, len(stance.home_qpos)))


# --------------------------------------------------------------------------- #
# Optional: exact CoM via MuJoCo (leg mass included)                          #
# --------------------------------------------------------------------------- #
def com_mujoco(model, qpos):
    """Exact whole-robot CoM for a full ``qpos`` (uses MuJoCo). For when the
    base-origin proxy is too crude and you want leg mass accounted for."""
    import mujoco

    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    free = int(np.argmax(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE))
    return np.array(data.subtree_com[model.jnt_bodyid[free]])


# --------------------------------------------------------------------------- #
def _demo():
    from pathlib import Path
    import mujoco

    root = Path(__file__).resolve().parents[2]
    model = mujoco.MjModel.from_xml_path(str(root / "models" / "hexapod.xml"))
    stance = Stance.from_model(model)

    planted = [0, 2, 4]  # alternating tripod
    home = stance.home_qpos
    base0 = np.array([home[0], home[1], home[2], 0.0, 0.0, 0.0])

    # Wide box so we exercise both boundaries: big xy/z shifts push feet out of
    # reach, big xy shifts push the CoM outside the support triangle.
    #   +/- 18 cm x/y, [-14, +6] cm z, +/- 0.4 rad roll/pitch/yaw.
    lo = base0 + np.array([-0.18, -0.18, -0.14, -0.4, -0.4, -0.4])
    hi = base0 + np.array([0.18, 0.18, 0.06, 0.4, 0.4, 0.4])

    poses, qpos = sample_reachable(stance, planted, lo, hi, n=3000)
    print(f"planted feet {planted}: {len(poses)}/3000 body poses reachable")

    # Stability over the reachable set.
    margins = np.array([stance.stability(p, planted)[1] for p in poses])
    stable = margins > 0
    print(f"statically stable (CoM in support triangle): {stable.sum()}/{len(poses)}")
    print(f"margin range: [{margins.min():+.3f}, {margins.max():+.3f}] m  "
          f"(home margin {stance.stability(base0, planted)[1]:+.3f})")

    # Stability test discriminates (decoupled from reach): shove the base far
    # past the triangle edge -> CoM outside -> unstable, negative margin.
    for label, shift in [("home", 0.0), ("+50cm x", 0.5)]:
        p = base0 + np.array([shift, 0, 0, 0, 0, 0])
        s, m = stance.stability(p, planted)
        print(f"stability probe {label:>8}: stable={s!s:>5}  margin={m:+.3f} m")

    # End-to-end check: assembled qpos actually holds the planted feet at target.
    data = mujoco.MjData(model)
    err = 0.0
    for q in qpos[:200]:
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        for i in planted:
            sidi = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"foot{i}")
            err = max(err, np.linalg.norm(data.site_xpos[sidi] - stance.foot_home[i]))
    print(f"max planted-foot placement error over 200 poses: {err:.2e} m")


if __name__ == "__main__":
    _demo()
