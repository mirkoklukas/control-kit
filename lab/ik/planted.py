"""Solve a hexapod configuration ``q`` (``qpos``) for a given body pose while a
fixed set of feet stay planted at fixed world positions.

The body pose (base position + orientation) is the *free* variable; the leg
joints are whatever keeps the planted feet at their targets. Because each leg is
an independent 3R chain (coxa yaw -> femur lift -> tibia knee), planting foot ``i``
is a square 3-DOF-to-3D-point inverse-kinematics problem solved *per leg*, with
no coupling between legs. Non-planted legs are left at the seed pose.

Per-leg IK is damped-least-squares Newton on the leg's three joints:

    err = target - p_foot(q)                       # 3-vector, world frame
    J   = d p_foot / d q_leg                        # 3x3 site Jacobian block
    dq  = J^T (J J^T + lambda^2 I)^{-1} err         # damped step
    q_leg <- clamp(q_leg + dq, joint_range)

using MuJoCo's own kinematics (``mj_jacSite``), so the coxa frame offsets in the
MJCF are handled by the model rather than re-derived by hand. This is CPU MuJoCo
only -- no MJX / jax.

Coordinates. ``qpos`` (``nq = 25``) is the configuration array: freejoint root
(3 position + 4 quaternion, ``wxyz``) then 18 hinge angles. The quaternion is 4
numbers for 3 rotational DOF, so the true DOF count is ``nv = 24`` -- and
Jacobian columns are indexed by *dof* address, not *qpos* address. Those two
addresses coincide for the hinges but differ past the root because of the extra
quaternion slot, which is why the two are tracked separately below. Planting all
six feet removes ``6 x 3 = 18`` of the 24 DOF, leaving the 6-DOF body pose free
-- exactly the free variable this solver takes as input.

Run ``uv run python -m lab.ik.planted`` for a self-check.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models" / "hexapod.xml"

# Leg labels in <replicate> / MJCF order (see models/hexapod.xml):
#   0=FL(30)  1=ML(90)  2=BL(150)  3=BR(210)  4=MR(270)  5=FR(330)
N_LEGS = 6


def _name2id(model, objtype, name):
    """Resolve a MuJoCo name to its id, raising if it is missing (name lookups
    happen once at construction, never in the solve loop)."""
    i = mujoco.mj_name2id(model, objtype, name)
    if i < 0:
        raise KeyError(f"no {name!r} in model")
    return i


class PlantedSolver:
    """Per-body-pose IK with a fixed planted-foot set.

    Parameters
    ----------
    model : mujoco.MjModel
        The hexapod model.
    planted : iterable of int
        Foot indices (0..5) to keep planted.
    targets : dict[int, array-like] | None
        World-frame plant positions per foot index. Defaults to each planted
        foot's position in the ``home`` keyframe (feet on the ground at rest).
    seed_key : str
        Keyframe used to seed leg joints (and to source default targets).
    tol, max_iters, damping : float, int, float
        Newton tolerance (m), iteration cap, and DLS damping ``lambda``.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        planted,
        targets: dict | None = None,
        seed_key: str = "home",
        tol: float = 1e-8,
        max_iters: int = 50,
        damping: float = 1e-4,
    ):
        self.model = model
        self.data = mujoco.MjData(model)  # scratch, reused across solves
        self.planted = list(planted)
        self.tol = tol
        self.max_iters = max_iters
        self.damping = damping

        # Seed qpos from the keyframe. The seed sets the non-planted legs (left
        # untouched by solve) and the starting point for Newton on the planted
        # legs; only the root slice is overwritten per solve.
        key = _name2id(model, mujoco.mjtObj.mjOBJ_KEY, seed_key)
        self.seed_qpos = model.key_qpos[key].copy()

        # Root (freejoint) qpos slice start: 3 pos + 4 quat (wxyz) live at
        # [root_adr : root_adr + 7]. It is 0 for this model, but resolve it from
        # the model so the code is not tied to joint ordering.
        free = int(np.argmax(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE))
        self.root_adr = int(model.jnt_qposadr[free])

        # Per-leg ids, resolved once by name. Two *different* addresses per joint:
        #   qadr  -> index into qpos   (the value we update / clamp)
        #   dofadr-> index into qvel   (the column to pull from the site Jacobian)
        # They diverge after the root because the quaternion occupies 4 qpos slots
        # but only 3 dof slots (see module docstring).
        self.foot_site = {}
        self.qadr = {}  # leg -> (3,) qpos addresses of coxa/femur/tibia
        self.dofadr = {}  # leg -> (3,) dof (Jacobian column) addresses
        self.jnt_lo = {}  # leg -> (3,) lower joint limits (radians)
        self.jnt_hi = {}  # leg -> (3,) upper joint limits (radians)
        for i in range(N_LEGS):
            self.foot_site[i] = _name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"foot{i}")
            jids = [
                _name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{seg}{i}")
                for seg in ("coxa", "femur", "tibia")
            ]
            self.qadr[i] = np.array([model.jnt_qposadr[j] for j in jids])
            self.dofadr[i] = np.array([model.jnt_dofadr[j] for j in jids])
            # jnt_range is stored in radians (MJCF angle="degree" is converted at
            # compile time); columns are (lo, hi).
            rng = np.array([model.jnt_range[j] for j in jids])  # (3, 2)
            self.jnt_lo[i] = rng[:, 0]
            self.jnt_hi[i] = rng[:, 1]

        # Plant targets: per foot, the fixed world point its site must hold.
        # Default to the foot's position in the seed keyframe (feet resting on the
        # ground), overridable per index via ``targets``.
        self.targets = {}
        home_sites = self._forward_sites(self.seed_qpos)
        for i in self.planted:
            self.targets[i] = (
                np.asarray(targets[i], float) if targets and i in targets
                else home_sites[i].copy()
            )

    def _forward_sites(self, qpos):
        """Return (6, 3) foot-site world positions for a full ``qpos``.

        Runs forward kinematics into the scratch ``data``; used only at
        construction to read the default (home) plant targets.
        """
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)
        return np.array([self.data.site_xpos[self.foot_site[i]] for i in range(N_LEGS)])

    def solve(self, base_pos, base_quat, return_info: bool = False):
        """Solve for ``qpos`` at the given body pose with planted feet fixed.

        ``base_pos`` is (3,) world position; ``base_quat`` is (4,) ``wxyz`` (it is
        normalized here). Returns the solved ``qpos`` (25,), or ``None`` if any
        planted foot could not be reached within tolerance/limits. With
        ``return_info=True`` returns ``(qpos, info)`` where ``info`` maps each
        planted foot index to its final residual norm and always returns the
        (possibly imperfect) qpos.
        """
        # Start from the seed, then overwrite the root with the requested body
        # pose. The quaternion is normalized here so callers can pass an
        # un-normalized (e.g. perturbed) orientation.
        q = self.seed_qpos.copy()
        q[self.root_adr : self.root_adr + 3] = np.asarray(base_pos, float)
        quat = np.asarray(base_quat, float)
        quat = quat / np.linalg.norm(quat)
        q[self.root_adr + 3 : self.root_adr + 7] = quat

        d = self.data
        d.qpos[:] = q
        jacp = np.zeros((3, self.model.nv))  # site translational Jacobian buffer
        info = {}  # foot index -> final residual norm (m)
        ok = True

        # Each leg is solved independently: the root is fixed for this pose, so
        # moving leg i's joints cannot affect any other foot.
        for i in self.planted:
            qa, da = self.qadr[i], self.dofadr[i]  # this leg's qpos / dof cols
            lo, hi = self.jnt_lo[i], self.jnt_hi[i]
            site, target = self.foot_site[i], self.targets[i]
            lam2 = self.damping ** 2  # DLS regularizer

            err_norm = np.inf
            for _ in range(self.max_iters):
                # Refresh kinematics for the current qpos, then measure how far
                # this foot is from its plant target.
                mujoco.mj_forward(self.model, d)
                err = target - d.site_xpos[site]  # (3,) world-frame residual
                err_norm = np.linalg.norm(err)
                if err_norm < self.tol:
                    break
                # Damped-least-squares Newton step on the leg's 3 joints:
                #   dq = J^T (J J^T + lambda^2 I)^{-1} err
                # The damping keeps the step finite near singularities (e.g. a
                # fully extended leg where J loses rank).
                mujoco.mj_jacSite(self.model, d, jacp, None, site)
                J = jacp[:, da]  # (3, 3) block: d(foot pos) / d(leg joints)
                dq = J.T @ np.linalg.solve(J @ J.T + lam2 * np.eye(3), err)
                # Apply and clamp to joint limits (an unreachable target then
                # stalls against a limit and is reported below).
                d.qpos[qa] = np.clip(d.qpos[qa] + dq, lo, hi)

            info[i] = float(err_norm)
            # Treat as failed if it never got within a loose 0.1 mm (the tol used
            # for success is the coarser of the Newton tol and this floor).
            if err_norm >= max(self.tol, 1e-4):
                ok = False

        q_out = d.qpos.copy()
        if return_info:
            return q_out, info  # always hand back the best-effort qpos + residuals
        return q_out if ok else None


def _selfcheck():
    """Plant all six feet, jitter the body pose, and verify the feet stay put.

    Sanity check that the solved qpos actually holds every planted foot at its
    target as the body translates and yaws around the home pose -- prints the
    worst per-pose foot residual (should be ~1e-4 m or better).
    """
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    solver = PlantedSolver(model, planted=range(6))

    home = solver.seed_qpos
    base_pos0 = home[:3].copy()
    base_quat0 = home[3:7].copy()

    rng = np.random.default_rng(0)
    print(f"{'dx':>18} {'dz':>6} {'dyaw':>6} | {'max foot err (m)':>16}  status")
    for _ in range(8):
        # Small body-pose jitter: +/-3 cm horizontal, -4/+3 cm vertical, +/-0.15 rad yaw.
        dpos = rng.uniform([-0.03, -0.03, -0.04], [0.03, 0.03, 0.03])
        dyaw = rng.uniform(-0.15, 0.15)
        # Yaw perturbation about z (quat [cos(a/2), 0, 0, sin(a/2)]) composed
        # onto the home orientation.
        dq = np.array([np.cos(dyaw / 2), 0, 0, np.sin(dyaw / 2)])
        base_pos = base_pos0 + dpos
        base_quat = _quat_mul(dq, base_quat0)

        # return_info=True: keep the residuals even if a pose is unreachable.
        q, err = solver.solve(base_pos, base_quat, return_info=True)
        max_err = max(err.values())
        status = "ok" if max_err < 1e-4 else "UNREACHABLE"
        print(
            f"[{dpos[0]:+.3f},{dpos[1]:+.3f}] {dpos[2]:+.3f} {dyaw:+.3f} "
            f"| {max_err:16.2e}  {status}"
        )


def _quat_mul(a, b):
    """Hamilton product of two ``wxyz`` quaternions (a then b), returned ``wxyz``."""
    w0, x0, y0, z0 = a
    w1, x1, y1, z1 = b
    return np.array([
        w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
        w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
        w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
        w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
    ])


if __name__ == "__main__":
    _selfcheck()
