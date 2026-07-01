"""JAX (vmap/jit) version of the reachability + static-stability pipeline.

Mirrors :mod:`lab.ik.stance` but as a pure-jnp function you can ``vmap`` over a
pose batch and ``jit``. Use it to generate/score stable stances at scale.

Requires jax -- run under the mjx extra: ``uv run --extra mjx python -m
lab.ik.stance_jax``. The host-side model read still goes through the numpy
:class:`~lab.ik.stance.Stance`; this module just converts its constants to jnp
and provides the batched kernel.

Design notes
------------
* The plant targets are pose-independent, so the support polygon is constant over
  a pose batch. It is ordered CCW once on the host, so no convex hull (and no
  data-dependent control flow) is needed inside jax.
* The planted set is *static* per built kernel (closed over), so the polygon size
  ``k`` and the qpos scatter indices are compile-time constants.
* ``leg_ik`` here is the closed-form knee-"down" branch inlined in jnp.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from .stance import Stance, _rotmat_wxyz, _convex_hull_ccw


# --------------------------------------------------------------------------- #
# jnp helpers                                                                 #
# --------------------------------------------------------------------------- #
def _rotmat(q):
    """Rotation matrix from a ``wxyz`` quaternion (jnp; assumes unit norm)."""
    w, x, y, z = q
    return jnp.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


def _quat_from_rpy(rpy):
    """``wxyz`` quaternion for R = Rz(yaw) Ry(pitch) Rx(roll) (jnp)."""
    roll, pitch, yaw = rpy
    cr, sr = jnp.cos(roll / 2), jnp.sin(roll / 2)
    cp, sp = jnp.cos(pitch / 2), jnp.sin(pitch / 2)
    cy, sy = jnp.cos(yaw / 2), jnp.sin(yaw / 2)
    return jnp.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def _leg_down_angles(p, lengths):
    """Knee-"down" angles ``(3,)`` and a ``reachable`` flag for a foot expressed
    in the shoulder frame (``p``). Closed-form 3R; see :mod:`lab.ik.leg_ik`."""
    Lc, Lf, Lt = lengths
    theta1 = jnp.arctan2(p[1], p[0])
    u = jnp.hypot(p[0], p[1]) - Lc
    xi = -p[2]
    c3 = (u * u + xi * xi - Lf * Lf - Lt * Lt) / (2 * Lf * Lt)
    reachable = (c3 >= -1.0) & (c3 <= 1.0)
    theta3 = jnp.arccos(jnp.clip(c3, -1.0, 1.0))  # +arccos = elbow "down"
    theta2 = jnp.arctan2(xi, u) - jnp.arctan2(Lt * jnp.sin(theta3), Lf + Lt * jnp.cos(theta3))
    return jnp.array([theta1, theta2, theta3]), reachable


def _polygon_margin(p, poly):
    """Signed distance from ``p`` to a CCW convex polygon (jnp, fixed size).

    Positive inside (distance to nearest edge), negative outside.
    """
    a = poly
    b = jnp.roll(poly, -1, axis=0)
    e = b - a
    d = (e[:, 0] * (p[1] - a[:, 1]) - e[:, 1] * (p[0] - a[:, 0])) / jnp.linalg.norm(e, axis=1)
    return jnp.min(d)


# --------------------------------------------------------------------------- #
# Static geometry bundle (jnp)                                                #
# --------------------------------------------------------------------------- #
def geom_from_stance(stance: Stance) -> dict:
    """Convert a numpy :class:`Stance` into jnp constants for the kernel."""
    shoulder_R = np.stack([_rotmat_wxyz(q) for q in stance.shoulder_quat])  # (6,3,3)
    return dict(
        shoulder_pos=jnp.asarray(stance.shoulder_pos),
        shoulder_R=jnp.asarray(shoulder_R),
        lengths=jnp.asarray(stance.lengths),
        jnt_lo=jnp.asarray(stance.jnt_lo),
        jnt_hi=jnp.asarray(stance.jnt_hi),
        foot_home=jnp.asarray(stance.foot_home),
        qadr=stance.qadr,  # int indices, kept as numpy (static)
        home_qpos=jnp.asarray(stance.home_qpos),
        com_offset=jnp.asarray(stance.com_offset),
    )


def make_kernel(geom: dict, planted, targets=None):
    """Build a ``jit(vmap)`` kernel for a fixed planted set.

    ``planted`` is a tuple of leg indices (static). ``targets`` optionally
    overrides the plant points (``{leg: (3,) world}``); defaults to home feet.
    Returns a function ``poses (M, 6) -> dict`` with keys ``qpos (M, nq)``,
    ``reachable (M,)``, ``margin (M,)``, ``stable (M,)``.
    """
    planted = tuple(int(i) for i in planted)
    tgt = {i: (np.asarray(targets[i], float) if targets and i in targets
               else np.asarray(geom["foot_home"][i])) for i in planted}

    # Support polygon: planted feet ordered CCW, once, on the host.
    poly = jnp.asarray(_convex_hull_ccw(np.array([tgt[i][:2] for i in planted])))
    tgt_j = {i: jnp.asarray(tgt[i]) for i in planted}

    eps = 1e-9

    def solve_one(pose6):
        base_pos = pose6[:3]
        R_base = _rotmat(_quat_from_rpy(pose6[3:6]))

        q = geom["home_qpos"]
        q = q.at[:3].set(base_pos)
        q = q.at[3:7].set(_quat_from_rpy(pose6[3:6]))

        flags = []
        for leg in planted:
            x_sh = base_pos + R_base @ geom["shoulder_pos"][leg]
            R_sh = R_base @ geom["shoulder_R"][leg]
            p = R_sh.T @ (tgt_j[leg] - x_sh)  # foot in the shoulder frame
            angles, reach = _leg_down_angles(p, geom["lengths"][leg])
            within = jnp.all((angles >= geom["jnt_lo"][leg] - eps)
                             & (angles <= geom["jnt_hi"][leg] + eps))
            flags.append(reach & within)
            q = q.at[jnp.asarray(geom["qadr"][leg])].set(angles)

        reachable = jnp.all(jnp.stack(flags))
        com = base_pos + R_base @ geom["com_offset"]
        margin = _polygon_margin(com[:2], poly)
        return dict(qpos=q, reachable=reachable, margin=margin, stable=margin > 0.0)

    return jax.jit(jax.vmap(solve_one))


# --------------------------------------------------------------------------- #
def _demo():
    from pathlib import Path
    import time
    import mujoco

    jax.config.update("jax_enable_x64", True)  # tight parity with numpy reference

    root = Path(__file__).resolve().parents[2]
    model = mujoco.MjModel.from_xml_path(str(root / "models" / "hexapod.xml"))
    stance = Stance.from_model(model)
    geom = geom_from_stance(stance)

    planted = (0, 2, 4)
    kernel = make_kernel(geom, planted)

    home = np.asarray(stance.home_qpos)
    base0 = np.array([home[0], home[1], home[2], 0.0, 0.0, 0.0])
    lo = base0 + np.array([-0.18, -0.18, -0.14, -0.4, -0.4, -0.4])
    hi = base0 + np.array([0.18, 0.18, 0.06, 0.4, 0.4, 0.4])

    M = 200_000
    rng = np.random.default_rng(0)
    poses = jnp.asarray(rng.uniform(lo, hi, size=(M, 6)))

    out = kernel(poses)  # compile
    jax.block_until_ready(out)
    t = time.perf_counter()
    out = kernel(poses)
    jax.block_until_ready(out)
    dt = time.perf_counter() - t

    reach = np.asarray(out["reachable"])
    stable = np.asarray(out["stable"]) & reach
    print(f"batch {M} poses in {dt * 1e3:.1f} ms  ({M / dt / 1e6:.1f} M poses/s)")
    print(f"reachable: {reach.sum()}  |  reachable & stable: {stable.sum()}")

    # Parity vs the numpy reference on a few reachable poses.
    idx = np.where(reach)[0][:5]
    err = 0.0
    for k in idx:
        ok, q_np = stance.reachable(np.asarray(poses[k]), planted)
        err = max(err, np.abs(np.asarray(out["qpos"][k]) - q_np).max())
        m_np = stance.stability(np.asarray(poses[k]), planted)[1]
        err = max(err, abs(float(out["margin"][k]) - m_np))
    print(f"max |jax - numpy| (qpos & margin) over 5 poses: {err:.2e}")


if __name__ == "__main__":
    _demo()
