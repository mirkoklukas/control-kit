"""Prototype: MuJoCo feed-forward stability check for ONE hexapod stance (CPU).

Standalone, plain MuJoCo -- no jax, no imports from the rest of lab/ik, so it runs
in the base venv: ``uv run python -m lab.ik.mj_stability``.

The "state" we check: joints held at their angles by a feed-forward gravity-
compensation torque, base floating on the planted feet. We then do a single
``mj_forward`` and read whether the contacts can balance the body.

Recipe (one pose, one planted set):
  1. disable contacts on the free (non-planted) legs -> only the planted feet support;
  2. feed-forward the gravity-comp torque on the joints so they don't buckle;
  3. one ``mj_forward``; read
       - base qacc  (small angular accel => not tipping),
       - planted-foot normal forces (should be > 0, pushing up),
       - required holding torque vs the actuator limit (+/-8 Nm).

This is the cheap static check; a short settle (also here) is the honest cross-check.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models" / "hexapod.xml"
TORQUE_LIMIT = 8.0   # actuator forcerange in the model


def load():
    return mujoco.MjModel.from_xml_path(str(MODEL))


def sink(qpos, dz=0.002):
    """Lower the base by ``dz`` so the planted feet penetrate slightly and the
    contacts are actually loaded. This is a *placement adjustment* done by the
    caller before checking -- the stability methods check whatever pose they get.
    (A pose with feet exactly on the surface has zero penetration => no contact
    force => the static one-forward check sees free-fall.)
    """
    q = np.array(qpos, dtype=float).copy()
    q[2] -= dz
    return q


def _bid(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def _leg_geom_ids(model, legs):
    bodies = {_bid(model, f"{s}{i}") for i in legs for s in ("coxa", "femur", "tibia")}
    return [g for g in range(model.ngeom) if model.geom_bodyid[g] in bodies]


def _joint_dofs(model):
    """Dof indices of the 18 hinge joints (everything past the freejoint)."""
    free = int(np.argmax(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE))
    base = model.jnt_dofadr[free]
    return np.array([d for d in range(model.nv) if not (base <= d < base + 6)])


def set_free_collisions(model, planted, on):
    """Toggle floor contacts for the non-planted legs (conaffinity 0/1)."""
    free = [i for i in range(6) if i not in planted]
    for g in _leg_geom_ids(model, free):
        model.geom_conaffinity[g] = 1 if on else 0


def _foot_normal_forces(model, data, planted):
    """Total contact normal force under each planted foot."""
    foot = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"foot{i}"): i for i in planted}
    out = {i: 0.0 for i in planted}
    f6 = np.zeros(6)
    for c in range(data.ncon):
        con = data.contact[c]
        if con.geom1 in foot or con.geom2 in foot:
            mujoco.mj_contactForce(model, data, c, f6)
            gid = con.geom1 if con.geom1 in foot else con.geom2
            out[foot[gid]] += float(f6[0])   # f6[0] = normal component
    return out


def check_static(model, qpos, planted):
    """Feed-forward-hold static stability check. Returns a report dict.

    qpos    : (nq,) full configuration.
    planted : list of planted leg indices (the rest are ignored / non-colliding).
    """
    set_free_collisions(model, planted, on=False)
    jdof = _joint_dofs(model)

    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)                       # compute qfrc_bias (gravity)

    tau = data.qfrc_bias[jdof].copy()                    # torque needed to hold the joints
    data.qfrc_applied[jdof] = tau                        # feed it forward
    mujoco.mj_forward(model, data)                       # re-solve with joints held

    forces = _foot_normal_forces(model, data, planted)
    return {
        "base_lin_acc": float(np.linalg.norm(data.qacc[0:3])),   # m/s^2
        "base_ang_acc": float(np.linalg.norm(data.qacc[3:6])),   # rad/s^2  <- tipping signal
        "foot_forces": forces,                                   # N per planted foot
        "min_foot_force": float(min(forces.values())),
        "max_hold_torque": float(np.max(np.abs(tau))),           # Nm
        "within_limits": bool(np.all(np.abs(tau) <= TORQUE_LIMIT)),
    }


def settle(model, qpos, planted, seconds=0.4):
    """Honest cross-check: hold the joints (servo + gravity comp) and integrate;
    report how far the base drops and tilts. Small drop/tilt => held."""
    set_free_collisions(model, planted, on=False)
    jdof = _joint_dofs(model)
    theta = qpos[7:]                                      # joint targets (18,)

    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    z0 = float(data.qpos[2])
    quat0 = data.qpos[3:7].copy()

    for _ in range(int(seconds / model.opt.timestep)):
        data.qfrc_applied[jdof] = data.qfrc_bias[jdof]   # gravity comp
        data.ctrl[:] = theta                             # position servos hold the rest
        mujoco.mj_step(model, data)

    # tilt = angle between the settled and initial body orientations
    dq = np.zeros(4)
    mujoco.mju_mulQuat(dq, data.qpos[3:7], np.array([quat0[0], -quat0[1], -quat0[2], -quat0[3]]))
    tilt = 2.0 * np.arccos(np.clip(abs(dq[0]), 0.0, 1.0))
    return {"z_drop": z0 - float(data.qpos[2]), "tilt_deg": float(np.degrees(tilt))}


def _demo():
    model = load()
    home = model.key_qpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")].copy()

    cases = {
        "alternating tripod [0,2,4] (should be STABLE)": [0, 2, 4],
        "adjacent legs   [0,1,2] (should TIP, CoM outside)": [0, 1, 2],
        "two legs        [0,3]   (line support, should TIP)": [0, 3],
    }
    for name, planted in cases.items():
        model = load()   # fresh model each case (set_free_collisions mutates it)
        s = check_static(model, sink(home), planted)   # hand in an adjusted (sunk) pose
        d = settle(load(), home, planted)              # settle engages contacts on its own
        print(f"\n{name}")
        print(f"  static: base_ang_acc={s['base_ang_acc']:7.2f} rad/s^2  "
              f"base_lin_acc={s['base_lin_acc']:6.2f}  min_foot_force={s['min_foot_force']:6.2f} N  "
              f"max_tau={s['max_hold_torque']:.2f} Nm (ok={s['within_limits']})")
        print(f"  settle: z_drop={d['z_drop']*1000:6.1f} mm   tilt={d['tilt_deg']:6.1f} deg")


if __name__ == "__main__":
    _demo()
