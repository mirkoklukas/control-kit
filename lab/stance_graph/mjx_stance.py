"""Static stance forces: joint torques + foot reaction forces for a planted posture.

Scenario A -- given a body pose + joint angles + which feet are planted, solve the
statics (no stepping, no welds): the planted feet must supply the gravity wrench on
the body. This is one ``mjx.forward`` + a small least-squares, differentiable and
vmappable over N postures. (A weld only matters if you also want to run dynamics;
for the force *readout* the planted set just selects which feet's Jacobians bear
load. A single forward of a weld does NOT give equilibrium forces -- this does.)

Method -- static equilibrium at qvel = qacc = 0:

    sum_i  Jp_i^T f_i  =  qfrc_bias            (per dof; qfrc_bias = gravity gen. force)

  * base (free-joint) rows -> solve the foot forces f_i (feet support the body),
  * joint rows             -> tau = qfrc_bias_joint - sum_i Jp_i,joint^T f_i.

A planted foot with normal force <= 0 means the stance can't be held there (the
foot would have to pull / it lifts) -- a tip-over signal.

Run under ``uv run --extra mjx``.
"""
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from jaxlie import SE3

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models" / "weld0.xml"


def load_model(path=MODEL):
    """Load the model and locate the foot bodies.

    Args:
        path: model XML path.

    Returns:
        mjx_model: the ``put_model``'d model.
        foot_ids: (6,) foot body ids, leg order 0..5.
    """
    m = mujoco.MjModel.from_xml_path(str(path))
    mx = mjx.put_model(m)
    foot_ids = jnp.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"foot{i}")
                          for i in range(6)])
    return mx, foot_ids


def to_qpos(bodies: SE3, thetas):
    """Assemble qpos from base poses and joint angles.

    Args:
        bodies: (N,) SE3 base poses.
        thetas: (N, 6, 3) joint angles [coxa, femur, tibia] per leg.

    Returns:
        (N, nq) qpos: ``[x y z qw qx qy qz | 18 joint angles]``.
    """
    wxyz_xyz = bodies.wxyz_xyz                                    # (N,7): qw qx qy qz x y z
    free = jnp.concatenate([wxyz_xyz[:, 4:], wxyz_xyz[:, :4]], axis=1)
    return jnp.concatenate([free, thetas.reshape(thetas.shape[0], -1)], axis=1)


def stance_forces(mjx_model, foot_ids, qpos, stance):
    """Static foot reaction forces + joint hold torques for one planted posture.

    Args:
        mjx_model: the ``put_model``'d model.
        foot_ids: (6,) foot body ids.
        qpos: (nq,) configuration.
        stance: (6,) bool -- True = that leg's foot is planted (bears load).

    Returns:
        foot_forces: (6, 3) world-frame reaction force per foot; 0 for unplanted
            legs. A negative z means the stance would tip (the foot would lift).
        joint_torques: (6, 3) actuator torque [coxa, femur, tibia] per leg needed
            to hold the pose under those foot reactions.
    """
    nv = mjx_model.nv
    d = mjx.forward(mjx_model, mjx.make_data(mjx_model).replace(
        qpos=qpos, qvel=jnp.zeros(nv)))
    g = d.qfrc_bias                                              # (nv,) gravity gen. force
    Jp = jax.vmap(lambda fid: mjx.jac(mjx_model, d, d.xpos[fid], fid)[0])(foot_ids)  # (6,nv,3)
    w = stance.astype(g.dtype)

    # base (free-joint) rows: sum_i w_i Jp[i,:6,:] @ f_i = g[:6]  ->  solve foot forces
    M = jnp.transpose(Jp[:, :6, :] * w[:, None, None], (1, 0, 2)).reshape(6, 3 * foot_ids.shape[0])
    f = jnp.linalg.lstsq(M, g[:6])[0].reshape(-1, 3) * w[:, None]        # (6, 3)

    # joint rows: tau = g_joint - sum_i Jp[i,6:,:] @ f_i
    tau = g[6:] - jnp.einsum("inj,ij->n", Jp[:, 6:, :], f)              # (nv-6,)
    return f, tau.reshape(-1, 3)


def stance_wrenches(mjx_model, foot_ids, qpos, stance):
    """Static 6-DOF version of :func:`stance_forces`: each planted foot may exert a
    full **wrench** (force + moment), as with a ``torquescale>0`` weld / a rigid
    grip that resists the foot twisting.

    Same statics as ``stance_forces`` but each foot contributes ``Jp^T f + Jr^T m``
    (translational Jacobian times force, plus rotational Jacobian times moment), so
    the per-foot unknown is a 6-vector ``[f, m]``. Solving the base rows now
    distributes the load into moments too (even more underdetermined -> min-norm).

    Args:
        mjx_model: the ``put_model``'d model.
        foot_ids: (6,) foot body ids.
        qpos: (nq,) configuration.
        stance: (6,) bool -- True = that leg's foot is planted (grips).

    Returns:
        foot_forces: (6, 3) world-frame reaction force per foot.
        foot_moments: (6, 3) world-frame reaction moment per foot.
        joint_torques: (6, 3) actuator torque [coxa, femur, tibia] per leg.
    """
    nv, nf = mjx_model.nv, foot_ids.shape[0]
    d = mjx.forward(mjx_model, mjx.make_data(mjx_model).replace(
        qpos=qpos, qvel=jnp.zeros(nv)))
    g = d.qfrc_bias
    # per foot: [Jp | Jr] -> (nv, 6); columns 0:3 map force, 3:6 map moment.
    Jw = jax.vmap(lambda fid: jnp.concatenate(
        mjx.jac(mjx_model, d, d.xpos[fid], fid), axis=-1))(foot_ids)    # (6, nv, 6)
    w = stance.astype(g.dtype)

    # base rows: sum_i w_i Jw[i,:6,:] @ x_i = g[:6],  x_i = [f_i(3), m_i(3)]
    M = jnp.transpose(Jw[:, :6, :] * w[:, None, None], (1, 0, 2)).reshape(6, 6 * nf)
    x = jnp.linalg.lstsq(M, g[:6])[0].reshape(nf, 6) * w[:, None]       # (6, 6)

    # joint rows: tau = g_joint - sum_i Jw[i,6:,:] @ x_i
    tau = g[6:] - jnp.einsum("inj,ij->n", Jw[:, 6:, :], x)
    return x[:, :3], x[:, 3:], tau.reshape(-1, 3)


def stance_forces_batch(mjx_model, foot_ids, qpos, stances):
    """Vectorized :func:`stance_forces` over N postures.

    Args:
        mjx_model: the ``put_model``'d model.
        foot_ids: (6,) foot body ids.
        qpos: (N, nq) configurations.
        stances: (N, 6) bool planted masks.

    Returns:
        foot_forces: (N, 6, 3), joint_torques: (N, 6, 3).
    """
    return jax.vmap(lambda q, s: stance_forces(mjx_model, foot_ids, q, s))(qpos, stances)


def stance_wrenches_batch(mjx_model, foot_ids, qpos, stances):
    """Vectorized :func:`stance_wrenches` over N postures.

    Args:
        mjx_model: the ``put_model``'d model.
        foot_ids: (6,) foot body ids.
        qpos: (N, nq) configurations.
        stances: (N, 6) bool planted masks.

    Returns:
        foot_forces: (N, 6, 3), foot_moments: (N, 6, 3), joint_torques: (N, 6, 3).
    """
    return jax.vmap(lambda q, s: stance_wrenches(mjx_model, foot_ids, q, s))(qpos, stances)


if __name__ == "__main__":
    import numpy as np

    m = mujoco.MjModel.from_xml_path(str(MODEL))
    mx, foot_ids = load_model()
    weight = float(m.body_mass.sum() * abs(m.opt.gravity[2]))
    home = jnp.array(m.key_qpos[0])

    stances = jnp.array([[1, 0, 1, 0, 1, 0],       # tripod
                         [1, 1, 1, 1, 1, 1],       # all six
                         [1, 0, 0, 1, 0, 0]], bool)  # two legs
    qpos = jnp.broadcast_to(home, (stances.shape[0], home.shape[0]))
    ff, tau = jax.jit(stance_forces_batch)(mx, foot_ids, qpos, stances)
    ff, tau = np.asarray(ff), np.asarray(tau)
    print(f"weight = {weight:.1f} N")
    for i, name in enumerate(["tripod 0,2,4", "all six", "two 0,3"]):
        print(f"  {name:14s} sum Fz = {ff[i,:,2].sum():6.1f} N | "
              f"max |tau| = {np.abs(tau[i]).max():5.2f} Nm | "
              f"min planted Fz = {ff[i,:,2][ff[i,:,2] != 0].min():6.2f} N")
