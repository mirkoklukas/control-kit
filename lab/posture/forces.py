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

This is the ``lab/posture`` variant of ``lab/stance_graph/mjx_stance``, wired to the
4-DOF-per-leg model ``6x4DOF.xml`` (24 joint DOF). Everything below is written in
terms of ``nf`` legs and ``(nv - 6) / nf`` joints per leg, so it does not assume a
particular joint count.

Run under ``uv run --extra mjx``.
"""
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from .kinematics import Posture, NUM_LEGS as LEGS, NUM_JOINTS as JOINTS

MODEL = Path(__file__).resolve().parent / "6x4DOF.xml"


def load_model(path=MODEL):
    """Load the model and locate the foot bodies.

    Args:
        path: model XML path.

    Returns:
        mjx_model: the ``put_model``'d model.
        foot_ids: (LEGS,) foot body ids, leg order 0..LEGS-1.
    """
    m = mujoco.MjModel.from_xml_path(str(path))
    mx = mjx.put_model(m)
    foot_ids = jnp.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"foot{i}")
                          for i in range(LEGS)])
    return mx, foot_ids


def to_qpos(posture: Posture):
    """Assemble a model ``qpos`` from a posture.

    Reorders the base pose from jaxlie's ``wxyz_xyz`` (quat-first) to MuJoCo's
    free-joint layout (position-first) and appends the flattened joint angles.
    vmap over a batched ``Posture`` for the ``(N, nq)`` stack.

    Args:
        posture: a :class:`..kinematics.Posture` (single; ``body`` SE3 +
            ``(NUM_LEGS, NUM_JOINTS)`` ``thetas``).

    Returns:
        (nq,) qpos ``[x y z qw qx qy qz | NUM_LEGS*NUM_JOINTS joint angles]``.
    """
    wxyz_xyz = posture.body.wxyz_xyz                             # (7,): qw qx qy qz x y z
    free = jnp.concatenate([wxyz_xyz[4:], wxyz_xyz[:4]])        # -> x y z qw qx qy qz
    return jnp.concatenate([free, jnp.asarray(posture.thetas).reshape(-1)])


def stance_forces(mjx_model, foot_ids, qpos, support):
    """Static foot reaction forces + joint hold torques for one planted posture.

    Args:
        mjx_model: the ``put_model``'d model.
        foot_ids: (nf,) foot body ids.
        qpos: (nq,) configuration.
        support: (nf,) bool -- True = that leg's foot is planted (bears load).

    Returns:
        foot_forces: (nf, 3) world-frame reaction force per foot; 0 for unplanted
            legs. A negative z means the stance would tip (the foot would lift).
        joint_torques: (nf, nj) actuator torque per leg (nj = joints per leg)
            needed to hold the pose under those foot reactions.
    """
    nv = mjx_model.nv
    d = mjx.forward(mjx_model, mjx.make_data(mjx_model).replace(
        qpos=qpos, qvel=jnp.zeros(nv)))
    g = d.qfrc_bias                                            # (nv,) gravity gen. force
    Jp = jax.vmap(lambda fid: mjx.jac(mjx_model, d, d.xpos[fid], fid)[0])(foot_ids)  # (LEGS,nv,3)
    w = support.astype(g.dtype)

    # base (free-joint) rows: sum_i w_i Jp[i,:6,:] @ f_i = g[:6]  ->  solve foot forces
    M = jnp.transpose(Jp[:, :6, :] * w[:, None, None], (1, 0, 2)).reshape(6, 3 * LEGS)
    f = jnp.linalg.lstsq(M, g[:6])[0].reshape(LEGS, 3) * w[:, None]      # (LEGS, 3)

    # joint rows: tau = g_joint - sum_i Jp[i,6:,:] @ f_i
    tau = g[6:] - jnp.einsum("inj,ij->n", Jp[:, 6:, :], f)              # (nv-6,)
    return f, tau.reshape(LEGS, JOINTS)


def posture_forces(mjx_model, foot_ids, posture: Posture):
    """:func:`stance_forces` for a :class:`..kinematics.Posture`.

    Builds ``qpos`` via :func:`to_qpos` and uses ``posture.stance.support`` as the
    planted mask. vmap over a batched ``Posture`` for N postures.

    Args:
        mjx_model: the ``put_model``'d model.
        foot_ids: (LEGS,) foot body ids.
        posture: a :class:`..kinematics.Posture` (single).

    Returns:
        foot_forces: (LEGS, 3) world-frame reaction force per foot (0 for unplanted
            legs); joint_torques: (LEGS, JOINTS) actuator torque per leg.
    """
    return stance_forces(mjx_model, foot_ids, to_qpos(posture), posture.stance.support)


def stance_wrenches(mjx_model, foot_ids, qpos, support):
    """Static 6-DOF version of :func:`stance_forces`: each planted foot may exert a
    full **wrench** (force + moment), as with a ``torquescale>0`` weld / a rigid
    grip that resists the foot twisting.

    Same statics as ``stance_forces`` but each foot contributes ``Jp^T f + Jr^T m``
    (translational Jacobian times force, plus rotational Jacobian times moment), so
    the per-foot unknown is a 6-vector ``[f, m]``. Solving the base rows now
    distributes the load into moments too (even more underdetermined -> min-norm).

    Args:
        mjx_model: the ``put_model``'d model.
        foot_ids: (nf,) foot body ids.
        qpos: (nq,) configuration.
        support: (nf,) bool -- True = that leg's foot is planted (grips).

    Returns:
        foot_forces: (nf, 3) world-frame reaction force per foot.
        foot_moments: (nf, 3) world-frame reaction moment per foot.
        joint_torques: (nf, nj) actuator torque per leg.
    """
    nv = mjx_model.nv
    d = mjx.forward(mjx_model, mjx.make_data(mjx_model).replace(
        qpos=qpos, qvel=jnp.zeros(nv)))
    g = d.qfrc_bias
    # per foot: [Jp | Jr] -> (nv, 6); columns 0:3 map force, 3:6 map moment.
    Jw = jax.vmap(lambda fid: jnp.concatenate(
        mjx.jac(mjx_model, d, d.xpos[fid], fid), axis=-1))(foot_ids)    # (LEGS, nv, 6)
    w = support.astype(g.dtype)

    # base rows: sum_i w_i Jw[i,:6,:] @ x_i = g[:6],  x_i = [f_i(3), m_i(3)]
    M = jnp.transpose(Jw[:, :6, :] * w[:, None, None], (1, 0, 2)).reshape(6, 6 * LEGS)
    x = jnp.linalg.lstsq(M, g[:6])[0].reshape(LEGS, 6) * w[:, None]     # (LEGS, 6)

    # joint rows: tau = g_joint - sum_i Jw[i,6:,:] @ x_i
    tau = g[6:] - jnp.einsum("inj,ij->n", Jw[:, 6:, :], x)
    return x[:, :3], x[:, 3:], tau.reshape(LEGS, JOINTS)


def posture_wrenches(mjx_model, foot_ids, posture: Posture):
    """:func:`stance_wrenches` for a :class:`..kinematics.Posture`.

    Builds ``qpos`` via :func:`to_qpos` and uses ``posture.stance.support`` as the
    planted mask. vmap over a batched ``Posture`` for N postures.

    Args:
        mjx_model: the ``put_model``'d model.
        foot_ids: (LEGS,) foot body ids.
        posture: a :class:`..kinematics.Posture` (single).

    Returns:
        foot_forces: (LEGS, 3), foot_moments: (LEGS, 3), joint_torques: (LEGS, JOINTS).
    """
    return stance_wrenches(mjx_model, foot_ids, to_qpos(posture), posture.stance.support)


def stance_sensitivity(mjx_model, foot_ids, qpos, support):
    """Static stance forces plus their sensitivity to a wrench applied to the body.

    Same base-equilibrium solve as :func:`stance_forces`, but through the
    pseudoinverse ``M+`` of ``M = J_base^T`` (the base-dof rows of the planted
    feet's Jacobians). ``M+`` gives the nominal forces (``f = M+ g_base``) *and* the
    linearized response to an external body wrench ``w`` -- a stance-stability map:
    applying ``w`` perturbs the base balance by ``-w``, so ``df = -M+ w`` and
    ``dtau = -sum_i J_joint_i @ df_i``. Small ``|dfdw|`` / staying admissible under
    the wrenches you expect => a robust stance.

    Args:
        mjx_model: the ``put_model``'d model.
        foot_ids: (nf,) foot body ids.
        qpos: (nq,) configuration.
        support: (nf,) bool planted mask.

    Returns:
        foot_forces: (nf, 3) nominal reaction force per foot (matches stance_forces).
        joint_torques: (nf, nj) nominal actuator torque per leg.
        dfdw: (nf, 3, 6) d(foot force) / d(body wrench). The wrench is
            ``[Fx, Fy, Fz, Tx, Ty, Tz]`` on the base dofs (force in world frame,
            torque in the base frame).
        dtaudw: (nf, nj, 6) d(joint torque) / d(body wrench).
    """
    nv = mjx_model.nv
    d = mjx.forward(mjx_model, mjx.make_data(mjx_model).replace(
        qpos=qpos, qvel=jnp.zeros(nv)))
    g = d.qfrc_bias
    Jp = jax.vmap(lambda fid: mjx.jac(mjx_model, d, d.xpos[fid], fid)[0])(foot_ids)  # (LEGS,nv,3)
    w = support.astype(g.dtype)

    M = jnp.transpose(Jp[:, :6, :] * w[:, None, None], (1, 0, 2)).reshape(6, 3 * LEGS)
    Mpinv = jnp.linalg.pinv(M)                                 # (3*LEGS, 6); zero rows for lifted feet
    f = (Mpinv @ g[:6]).reshape(LEGS, 3)                       # nominal foot forces (= lstsq)
    dfdw = -Mpinv.reshape(LEGS, 3, 6)                          # d(foot force) / d(body wrench)

    Jj = Jp[:, 6:, :]                                          # (LEGS, nv-6, 3)
    tau = g[6:] - jnp.einsum("inj,ij->n", Jj, f)
    dtaudw = -jnp.einsum("inj,ijk->nk", Jj, dfdw)             # (nv-6, 6)
    return f, tau.reshape(LEGS, JOINTS), dfdw, dtaudw.reshape(LEGS, JOINTS, 6)


def stance_forces_batch(mjx_model, foot_ids, qpos, supports):
    """Vectorized :func:`stance_forces` over N postures.

    Args:
        mjx_model: the ``put_model``'d model.
        foot_ids: (nf,) foot body ids.
        qpos: (N, nq) configurations.
        supports: (N, nf) bool planted masks.

    Returns:
        foot_forces: (N, nf, 3), joint_torques: (N, nf, nj).
    """
    return jax.vmap(lambda q, s: stance_forces(mjx_model, foot_ids, q, s))(qpos, supports)


def stance_wrenches_batch(mjx_model, foot_ids, qpos, supports):
    """Vectorized :func:`stance_wrenches` over N postures.

    Args:
        mjx_model: the ``put_model``'d model.
        foot_ids: (nf,) foot body ids.
        qpos: (N, nq) configurations.
        supports: (N, nf) bool planted masks.

    Returns:
        foot_forces: (N, nf, 3), foot_moments: (N, nf, 3), joint_torques: (N, nf, nj).
    """
    return jax.vmap(lambda q, s: stance_wrenches(mjx_model, foot_ids, q, s))(qpos, supports)


def stance_sensitivity_batch(mjx_model, foot_ids, qpos, supports):
    """Vectorized :func:`stance_sensitivity` over N postures.

    Returns foot_forces (N,nf,3), joint_torques (N,nf,nj), dfdw (N,nf,3,6),
    dtaudw (N,nf,nj,6).
    """
    return jax.vmap(lambda q, s: stance_sensitivity(mjx_model, foot_ids, q, s))(qpos, supports)


if __name__ == "__main__":
    import numpy as np

    m = mujoco.MjModel.from_xml_path(str(MODEL))
    mx, foot_ids = load_model()
    weight = float(m.body_mass.sum() * abs(m.opt.gravity[2]))
    home = jnp.array(m.key_qpos[0])

    supports = jnp.array([[1, 0, 1, 0, 1, 0],       # tripod
                         [1, 1, 1, 1, 1, 1],       # all six
                         [1, 0, 0, 1, 0, 0]], bool)  # two legs
    qpos = jnp.broadcast_to(home, (supports.shape[0], home.shape[0]))
    ff, tau = jax.jit(stance_forces_batch)(mx, foot_ids, qpos, supports)
    ff, tau = np.asarray(ff), np.asarray(tau)
    print(f"weight = {weight:.1f} N | nq={m.nq} nv={m.nv} tau/leg={tau.shape[-1]}")
    for i, name in enumerate(["tripod 0,2,4", "all six", "two 0,3"]):
        print(f"  {name:14s} sum Fz = {ff[i,:,2].sum():6.1f} N | "
              f"max |tau| = {np.abs(tau[i]).max():5.2f} Nm | "
              f"min planted Fz = {ff[i,:,2][ff[i,:,2] != 0].min():6.2f} N")
