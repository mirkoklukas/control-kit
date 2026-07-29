"""Static stance forces: joint torques + foot reaction forces for a planted posture.

The statics layer, one above pure kinematics (:mod:`controlkit.kinematics`): where
kinematics maps a configuration to frames, this maps a *planted* configuration to
the forces that hold it. It is mjx-backed -- it needs a dynamics model for the
gravity generalized force and the foot Jacobians -- so it lives here, not in
``kinematics`` (which stays pure JAX).

Scenario -- given a body pose + joint angles + which feet are planted, solve the
statics (no stepping, no welds): the planted feet must supply the gravity wrench on
the body. One ``mjx.forward`` + a small least-squares, differentiable and vmappable
over N postures. (A weld only matters to *run* dynamics; for the force readout the
planted set just selects which feet's Jacobians bear load. A single forward of a
weld does not give equilibrium forces -- this does.)

Method -- static equilibrium at qvel = qacc = 0::

    sum_i  Jp_i^T f_i  =  qfrc_bias            (per dof; qfrc_bias = gravity gen. force)

  * base (free-joint) rows -> solve the foot forces f_i (feet support the body),
  * joint rows             -> tau = qfrc_bias_joint - sum_i Jp_i,joint^T f_i.

A planted foot with normal force <= 0 means the stance can't be held there (the
foot would have to pull / it lifts) -- a tip-over signal.

**Two layers here.** The ``stance_*`` core is model-agnostic: it takes
``(mjx_model, foot_ids, qpos, support)`` and reads the leg/joint counts off the
model (``nf = len(foot_ids)``, ``nj = (nv - 6) / nf``), assuming a floating base on a
6-DOF free joint. The ``posture_*`` / :func:`model_from_robot` bridge ties it to
:mod:`controlkit.kinematics` types -- a :class:`~controlkit.kinematics.Posture` and a
:class:`~controlkit.kinematics.Robot` (via its :meth:`Robot.to_mujoco`).

Run under ``uv run --extra mjx``.
"""

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from controlkit.kinematics import Posture, Robot


# --------------------------------------------------------------------------- #
# Model-agnostic core                                                         #
# --------------------------------------------------------------------------- #
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
    nf = foot_ids.shape[0]
    nj = (nv - 6) // nf
    d = mjx.forward(mjx_model, mjx.make_data(mjx_model).replace(
        qpos=qpos, qvel=jnp.zeros(nv)))
    g = d.qfrc_bias                                            # (nv,) gravity gen. force
    Jp = jax.vmap(lambda fid: mjx.jac(mjx_model, d, d.xpos[fid], fid)[0])(foot_ids)  # (nf,nv,3)
    w = support.astype(g.dtype)

    # base (free-joint) rows: sum_i w_i Jp[i,:6,:] @ f_i = g[:6]  ->  solve foot forces
    M = jnp.transpose(Jp[:, :6, :] * w[:, None, None], (1, 0, 2)).reshape(6, 3 * nf)
    f = jnp.linalg.lstsq(M, g[:6])[0].reshape(nf, 3) * w[:, None]        # (nf, 3)

    # joint rows: tau = g_joint - sum_i Jp[i,6:,:] @ f_i
    tau = g[6:] - jnp.einsum("inj,ij->n", Jp[:, 6:, :], f)              # (nv-6,)
    return f, tau.reshape(nf, nj)


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
    nf = foot_ids.shape[0]
    nj = (nv - 6) // nf
    d = mjx.forward(mjx_model, mjx.make_data(mjx_model).replace(
        qpos=qpos, qvel=jnp.zeros(nv)))
    g = d.qfrc_bias
    # per foot: [Jp | Jr] -> (nv, 6); columns 0:3 map force, 3:6 map moment.
    Jw = jax.vmap(lambda fid: jnp.concatenate(
        mjx.jac(mjx_model, d, d.xpos[fid], fid), axis=-1))(foot_ids)    # (nf, nv, 6)
    w = support.astype(g.dtype)

    # base rows: sum_i w_i Jw[i,:6,:] @ x_i = g[:6],  x_i = [f_i(3), m_i(3)]
    M = jnp.transpose(Jw[:, :6, :] * w[:, None, None], (1, 0, 2)).reshape(6, 6 * nf)
    x = jnp.linalg.lstsq(M, g[:6])[0].reshape(nf, 6) * w[:, None]       # (nf, 6)

    # joint rows: tau = g_joint - sum_i Jw[i,6:,:] @ x_i
    tau = g[6:] - jnp.einsum("inj,ij->n", Jw[:, 6:, :], x)
    return x[:, :3], x[:, 3:], tau.reshape(nf, nj)


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
    nf = foot_ids.shape[0]
    nj = (nv - 6) // nf
    d = mjx.forward(mjx_model, mjx.make_data(mjx_model).replace(
        qpos=qpos, qvel=jnp.zeros(nv)))
    g = d.qfrc_bias
    Jp = jax.vmap(lambda fid: mjx.jac(mjx_model, d, d.xpos[fid], fid)[0])(foot_ids)  # (nf,nv,3)
    w = support.astype(g.dtype)

    M = jnp.transpose(Jp[:, :6, :] * w[:, None, None], (1, 0, 2)).reshape(6, 3 * nf)
    Mpinv = jnp.linalg.pinv(M)                                 # (3*nf, 6); zero rows for lifted feet
    f = (Mpinv @ g[:6]).reshape(nf, 3)                        # nominal foot forces (= lstsq)
    dfdw = -Mpinv.reshape(nf, 3, 6)                           # d(foot force) / d(body wrench)

    Jj = Jp[:, 6:, :]                                          # (nf, nv-6, 3)
    tau = g[6:] - jnp.einsum("inj,ij->n", Jj, f)
    dtaudw = -jnp.einsum("inj,ijk->nk", Jj, dfdw)            # (nv-6, 6)
    return f, tau.reshape(nf, nj), dfdw, dtaudw.reshape(nf, nj, 6)


def stance_sigma_min(mjx_model, foot_ids, qpos, support, *, length):
    """Smallest singular value of the (force-model) base map ``M = J_baseᵀ``.

    ``M`` (6 x 3*nf) sends planted-foot forces to the net body wrench. Its smallest
    singular value is a stance degeneracy / fragility score for the **force** model:
    ``0`` means a free mode (a body wrench no foot force can resist, equivalently a
    base twist that moves the body with the feet fixed); tiny means supportable only
    with ``~1/sigma_min`` larger forces; healthy means well supported. (The wrench
    model adds per-foot moment columns and is usually full rank, so this does not
    diagnose it -- use a support-geometry check there.)

    The 6 rows mix force (N) and torque (N*m), so the torque rows (3:6) are scaled
    by ``1/length`` to make them commensurate before the SVD.

    Args:
        mjx_model: the ``put_model``'d model.
        foot_ids: (nf,) foot body ids.
        qpos: (nq,) configuration.
        support: (nf,) bool planted mask.
        length: characteristic length used to nondimensionalize the torque rows
            (e.g. the shoulder circumradius). Model-dependent, so it has no default.

    Returns:
        sigma_min: scalar smallest singular value (nondimensionalized).
    """
    nv = mjx_model.nv
    nf = foot_ids.shape[0]
    d = mjx.forward(mjx_model, mjx.make_data(mjx_model).replace(
        qpos=qpos, qvel=jnp.zeros(nv)))
    Jp = jax.vmap(lambda fid: mjx.jac(mjx_model, d, d.xpos[fid], fid)[0])(foot_ids)  # (nf,nv,3)
    w = support.astype(Jp.dtype)
    M = jnp.transpose(Jp[:, :6, :] * w[:, None, None], (1, 0, 2)).reshape(6, 3 * nf)
    M = M.at[3:, :].multiply(1.0 / length)                     # torque rows -> force-commensurate
    sv = jnp.linalg.svd(M, compute_uv=False)                   # min(6, 3*nf) values
    # The base map R^{3*nf} (foot forces) -> R^6 (body wrench): its smallest singular
    # value *in the 6-dim wrench space* is 0 whenever the feet can't span it, i.e.
    # 3*nf < 6 (a single leg). svd returns only min(6, 3*nf) values, so that
    # structural zero is absent from `sv`; supply it. With >= 2 legs, `sv[-1]`.
    return sv[-1] if 3 * nf >= 6 else jnp.asarray(0.0, sv.dtype)


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

    Returns foot_forces (N, nf, 3), foot_moments (N, nf, 3), joint_torques (N, nf, nj).
    """
    return jax.vmap(lambda q, s: stance_wrenches(mjx_model, foot_ids, q, s))(qpos, supports)


def stance_sensitivity_batch(mjx_model, foot_ids, qpos, supports):
    """Vectorized :func:`stance_sensitivity` over N postures.

    Returns foot_forces (N,nf,3), joint_torques (N,nf,nj), dfdw (N,nf,3,6),
    dtaudw (N,nf,nj,6).
    """
    return jax.vmap(lambda q, s: stance_sensitivity(mjx_model, foot_ids, q, s))(qpos, supports)


def stance_sigma_min_batch(mjx_model, foot_ids, qpos, supports, *, length):
    """Vectorized :func:`stance_sigma_min` over N postures.

    Args:
        mjx_model: the ``put_model``'d model.
        foot_ids: (nf,) foot body ids.
        qpos: (N, nq) configurations.
        supports: (N, nf) bool planted masks.
        length: characteristic length (shared across the batch; see
            :func:`stance_sigma_min`).

    Returns:
        sigma_min: (N,) smallest singular value per posture.
    """
    return jax.vmap(lambda q, s: stance_sigma_min(
        mjx_model, foot_ids, q, s, length=length))(qpos, supports)


# --------------------------------------------------------------------------- #
# Bridge to controlkit.kinematics (Posture / Robot)                           #
# --------------------------------------------------------------------------- #
def to_qpos(posture: Posture) -> jax.Array:
    """Assemble a model ``qpos`` from a :class:`~controlkit.kinematics.Posture`.

    Reorders the base pose from jaxlie's ``wxyz_xyz`` (quat-first) to MuJoCo's
    free-joint layout (position-first) and appends the flattened joint angles. This
    matches the model :meth:`Robot.to_mujoco` emits (a ``root`` free joint, then the
    per-leg hinges in leg-major order). vmap over a batched ``Posture`` for ``(N,
    nq)``.

    Args:
        posture: a ``Posture`` (single; ``body`` SE3 + ``(num_legs, num_joints)``
            ``thetas``).

    Returns:
        (nq,) qpos ``[x y z qw qx qy qz | num_legs*num_joints joint angles]``.
    """
    wxyz_xyz = posture.body.wxyz_xyz                            # (7,): qw qx qy qz x y z
    free = jnp.concatenate([wxyz_xyz[4:], wxyz_xyz[:4]])       # -> x y z qw qx qy qz
    return jnp.concatenate([free, jnp.asarray(posture.thetas).reshape(-1)])


def model_from_robot(robot: Robot, **mjcf_kwargs):
    """Build an mjx model + foot ids from a :class:`~controlkit.kinematics.Robot`.

    Compiles :meth:`Robot.to_mujoco` (kwargs forwarded), puts it on device, and
    locates the ``foot{i}`` bodies. Ready to feed the ``stance_*`` solvers.

    Mass comes from geom density: unless you pass ``body_density`` / ``leg_density``
    / ``foot_density`` (forwarded to :meth:`Robot.to_mjcf`), the model uses MuJoCo's
    default 1000 kg/m^3 and force *magnitudes* are only as good as that inertia.
    Relative stance analysis (degeneracy, which feet bear load) is fine regardless;
    for true magnitudes pass the real densities.

    Args:
        robot: the robot.
        **mjcf_kwargs: forwarded to :meth:`Robot.to_mjcf` (link/foot radius, the
            per-part densities, etc.).

    Returns:
        ``(mjx_model, foot_ids)`` -- the ``put_model``'d model and (num_legs,) foot
        body ids in leg order.
    """
    m = robot.to_mujoco(**mjcf_kwargs)
    mx = mjx.put_model(m)
    foot_ids = jnp.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"foot{i}")
                          for i in range(robot.num_legs)])
    return mx, foot_ids


def posture_forces(mjx_model, foot_ids, posture: Posture, support):
    """:func:`stance_forces` for a ``Posture`` (via :func:`to_qpos`)."""
    return stance_forces(mjx_model, foot_ids, to_qpos(posture), support)


def posture_wrenches(mjx_model, foot_ids, posture: Posture, support):
    """:func:`stance_wrenches` for a ``Posture`` (via :func:`to_qpos`)."""
    return stance_wrenches(mjx_model, foot_ids, to_qpos(posture), support)


def posture_sigma_min(mjx_model, foot_ids, posture: Posture, support, *, length):
    """:func:`stance_sigma_min` for a ``Posture`` (via :func:`to_qpos`)."""
    return stance_sigma_min(mjx_model, foot_ids, to_qpos(posture), support, length=length)
