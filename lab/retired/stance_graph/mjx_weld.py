"""Run N MJX hexapods in parallel, welding the planted feet in place (dynamics).

Scenario A, *dynamics* variant -- the physical cross-check to ``mjx_stance``'s
statics. ``weld0.xml`` ships 6 inactive position-welds ``wf0..wf5`` (``foot_i`` <->
world). Per env we activate the welds for the planted legs, point each active weld
at that foot's world position in the given posture, and settle under gravity. A
planted foot that barely drifts (and a base that stays put) means the pose holds.

Batching note: ``eq_active`` lives in ``Data`` (so it just batches), but the weld
*target* ``eq_data`` lives in ``Model``. So we batch the model over ``eq_data`` via
a ``None``-except-``eq_data`` ``in_axes`` tree (the standard MJX domain-randomization
pattern) and share every other model field.

Run under ``uv run --extra mjx``.
"""
import jax
import jax.numpy as jnp
from mujoco import mjx
from jaxlie import SE3, SO3

from lab.retired.stance_graph.mjx_stance import MODEL, load_model, to_qpos


def weld_in_place(mjx_model, foot_ids, qpos, stances, n_steps=200):
    """Weld each env's planted feet at their posture positions, then settle.

    Args:
        mjx_model: the ``put_model``'d weld0 (welds ``wf_i`` are ``foot_i`` <-> world,
            inactive by default).
        foot_ids: (6,) foot body ids, in weld / stance-column order.
        qpos: (N, nq) per-env configurations.
        stances: (N, 6) bool -- True welds that leg's foot (planted).
        n_steps: physics steps to settle under gravity.

    Returns:
        data: the settled batched ``mjx.Data``.
        foot_pos0: (N, 6, 3) pre-settle foot world positions (for a drift check).
    """
    # per env: foot world poses at the posture. The welds are body1=foot, body2=world,
    # so the target relpose (world-in-foot) is the *inverse* foot pose.
    def env(q):
        d = mjx.kinematics(mjx_model, mjx.make_data(mjx_model).replace(qpos=q))
        fpos, fquat = d.xpos[foot_ids], d.xquat[foot_ids]           # (6,3), (6,4) wxyz
        Tinv = jax.vmap(lambda p, w: SE3.from_rotation_and_translation(
            SO3(w), p).inverse())(fpos, fquat)
        return fpos, Tinv.translation(), Tinv.rotation().wxyz
    foot_pos0, rel_pos, rel_quat = jax.vmap(env)(qpos)

    # per-env eq_data: relpose = [pos(3), quat(4)] into each weld's slot.
    base = mjx_model.eq_data                                        # (neq, 11)
    eq_data = jax.vmap(lambda rp, rq: base.at[:, 3:6].set(rp).at[:, 6:10].set(rq))(
        rel_pos, rel_quat)                                          # (N, neq, 11)
    eq_active = stances.astype(bool)                               # (N, neq), Data field

    # batch the model over eq_data only; batch data over qpos + eq_active.
    model_b = mjx_model.replace(eq_data=eq_data)
    axes = jax.tree_util.tree_map(lambda _: None, mjx_model).replace(eq_data=0)
    data_b = jax.vmap(lambda q, a: mjx.make_data(mjx_model).replace(
        qpos=q, qvel=jnp.zeros(mjx_model.nv), eq_active=a))(qpos, eq_active)

    # settle; the active welds hold the planted feet.
    def settle(m, d):
        d, _ = jax.lax.scan(lambda d, _: (mjx.step(m, d), None), d, None, length=n_steps)
        return d
    return jax.vmap(settle, in_axes=(axes, 0))(model_b, data_b), foot_pos0


if __name__ == "__main__":
    import numpy as np
    import mujoco

    mx, foot_ids = load_model()
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    home = jnp.array(m.key_qpos[0])

    N = 4
    pos = jnp.broadcast_to(jnp.array(home[:3]), (N, 3))
    rpy = jnp.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0.05, 0.05, 0]])   # small tilts
    bodies = jax.vmap(lambda p, e: SE3.from_rotation_and_translation(
        SO3.from_rpy_radians(*e), p))(pos, rpy)
    thetas = jnp.broadcast_to(jnp.array(home[7:]).reshape(6, 3), (N, 6, 3))
    stances = jnp.array([[1, 0, 1, 0, 1, 0]] * N, dtype=bool)      # tripod 0,2,4

    qpos = to_qpos(bodies, thetas)
    out, foot0 = weld_in_place(mx, foot_ids, qpos, stances, n_steps=200)
    foot_now = jax.vmap(lambda d: d.xpos[foot_ids])(out)          # (N,6,3)
    drift = jnp.linalg.norm(foot_now - foot0, axis=-1)            # (N,6)
    print("max planted-foot drift per env (m):",
          np.round(np.asarray(jnp.max(jnp.where(stances, drift, 0.0), axis=1)), 4))
    print("max free-foot drift per env    (m):",
          np.round(np.asarray(jnp.max(jnp.where(~stances, drift, 0.0), axis=1)), 3))
