"""Step 2: reachability over the foot grid, masked sampling for the free feet, IK.

Given a current body pose and the surface candidate points (N,3) from foot_grid:
  * reach (N, 6): for each candidate and each leg, is it reachable (are_connected).
  * fixed feet stay put; free feet are re-sampled from the candidates, masked by
    reachability (fixed shape -- sample from all N with prob 0 on unreachable).
  * infer_theta solves the leg angles for the resulting foot targets.

Renders one sampled pose. Run under ``uv run --extra mjx python -m lab.ik.debug.foot_sampling``.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg

from jaxlie import SE3
from lab.ik.core import SHOULDERS, LENGTHS, from_te, foot_transform, are_connected, infer_theta
from lab.ik.debug.foot_grid import sample_surface, MODEL, SCRATCH

HOME_THETA = jnp.array([0.0, 0.0523599, 1.46608])


def joint_limits(model):
    """Read [min, max] joint range (radians) per leg from the model. Returns (6, 3, 2)
    for legs 0..5 x (coxa, femur, tibia)."""
    lims = np.zeros((6, 3, 2))
    for i in range(6):
        for j, seg in enumerate(("coxa", "femur", "tibia")):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{seg}{i}")
            lims[i, j] = model.jnt_range[jid]
    return jnp.asarray(lims)


def reachability(body: SE3, grid, limits):
    """(N, 6) bool: candidate n reachable by leg l -- in the reach annulus AND the
    elbow-down IK angles inside the joint limits. ``limits`` is (6, 3, 2)."""
    shoulders = body @ SHOULDERS                                   # (6,) world shoulders

    def per_leg(sh, lim, p):
        ok, theta = infer_theta(sh, p, LENGTHS)                    # theta (2,3), branch 0 = elbow-down
        th = theta[0]
        within = jnp.all((th >= lim[:, 0]) & (th <= lim[:, 1]))
        return ok & within

    return jax.vmap(lambda p: jax.vmap(per_leg, (0, 0, None))(shoulders, limits, p))(grid)


def sample_free(key, grid, reach, free_ids, S):
    """Sample S candidate positions per free leg, masked by reachability.
    Returns (len(free_ids), S, 3)."""
    def one(k, l):
        m = reach[:, l].astype(jnp.float32)
        idx = jax.random.choice(k, grid.shape[0], (S,), p=m / m.sum())   # 0 prob on unreachable
        return grid[idx]
    keys = jax.random.split(key, len(free_ids))
    return jnp.stack([one(keys[i], int(free_ids[i])) for i in range(len(free_ids))])


def build_poses(base, S=10, free=(0, 5), seed=0):
    """Return (qpos_all (S, nq)) -- S poses, fixed feet held, free feet re-sampled."""
    body = from_te(jnp.asarray(base), jnp.zeros(3))
    grid = jnp.asarray(sample_surface(base, n=500))
    shoulders = body @ SHOULDERS
    limits = joint_limits(mujoco.MjModel.from_xml_path(str(MODEL)))
    reach = reachability(body, grid, limits)
    print("reach shape:", reach.shape, " reachable per leg (limit-checked):", np.asarray(reach.sum(0)))

    free = jnp.asarray(free)
    cur = np.asarray(jax.vmap(lambda sh: foot_transform(sh, HOME_THETA, LENGTHS).translation())(shoulders))
    samples = np.asarray(sample_free(jax.random.PRNGKey(seed), grid, reach, free, S))  # (len_free, S, 3)

    poses = []
    for s in range(S):
        feet = cur.copy()
        for i, l in enumerate(np.asarray(free)):
            feet[int(l)] = samples[i, s]
        theta = np.stack([np.asarray(infer_theta(shoulders[l], jnp.asarray(feet[l]), LENGTHS)[1][0])
                          for l in range(6)])                      # (6,3), elbow-down branch
        poses.append(np.concatenate([base, [1, 0, 0, 0], theta.reshape(-1)]))
    return np.array(poses)


def render_montage(qpos_all, base, out=SCRATCH / "foot_samples.png", cols=5):
    from lab.ik.viz import tile
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    m.vis.global_.offwidth, m.vis.global_.offheight = 700, 700
    d = mujoco.MjData(m)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [base[0] + 0.2, 0.0, 0.3]
    cam.distance, cam.azimuth, cam.elevation = 1.7, 120.0, -14.0
    r = mujoco.Renderer(m, 360, 360)
    frames = []
    for q in qpos_all:
        d.qpos[:] = q; mujoco.mj_forward(m, d); r.update_scene(d, cam); frames.append(r.render().copy())
    r.close()
    mpimg.imsave(out, tile(np.array(frames), cols=cols))
    print(f"montage of {len(qpos_all)} poses -> {out}")


def _demo():
    base = np.array([0.6, 0.0, 0.241185])
    qpos_all = build_poses(base, S=10)
    render_montage(qpos_all, base)
    # scrubbable rollout: hold each pose ~0.3 s
    q = np.repeat(qpos_all, 75, axis=0)
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    np.savez(SCRATCH / "foot_samples.npz", qpos=q, qvel=np.zeros((len(q), m.nv)),
             timestep=m.opt.timestep, model="models/climb0.xml")
    print("rollout -> lab/ik/scratch/foot_samples.npz  (uv run ctk play lab/ik/scratch/foot_samples.npz)")


if __name__ == "__main__":
    _demo()
