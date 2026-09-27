"""Boolean collision check for a posture, through MuJoCo. Runtime-first.

The robot's links are capsules and the body is a box -- real geometries, not points
-- so collision is geometry-vs-geometry. MuJoCo already does all those pairs exactly
(capsule-box, box-box) *and* self-collision (leg-vs-leg, leg-vs-body) for free, so
we reuse its engine rather than hand-rolling.

The model is *generated from the* :class:`Robot` *spec* (same mounts, lengths, axis
pattern), so it cannot drift from the analytic kinematics the planner uses -- see
:func:`verify` for the check that the two agree.

Collision filtering: robot links and obstacles are ``contype=conaffinity=1`` so
they collide with each other and the robot with itself. The floor is visual only
(``0/0``) -- the planted feet rest on it by design, that is not a collision to
avoid. So ``data.ncon > 0`` means "a link hit an obstacle, or the robot hit itself".

Run:  uv run --extra mjx python lab/posture_graph/collision.py
"""

import time

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from controlkit.kinematics import Posture
from robots import QUAD_4DOF
from controlkit.se3 import SE3

ROBOT = QUAD_4DOF
LINK_RADIUS = 0.015


def build_model(robot, boxes=(), radius=LINK_RADIUS) -> mujoco.MjModel:
    """Emit a MuJoCo model matching ``robot``, plus a box per obstacle.

    Each leg is a nested chain of hinge bodies with a capsule per link; the body is
    a free joint. Everything is generated from the spec so the geometry is the same
    the planner reasons about.

    Args:
        robot: the :class:`Robot`.
        boxes: iterable of ``(center_xyz, half_xyz)`` obstacle boxes.
        radius: capsule radius for the links.

    Returns:
        A compiled ``MjModel``.
    """
    lengths = [float(x) for x in np.asarray(robot.leg.lengths)]
    axes = np.asarray(robot.leg.axes)                         # (n, 3) unit vectors
    mounts = robot.mounts
    mpos = np.asarray(mounts.translation())
    mquat = np.asarray(mounts.rotation().wxyz)                 # (num_legs, 4) wxyz

    def leg_body(i):
        # nested hinge chain: seg k rotates about axes[k], then the link runs +x by
        # lengths[k]; the last link is the foot segment.
        s = (f'<body name="l{i}_0" pos="{mpos[i,0]} {mpos[i,1]} {mpos[i,2]}" '
             f'quat="{mquat[i,0]} {mquat[i,1]} {mquat[i,2]} {mquat[i,3]}">\n')
        for k in range(len(lengths)):
            s += f'<joint name="l{i}_j{k}" type="hinge" axis="{axes[k,0]} {axes[k,1]} {axes[k,2]}"/>\n'
            s += f'<geom type="capsule" fromto="0 0 0 {lengths[k]} 0 0" size="{radius}"/>\n'
            if k < len(lengths) - 1:
                s += f'<body name="l{i}_{k+1}" pos="{lengths[k]} 0 0">\n'
        s += "</body>\n" * len(lengths)
        return s

    obstacles = "".join(
        f'<geom type="box" pos="{c[0]} {c[1]} {c[2]}" size="{h[0]} {h[1]} {h[2]}" '
        f'contype="1" conaffinity="1" rgba="0.8 0.3 0.3 0.5"/>\n'
        for c, h in boxes)

    xml = f"""<mujoco model="{robot.__class__.__name__}_collision">
      <compiler angle="radian"/>
      <default>
        <geom contype="1" conaffinity="1"/>
      </default>
      <worldbody>
        <geom name="floor" type="plane" size="5 5 0.1" contype="0" conaffinity="0"/>
        {obstacles}
        <body name="base">
          <freejoint/>
          <geom type="box" size="0.05 0.05 0.02"/>
          {''.join(leg_body(i) for i in range(robot.num_legs))}
        </body>
      </worldbody>
    </mujoco>"""
    return mujoco.MjModel.from_xml_string(xml)


def set_posture(model, data, posture: Posture):
    """Write a posture into ``data.qpos``: free joint (body) + hinges (thetas)."""
    body = posture.body
    data.qpos[0:3] = np.asarray(body.translation())
    data.qpos[3:7] = np.asarray(body.rotation().wxyz)          # MuJoCo quat is wxyz
    data.qpos[7:] = np.asarray(posture.thetas).reshape(-1)     # leg-major, joint order


def collides(model, data, posture: Posture, *, margin=0.0) -> bool:
    """Is the posture in collision (link-vs-obstacle or self)? Boolean.

    Args:
        model, data: the MuJoCo pair.
        posture: the configuration.
        margin: inflate contacts by this distance (report near-misses too).

    Returns:
        True if any contact exists.
    """
    set_posture(model, data, posture)
    if margin:
        model.geom_margin[:] = margin
    mujoco.mj_forward(model, data)
    return data.ncon > 0


def verify(robot, model):
    """Check the generated model's kinematics match ``robot.forward``.

    The whole approach rests on this: if MuJoCo's geoms sit where the analytic FK
    says, the collision is checked against the geometry the planner reasons about.
    """
    data = mujoco.MjData(model)
    key = jax.random.PRNGKey(0)
    body = SE3.from_te(jnp.array([0.02, -0.01, 0.33]),
                       jnp.deg2rad(jnp.array([3.0, -2.0, 8.0])), "xyz")
    thetas = jax.random.uniform(key, (robot.num_legs, robot.leg.num_joints),
                                minval=-0.6, maxval=0.6)
    post = Posture(body, thetas)

    set_posture(model, data, post)
    mujoco.mj_forward(model, data)

    # analytic foot positions vs MuJoCo's last-link body frames
    feet_analytic = np.asarray(robot.feet(post))
    err = 0.0
    for i in range(robot.num_legs):
        bid = model.body(f"l{i}_{robot.leg.num_joints - 1}").id
        # the foot is `last length` along the last body's local x, in world
        last = float(robot.leg.lengths[-1])
        foot_mj = data.xpos[bid] + data.xmat[bid].reshape(3, 3) @ np.array([last, 0, 0])
        err = max(err, np.linalg.norm(foot_mj - feet_analytic[i]))
    return err


def main():
    print("building MuJoCo collision model from QUAD_4DOF ...")
    model = build_model(ROBOT)
    data = mujoco.MjData(model)
    print("  qpos dim %d  (7 free + %d hinges)   geoms %d"
          % (model.nq, model.nq - 7, model.ngeom))

    err = verify(ROBOT, model)
    print("  FK match vs robot.forward: max foot error = %.2e m  %s"
          % (err, "OK" if err < 1e-5 else "MISMATCH -- model is wrong"))

    # a clean standing posture: should NOT collide
    support_body = SE3.from_te(jnp.array([0.0, 0.0, 0.35]), jnp.zeros((1,)), "z")
    from controlkit.kinematics import Foothold, Support
    ids = jnp.array([0, 1, 2])
    sh = ROBOT.shoulders(support_body)[ids]
    sites = Foothold(jax.vmap(lambda s: s.apply(jnp.array([0.30, 0.0, -0.35])))(sh),
                     jnp.tile(jnp.array([0.0, 0.0, 1.0]), (3, 1)))
    _, post = ROBOT.sample_posture(jax.random.PRNGKey(1), support_body,
                                   Support(sites, ids))
    print("\nclean posture, no obstacle : collides = %s  (expect False)"
          % collides(model, data, post))

    # drop a box right on a planted foot: should collide
    foot0 = np.asarray(ROBOT.feet(post))[0]
    model_obs = build_model(ROBOT, boxes=[(foot0, (0.05, 0.05, 0.05))])
    data_obs = mujoco.MjData(model_obs)
    print("box on a foot              : collides = %s  (expect True)"
          % collides(model_obs, data_obs, post))

    # ---- runtime ----
    keys = jax.random.split(jax.random.PRNGKey(2), 2000)
    _, posts = jax.vmap(lambda k: ROBOT.sample_posture(k, support_body,
                                                       Support(sites, ids)))(keys)
    qpos = np.concatenate([np.asarray(posts.body.translation()),
                           np.asarray(posts.body.rotation().wxyz),
                           np.asarray(posts.thetas).reshape(2000, -1)], axis=1)

    N = 2000
    t0 = time.time()
    ncol = 0
    for i in range(N):
        data_obs.qpos[:] = qpos[i]
        mujoco.mj_forward(model_obs, data_obs)
        ncol += data_obs.ncon > 0
    dt = time.time() - t0
    print("\nruntime: %d checks in %.3fs -> %.1f us/check  (%.0f k/s)   [%d collided]"
          % (N, dt, dt / N * 1e6, N / dt / 1e3, ncol))
    print("  => %d A* neighbours would cost ~%.2f s" % (30000, 30000 * dt / N))


if __name__ == "__main__":
    main()
