"""A minimal posture roadmap: one contact mode, brute force, end to end.

The concept, in three pieces:

**nodes** -- a posture with the *same* 3 legs planted on the *same* footholds. The
planted legs are solved from the body pose (best valid IK branch), so a node is
really just a body pose; the free leg rides along at a fixed config. Solving rather
than sampling the redundancy matters: ``sample_planted`` draws the yaw at random, so
two postures with the same body would differ by ~90 degrees of hip yaw and no edge
would ever pass the joint gate.

**edges** -- two postures connect if you can move between them in one small step.
This is a *box test*, not a distance: each DOF is compared against its own threshold
in its own unit, so metres and radians never have to be traded off against each
other (there is no principled exchange rate, and inventing one is what makes scalar
posture "distances" arbitrary). A body 1 cm away but yawed 90 degrees is correctly
*not* close, which an averaged distance would happily miss.

**plan** -- shortest path, cost = time = ``max_j |dtheta_j| / theta_dot_max``, i.e.
how long the slowest joint takes. That makes the joint threshold a time budget
rather than a tuning knob.

This is one *mode* (a fixed contact set). A full roadmap layers modes and adds
lift/place edges between them, plus stability and force filters -- none of that is
here. This is the skeleton.

Run:  uv run --extra mjx python lab/retired/posture_graph/roadmap.py
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from controlkit.kinematics import Foothold, Posture, Support, candidates
from controlkit.kinematics import sample_pose
from robots import QUAD_4DOF
from controlkit.se3 import SE3, SO3

# --------------------------------------------------------------------------- #
# Scene                                                                        #
# --------------------------------------------------------------------------- #
ROBOT = QUAD_4DOF
PLANTED = jnp.array([0, 1, 2])                    # legs on the ground
FREE_THETA = jnp.deg2rad(jnp.array([0.0, 0.0, -60.0, 120.0]))   # the swing leg, parked

BODY0 = SE3.from_te(jnp.array([0.0, 0.0, 0.35]), jnp.zeros((1,)), "z")

# How far the body may roam. Start and goal sit at opposite corners of this box.
XYZ_LIMITS = jnp.array([[-0.04, 0.04], [-0.03, 0.03], [-0.02, 0.02]])
RPY_LIMITS = jnp.deg2rad(jnp.array([[-5.0, 5.0], [-5.0, 5.0], [-10.0, 10.0]]))

# One step of motion: every DOF must be within its own threshold.
#
# These three have to agree with each other, which is not obvious: the legs
# *amplify* the body. Hip yaw points at the foot, so body rotation AND body
# translation both feed it (4 cm of shift at a 0.3 m foot is already ~8 degrees).
# Measured, a 1.5 cm / 4 deg body step implies ~20 deg of joint travel -- so a
# tight joint gate silently forbids the very steps the body gates allow, and the
# graph comes out as dust. TAU_TH is set from that measured ratio, not guessed.
TAU_T = 0.015                      # m, per body-translation axis
TAU_R = jnp.deg2rad(4.0)           # rad, per axis of the *relative* rotation
TAU_TH = jnp.deg2rad(25.0)         # rad, per joint  (~0.14 s at THETA_DOT_MAX)
THETA_DOT_MAX = 3.0                # rad/s, for the time cost

N_SAMPLES = 2000


def sites_under(body: SE3) -> Support:
    """Footholds on the ground under three of the shoulders."""
    sh = ROBOT.shoulders(body)[PLANTED]
    pos = jax.vmap(lambda s: s.apply(jnp.array([0.30, 0.0, -0.35])))(sh)
    return Support(Foothold(pos, jnp.tile(jnp.array([0.0, 0.0, 1.0]), (3, 1))), PLANTED)


# --------------------------------------------------------------------------- #
# Nodes: a posture is determined by the body pose                              #
# --------------------------------------------------------------------------- #
def plant(leg, site: Foothold):
    """Best valid leg config standing on ``site``. Deterministic.

    The redundancy is spent by pointing the hip yaw straight at the foot -- what a
    leg actually does. The IK then hands back every branch; we gate on
    reachable-and-in-limits and pick the one meeting the surface squarest. Picking
    (rather than sampling) is what makes a node a function of the body pose alone.

    Not ``ik_from_foot_and_normal``: forcing a *vertical* ground normal into the
    pitch plane pins the roll to a value well outside its limit, and only 1% of body
    poses survive. That solver is for pushing along a normal, not for standing on
    flat ground.

    Args:
        leg: the leg.
        site: foothold, in the leg's base frame.

    Returns:
        ``(ok, theta)`` -- scalar bool and (n,).
    """
    yaw = jnp.arctan2(site.position[1], site.position[0])      # point the leg at the foot
    ok, thetas = leg.ik_from_foot_and_yaw(site.position, yaw)
    ok = ok & candidates.in_limits(thetas, leg.limits)
    vs = jax.vmap(leg.contact_vector)(thetas)
    i = candidates.best(candidates.contact_alignment(vs, site.normal), ok)
    return ok[i], thetas[i]


def posture_at(body: SE3, support: Support):
    """The posture this body pose implies, given the planted feet.

    Args:
        body: SE3 body pose.
        support: the planted legs and their footholds (world frame).

    Returns:
        ``(valid, posture)`` -- valid iff every planted leg found a config.
    """
    sh = ROBOT.shoulders(body)[support.ids]
    thetas = jnp.tile(FREE_THETA, (ROBOT.num_legs, 1))
    oks, planted = jax.vmap(
        lambda s, site: plant(ROBOT.leg, site.transform(s.inverse()))
    )(sh, support.sites)
    thetas = thetas.at[support.ids].set(planted)
    return jnp.all(oks), Posture(body, thetas)


# --------------------------------------------------------------------------- #
# Edges: the box test                                                          #
# --------------------------------------------------------------------------- #
def step_deltas(p1: Posture, p2: Posture):
    """Per-DOF differences between two postures, each in its own unit.

    The rotation is the *relative* one: ``log(R1^T R2)`` is the rotation you would
    actually have to perform, as a 3-vector whose norm is the geodesic angle.
    Differencing absolute Euler angles instead would break at gimbal lock and on
    wrapping, and would not measure how far you must turn.

    Args:
        p1: first posture.
        p2: second posture.

    Returns:
        ``(dt, dr, dth)`` -- (3,) metres, (3,) radians, (num_legs, n) radians.
    """
    dt = p2.body.translation() - p1.body.translation()
    dr = (p1.body.rotation().inverse() @ p2.body.rotation()).log()
    dth = p2.thetas - p1.thetas
    return dt, dr, dth


def close_enough(p1: Posture, p2: Posture):
    """Can we move p1 -> p2 in one small step? Every DOF within its threshold.

    Returns:
        ``(ok, seconds)`` -- whether the step is allowed, and how long it takes
        (the slowest joint's travel). ``seconds`` is meaningless when ``ok`` is false.
    """
    dt, dr, dth = step_deltas(p1, p2)
    ok = (
        (jnp.abs(dt).max() <= TAU_T)
        & (jnp.abs(dr).max() <= TAU_R)
        & (jnp.abs(dth).max() <= TAU_TH)
    )
    return ok, jnp.abs(dth).max() / THETA_DOT_MAX


def all_pairs(p: Posture):
    """``close_enough`` for every ordered pair. Returns (N, N) ``(ok, seconds)``.

    The pairs are gathered into flat, *equal-shaped* batches and pushed through one
    batched call. That shape is not incidental:

    **jaxlie's SO3 multiply is wrong for mismatched batch shapes.** Both the obvious
    formulations -- nested ``vmap`` (``vmap(vmap(f,(None,0)),(0,None))``) and
    broadcasting ``(N,1) @ (1,N)`` -- silently produce garbage past ~64 rows, while
    single ``vmap`` and equal-shape batching are exact. It comes from ``multiply``
    doing ``q_outer[..., terms_i, terms_j]`` -- advanced indexing under an ellipsis.
    It fails *quietly*: no error, just wrong numbers, so the graph fills with edges
    that violate their own gate. Reproduces on raw jaxlie with no controlkit
    imported. Do not "simplify" this back into a nested vmap.
    """
    n = p.thetas.shape[0]
    i, j = (x.ravel() for x in jnp.meshgrid(jnp.arange(n), jnp.arange(n), indexing="ij"))
    t = p.body.translation()
    q = p.body.rotation().wxyz
    th = p.thetas.reshape(n, -1)

    dt = t[j] - t[i]                                              # (n*n, 3)
    dr = (SO3(q[i]).inverse() @ SO3(q[j])).log()                  # (n*n, 3) equal shapes
    dth = th[j] - th[i]                                           # (n*n, num_legs*n_joints)

    ok = (
        (jnp.abs(dt).max(-1) <= TAU_T)
        & (jnp.abs(dr).max(-1) <= TAU_R)
        & (jnp.abs(dth).max(-1) <= TAU_TH)
    )
    return ok.reshape(n, n), (jnp.abs(dth).max(-1) / THETA_DOT_MAX).reshape(n, n)


def main():
    print("posture roadmap -- one mode: legs %s planted, leg 3 free\n"
          % np.asarray(PLANTED))

    support = sites_under(BODY0)

    # ---- nodes: start, goal, then random body poses in between ----------------
    start_body = BODY0 @ SE3.from_te(jnp.array([-0.04, -0.03, 0.0]),
                                     jnp.deg2rad(jnp.array([0.0, 0.0, -10.0])), "xyz")
    goal_body = BODY0 @ SE3.from_te(jnp.array([0.04, 0.03, 0.0]),
                                    jnp.deg2rad(jnp.array([0.0, 0.0, 10.0])), "xyz")
    keys = jax.random.split(jax.random.PRNGKey(0), N_SAMPLES)
    rand = jax.vmap(lambda k: sample_pose(k, BODY0, XYZ_LIMITS, RPY_LIMITS, "xyz"))(keys)
    bodies = SE3(jnp.concatenate(
        [start_body.wxyz_xyz[None], goal_body.wxyz_xyz[None], rand.wxyz_xyz], axis=0))

    valid, postures = jax.jit(jax.vmap(posture_at, (0, None)))(bodies, support)
    valid = np.asarray(valid)
    print("nodes: %d sampled, %d kinematically valid" % (len(valid), valid.sum()))
    assert valid[0] and valid[1], "start/goal must be reachable"
    keep = np.where(valid)[0]
    remap = {int(o): i for i, o in enumerate(keep)}
    START, GOAL = remap[0], remap[1]
    postures = Posture(SE3(postures.body.wxyz_xyz[keep]), postures.thetas[keep])
    N = len(keep)

    # ---- edges: brute force all pairs ---------------------------------------
    t0 = time.time()
    ok, secs = jax.jit(all_pairs)(postures)
    ok = np.array(ok)                                 # copy: jax arrays are read-only
    np.fill_diagonal(ok, False)                       # no self-edges
    secs = np.asarray(secs)
    print("edges: %d pairs tested in %.1fs -> %d edges (%.2f%% of pairs, ~%.0f per node)"
          % (N * N, time.time() - t0, ok.sum(), 100 * ok.sum() / (N * N), ok.sum() / N))

    # ---- plan: shortest path, cost = seconds --------------------------------
    W = np.where(ok, np.maximum(secs, 1e-6), 0.0)     # 0 = no edge, for the sparse graph
    dist, pred = dijkstra(csr_matrix(W), indices=START, return_predecessors=True)
    if not np.isfinite(dist[GOAL]):
        print("\nno path: graph is disconnected between start and goal.")
        return

    path = []
    n = GOAL
    while n != START:
        path.append(n)
        n = pred[n]
    path = [START] + path[::-1]

    print("\npath found: %d hops, %.2f s of motion" % (len(path) - 1, dist[GOAL]))
    d0, r0, _ = step_deltas(postures[START], postures[GOAL])
    print("  start -> goal directly: %.3f m, %.1f deg apart  (a single step allows %.3f m, %.1f deg)"
          % (jnp.abs(d0).max(), jnp.rad2deg(jnp.abs(r0).max()), TAU_T, jnp.rad2deg(TAU_R)))
    print("  so it must be walked in steps:\n")
    print("    hop      dx(m)   drot(deg)  djoint(deg)   dt(s)")
    for a, b in zip(path[:-1], path[1:]):
        dt, dr, dth = step_deltas(postures[a], postures[b])
        print("    %4d->%-4d %6.3f %8.2f %11.2f %7.3f"
              % (a, b, jnp.abs(dt).max(), jnp.rad2deg(jnp.abs(dr).max()),
                 jnp.rad2deg(jnp.abs(dth).max()), jnp.abs(dth).max() / THETA_DOT_MAX))
    return postures, path, support


if __name__ == "__main__":
    main()
