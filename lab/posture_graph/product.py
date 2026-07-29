"""A posture roadmap as a *product* of two factors. One mode, brute force.

The problem with sampling whole postures jointly: the mode has 13 DOF (6 body +
3 planted redundancies + 4 free arm), and neighbour count scales as ``s^13``.
Measured, 10M joint samples sit exactly on the connectivity cliff -- drop the step
size 25% and the graph turns to dust.

But the arm touches nothing, so the gate *separates*:

    edge((b,f), (b',f'))  <=>  [body gates](b,b')  AND  [arm gate](f,f')

Nothing couples them, and validity separates too (a node is reachable iff its
*body* is; ``sample_free`` respects limits by construction). So the graph is the
**strong product** ``B (x) F`` -- strong, not Cartesian, because a step may move
both factors at once (``0`` passes any threshold, so "stay put" is a legal move in
either factor).

That buys three things:

1. **Sampling.** Draw ``N_b + N_f`` configs, get ``N_b * N_f`` nodes. Two low-dim
   problems (9 and 4) instead of one 13-dim one.
2. **A heuristic, for free.** ``d_B`` and ``d_F`` are exact distances in a
   projection of the state space -- i.e. pattern databases -- and the max of
   admissible heuristics is admissible. With no infeasible nodes it is *exact*, so
   A* expands only the path.
3. Cheap neighbour enumeration, as a cross product of the factors'.

Costs are hop counts here, which makes ``d_(x) = max(d_B, d_F)`` exact. With
time-weighted edges ``max`` is only a lower bound (still admissible, so A* stays
correct -- it just expands more).

**Where scoring goes:** :func:`feasible`. It is the one thing that does *not*
factor -- the swing leg's mass moves the CoM, so stability depends on ``b`` and
``f`` jointly -- which punctures the product with holes. See its docstring.

Run:  uv run --extra mjx python lab/posture_graph/product.py
"""

import heapq
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from controlkit.kinematics import Foothold, Posture, Support
from controlkit.kinematics import sample_pose
from robots import QUAD_4DOF
from controlkit.se3 import SE3, SO3

ROBOT = QUAD_4DOF
PLANTED = jnp.array([0, 1, 2])
FREE = 3

BODY0 = SE3.from_te(jnp.array([0.0, 0.0, 0.35]), jnp.zeros((1,)), "z")
XYZ_LIMITS = jnp.array([[-0.04, 0.04], [-0.03, 0.03], [-0.02, 0.02]])
RPY_LIMITS = jnp.deg2rad(jnp.array([[-5.0, 5.0], [-5.0, 5.0], [-10.0, 10.0]]))

TAU_T = 0.015                      # m,  per body-translation axis
TAU_R = float(jnp.deg2rad(4.0))    # rad, per axis of the *relative* rotation
TAU_TH = float(jnp.deg2rad(25.0))  # rad, per joint

N_B = 30_000                       # body-factor samples  (6 body + 3 redundancy dims)
N_F = 3_000                        # arm-factor samples   (4 dims)


def support_at(body: SE3) -> Support:
    sh = ROBOT.shoulders(body)[PLANTED]
    pos = jax.vmap(lambda s: s.apply(jnp.array([0.30, 0.0, -0.35])))(sh)
    return Support(Foothold(pos, jnp.tile(jnp.array([0.0, 0.0, 1.0]), (3, 1))), PLANTED)


# --------------------------------------------------------------------------- #
# Where scoring lands                                                         #
# --------------------------------------------------------------------------- #
def feasible(b_idx, f_idx, body, planted, arm):
    """Is the posture (body-node ``b_idx``, arm-node ``f_idx``) physically OK?

    **This is the hook for stability / self-collision / torque limits.** It is
    stubbed to ``True`` here so the structure can be seen without it.

    It is also the one thing that does *not* factor: the swing leg's mass moves the
    CoM, so whether the robot topples depends on ``b`` and ``f`` together. That
    punctures the product with holes, and ``d_(x) = max(d_B, d_F)`` stops being the
    answer -- you can no longer just plan each factor and zip. But ``max(d_B, d_F)``
    remains a *lower bound* (removing nodes only lengthens paths), so it stays an
    admissible A* heuristic and the search below still returns the true optimum.

    The cheap way to add it: precompute the per-factor pieces once --
    ``CoM_body+planted(b)`` over ``N_b`` and ``com_arm(f)`` over ``N_f`` -- then each
    pair is one rotate-and-add plus a point-in-triangle test against the (fixed)
    support polygon. Tens of flops, vectorisable over a whole batch of neighbours.

    Better still, don't call it per node at all: plan first, then score only the
    nodes on the returned path (plus a tube of their neighbours, batched), and
    re-plan if any fail. That is Lazy PRM / LazySP, and it stays optimal precisely
    because "unscored = assumed feasible" is optimistic.
    """
    return True


# --------------------------------------------------------------------------- #
# Factor 1: the body (body pose + the 3 planted legs, redundancy sampled)     #
# --------------------------------------------------------------------------- #
def sample_body_nodes(key, n, support):
    """Sample ``n`` body-factor nodes: a body pose plus its planted legs.

    Uses ``sample_posture`` and keeps only the planted legs -- the arm is a separate
    factor. The hip-yaw redundancy stays *sampled* (not solved), which is what makes
    this factor 9-dim rather than 6.

    Returns:
        ``(valid, body, planted)`` -- (n,) bool, (n,) SE3, (n, 3, 4) angles.
    """
    def one(k):
        ok, post = ROBOT.sample_posture(k, BODY0, support,
                                        xyz_limits=XYZ_LIMITS, rpy_limits=RPY_LIMITS)
        return ok, post.body, post.thetas[PLANTED]
    return jax.jit(jax.vmap(one))(jax.random.split(key, n))


def body_at(key, body: SE3, support):
    """Plant the legs at one *given* body pose (zero-width limits -> body unchanged)."""
    ok, post = ROBOT.sample_posture(key, body, support)
    return ok, post.body, post.thetas[PLANTED]


def body_adjacency(body: SE3, planted, chunk=2000):
    """Edges of the body factor: all three body gates + the planted-joint gate.

    Brute force in row chunks. The pairs go through as flat, *equal-shaped* batches:
    jaxlie's SO3 multiply is silently wrong for mismatched batch shapes (nested vmap
    or ``(N,1) @ (1,N)``), producing garbage past ~64 rows with no error. Do not
    "simplify" this into a nested vmap.
    """
    n = body.shape[0]
    t = body.translation()
    q = body.rotation().wxyz
    th = planted.reshape(n, -1)
    rows = []

    @jax.jit
    def block(lo):
        idx = lo + jnp.arange(chunk)
        i = jnp.repeat(idx, n)
        j = jnp.tile(jnp.arange(n), chunk)
        ok = (jnp.abs(t[j] - t[i]).max(-1) <= TAU_T)
        ok &= (jnp.abs(th[j] - th[i]).max(-1) <= TAU_TH)
        ok &= (jnp.abs((SO3(q[i]).inverse() @ SO3(q[j])).log()).max(-1) <= TAU_R)
        return ok.reshape(chunk, n)

    for lo in range(0, n, chunk):
        m = np.array(block(lo))[: min(chunk, n - lo)]     # copy: jax arrays are read-only
        for r in range(len(m)):
            m[r, lo + r] = False                      # no self-edge
            rows.append(np.where(m[r])[0])
    return rows


# --------------------------------------------------------------------------- #
# Factor 2: the free arm (4 joints, touching nothing)                         #
# --------------------------------------------------------------------------- #
def sample_arm_nodes(key, n):
    """Sample ``n`` free-arm configs, uniform in the joint limits. (n, 4)."""
    return jax.jit(jax.vmap(ROBOT.leg.sample_free))(jax.random.split(key, n))


def arm_adjacency(arm):
    """Edges of the arm factor: the joint gate on its 4 joints."""
    a = np.asarray(arm)
    d = np.abs(a[:, None, :] - a[None, :, :]).max(-1)
    m = d <= TAU_TH
    np.fill_diagonal(m, False)
    return [np.where(r)[0] for r in m]


def hop_distances(adj, goal):
    """Hop distance from every node to ``goal``. One backward Dijkstra -> a PDB."""
    n = len(adj)
    rows = np.concatenate([np.full(len(a), i) for i, a in enumerate(adj)])
    cols = np.concatenate(adj) if len(adj) else np.array([], int)
    g = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    return dijkstra(g, indices=goal, unweighted=True)      # symmetric, so backward == forward


# --------------------------------------------------------------------------- #
# A* on the implicit product                                                  #
# --------------------------------------------------------------------------- #
def astar(adj_b, adj_f, d_b, d_f, start, goal, nodes):
    """Shortest path in ``B (x) F``, never materialising the product.

    Args:
        adj_b, adj_f: per-factor adjacency (lists of neighbour arrays).
        d_b, d_f: per-factor distance-to-goal (the pattern databases).
        start, goal: ``(b, f)`` pairs.
        nodes: ``(body, planted, arm)``, passed through to :func:`feasible`.

    Returns:
        ``(path, expanded)`` -- the list of ``(b, f)`` nodes, and how many were
        expanded (the number that decides whether this is fast).
    """
    body, planted, arm = nodes
    nf = len(adj_f)
    enc = lambda b, f: b * nf + f
    h = lambda b, f: max(d_b[b], d_f[f])

    g = {enc(*start): 0}
    came = {}
    pq = [(h(*start), 0, start)]
    seen = set()
    expanded = 0

    while pq:
        _, gc, (b, f) = heapq.heappop(pq)
        if (b, f) == goal:
            path, cur = [], goal
            while cur != start:
                path.append(cur)
                cur = came[cur]
            return [start] + path[::-1], expanded
        if enc(b, f) in seen:
            continue
        seen.add(enc(b, f))
        expanded += 1

        # neighbours = cross product of the factors' (each may also stay put)
        for nb in np.append(adj_b[b], b):
            for nf_ in np.append(adj_f[f], f):
                if nb == b and nf_ == f:
                    continue
                if not feasible(nb, nf_, body, planted, arm):   # <-- scoring lands here
                    continue
                k = enc(nb, nf_)
                if k in seen:
                    continue
                ng = gc + 1                                     # hop cost
                if ng < g.get(k, 1 << 30):
                    g[k] = ng
                    came[(nb, nf_)] = (b, f)
                    heapq.heappush(pq, (ng + h(nb, nf_), ng, (nb, nf_)))
    return None, expanded


def posture_of(b, f, body, planted, arm) -> Posture:
    """Reassemble a product node ``(b, f)`` back into a whole posture.

    The factors only exist to make the search cheap; a node is still just a robot.

    Args:
        b: body-factor index. f: arm-factor index.
        body, planted, arm: the factor node tables.

    Returns:
        The posture: body pose from ``b``, planted legs from ``b``, arm from ``f``.
    """
    th = jnp.zeros((ROBOT.num_legs, ROBOT.leg.num_joints))
    th = th.at[PLANTED].set(planted[b]).at[FREE].set(arm[f])
    return Posture(body[b], th)


def log_path(path, body, planted, arm, support, *, save=None, spawn=False):
    """Draw the planned postures in rerun, one per step of a scrubbable timeline.

    The footholds are logged too -- they are what the whole plan is pinned to, so
    seeing the feet stay put while the body walks is the point.

    Args:
        path: the ``(b, f)`` nodes from :func:`astar`.
        body, planted, arm: the factor node tables.
        support: the planted feet (drawn as contact tiles).
        save: ``.rrd`` path to write; ignored when ``spawn``.
        spawn: open a viewer instead of writing a file.
    """
    import controlkit.rerun_viz as rv

    rv.init_viewer("posture_path", spawn=spawn, save=None if spawn else save)
    sites = support.sites
    feet = []
    for i, (b, f) in enumerate(path):
        rv.set_time(i)
        post = posture_of(b, f, body, planted, arm)
        rv.log_posture("world/robot", ROBOT, post)
        rv.log_contacts("world/sites", np.asarray(sites.position),
                        np.asarray(sites.normal), show_normals=True)
        feet.append(np.asarray(ROBOT.feet(post)))
        # body trail so the walk is legible when scrubbing
        rv.draw.log_points("world/trail",
                           np.stack([np.asarray(posture_of(*p, body, planted, arm)
                                                .body.translation()) for p in path[: i + 1]]),
                           radius=0.004, color=(255, 255, 255))
    feet = np.stack(feet)                                  # (hops, num_legs, 3)
    moved = np.linalg.norm(feet[-1] - feet[0], axis=-1)
    print("\nrerun: %d frames" % len(path))
    print("  planted feet drift over the whole path: %s m  (should be ~0)"
          % np.round(moved[np.asarray(PLANTED)], 6))
    print("  swing foot travelled: %.3f m" % moved[FREE])


def main():
    """Plan a motion across one contact mode, via the product of two factors.

    The pipeline, and why each stage exists:

    1. **Scene.** Three feet pinned to fixed world sites. Nothing below ever moves
       them -- that constraint is what the whole plan is for.
    2. **Two factors, sampled independently.** This is the load-bearing step. The
       arm touches nothing, so its gate is independent of the body's; sampling them
       separately buys ``N_B * N_F`` nodes for ``N_B + N_F`` samples, and splits one
       13-dim space into a 9-dim and a 4-dim one.
    3. **A graph per factor.** Only ~1e9 + ~1e7 pair tests, versus ~1e16 for the
       product. In both, node 0 is the start and node 1 the goal, by construction.
    4. **A Dijkstra per factor** -> distance-to-goal tables. These are pattern
       databases: exact distances in a projection of the state space.
    5. **A\\* on the implicit product**, with ``h = max(d_B, d_F)``. The product is
       never built; nodes are ``(b, f)`` pairs conjured on demand.

    Returns:
        The path as a list of ``(b, f)`` nodes.
    """
    key = jax.random.PRNGKey(0)
    k_start, k_goal, k_body, k_arm = jax.random.split(key, 4)

    # --- 1. the scene: three feet, pinned ---------------------------------- #
    support = support_at(BODY0)

    # --- 2a. factor 1: the body (6 body DOF + 3 sampled redundancies) ------- #
    # Start and goal are opposite corners of the roam box, so no single step can
    # cross between them and the planner has to actually walk it.
    t0 = time.time()
    start_body = BODY0 @ SE3.from_te(jnp.array([-0.04, -0.03, 0.0]),
                                     jnp.deg2rad(jnp.array([0.0, 0.0, -10.0])), "xyz")
    goal_body = BODY0 @ SE3.from_te(jnp.array([0.04, 0.03, 0.0]),
                                    jnp.deg2rad(jnp.array([0.0, 0.0, 10.0])), "xyz")
    ok_s, b_start, p_start = body_at(k_start, start_body, support)
    ok_g, b_goal, p_goal = body_at(k_goal, goal_body, support)
    assert bool(ok_s) and bool(ok_g), "start/goal bodies must be reachable"

    ok, b_rand, p_rand = sample_body_nodes(k_body, N_B, support)   # the roadmap filler
    ok = np.asarray(ok)
    # Node order is the contract the rest of the file relies on: 0 = start, 1 = goal.
    body = SE3(jnp.concatenate(
        [b_start.wxyz_xyz[None], b_goal.wxyz_xyz[None], b_rand.wxyz_xyz[ok]], 0))
    planted = jnp.concatenate([p_start[None], p_goal[None], p_rand[ok]], 0)
    n_body = body.shape[0]
    print("body factor : %d nodes (%d sampled, %.0f%% valid)  %.1fs"
          % (n_body, N_B, 100 * ok.mean(), time.time() - t0))

    # --- 2b. factor 2: the free arm (4 DOF, independent of the body) -------- #
    t0 = time.time()
    arm = sample_arm_nodes(k_arm, N_F)
    # Same contract: 0 = start, 1 = goal. Take the arm farthest from node 0 as the
    # goal (so the arm has real distance to cover) and swap it into slot 1.
    far = int(jnp.argmax(jnp.abs(arm - arm[0]).max(-1)))
    a1, a_far = arm[1], arm[far]
    arm = arm.at[1].set(a_far).at[far].set(a1)
    n_arm = arm.shape[0]
    print("arm  factor : %d nodes  %.1fs" % (n_arm, time.time() - t0))

    # --- 3. a graph per factor --------------------------------------------- #
    t0 = time.time()
    adj_b = body_adjacency(body, planted)
    deg_b = np.mean([len(a) for a in adj_b])
    print("body edges  : %d  (mean degree %.1f)  %.1fs"
          % (sum(len(a) for a in adj_b) // 2, deg_b, time.time() - t0))

    t0 = time.time()
    adj_f = arm_adjacency(arm)
    deg_f = np.mean([len(a) for a in adj_f])
    print("arm  edges  : %d  (mean degree %.1f)  %.1fs"
          % (sum(len(a) for a in adj_f) // 2, deg_f, time.time() - t0))

    # Degrees multiply because a step may move either factor or both (strong
    # product); the "+1" per factor is the legal "stay put" move.
    print("=> product  : %d nodes, mean degree %.0f  (never materialised)"
          % (n_body * n_arm, (deg_b + 1) * (deg_f + 1) - 1))

    # --- 4. one Dijkstra per factor -> the pattern databases ---------------- #
    # Cheap precisely because the factors are small (1e4, 1e3), which is the whole
    # reason for factoring. Goal is node 1 in each.
    t0 = time.time()
    d_b = hop_distances(adj_b, goal=1)
    d_f = hop_distances(adj_f, goal=1)
    print("\nPDBs        : d_B[start]=%s  d_F[start]=%s   %.2fs"
          % (d_b[0], d_f[0], time.time() - t0))
    if not (np.isfinite(d_b[0]) and np.isfinite(d_f[0])):
        print("a factor is disconnected between start and goal -- raise N_B / N_F.")
        return

    # --- 5. A* on the implicit product ------------------------------------- #
    t0 = time.time()
    path, expanded = astar(adj_b, adj_f, d_b, d_f,
                           start=(0, 0), goal=(1, 1), nodes=(body, planted, arm))
    if path is None:
        print("no path (expanded %d nodes)" % expanded)
        return
    print("path        : %d hops, expanded %d nodes, %.2fs"
          % (len(path) - 1, expanded, time.time() - t0))

    # With no infeasible nodes the heuristic is exact, so the path should land
    # exactly on the bound. If it ever exceeds it, `feasible` is punching holes.
    bound = int(max(d_b[0], d_f[0]))
    print("              lower bound max(d_B,d_F) = %d  -> %s"
          % (bound, "OPTIMAL (heuristic exact, no holes)" if len(path) - 1 == bound
             else "longer than the bound (holes present)"))

    # Per-hop trace: which factor moved. Both move until the arm arrives, then it
    # stalls while the body finishes -- that is max(d_B, d_F), not d_B + d_F.
    print("\n  hop   body   arm     (body moves / arm moves)")
    for n, (b, f) in enumerate(path):
        prev = path[n - 1] if n else None
        mv = "" if prev is None else "%s %s" % ("B" if b != prev[0] else ".",
                                                "F" if f != prev[1] else ".")
        print("  %3d  %5d %5d     %s" % (n, b, f, mv))

    # --- 6. draw it -------------------------------------------------------- #
    spawn = "--spawn" in sys.argv
    log_path(path, body, planted, arm, support,
             save="/tmp/posture_path.rrd", spawn=spawn)
    if not spawn:
        print("  view with:  rerun /tmp/posture_path.rrd   (or re-run with --spawn)")
    return path


if __name__ == "__main__":
    main()
