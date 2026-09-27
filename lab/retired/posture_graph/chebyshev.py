"""The same roadmap as ``roadmap.py``, but as one L-infinity metric on a kd-tree.

Where ``roadmap.py`` gates each DOF against its own threshold in its own unit (a
*box test*, no scalar distance), this collapses the whole thing to a single number:

    d(p, p') = max_j |theta_j - theta'_j|                 # Chebyshev / L-infinity

over all ``num_legs * n`` joints. Two things make that legitimate here rather than
the arbitrary metre-vs-radian trade the box test avoids:

1. **It is a box.** An L-infinity ball *is* an axis-aligned box -- the AND of the
   per-joint slabs ``|dtheta_j| <= tau``. So "d <= tau" is exactly roadmap's joint
   gate, ``jnp.abs(dth).max() <= TAU_TH``. Same edges; the difference is purely
   *how we find them*.

2. **The body rides in the joints.** In one fixed support the three planted feet
   are pinned, so moving the body moves those legs' 12 joint angles (they are
   solved from the body pose). The joint-only max therefore still sees body motion
   -- roadmap's separate ``TAU_T`` / ``TAU_R`` gates are, in this mode, largely
   redundant with the joint gate. (This stops being true across supports, where a
   planted foot may not slide; then you need the body terms back.)

Because it is a metric, a metric tree applies: ``cKDTree.query_pairs(r=TAU_TH,
p=inf)`` enumerates every within-box pair in ~``O(N log N)`` instead of
``roadmap.all_pairs``'s ``O(N^2)``. That is the whole point -- roadmap tops out
around a few thousand nodes; this is meant for a stack of 1e5-1e6.

Cost is unchanged: ``max_j |dtheta_j| / THETA_DOT_MAX``, the makespan of a
synchronised joint move (the slowest joint sets the time). Nodes, support, start
and goal are imported from ``roadmap`` verbatim, so the two graphs are built on the
*same* stack and can be compared directly.

Contrast also ``metrics.py``, which argues for point-cloud RMSD over any
joint-space metric; the ``max_m ||dp_m||`` variant noted there is the task-space
sibling of this one. This file is the deliberately-simple L-infinity baseline.

Run:  uv run --extra mjx python lab/retired/posture_graph/chebyshev.py [N_SAMPLES]
"""

import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from controlkit.kinematics import Posture
from controlkit.se3 import SE3

# Node construction, scene and cost are roadmap's -- imported, not restated, so the
# two experiments plan over an identical stack. (Runs as a script; the sibling
# `roadmap` is importable because its directory is on sys.path.)
from roadmap import (
    BODY0, PLANTED, RPY_LIMITS, THETA_DOT_MAX, TAU_TH, XYZ_LIMITS,
    posture_at, sites_under, sample_pose, step_deltas,
)

N_SAMPLES = 50_000                 # target regime is 1e5-1e6; roadmap caps ~2e3


def build_stack(n_samples: int, seed: int = 0):
    """Sample the stack: start, goal, then ``n_samples`` random body poses.

    Identical construction to ``roadmap.main`` -- one fixed support, planted legs
    solved from the body pose, free leg parked -- so the node set is comparable.

    Args:
        n_samples: random postures drawn (start and goal are prepended).
        seed: PRNG seed for the body-pose draws.

    Returns:
        ``(postures, support, START, GOAL)`` -- the kinematically-valid stack, its
        support, and the stack indices of the start and goal nodes.
    """
    support = sites_under(BODY0)

    start_body = BODY0 @ SE3.from_te(jnp.array([-0.04, -0.03, 0.0]),
                                     jnp.deg2rad(jnp.array([0.0, 0.0, -10.0])), "xyz")
    goal_body = BODY0 @ SE3.from_te(jnp.array([0.04, 0.03, 0.0]),
                                    jnp.deg2rad(jnp.array([0.0, 0.0, 10.0])), "xyz")
    keys = jax.random.split(jax.random.PRNGKey(seed), n_samples)
    rand = jax.vmap(lambda k: sample_pose(k, BODY0, XYZ_LIMITS, RPY_LIMITS, "xyz"))(keys)
    bodies = SE3(jnp.concatenate(
        [start_body.wxyz_xyz[None], goal_body.wxyz_xyz[None], rand.wxyz_xyz], axis=0))

    valid, postures = jax.jit(jax.vmap(posture_at, (0, None)))(bodies, support)
    valid = np.asarray(valid)
    print("nodes: %d sampled, %d kinematically valid" % (len(valid), valid.sum()))
    assert valid[0] and valid[1], "start/goal must be reachable"

    keep = np.where(valid)[0]
    remap = {int(o): i for i, o in enumerate(keep)}
    postures = Posture(SE3(postures.body.wxyz_xyz[keep]), postures.thetas[keep])
    return postures, support, remap[0], remap[1]


def linf_graph(postures: Posture, tau: float):
    """Edges of the L-infinity roadmap: every pair of postures within a joint box.

    The joint angles are the feature vector; a kd-tree with ``p=inf`` enumerates
    all pairs inside the ``tau``-box (``max_j |dtheta_j| <= tau``). Edge weight is
    the transition time ``max_j |dtheta_j| / THETA_DOT_MAX``.

    Args:
        postures: (N,) the stack; only ``thetas`` is read.
        tau: box half-width per joint, radians (roadmap's ``TAU_TH``).

    Returns:
        ``(W, n_edges)`` -- an (N, N) symmetric CSR of edge times, and the pair count.
    """
    TH = np.asarray(postures.thetas).reshape(len(postures.thetas), -1)   # (N, 16)

    tree = cKDTree(TH)
    pairs = tree.query_pairs(r=float(tau), p=np.inf, output_type="ndarray")  # (E, 2)
    i, j = pairs[:, 0], pairs[:, 1]

    secs = np.abs(TH[i] - TH[j]).max(axis=1) / THETA_DOT_MAX                 # Chebyshev
    secs = np.maximum(secs, 1e-6)                                            # keep edge > 0

    N = TH.shape[0]
    W = coo_matrix((np.concatenate([secs, secs]),                           # symmetrise
                    (np.concatenate([i, j]), np.concatenate([j, i]))),
                   shape=(N, N)).tocsr()
    return W, len(pairs)


def plan(postures: Posture, W, start: int, goal: int):
    """Shortest path between two stack indices; cost = seconds of motion.

    Args:
        postures: the stack (for the per-hop report).
        W: (N, N) CSR edge-time graph from :func:`linf_graph`.
        start: source posture index.
        goal: target posture index.

    Returns:
        The path as a list of indices, or ``None`` if start and goal are in
        different components.
    """
    dist, pred = dijkstra(W, indices=start, return_predecessors=True)
    if not np.isfinite(dist[goal]):
        print("\nno path: start and goal are in different components.")
        return None

    path, n = [], goal
    while n != start:
        path.append(n)
        n = pred[n]
    path = [start] + path[::-1]

    print("\npath found: %d hops, %.2f s of motion" % (len(path) - 1, dist[goal]))
    print("\n    hop      dx(m)   drot(deg)  djoint(deg)   dt(s)")
    for a, b in zip(path[:-1], path[1:]):
        dt, dr, dth = step_deltas(postures[a], postures[b])
        print("    %4d->%-4d %6.3f %8.2f %11.2f %7.3f"
              % (a, b, jnp.abs(dt).max(), jnp.rad2deg(jnp.abs(dr).max()),
                 jnp.rad2deg(jnp.abs(dth).max()), jnp.abs(dth).max() / THETA_DOT_MAX))
    return path


def main():
    n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else N_SAMPLES
    print("L-infinity posture roadmap (kd-tree) -- legs %s planted, leg 3 free\n"
          % np.asarray(PLANTED))

    postures, support, START, GOAL = build_stack(n_samples)
    N = len(postures.thetas)

    t0 = time.time()
    W, n_edges = linf_graph(postures, TAU_TH)
    print("edges: %d pairs within the joint box in %.2fs (~%.0f per node)"
          % (n_edges, time.time() - t0, 2 * n_edges / N))

    d0, r0, j0 = step_deltas(postures[START], postures[GOAL])
    print("\nstart -> goal directly: %.3f m, %.1f deg body, %.1f deg joint  "
          "(one step allows %.1f deg joint)"
          % (jnp.abs(d0).max(), jnp.rad2deg(jnp.abs(r0).max()),
             jnp.rad2deg(jnp.abs(j0).max()), jnp.rad2deg(TAU_TH)))

    plan(postures, W, START, GOAL)
    return postures, support, W, START, GOAL


if __name__ == "__main__":
    main()
