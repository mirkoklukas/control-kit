"""Comparing postures: distance and nearest neighbours.

A posture mixes metres (body position) with radians (body rotation, joint angles),
and there is no principled exchange rate between them -- SE(3) admits no
bi-invariant metric, so any ``w_p*|dt|^2 + w_r*phi^2 + sum w_j*dtheta_j^2`` invents
its weights. So compare postures by where the robot's parts physically *are*:
:meth:`Robot.features` maps a posture to the world positions of every joint and
foot, and the distance is Euclidean on that. Everything is metres, nothing to tune,
and the body pose comes along for free (the shoulders are in the cloud).

It is also the *principled* version of a weighted joint-space metric. A
perturbation ``dtheta_j`` moves a distal point at radius ``r`` by about
``r*dtheta_j``, so summing squared displacements gives
``sum_j (sum_distal r^2) dtheta_j^2`` -- i.e. the right joint weights are the
summed squared distances to everything downstream. The point cloud computes exactly
that, without linearising.

**The clouds are corresponded** -- joint ``j`` of leg ``i`` matches joint ``j`` of
leg ``i``. So this is a per-point distance, not a matching problem: do *not* reach
for Chamfer / EMD / Hausdorff, which solve matching and would cheerfully pair leg
1's knee with leg 4's hip and call two different postures close.

Aggregating the per-point displacements, RMSD is the choice here::

    rmsd(P1, P2) = sqrt(mean_m ||dp_m||^2) = ||v1 - v2|| / sqrt(M)

because it *is* the Euclidean norm of the flattened vectors, so the whole pairwise
matrix is one matmul and any metric tree (KD/ball) applies. The alternatives are
worth knowing but cost you that: ``max_m ||dp_m||`` ("no part moved more than d")
bounds the swept volume and, over joint speed, lower-bounds transition time -- a
good *second-stage* cost; ``mean_m ||dp_m||`` is outlier-robust.

Note this deliberately does **not** Kabsch-align the clouds the way structural
RMSD does: the body pose should count, so the comparison is in the world frame.

Two caveats when using this for a transition graph:

- ``Robot.sample_posture`` gives free legs a *uniform random* config, so an
  unweighted distance largely measures free-leg noise. Weight them out
  (``Robot.features(..., weights=...)``) or compare within a support class.
- Weights fold into the feature vector, so they must be the same for every posture
  in the stack. A free-leg mask depends on each posture's support -- so group
  postures by support and run kNN within a group. Cross-support pairs need the
  expensive check anyway: a planted foot cannot slide.
"""

import jax
import jax.numpy as jnp


def pairwise_sqdist(V: jax.Array) -> jax.Array:
    """Squared Euclidean distances between every pair of feature vectors.

    Uses ``|a-b|^2 = |a|^2 + |b|^2 - 2 a.b``, so it is a single matmul rather than
    an (N, N, 3M) broadcast.

    Args:
        V: (N, D) feature vectors, e.g. ``vmap(robot.features)(postures)``.

    Returns:
        (N, N) squared distances, clamped at 0 (the identity above can go slightly
        negative for near-identical rows in float32).
    """
    sq = jnp.sum(V * V, axis=-1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (V @ V.T)
    return jnp.maximum(d2, 0.0)


def knn(V: jax.Array, k: int, *, exclude_self: bool = True):
    """The ``k`` nearest neighbours of every feature vector.

    Args:
        V: (N, D) feature vectors, e.g. ``vmap(robot.features)(postures)``.
        k: neighbours per item.
        exclude_self: drop each item's own zero-distance match.

    Returns:
        ``(idx, dist)`` with shapes (N, k) and (N, k), nearest first. ``dist`` is
        the Euclidean feature distance; for unweighted features divide by
        ``sqrt(D / 3)`` to read it as an RMSD in metres.
    """
    d2 = pairwise_sqdist(V)
    if exclude_self:
        d2 = d2.at[jnp.diag_indices(V.shape[0])].set(jnp.inf)
    neg, idx = jax.lax.top_k(-d2, k)
    return idx, jnp.sqrt(jnp.maximum(-neg, 0.0))


def rmsd(V: jax.Array, dist: jax.Array) -> jax.Array:
    """Read a feature distance as an RMSD in metres.

    Args:
        V: (N, D) the feature matrix the distance came from; supplies ``M = D/3``.
        dist: any Euclidean feature distance(s).

    Returns:
        ``dist / sqrt(M)``. Only meaningful for unweighted features -- weighted
        ones normalise by ``sqrt(sum w)`` instead.
    """
    return dist / jnp.sqrt(V.shape[-1] / 3)
