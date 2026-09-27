"""MJX/terrain foot proposals -- the non-planar counterpart to ``sample.py``.

``sample.py``'s ``foot_sampler`` scatters a per-leg xy Gaussian and pins z=0 (flat
ground). Here feet are drawn from a *terrain* pool instead: surface points
pre-sampled on the scene collision geometry (see ``mjx_terrain.py``), so footholds
sit on the real surface (floor + climbing box). Per free leg the pool is weighted
by a locality kernel around the leg's target and by IK-reachability from the node
body ``b0`` (both feasible-region biases; the exact per-body check happens
downstream in ``infer_posture``).

``body_sampler`` is unchanged, so it is re-used from ``sample.py``.

Pure jax; run under ``uv run --extra mjx``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from lab.retired.stance_graph.kinematics import (
    _infer_theta,
    SHOULDERS,
    LENGTHS,
    JOINT_RANGES,
)
from lab.retired.stance_graph.sample import body_sampler, XYZ_DELTA, RPY_DELTA  # noqa: F401 (re-export)


# Locality-kernel std (metres) on the terrain, centred on each free leg's current
# foothold nudged forward by SHIFT.
FOOT_STD = 0.1
FOOT_SHIFT = jnp.array([0.1, 0.0, 0.0])


def _leg_reach_mask(shoulder, pool_xyz):
    """Per-pool-point IK feasibility for one leg: in the reach annulus AND in
    joint limits on the elbow-down branch.

    Args:
        shoulder: SE3 world pose of the leg's shoulder frame.
        pool_xyz: (M, 3) terrain surface points.

    Returns:
        (M,) bool mask -- whether each point is a feasible foothold for this leg.
    """
    reachable, theta = jax.vmap(_infer_theta, (None, 0, None))(shoulder, pool_xyz, LENGTHS)
    theta = theta[:, 0]                                             # elbow-down, (M, 3)
    in_lim = jnp.all((JOINT_RANGES[:, 0] <= theta) & (theta <= JOINT_RANGES[:, 1]), axis=-1)
    return reachable & in_lim

def foot_sampler(key, b0, mean, pool_xyz, N=100, std=FOOT_STD, shift=FOOT_SHIFT):
    """Draw ``N`` foot layouts from the terrain pool.

    For each of the 6 legs, weight the pool by a locality kernel around
    ``mean[leg] + shift`` (Gaussian on xy, ``std`` metres) times an IK-reach mask
    from ``b0``, then sample ``N`` footholds. If a leg has no reachable pool point
    in range, it falls back to the locality kernel alone.

    Args:
        key: PRNG key.
        b0: SE3 node body pose (the reachability reference).
        mean: (6, 3) current footholds, one per leg.
        pool_xyz: (M, 3) terrain surface points.
        N: number of foot layouts to draw.
        std: locality-kernel std (metres) on xy.
        shift: (3,) forward nudge added to each leg's target.

    Returns:
        (N, 6, 3) foot layouts; feet carry their real terrain z.
    """
    M = pool_xyz.shape[0]
    shoulders = b0 @ SHOULDERS                                     # (6,) SE3
    targets = mean[:, :2] + shift[None, :2]                        # (6, 2)

    d2 = jnp.sum((pool_xyz[None, :, :2] - targets[:, None, :]) ** 2, axis=-1)   # (6, M)
    w_region = jnp.exp(-0.5 * d2 / std ** 2)
    reach = jax.vmap(_leg_reach_mask, (0, None))(shoulders, pool_xyz)           # (6, M)

    w = w_region * reach
    w = jnp.where(w.sum(-1, keepdims=True) > 0, w, w_region)       # fallback if none reachable
    p = w / w.sum(-1, keepdims=True)

    keys = jax.random.split(key, 6)
    idx = jax.vmap(lambda k, pp: jax.random.choice(k, M, (N,), p=pp))(keys, p)  # (6, N)
    return jnp.transpose(pool_xyz[idx], (1, 0, 2))                 # (N, 6, 3)
