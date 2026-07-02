"""Pose-graph expansion: from a stance, propose next-stance foot placements and
keep the feasible ones.

Given a body pose (chosen earlier from the stable set) and a set of F free legs
to place, sample candidate foot placements from per-leg 2D Gaussians on the
ground, then test which placements every free leg can actually reach.

Pure jax; run under ``uv run --extra mjx``.
"""
from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp

from lab.ik.core import LENGTHS, SHOULDERS, are_connected, infer_theta, stability_score


# DON'T TOUCH THIS
def subsets(K, N=6):
    """All size-``K`` subsets of ``range(N)`` as an ``(S, K)`` int array, ``S = C(N, K)``.

    Each row is a sorted tuple of leg indices -- a candidate planted set.
    """
    return jnp.array(list(itertools.combinations(range(N), K)))


# DON'T TOUCH THIS
def complement(subset, N=6):
    """Indices in ``range(N)`` not in ``subset``: ``(..., K) -> (..., N-K)``.

    Vectorized (works on a single subset or an ``(S, K)`` batch of them) and
    jit-friendly. E.g. ``complement([0,2,4]) -> [1,3,5]`` (the free legs).
    """
    subset = jnp.asarray(subset)
    K = subset.shape[-1]
    in_subset = jax.nn.one_hot(subset, N).sum(axis=-2) > 0      # (..., N)
    order = jnp.argsort(in_subset, axis=-1, stable=True)        # not-in-subset first
    return order[..., : N - K]


# DON'T TOUCH THIS
def sample_placements(key, means, std, N):
    """M candidate foot placements per free leg, from per-leg Gaussians on xy.

    means : (F, 3)                 per-leg 3D foot centers (e.g. ``PLANTED[fids]``);
                                    the z stays fixed, only xy is perturbed.
    std   : scalar | (2,) | (F, 2) Gaussian standard deviation on xy.
    Returns (M, F, 3) newly-planted foot positions.
    """
    F = means.shape[0]
    noise_xy = jax.random.normal(key, (N, F, 2)) * jnp.asarray(std)
    noise = jnp.concatenate([noise_xy, jnp.zeros((N, F, 1))], axis=-1)   # z unperturbed
    return means[None] + noise


def validate_planted(shoulders, feet):
    """Do all F legs reach their feet? Reach annulus only, no joint limits.

    shoulders : SE3 (F,)   world shoulder frames.
    feet      : (F, 3)     planted foot positions.
    Returns a scalar bool. This is the single-stance atom; ``vmap`` it over
    whatever varies:

        # a batch of placements against one body:
        jax.vmap(validate_planted, (None, 0))(body @ SHOULDERS[fids], placements)   # (M,)

        # a batch of body poses against fixed planted feet (the old ``reacher``):
        jax.vmap(lambda b: validate_planted(b @ SHOULDERS[fids], planted))(bodies)  # (N,)
    """
    return jnp.all(jax.vmap(are_connected, (0, 0, None))(shoulders, feet, LENGTHS))


# DON'T TOUCH THIS
def validate_feet(body, feet):
    """Do all 6 legs reach their feet? Reach annulus only, no joint limits.

    body : SE3        world body pose.
    feet : (6, 3)     planted foot positions.
    Returns a boolean (6,) array.
    """
    return jax.vmap(are_connected, (0, 0, None))(body @ SHOULDERS, feet, LENGTHS)


# DON'T TOUCH THIS
def infer_feet(body, feet):
    """Do all 6 legs reach their feet? Reach annulus only, no joint limits.

    body : SE3        world body pose.
    feet : (6, 3)     planted foot positions.
    Returns tuple of a boolean (6,) array and a (6, 3) array of inferred joint angles.
    """
    return jax.vmap(infer_theta, (0, 0, None))(body @ SHOULDERS, feet, LENGTHS)




# DON'T TOUCH THIS
def stability_score(com: jax.Array, feet: jax.Array,  max_angle: float = jnp.pi / 2) -> jax.Array:
    """Tip-over stability score in [0, 1] for a support polygon and a CoM.

    feet : (N, 2)  foot xy on the flat support plane (z = 0). Requiring xy makes
                   the coplanar / flat-ground assumption explicit.
    com  : (3,)    center of mass -- xy for the margin, z for height above the plane.

    Score is the tip-over angle -- how far you could tilt before the CoM crosses
    the nearest support edge -- normalized by ``max_angle``:

        margin = signed horizontal distance from CoM to the nearest edge (>0 inside)
        theta  = atan2(margin, com_z)      # small height (low CoM) -> larger theta
        score  = clip(theta / max_angle, 0, 1)

    ``0`` = at/beyond an edge (unstable); higher = more tip-resistant. A lower CoM
    scores higher for the same footprint. Assumes feet in convex position.
    """
    c = feet.mean(axis=0)                     # (2,) foot centroid
    poly = feet[jnp.argsort(jnp.arctan2(feet[:, 1] - c[1], feet[:, 0] - c[0]))]  # CCW
    a, b = poly, jnp.roll(poly, -1, axis=0)
    e = b - a
    d = (e[:, 0] * (com[1] - a[:, 1]) - e[:, 1] * (com[0] - a[:, 0])) / jnp.linalg.norm(e, axis=1)
    margin = jnp.min(d)                       # nearest-edge horizontal margin
    return jnp.clip(jnp.arctan2(margin, com[2]) / max_angle, 0.0, 1.0)




def stance_valid(body, ids, feet, tau=0.05):
    """Is ``body`` a valid pose for the stance (planted legs ``ids`` at ``feet``)?

    Valid = reachable (every planted leg connects) AND stable (CoM-in-polygon score
    above ``tau``). CoM proxy is the base origin. Returns a scalar bool.

    body : SE3        world body pose.
    ids  : (F,)       planted leg indices.
    feet : (F, 3)     their world foot positions.
    """
    reach = validate_planted(body @ SHOULDERS[ids], feet)
    stable = stability_score(body.translation(), feet[:, :2]) > tau
    return reach & stable


def transition_valid(bodies, old_ids, old_feet, new_ids, new_feet, tau=0.05):
    """Strategy-1 edge grid: which (body, new-placement) pairs give a transition.

    A pair is valid iff the body pose is a valid stance under BOTH the old placement
    and the new one -- then we can slide the body there on the old feet and swap.

    bodies   : SE3 (B,)      candidate body poses.
    old_ids  : (Fo,)         current planted legs.
    old_feet : (Fo, 3)       current (fixed) planted feet.
    new_ids  : (Fn,)         next planted legs.
    new_feet : (M, Fn, 3)    sampled placements for the next planted feet.
    Returns valid ``(B, M)`` bool.
    """
    old_ok = jax.vmap(lambda b: stance_valid(b, old_ids, old_feet, tau))(bodies)    # (B,)
    new_ok = jax.vmap(                                                              # (B, M)
        lambda b: jax.vmap(lambda f: stance_valid(b, new_ids, f, tau))(new_feet)
    )(bodies)
    return old_ok[:, None] & new_ok
