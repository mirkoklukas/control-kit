"""Proposal samplers for branching: draw candidate bodies (and, later, feet).

Goal-agnostic proposals -- they scatter candidates around a base pose; the goal
enters later as a separate ``goal_score`` (see the design in the notebook /
README). Each sampler returns its samples AND their proposal log-density, so the
downstream scoring can importance-weight against the proposal.

Pure jax; run under ``uv run --extra mjx``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from lab.gait_graph.kinematics import from_te
from lab.gait_graph.stance import complement


# Default half-widths of the uniform proposal box around the base body pose,
# matching notebook 04_gait_graph. xyz in metres, rpy in radians.
XYZ_DELTA = jnp.array([0.2, 0.2, 0.1])
RPY_DELTA = jnp.deg2rad(jnp.array([10.0, 10.0, 30.0]))

# Default proposal for foot placements: per-leg 2D Gaussian on xy (std, metres),
# centred on the free legs' home positions nudged forward by SHIFT.
FOOT_STD = 0.1
FOOT_SHIFT = jnp.array([0.1, 0.0, 0.0])
# FOOT_SHIFT = jnp.array([0.0, 0.0, 0.0])


def body_sampler(key, body0, xyz_delta=XYZ_DELTA, rpy_delta=RPY_DELTA, N=100):
    """Sample ``N`` witness bodies uniformly in a box around ``body0``.

    Draws xyz uniformly in ``body0.xyz ± xyz_delta`` and rpy uniformly in
    ``body0.rpy ± rpy_delta`` (independent axes), then rebuilds the SE3.

    # TODO: condition on the goal -- bias the box toward the desired motion
    #       (importance sampling), which is why we carry the log-prob out.

    body0     : SE3            base body pose to scatter around (single, not batched).
    xyz_delta : (3,) | scalar  half-width of the xyz box (metres).
    rpy_delta : (3,) | scalar  half-width of the rpy box (radians).

    Returns ``(bodies, logp)`` -- a batched SE3 of shape ``(N,)`` and the proposal
    log-density ``(N,)`` in ``(xyz, rpy)`` coordinates. It is constant across
    samples (uniform box) but returned per-sample to compose with other samplers.
    """
    x0 = body0.translation()
    rpy0 = jnp.asarray(body0.rotation().as_rpy_radians())

    key_xyz, key_rpy = jax.random.split(key)
    xs = jax.random.uniform(key_xyz, (N, 3), minval=x0 - xyz_delta, maxval=x0 + xyz_delta)
    rpys = jax.random.uniform(key_rpy, (N, 3), minval=rpy0 - rpy_delta, maxval=rpy0 + rpy_delta)
    bodies = jax.vmap(from_te)(xs, rpys)

    # Uniform over a box of side 2*delta per axis -> density = 1 / total volume.
    log_vol = jnp.sum(jnp.log(2.0 * xyz_delta)) + jnp.sum(jnp.log(2.0 * rpy_delta))
    logp = jnp.full((N,), -log_vol)
    return bodies, logp


def foot_sampler(key, posture, move_ids, N=100):
    """Sample ``N`` candidate foot layouts: swing legs stride from last footholds.
    ...

    Returns ``(foot_layout, logp)`` -- ``foot_layout`` is ``(N, 6, 3)``, a full
    set of foot positions per sample indexed by leg (``next_stances =
    foot_layout[next_ids]``); ``logp`` is ``(N,)``, the proposal log-density
    (product of the free-leg xy Gaussians; planted feet and z are deterministic).
    """
    free_ids = move_ids
    keep_ids = complement(free_ids)

    means = posture.foot_positions[free_ids].at[:,2].set(0.0) + FOOT_SHIFT[None]   # (F, 3)
    
    F = free_ids.shape[0]
    std = jnp.asarray(0.1)
    noise_xy = jax.random.normal(key, (N, F, 2)) * std          # (N, F, 2)
    free_feet = means[None] + jnp.concatenate(                  # (N, F, 3), z unperturbed
        [noise_xy, jnp.zeros((N, F, 1))], axis=-1) 

    # Assemble the full layout: planted feet fixed (broadcast over N), free sampled.
    layout = jnp.zeros((N, 6, 3))
    layout = layout.at[:, keep_ids].set(posture.foot_positions[keep_ids][None])   # (N, 6, 3)
    layout = layout.at[:, free_ids].set(free_feet)

    # Proposal log-density: sum of the per-free-leg 2D Gaussian logs (xy only).
    z = noise_xy / std
    logp = jnp.sum(-0.5 * z**2 - 0.5 * jnp.log(2 * jnp.pi) - jnp.log(std), axis=(1, 2))
    return layout, logp
