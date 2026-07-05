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

from lab.stance_graph.kinematics import from_te, complement


# Default half-widths of the uniform proposal box around the base body pose,
# matching notebook 04_gait_graph. xyz in metres, rpy in radians.
XYZ_DELTA = jnp.array([0.2, 0.2, 0.1])
RPY_DELTA = jnp.deg2rad(jnp.array([10.0, 10.0, 30.0]))

# Default proposal for foot placements: per-leg 2D Gaussian on xy (std, metres),
# centred on the free legs' home positions nudged forward by SHIFT.
FOOT_STD = 0.1
FOOT_SHIFT = jnp.array([0.1, 0.0, 0.0])


def body_sampler(key, body0, xyz_delta=XYZ_DELTA, rpy_delta=RPY_DELTA, N=100):
    x0 = body0.translation()
    rpy0 = jnp.asarray(body0.rotation().as_rpy_radians())

    key_xyz, key_rpy = jax.random.split(key)
    xs = jax.random.uniform(key_xyz, (N, 3), minval=x0 - xyz_delta, maxval=x0 + xyz_delta)
    rpys = jax.random.uniform(key_rpy, (N, 3), minval=rpy0 - rpy_delta, maxval=rpy0 + rpy_delta)
    bodies = jax.vmap(from_te)(xs, rpys)

    return bodies


def foot_sampler(key, mean, N=100, std=0.1, shift=jnp.array([0.1, 0.0, 0.0])):
    """Sample ``N`` candidate foot layouts: swing legs stride from last footholds.
    """
    # Sample foot positions from a Gaussian distribution around the mean

    samples = (
        std*jax.random.normal(key, (N, *mean.shape)) + 
        mean  +
        jnp.broadcast_to(shift, mean.shape)
        )
    samples = samples.at[..., 2].set(0.0)  # Set z-coordinate to 0 for all samples
    
    return samples