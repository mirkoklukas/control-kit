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


def _sample_box(key, center, spec, N):
    """Uniform samples in a box around ``center``.

    ``spec`` is either a symmetric half-width (scalar or ``(3,)`` delta, so the box
    is ``center +/- spec``) or a ``(3, 2)`` ``[lo, hi]`` offset range per axis (box
    is ``center + [lo, hi]``, allowing asymmetric bounds). ``delta d`` == range
    ``[-d, +d]``.
    """
    spec = jnp.asarray(spec)
    if spec.ndim <= 1:                                  # scalar / (3,) delta -> symmetric
        lo, hi = center - spec, center + spec
    else:                                               # (3, 2) [lo, hi] offset range
        lo, hi = center + spec[:, 0], center + spec[:, 1]
    return jax.random.uniform(key, (N, 3), minval=lo, maxval=hi)


def body_sampler(key, body0, xyz_delta=XYZ_DELTA, rpy_delta=RPY_DELTA, N=100):
    """Sample ``N`` bodies uniformly in a box around ``body0``.

    ``xyz_delta`` / ``rpy_delta`` are each either a ``(3,)`` symmetric half-width
    (delta) or a ``(3, 2)`` ``[lo, hi]`` offset range per axis (see ``_sample_box``).
    """
    x0 = body0.translation()
    rpy0 = jnp.asarray(body0.rotation().as_rpy_radians())

    key_xyz, key_rpy = jax.random.split(key)
    xs = _sample_box(key_xyz, x0, xyz_delta, N)
    rpys = _sample_box(key_rpy, rpy0, rpy_delta, N)
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