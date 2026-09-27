"""Concrete robot builds for the posture-graph experiments.

The ``controlkit.kinematics`` package is machinery -- ``Leg``, ``Robot``, the
solvers. A *particular* robot's dimensions are a fact about that robot, not about
kinematics, so the concrete instances live here, next to the experiments that use
them, rather than in the library.

``QUAD_4DOF`` is the one the experiments actually run on; ``HEX_4DOF`` is the
platform goal, kept for when the pipeline moves to it.
"""

import jax.numpy as jnp

from controlkit.kinematics import Leg4DOF, Robot, radial_mounts

_LIMITS_4DOF = jnp.deg2rad(jnp.array([
    [-80.0, 80.0],    # hip-yaw
    [-90.0, 90.0],    # hip-roll
    [-120.0, 120.0],  # hip-pitch
    [0.0, 120.0],     # knee-pitch
]))

HEX_4DOF = Robot(
    mounts=radial_mounts(6, radius=0.15),
    leg=Leg4DOF.from_lengths(jnp.array([0.025, 0.025, 0.2, 0.25]), _LIMITS_4DOF),
)

QUAD_4DOF = Robot(
    mounts=radial_mounts(4, radius=0.15),
    leg=Leg4DOF.from_lengths(jnp.array([0.05, 0.1, 0.3, 0.35]), _LIMITS_4DOF),
)
