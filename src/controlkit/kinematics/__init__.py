"""Kinematics for legged robots.

Layering, bottom up -- each layer knows nothing of the one above:

- ``chain``    forward kinematics for a hinge chain, any DOF.
- ``planar``   the two-link subproblem every leg type bottoms out in.
- ``candidates`` gates, scores and selectors over the branches a solver returns.
- ``leg``      the :class:`Leg` base: a hinge chain (offsets/axes/tool) and the
  forward kinematics it induces. The chain is data, so a bare ``Leg`` is usable.
- ``leg4dof``  a leg type: a :class:`Leg` subclass that adds analytic inverse
  kinematics and sampling for one axis/link structure. One module per leg type.
- ``robot``    the :class:`Robot`: a body with identical legs, plus the pose
  helpers. Concrete robot builds (specific dimensions) are the caller's, not the
  library's.

A leg carries its own kinematics -- it is a :class:`Leg` subclass, so generic code
calls ``leg.forward(...)`` / ``leg.sample_planted(...)`` without knowing the
type. Legs are homogeneous by construction.
"""

# Import order is dependency order: the leaf modules must be attributes of this
# package before `leg4dof` (which imports them from it) is itself imported.
from controlkit.kinematics import chain, planar, candidates, metrics, collision
from controlkit.kinematics.types import Foothold, Support, Posture
from controlkit.kinematics.leg import Leg
from controlkit.kinematics.robot import Robot, radial_mounts, sample_pose
from controlkit.kinematics import leg4dof
from controlkit.kinematics.leg4dof import Leg4DOF

__all__ = [
    "chain",
    "planar",
    "candidates",
    "metrics",
    "collision",
    "leg4dof",
    "Foothold",
    "Support",
    "Posture",
    "Leg",
    "Leg4DOF",
    "Robot",
    "radial_mounts",
    "sample_pose",
]
