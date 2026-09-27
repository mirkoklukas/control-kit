"""A hip-yaw / hip-pitch / knee-pitch leg.

The two pitch joints do the reaching; yaw alone orients the plane they reach in
(the *pitch frame*). Unlike :class:`.Leg4DOF` there is no redundant DOF: a foot
position determines the leg up to a finite set of branches -- two yaws a half-turn
apart (facing the foot, or facing away from it), each with two elbow branches.

:meth:`Leg3DOF.ik_from_foot` returns *every* branch, gated by a ``reachable`` mask;
picking one is the caller's job (by joint limits, or by sampling -- see
:meth:`Leg3DOF.sample_planted`).

Positions and directions are in the **leg's base frame**, i.e. the mount is at the
origin. Transform at the call site: ``mount.inverse().apply(foot)`` for points,
``mount.rotation().inverse().apply(n)`` for directions.
"""

from dataclasses import dataclass
from typing import ClassVar

import jax
import jax.numpy as jnp

from controlkit.kinematics import candidates, chain, planar
from controlkit.kinematics.leg import Leg
from controlkit.kinematics.types import Foothold
from controlkit.se3 import SE3


@jax.tree_util.register_dataclass
@dataclass
class Leg3DOF(Leg):
    """A 3-DOF leg: hip-yaw, hip-pitch, knee-pitch.

    A :class:`Leg` (a hinge chain) specialised by analytic inverse kinematics. The
    solver assumes this leg type's structure: the ``(z, y, y)`` axis pattern and
    colinear ``+x`` links. Build one with :meth:`from_lengths`.

    Forward kinematics is the base ``Leg``'s -- the chain is data, so nothing to
    override.
    """

    # This leg type's axis pattern: hip-yaw (z), hip-pitch (y), knee-pitch (y). Not
    # a dataclass field -- the canonical value `from_lengths` writes into `axes`.
    _AXES: ClassVar = jnp.array([
        [0.0, 0.0, 1.0],   # hip-yaw    (z)
        [0.0, 1.0, 0.0],   # hip-pitch  (y)
        [0.0, 1.0, 0.0],   # knee-pitch (y)
    ])

    @classmethod
    def from_lengths(cls, lengths: jax.Array, limits=None) -> "Leg3DOF":
        """Build a 3-DOF leg from its straight-``x`` link lengths.

        A classmethod, not ``__init__``, for the same reason as
        :meth:`.Leg4DOF.from_lengths`: ``register_dataclass`` rebuilds the leg from
        the chain fields on every unflatten.

        Args:
            lengths: (3,) link lengths ``[coxa, femur, tibia]``; ``lengths[-1]`` is
                the foot segment.
            limits: (3, 2) ``[lo, hi]`` per joint, radians; None -> full range.

        Returns:
            A :class:`Leg3DOF`.
        """
        offsets, tool = chain.x_offsets(jnp.asarray(lengths))
        return cls(offsets=offsets, axes=cls._AXES, tool=tool, limits=limits)

    # # # # # # # # # # # # # # #
    #   Inverse kinematics
    # # # # # # # # # # # # # # #
    def pitch_frame(self, yaw: jax.Array) -> SE3:
        """Frame the two pitch joints operate in, given the hip yaw.

        Origin at the hip-pitch joint (``lengths[0]`` along the yawed x-axis), x
        along the chain; the reaching happens in its xz-plane. The plane contains
        the base z-axis, so it contains the foot exactly when yaw faces it.

        Args:
            yaw: hip-yaw angle, radians.

        Returns:
            SE3 pitch frame, in the leg's base frame.
        """
        return (
            chain.link(self.offsets[0], yaw, self.axes[0])
            @ chain.link(self.offsets[1], 0.0, self.axes[1])
        )

    def _pitches_for(self, foot: jax.Array, yaw: jax.Array):
        """Planar solve in the pitch frame implied by ``yaw``.

        Returns:
            ``(reachable, pitches)`` -- scalar bool and (2, 2), per
            :func:`planar.solve`.
        """
        u = self.pitch_frame(yaw).inverse().apply(foot)
        return planar.solve(u[0], u[2], self.lengths[-2], self.lengths[-1])

    def ik_from_foot(self, foot: jax.Array):
        """Joint angles reaching ``foot``.

        Yaw faces the foot (``atan2(y, x)``) or faces away from it (a half-turn
        off); each yaw admits two elbow branches. Degenerate when the foot lies on
        the base z-axis: every yaw qualifies and ``atan2(0, 0) = 0`` is returned.

        Args:
            foot: (3,) target foot position, leg base frame.

        Returns:
            ``(reachable, thetas)`` with shapes (4,) and (4, 3). Branch order is
            yaw-major (facing first), elbow-down first. Angles on unreachable
            branches are finite but meaningless -- gate on ``reachable``.
        """
        x, y, _ = foot
        yaws = jnp.array([jnp.arctan2(y, x), jnp.arctan2(-y, -x)])
        ok, pitches = jax.vmap(self._pitches_for, (None, 0))(foot, yaws)  # (2,), (2, 2, 2)
        shape = pitches.shape[:2]                                          # (2, 2)
        return (
            jnp.broadcast_to(ok[:, None], shape).reshape(-1),
            jnp.stack(
                [jnp.broadcast_to(yaws[:, None], shape), pitches[..., 0], pitches[..., 1]],
                axis=-1,
            ).reshape(-1, self.num_joints),
        )

    # # # # # # # # # # # # # # #
    #   Sampling
    # # # # # # # # # # # # # # #
    def sample_planted(self, key: jax.Array, foothold: Foothold, *,
                       sharpness: float = 0.0):
        """Sample a leg configuration standing on ``foothold``.

        No redundancy to sample -- just a draw among the (at most four) IK branches
        that are reachable and within ``self.limits``, in proportion to
        ``exp(sharpness * <contact_vector, normal>)``.

        Args:
            key: PRNG key.
            foothold: target foothold, in the leg's base frame.
            sharpness: inverse temperature of the branch weighting. 0 (default)
                draws uniformly among the valid branches; higher favours squarer
                footings.

        Returns:
            ``(valid, theta)`` -- scalar bool and (3,). When nothing valid exists a
            branch is still drawn and ``valid`` is false.
        """
        ok, thetas = self.ik_from_foot(foothold.position)
        ok = ok & candidates.in_limits(thetas, self.limits)
        vs = jax.vmap(self.contact_vector)(thetas)
        logits = sharpness * candidates.contact_alignment(vs, foothold.normal)

        i = candidates.sample(key, logits, ok)
        return ok[i], thetas[i]
