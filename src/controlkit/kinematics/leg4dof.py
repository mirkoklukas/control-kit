"""A hip-yaw / hip-roll / hip-pitch / knee-pitch leg.

The two pitch joints do the reaching; yaw and roll orient the plane they reach in
(the *pitch frame*). That leaves one redundant DOF: a foot position alone does not
determine the leg. Killing the redundancy takes one more scalar constraint, and
there are two natural choices -- hence two ``ik_from_*`` methods:

- :meth:`Leg4DOF.ik_from_foot_and_yaw` -- you pick the yaw.
- :meth:`Leg4DOF.ik_from_foot_and_normal` -- the normal must lie in the pitch plane.

Both return *every* branch, gated by a ``reachable`` mask; picking one is the
caller's job (by joint limits, or by scoring -- see :meth:`Leg4DOF.sample_planted`).

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
class Leg4DOF(Leg):
    """A 4-DOF leg: hip-yaw, hip-roll, hip-pitch, knee-pitch.

    A :class:`Leg` (a hinge chain) specialised by analytic inverse kinematics. The
    solvers below assume this leg type's structure: the ``(z, x, y, y)`` axis
    pattern and colinear ``+x`` links. Build one with :meth:`from_lengths`, which
    lays that structure out from scalar link lengths; constructing the base ``Leg``
    fields (``offsets``/``axes``/``tool``) directly is possible, but the IK will
    misbehave if they stray from it.

    Forward kinematics is the base ``Leg``'s -- the chain is data, so nothing to
    override.
    """

    # This leg type's axis pattern: hip-yaw (z), hip-roll (x), hip-pitch (y),
    # knee-pitch (y), as MuJoCo-style unit 3-vectors. Not a dataclass field -- the
    # canonical value `from_lengths` writes into the (inherited) `axes` field.
    _AXES: ClassVar = jnp.array([
        [0.0, 0.0, 1.0],   # hip-yaw    (z)
        [1.0, 0.0, 0.0],   # hip-roll   (x)
        [0.0, 1.0, 0.0],   # hip-pitch  (y)
        [0.0, 1.0, 0.0],   # knee-pitch (y)
    ])

    @classmethod
    def from_lengths(cls, lengths: jax.Array, limits=None) -> "Leg4DOF":
        """Build a 4-DOF leg from its straight-``x`` link lengths.

        The intended constructor: it fixes the ``(z, x, y, y)`` axis pattern the IK
        assumes and lays the links out along ``+x`` (:func:`chain.x_offsets`).

        A classmethod, not ``__init__``, on purpose: the stored state is the general
        chain (``offsets``/``axes``/``tool``), and ``register_dataclass`` rebuilds the
        leg by calling ``cls(**those_fields)`` on every unflatten (after ``jit`` /
        ``vmap`` / ``tree_map``). So ``__init__`` must take the chain fields;
        ``lengths`` is a construction convenience layered on top.

        Args:
            lengths: (n,) link lengths, base to foot; ``lengths[-1]`` is the foot
                segment.
            limits: (n, 2) ``[lo, hi]`` per joint, radians; None -> full range.

        Returns:
            A :class:`Leg4DOF`.
        """
        offsets, tool = chain.x_offsets(jnp.asarray(lengths))
        return cls(offsets=offsets, axes=cls._AXES, tool=tool, limits=limits)

    # # # # # # # # # # # # # # #
    #   Inverse kinematics
    # # # # # # # # # # # # # # #
    def pitch_frame(self, yaw: jax.Array, roll: jax.Array) -> SE3:
        """Frame the two pitch joints operate in, given the orienting joints.

        The third factor advances to the hip-pitch origin *without* turning it, so
        the frame sits where the pitch chain starts: origin at the hip-pitch joint,
        x along the chain, and the reaching happens in its xz-plane.

        Its origin is ``lengths[0] + lengths[1]`` along the yawed x-axis regardless
        of roll (both offsets lie on the roll axis, so roll cannot move them). The
        plane therefore always contains the mount origin -- which is what makes
        :meth:`ik_from_foot_and_normal` exact rather than approximate.

        Args:
            yaw: hip-yaw angle, radians.
            roll: hip-roll angle, radians.

        Returns:
            SE3 pitch frame, in the leg's base frame.
        """
        return (
            chain.link(self.offsets[0], yaw, self.axes[0])
            @ chain.link(self.offsets[1], roll, self.axes[1])
            @ chain.link(self.offsets[2], 0.0, self.axes[2])
        )

    @staticmethod
    def infer_hip_roll(foot: jax.Array, yaw: jax.Array) -> jax.Array:
        """Rolls putting the foot in the pitch plane, for a given yaw.

        One point constrains the plane's normal to be perpendicular to it, leaving
        two rolls a half-turn apart. Both span the same plane; they are distinct
        leg configurations, so both are returned.

        Args:
            foot: (3,) foot position, leg base frame.
            yaw: hip-yaw angle, radians.

        Returns:
            (2,) hip-roll angles.
        """
        x, y, z = foot
        w = jnp.sin(yaw) * x - jnp.cos(yaw) * y
        return jnp.array([
            jnp.arctan2(-z, w) - jnp.pi / 2,
            jnp.arctan2(z, -w) - jnp.pi / 2,
        ])

    def _pitches_for(self, foot: jax.Array, yaw: jax.Array, roll: jax.Array):
        """Planar solve in the pitch frame implied by ``(yaw, roll)``.

        Returns:
            ``(reachable, pitches)`` -- scalar bool and (2, 2), per
            :func:`planar.solve`.
        """
        u = self.pitch_frame(yaw, roll).inverse().apply(foot)
        return planar.solve(u[0], u[2], self.lengths[-2], self.lengths[-1])

    def _assemble(self, yaw: jax.Array, rolls: jax.Array, ok: jax.Array,
                  pitches: jax.Array):
        """Broadcast orienting angles against pitch branches into a flat branch axis.

        Args:
            yaw: () or (k,) hip-yaw. () when one yaw is shared by all branches (as
                in :meth:`ik_from_foot_and_yaw`); (k,) when one per orienting branch.
            rolls: (k,) hip-roll, one per orienting branch.
            ok: (k,) reachability, one per orienting branch.
            pitches: (k, 2, 2) pitch branches per orienting branch.

        Returns:
            ``(reachable, thetas)`` with shapes (2k,) and (2k, num_joints).
        """
        shape = pitches.shape[:2]  # (k, 2)
        return (
            jnp.broadcast_to(ok[:, None], shape).reshape(-1),
            jnp.stack(
                [
                    jnp.broadcast_to(jnp.asarray(yaw).reshape(-1, 1), shape),
                    jnp.broadcast_to(rolls[:, None], shape),
                    pitches[..., 0],
                    pitches[..., 1],
                ],
                axis=-1,
            ).reshape(-1, self.num_joints),
        )

    def ik_from_foot_and_yaw(self, foot: jax.Array, yaw: jax.Array):
        """Joint angles reaching ``foot``, with hip-yaw pinned to ``yaw``.

        Fixing yaw removes the redundancy. Roll then follows from the foot (two
        branches), and each roll admits two elbow branches.

        Args:
            foot: (3,) target foot position, leg base frame.
            yaw: hip-yaw angle, radians.

        Returns:
            ``(reachable, thetas)`` with shapes (4,) and (4, 4). Branch order is
            roll-major, elbow-down first. Angles on unreachable branches are finite
            but meaningless -- gate on ``reachable``.
        """
        rolls = self.infer_hip_roll(foot, yaw)
        ok, pitches = jax.vmap(self._pitches_for, (None, None, 0))(foot, yaw, rolls)
        return self._assemble(yaw, rolls, ok, pitches)

    def ik_from_foot_and_normal(self, foot: jax.Array, normal: jax.Array):
        """Joint angles reaching ``foot`` with ``normal`` lying in the pitch plane.

        That is the constraint this spends the redundancy on: the foot and the
        normal are then two directions in the plane, so its out-of-plane axis is
        ``n = foot x normal`` and yaw/roll are read straight off it -- solved
        *jointly*, not one then the other.

        Note what this does **not** give you: the foot is not square to the surface.
        Squareness implies the normal is in the pitch plane, but not conversely --
        the pitches are already spent reaching the foot, so the contact angle is
        whatever falls out (measured: median ~60 degrees off the normal, even taking
        the best branch). Normal-in-plane means the leg *can* push along the normal
        using pitch alone; it does not mean it does. To get squareness, spend the
        redundancy on it instead -- see :meth:`sample_planted`, which scores
        branches by ``<contact_vector, normal>`` -- or filter these by contact angle.

        Both ``<A, n> = 0`` and ``<B, n> = 0`` have two roots a half-turn apart,
        giving four (yaw, roll) frames spanning that plane, each with two elbow
        branches. See ``docs/kinematics.md``, "All branches".

        Degenerate when ``foot`` and ``normal`` are parallel: the cross product
        vanishes, every plane through them qualifies, and the returned angles are
        arbitrary. ``reachable`` does *not* catch this.

        Args:
            foot: (3,) target foot position, leg base frame.
            normal: (3,) surface normal at the foot, leg base frame.

        Returns:
            ``(reachable, thetas)`` with shapes (8,) and (8, 4). Angles on
            unreachable branches are finite but meaningless -- gate on ``reachable``.
        """
        n = jnp.cross(foot, normal)
        r = jnp.hypot(n[0], n[1])
        phi = jnp.arctan2(n[1], n[0])

        # <A, n> = 0 pins yaw to phi -/+ pi/2; that choice flips the sign of the
        # in-plane term in <B, n> = 0, whose own two roots then give roll. Four frames.
        s = jnp.array([-1.0, -1.0, 1.0, 1.0])  # yaw branch
        t = jnp.array([-1.0, 1.0, -1.0, 1.0])  # roll branch
        yaws = phi - s * jnp.pi / 2
        # The zero-roll pitch plane is the xz-plane, not the xy-plane, hence -pi/2.
        rolls = jnp.arctan2(n[2], s * r) + t * jnp.pi / 2 - jnp.pi / 2

        ok, pitches = jax.vmap(self._pitches_for, (None, 0, 0))(foot, yaws, rolls)
        return self._assemble(yaws, rolls, ok, pitches)

    # # # # # # # # # # # # # # #
    #   Sampling
    # # # # # # # # # # # # # # #
    def sample_planted(self, key: jax.Array, foothold: Foothold, *,
                           sharpness: float = 5.0):
        """Sample a leg configuration standing on ``foothold``.

        The redundancy is sampled, not solved: draw a yaw, solve the rest, then
        draw among the branches in proportion to ``exp(sharpness * <v, normal>)``.

        This is the other way to spend the redundant DOF. Where
        :meth:`ik_from_foot_and_normal` spends it on putting the normal in the pitch
        plane -- which leaves the contact angle to chance -- this spends it on the
        contact angle itself, favouring square footings without demanding them.
        Branches outside ``self.limits`` are rejected before the draw.

        Args:
            key: PRNG key.
            foothold: target foothold, in the leg's base frame.
            sharpness: inverse temperature of the branch weighting. Higher is
                squarer; 0 draws uniformly among the valid branches.

        Returns:
            ``(valid, theta)`` -- scalar bool and (4,). ``valid`` means the drawn
            branch is both reachable and within joint limits. When nothing valid
            exists a branch is still drawn and ``valid`` is false.
        """
        key_yaw, key_pick = jax.random.split(key)
        yaw = jax.random.uniform(key_yaw, minval=self.limits[0, 0], maxval=self.limits[0, 1])

        ok, thetas = self.ik_from_foot_and_yaw(foothold.position, yaw)
        ok = ok & candidates.in_limits(thetas, self.limits)
        vs = jax.vmap(self.contact_vector)(thetas)                # via our forward
        logits = sharpness * candidates.contact_alignment(vs, foothold.normal)

        i = candidates.sample(key_pick, logits, ok)
        return ok[i], thetas[i]
