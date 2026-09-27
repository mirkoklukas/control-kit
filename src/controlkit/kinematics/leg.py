"""The single leg: a hinge chain plus its forward kinematics.

A leg is a chain of hinges -- ``offsets``/``axes``/``tool``, exactly what
:mod:`.chain` consumes -- and that chain alone gives forward kinematics: the foot,
the contact vector, every position derive from :meth:`Leg.forward`, fixed here once.

The chain is **data, not code**: a plain :class:`Leg` is already a usable leg
(forward kinematics for any offsets/axes, e.g. one loaded from an MJCF). What a *leg
type* adds is inverse kinematics and sampling -- the analytic solvers a particular
axis/link structure admits (see :class:`.Leg4DOF`). So legs differ by their IK, not
by ``forward``.

The :class:`Robot` that mounts these lives in ``robot``, one level up.
"""

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from controlkit.kinematics import chain
from controlkit.kinematics.types import Foothold
from controlkit.se3 import SE3


@dataclass
class Leg:
    """A single leg: a hinge chain and the forward kinematics it induces.

    The chain follows :mod:`.chain`: ``offsets[k]`` places joint ``k``, ``axes[k]``
    is its rotation axis, ``tool`` is the foot segment. Homogeneous robots hold one
    ``Leg`` and ``vmap`` the solvers over feet, so a ``Leg`` is a single object,
    never batched over a leading axis.

    Args:
        offsets: (n,) SE3 fixed offsets, one before each hinge (a MuJoCo body's
            ``pos``/``quat``). ``offsets[0]`` is the base -> first-joint transform.
        axes: (n, 3) unit rotation axes (a MuJoCo joint's ``axis``).
        tool: SE3 from the last joint to the foot (the foot segment; no joint).
        limits: (n, 2) ``[lo, hi]`` per joint, radians. Defaults to the full
            ``[-pi, pi]`` per joint -- unconstrained, so limit checks are a no-op
            until real limits are supplied.
    """
    offsets: SE3
    axes: jax.Array
    tool: SE3
    limits: jax.Array = None

    def __post_init__(self):
        if self.limits is None:
            self.limits = jnp.broadcast_to(
                jnp.array([-jnp.pi, jnp.pi]), (self.num_joints, 2)
            )

    @property
    def num_joints(self) -> int:
        return self.axes.shape[0]

    @property
    def lengths(self) -> jax.Array:
        """Per-link lengths, base to foot -- a straight-``x`` convenience view.

        Link ``k`` runs from joint ``k`` to joint ``k+1`` (the foot for the last),
        so its length is the offset that places the next frame: ``|offsets[k+1]|``
        for the inner links, ``|tool|`` for the foot segment. Meaningful for a
        colinear-link leg (ours, and any MJCF whose intra-leg bodies are pure
        translations); for a general chain, read ``offsets`` directly.
        """
        seg = jnp.concatenate(
            [self.offsets.translation()[1:], self.tool.translation()[None]], axis=0
        )
        return jnp.linalg.norm(seg, axis=-1)

    def boxes(self, theta: jax.Array, *, radius: float = 0.015):
        """Oriented boxes for the leg's links, in the leg's base frame.

        One box per link: centred on the link, local x along it (length
        ``lengths[k]``), square ``radius`` x ``radius`` cross-section. It reads the
        link direction off each joint frame's x-axis, so like :attr:`lengths` it
        assumes colinear ``+x`` links (every leg type built with
        :func:`.chain.x_offsets`).

        Feed the result to :func:`..collision.overlap` for box-vs-box tests.

        Args:
            theta: (n,) joint angles.
            radius: half-thickness of the links.

        Returns:
            :class:`..collision.OBB` of ``n`` boxes.
        """
        from controlkit.kinematics import collision
        starts = self.forward(theta)[: self.num_joints]        # link starts (frames 0..n-1)
        rot = starts.rotation().as_matrix()                    # (n, 3, 3)
        length = self.lengths
        center = starts.translation() + 0.5 * length[:, None] * rot[..., 0]   # +x axis
        r = jnp.broadcast_to(radius, (self.num_joints,))
        half = jnp.stack([0.5 * length, r, r], axis=-1)
        return collision.OBB(center, half, rot)

    def forward(self, theta: jax.Array) -> SE3:
        """Frames of every joint and the foot, in the leg's base frame.

        Args:
            theta: (n,) joint angles.

        Returns:
            (n+1,) SE3 frames, base to foot. The last frame is the foot.
        """
        return chain.joint_frames(theta, self.offsets, self.axes, self.tool)

    def sample_free(self, key: jax.Array, limits=None) -> jax.Array:
        """Sample a random config uniform within the joint limits.

        "Free" as in unplanted -- the leg stands on nothing, so unlike
        ``sample_planted`` there is no foothold to reach, just a draw in the limit
        box. This is what a lifted leg gets in :meth:`Robot.sample_posture`.

        Args:
            key: PRNG key.
            limits: (n, 2) ``[lo, hi]`` bounds to sample within; defaults to
                ``self.limits``.

        Returns:
            (n,) joint angles.
        """
        if limits is None:
            limits = self.limits
        return jax.random.uniform(
            key, (self.num_joints,), minval=limits[:, 0], maxval=limits[:, 1]
        )

    def sample_planted(self, key: jax.Array, foothold: Foothold, **kwargs) -> tuple:
        """Sample a config that stands on a foothold, if reachable.

        "Planted" as in the leg is standing on something: the foot must reach the
        given ``foothold``. This is what a supported leg gets in :meth:`Robot.sample_posture`.

        Configs outside ``self.limits`` are rejected, so the draw is confined to the
        joint limits without a ``limits`` argument here. A concrete leg type may add
        its own keyword options (e.g. :meth:`Leg4DOF.sample_planted`'s ``sharpness``);
        :meth:`Robot.sample_posture` calls this with the foothold alone, so any such
        option needs a default.

        This base leg has forward kinematics but no inverse, so it cannot plant --
        a leg *type* with analytic IK must override this.

        Args:
            key: PRNG key.
            foothold: Foothold describing the target foothold, in the leg's base frame.
            **kwargs: leg-type-specific sampling options.

        Returns:
            ``(ok, theta)`` -- ``ok`` is true when a config was found, false when the
            foothold is unreachable; ``theta`` is the config (undefined if not ok).
        """
        raise NotImplementedError(
            "sample_planted needs inverse kinematics; use a leg type that defines it"
        )

    def reach(self, key: jax.Array, foothold: Foothold, *, samples: int = 16,
              **kwargs) -> jax.Array:
        """Can this leg stand on ``foothold``? A sampled reachability test.

        Draws ``samples`` planted configs with :meth:`sample_planted` and reports
        whether *any* is valid. Generic on purpose: it goes through
        ``sample_planted`` alone, so any leg type that can plant answers this and the
        base needs no IK of its own. Stochastic -- but the reachable set is one
        contiguous region, so a modest ``samples`` keeps false negatives rare; raise
        it if in doubt.

        Args:
            key: PRNG key.
            foothold: target foothold, in the leg's base frame.
            samples: number of plant attempts.
            **kwargs: forwarded to :meth:`sample_planted` (e.g.
                :meth:`Leg4DOF.sample_planted`'s ``sharpness``).

        Returns:
            Scalar bool: some attempt found a reachable, in-limit config.
        """
        oks, _ = jax.vmap(lambda k: self.sample_planted(k, foothold, **kwargs))(
            jax.random.split(key, samples))
        return jnp.any(oks)

    def foot(self, theta: jax.Array) -> jax.Array:
        """Foot position in the leg's base frame.

        Args:
            theta: (n,) joint angles.

        Returns:
            (3,) foot position.
        """
        return self.forward(theta)[-1].translation()

    def contact_vector(self, theta: jax.Array) -> jax.Array:
        """Unit vector at the foot, pointing back along the last link.

        Compare against a surface normal to score how squarely the foot meets it.

        Args:
            theta: (n,) joint angles.

        Returns:
            (3,) unit vector in the leg's base frame.
        """
        xpos = self.forward(theta).translation()      # (n+1, 3)
        v = xpos[-2] - xpos[-1]
        return v / jnp.linalg.norm(v)

    def collision_cloud(self, theta: jax.Array, *, delta: float) -> jax.Array:
        """Points along the leg's links, spaced at most ``delta`` apart. Base frame.

        Denser than :func:`..collision.densify`'s fixed per-link count: each link of
        length ``L`` gets ``ceil(L / delta) + 1`` points, so the actual spacing is
        ``<= delta`` regardless of how long the link is -- a uniform-density cloud
        for an SDF collision check.

        The per-link counts come from ``lengths`` (static geometry), so this needs
        the leg to be a concrete object, not one traced as a ``jit`` argument; the
        common captured-leg case is fine.

        Args:
            theta: (n,) joint angles.
            delta: maximum spacing between consecutive points.

        Returns:
            (P, 3) base-frame points, ``P = sum_k (ceil(lengths[k] / delta) + 1)``.
        """
        p = self.forward(theta).translation()          # (n+1, 3)
        # Counts are static geometry, but `float()` on a jax array is illegal mid-
        # trace even for a captured constant; this evaluates them at trace time.
        with jax.ensure_compile_time_eval():
            lens = self.lengths
            counts = [math.ceil(float(lens[k]) / delta) + 1
                      for k in range(self.num_joints)]
        segs = []
        for k, m in enumerate(counts):
            t = jnp.linspace(0.0, 1.0, m)
            segs.append(p[k] + t[:, None] * (p[k + 1] - p[k]))
        return jnp.concatenate(segs, axis=0)
