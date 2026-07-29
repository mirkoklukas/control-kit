from dataclasses import dataclass

import jax
import jax.numpy as jnp
from controlkit.se3 import SE3


# Alternative names: ContactSite, SupportSite
@jax.tree_util.register_dataclass
@dataclass
class Foothold: 
    """Location in the environment that is suitable for placing a foot."""
    position: jax.Array
    normal: jax.Array 
    # Zero normal means no contact and a floating position, i.e. 
    # the foothold is not used, but we may store a foot position in there.

    def __post_init__(self):
        # A zero normal is the "no contact" sentinel; leave it zero rather than
        # dividing by its length. Any genuine normal is normalized.
        norm = jnp.linalg.norm(self.normal, axis=-1, keepdims=True)
        self.normal = self.normal / jnp.where(norm > 0, norm, 1.0)

    def transform(self, tf: SE3) -> "Foothold":
        """Move the foothold by an SE3.

        The two fields transform differently, which is why this lives here and not
        on ``SE3``: ``position`` is a point (rotated and translated), ``normal`` is
        a direction (rotated only). Applying ``tf`` uniformly would translate the
        normal, which is wrong.

        Args:
            tf: SE3 to apply; may be batched to match a batched foothold.

        Returns:
            The transformed foothold.
        """
        return Foothold(
            position=tf.apply(self.position),
            normal=tf.rotation().apply(self.normal),
        )

    def __getitem__(self, index) -> "Foothold":
        """Index a batched foothold (leading axis) -> a foothold."""
        return Foothold(self.position[index], self.normal[index])

    @property
    def shape(self) -> tuple[int, ...]:
        return self.position.shape[:-1]

    def __iter__(self):
        """Iterate over the leading axis of a batched foothold."""
        for i in range(self.shape[0]):
            yield self[i]

    @property
    def pos(self) -> jax.Array:
        return self.position


@jax.tree_util.register_dataclass
@dataclass
class Support:
    """Which legs are planted, and on which footholds.

    The contact assignment: leg ``ids[i]`` stands on ``sites[i]``. Legs not named
    in ``ids`` are free. Only the ``k`` active contacts are stored, so this knows
    nothing of the robot's leg count -- the robot fills the free legs.

    Args:
        sites: (k,) footholds the planted legs stand on.
        ids: (k,) which leg each site belongs to.
    """

    sites: Foothold
    ids: jax.Array

    @property
    def k(self) -> int:
        """Number of active contacts."""
        return self.ids.shape[-1]

    @property
    def shape(self) -> tuple[int, ...]:
        return self.sites.shape[:-1]

    def mask(self, num_legs: int) -> jax.Array:
        """Boolean planted mask over all ``num_legs`` legs.

        ``mask[i]`` is True iff leg ``i`` appears in ``ids`` (is planted). ``Support``
        stores only the ``k`` active contacts and not the robot's leg count, so that
        count is supplied here. This is the ``(num_legs,)`` form the statics solvers
        take (e.g. :func:`controlkit.forces.stance_forces`'s ``support``).

        Args:
            num_legs: total number of legs.

        Returns:
            (num_legs,) bool.
        """
        return jnp.zeros(self.shape + (num_legs,), bool).at[...,self.ids].set(True)

    def __getitem__(self, index) -> "Support":
        """Index a batched support (leading axis) -> a support."""
        return Support(self.sites[index], self.ids[index])

    def __iter__(self):
        """Iterate over the leading axis of a batched support."""
        for i in range(self.shape[0]):
            yield self[i]



@jax.tree_util.register_dataclass
@dataclass
class Posture:
    """A robot configuration: body pose and per-leg joint angles.

    The robot-level analog of a leg's ``theta``, and pure kinematics -- the
    :class:`Support` it was produced for is a separate thing, paired at the call
    site rather than bundled in.

    Args:
        body: SE3 body pose.
        thetas: (num_legs, num_joints) joint angles.
    """

    body: SE3
    thetas: jax.Array

    def __getitem__(self, index) -> "Posture":
        """Index a batched posture (leading axis) -> a posture."""
        return Posture(self.body[index], self.thetas[index])

    @property
    def shape(self) -> tuple[int, ...]:
        return self.body.shape

    def __iter__(self):
        """Iterate over the leading axis of a batched posture."""
        for i in range(self.shape[0]):
            yield self[i]