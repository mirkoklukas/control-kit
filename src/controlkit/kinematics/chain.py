"""Forward kinematics for a serial chain of hinges. Any DOF count.

A chain is a list of links, each a fixed **offset** followed by a **hinge**:

    frame_k = frame_{k-1} @ offset_k @ hinge(theta_k, axis_k)

This mirrors a MuJoCo body/joint exactly: ``offset`` is the child body's
``pos``/``quat`` (a full :class:`SE3`), ``axis`` is the joint's ``axis`` (a
3-vector). So a chain transcribes an MJCF subtree 1:1, and we can go back and forth
without reinterpreting the geometry. The trailing ``tool`` offset is the last body
with no joint -- the foot segment.

This is the general product-of-exponentials revolute chain; the earlier version
special-cased ``offset = translate(x*x_hat)`` and a principal axis. The common
straight-``x`` / principal-axis leg is now just :func:`x_offsets` +
:func:`principal`, which reproduce that case exactly.

One MJCF feature deliberately not modelled yet: a joint ``pos`` (rotation anchor)
off the body origin. Every anchor in our model is 0; a nonzero one is a
translate/un-translate we add when a model needs it.

All frames here are in the leg's *base* frame (the mount sits at the origin).
The caller composes with the mount pose.
"""

import jax
import jax.numpy as jnp

from controlkit.se3 import SE3, SO3


# Principal axes as unit 3-vectors, for building the common leg by hand.
_PRINCIPAL = {
    "x": jnp.array([1.0, 0.0, 0.0]),
    "y": jnp.array([0.0, 1.0, 0.0]),
    "z": jnp.array([0.0, 0.0, 1.0]),
}


def principal(axes) -> jax.Array:
    """Principal-axis characters -> unit 3-vectors.

    Args:
        axes: an iterable of ``"x"``/``"y"``/``"z"`` (a string works).

    Returns:
        (n, 3) stacked unit axis vectors.
    """
    return jnp.stack([_PRINCIPAL[c] for c in axes])


def hinge(theta: jax.Array, axis: jax.Array) -> SE3:
    """A pure rotation by ``theta`` about a unit 3-vector, at the origin.

    Args:
        theta: hinge angle, radians.
        axis: (3,) unit rotation axis.

    Returns:
        The SE3 of the hinge (zero translation).
    """
    return SE3.from_rotation_and_translation(SO3.exp(theta * axis), jnp.zeros(3))


def link(offset: SE3, theta: jax.Array, axis: jax.Array) -> SE3:
    """One fixed offset followed by one hinge.

    Args:
        offset: SE3 from the previous joint frame to this joint's pre-motion frame
            (a MuJoCo body ``pos``/``quat``).
        theta: hinge angle, radians.
        axis: (3,) unit rotation axis, in the post-offset frame.

    Returns:
        The SE3 taking the previous joint's frame to this one's.
    """
    return offset @ hinge(theta, axis)


def joint_frames(theta: jax.Array, offsets: SE3, axes: jax.Array, tool: SE3) -> SE3:
    """Frames of every joint and of the foot.

    Args:
        theta: (n,) joint angles.
        offsets: (n,) SE3 fixed offsets, one before each hinge. ``offsets[0]`` is the
            offset from the base frame to the first joint (identity if the first
            joint sits at the origin).
        axes: (n, 3) unit rotation axes.
        tool: SE3 from the last joint to the foot (the foot segment; no joint).

    Returns:
        (n + 1,) SE3 frames, base to foot. The last is the foot.
    """
    n = theta.shape[0]
    tf = SE3.identity()
    frames = []
    for k in range(n):
        tf = tf @ offsets[k] @ hinge(theta[k], axes[k])
        frames.append(tf)
    frames.append(tf @ tool)
    return SE3.stack(frames)


def x_offsets(lengths: jax.Array) -> tuple[SE3, SE3]:
    """The straight-``x`` skeleton: offsets and tool for a colinear-link chain.

    Every link runs along the previous frame's +x, which is the common leg (and
    exactly our MJCF, where the only rotation-bearing offset is the mount, kept at
    the :class:`Robot` level). Pair with :func:`principal` for the hand-built case.

    Args:
        lengths: (n,) link lengths; ``lengths[-1]`` is the foot segment.

    Returns:
        ``(offsets, tool)`` -- ``offsets`` is (n,) SE3 (``offsets[0]`` identity,
        ``offsets[k]`` a translation of ``lengths[k-1]`` along +x); ``tool`` is a
        translation of ``lengths[-1]`` along +x.
    """
    x = jnp.array([1.0, 0.0, 0.0])
    n = lengths.shape[-1]
    offsets = [SE3.identity()] + [
        SE3.from_translation(lengths[k] * x) for k in range(n - 1)
    ]
    tool = SE3.from_translation(lengths[-1] * x)
    return SE3.stack(offsets), tool


# NOTE: contact_vector lives on `Leg` (derived from `Leg.forward`), not here. A
# chain-flavoured copy would re-derive geometry behind the leg's back and silently
# disagree with any leg type whose `forward` is not built from this module.
