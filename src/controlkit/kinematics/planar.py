"""The two-link planar subproblem, shared by every leg type.

Once a leg's orienting joints have fixed the pitch plane, what is left is always
the same: two links in a plane reaching for a point. 3DOF and 4DOF legs differ
only in how they arrive at that plane (see each leg module's ``pitch_frame``);
from here down they are identical.

Takes bare scalars, so it is blind to how the chain is parameterised.
"""

import jax
import jax.numpy as jnp


def solve(x: jax.Array, z: jax.Array, proximal: jax.Array, distal: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Both elbow branches placing a two-link arm's tip at ``(x, z)``.

    Angles are returned for *both* branches rather than resolved here -- picking
    is the caller's job, by joint limits (``ik_from_*``) or by scoring
    (``propose_*``), and those want different branches.

    Args:
        x: target coordinate along the plane's x axis.
        z: target coordinate along the plane's z axis.
        proximal: length of the base-side link (hip to knee).
        distal: length of the tip-side link (knee to foot).

    Returns:
        ``(reachable, theta)``. ``reachable`` is a scalar bool: the target lies in
        the reach annulus. ``theta`` is (2, 2) -- two branches, elbow-down first,
        each ``[hip_pitch, knee_pitch]``.

        When ``reachable`` is false the angles are finite but meaningless (the
        cosine arguments are clipped). They are deliberately not NaN: a NaN would
        survive multiplication by a zero mask and poison anything downstream that
        scores branches. Always gate on ``reachable``.
    """
    # Triangle notation for the cosine rule: sides a (base to target), b, c, with
    # interior angle alpha at the knee and gamma at the hip.
    b, c = proximal, distal
    a = jnp.hypot(x, z)
    reachable = (a <= (b + c)) & (a >= jnp.abs(b - c))

    offset = jnp.arctan2(z, x)
    alpha = jnp.arccos(jnp.clip((c**2 + b**2 - a**2) / (2 * c * b), -1.0, 1.0))
    # gamma's denominator carries `a`, which vanishes at the origin -- a target that
    # is unreachable unless b == c, but reachable *and* singular when it is. Guard
    # the denominator so the angle stays finite (meaningless, like the rest of an
    # unreachable branch) rather than NaN, which would survive the reachable mask.
    safe_a = jnp.where(a > 0, a, 1.0)
    gamma = jnp.arccos(jnp.clip((b**2 + a**2 - c**2) / (2 * b * safe_a), -1.0, 1.0))

    return reachable, jnp.array([
        [-(offset + gamma), -(-jnp.pi + alpha)],
        [-(offset - gamma), -jnp.pi + alpha],
    ])


def annulus(proximal: jax.Array, distal: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Radii of the reachable annulus in the pitch plane.

    The set of targets for which :func:`solve` reports ``reachable``. Samplers use
    this to draw reachable points directly instead of rejecting.

    Args:
        proximal: length of the base-side link (hip to knee).
        distal: length of the tip-side link (knee to foot).

    Returns:
        ``(r_min, r_max)``, being ``|proximal - distal|`` and their sum.
    """
    return jnp.abs(proximal - distal), proximal + distal
