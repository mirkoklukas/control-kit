"""Choosing among the candidate configs an ``ik_from_*`` hands back.

Every solver returns ``(ok, thetas)`` -- a reachability mask and a leading branch
axis (the candidates) -- and refuses to pick, because different callers want
different branches. This is the vocabulary they pick with:

- **gates** (``in_limits``) -- (B,) bools, composed with ``&`` onto ``ok``.
- **scores** (``contact_alignment``, ``contact_angle``) -- (B,) floats. Plain
  arrays, so a new score is a new function and nothing here changes shape.
- **selectors** (``best``, ``sample``) -- gate + score in, index out.

Leg-agnostic, like ``planar``: bare arrays in, no leg, no geometry -- it never
derives anything from a chain, it only weighs what it is handed. Angles on
unreachable branches are finite but meaningless (see ``planar.solve``), so scores
computed over them are garbage -- which is exactly why every selector takes the
gate rather than trusting the score.
"""

import jax
import jax.numpy as jnp


def in_limits(thetas: jax.Array, limits: jax.Array) -> jax.Array:
    """Whether every joint of a branch lies within its limits.

    Args:
        thetas: (B, n) branches of joint angles.
        limits: (n, 2) ``[lo, hi]`` per joint, radians.

    Returns:
        (B,) bool.
    """
    within = (limits[:, 0] <= thetas) & (thetas <= limits[:, 1])
    return jnp.all(within, axis=-1)


def contact_alignment(vs: jax.Array, normal: jax.Array) -> jax.Array:
    """How squarely each branch meets a surface.

    The cosine of the angle between each contact vector and the normal. This is the
    primitive: use it directly as a score, and gate with ``>= cos(limit)`` rather
    than round-tripping through :func:`contact_angle`.

    Takes the contact vectors rather than deriving them: only the leg knows its own
    geometry (via ``Leg.forward``), so the caller passes
    ``jax.vmap(leg.contact_vector)(thetas)``. Deriving them here would hardcode one
    chain representation and quietly break for any leg type that doesn't use it.

    Args:
        vs: (B, 3) contact vectors, one per branch, in the leg's base frame.
        normal: (3,) unit surface normal, in the leg's base frame.

    Returns:
        (B,) in [-1, 1]. 1 is dead square, 0 is grazing.
    """
    return jnp.clip(vs @ normal, -1.0, 1.0)


def contact_angle(vs: jax.Array, normal: jax.Array) -> jax.Array:
    """Angle between each branch's contact vector and a surface normal.

    The readable view of :func:`contact_alignment`, for reporting and for gates
    written as ``angle <= limit``.

    Args:
        vs: (B, 3) contact vectors, one per branch, in the leg's base frame.
        normal: (3,) unit surface normal, in the leg's base frame.

    Returns:
        (B,) radians in [0, pi]. 0 is dead square.
    """
    return jnp.arccos(contact_alignment(vs, normal))


def best(score: jax.Array, ok: jax.Array) -> jax.Array:
    """Index of the highest-scoring branch that passes the gate.

    Args:
        score: (B,) higher is better.
        ok: (B,) bool gate.

    Returns:
        Scalar int index. When nothing passes, returns 0 -- gate on ``ok[i]``
        rather than trusting the index.
    """
    return jnp.argmax(jnp.where(ok, score, -jnp.inf))


def sample(key: jax.Array, logits: jax.Array, ok: jax.Array) -> jax.Array:
    """Draw a branch in proportion to ``exp(logits)``, among those passing the gate.

    Args:
        key: PRNG key.
        logits: (B,) unnormalized log-weights; higher is likelier.
        ok: (B,) bool gate.

    Returns:
        Scalar int index. When nothing passes, draws uniformly -- gate on
        ``ok[i]`` rather than trusting the index.
    """
    masked = jnp.where(ok, logits, -jnp.inf)
    # All-rejected would make every logit -inf and the draw NaN, which
    # `jax.random.categorical` turns into a silently garbage index rather than an
    # error. Fall back to uniform so `ok[i]` stays the caller's single check.
    masked = jnp.where(jnp.any(ok), masked, jnp.zeros_like(masked))
    return jax.random.categorical(key, masked)
