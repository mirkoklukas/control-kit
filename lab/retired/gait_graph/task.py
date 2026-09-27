"""The task: what we want the gait to achieve, and how we score a transition.

A gait-graph edge is a transition ``(last_stance, last_body) -> (stance, body)``.
``task_scorer`` is the reward *term* that says how well a candidate transition
serves the task -- e.g. advancing the body by a desired step each transition.
It is one component of the transition reward; ``branch`` combines it with the
stability term (task progress gated by stability).

Pure jax; run under ``uv run --extra mjx``.
"""
from __future__ import annotations

import dataclasses
from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.tree_util.register_dataclass,
         data_fields=["step", "sigma"], meta_fields=[])
@dataclasses.dataclass
class Task:
    """A velocity/step task: advance the body by ``step`` (xy) each transition.

    step  : (2,)     desired body xy displacement per transition (metres).
    sigma : scalar   bandwidth -- how forgiving the score is around ``step``.
    """
    step: jax.Array
    sigma: jax.Array


def canonical_xy(stance):
    """Canonical body xy for a stance: its support centroid.

    A stance's *canonical body* is the pose that maximises stability over its
    support (the Chebyshev center of the support polygon). We approximate it by
    the support centroid -- cheap, deterministic, and ~equal to the true optimum
    for roughly-regular tripods. (The grid-based ``argmax_b stb`` is a more
    faithful but noisier alternative.) Works on a single stance ``(n, 3)`` ->
    ``(2,)`` or a batched one ``(S, n, 3)`` -> ``(S, 2)``.
    """
    return stance.foot_positions[..., :2].mean(axis=-2)


def body_neutral_score(bodies):
    z_scores = bodies.translation()[...,2]
    return jnp.exp(-0.5 * ((z_scores - 0.241185) / 0.1)**2)


def task_scorer(bodies, stb_old, stb_new, *, task: Task):
    """Reward term: how well each candidate STANCE advances the task.
    ...
    """

    # We score each body individually, these are the "test" bodies.
    x_scores = bodies.translation()[...,0]
    z_scores = bodies.translation()[...,2]
    body_scores = x_scores * (jnp.exp(-0.5 * ((z_scores - 0.241185) / 0.1)**2))

    mask0 = stb_old > 0.05
    mask1 = stb_new > 0.05
    score0 = jnp.max(body_scores[:,None] * mask0)
    score1 = jnp.max(body_scores[:,None] * mask1, axis=0)
    return jax.nn.sigmoid(score1[None, :] - score0[None,None])
