"""Stances, postures, and the atomic checks over them.

A **stance** is a set of planted feet: which legs (``foot_ids``) and where their
feet are (``foot_positions``, world frame). A **posture** is a body pose plus
joint angles -- the thing that connects two stances as a graph edge.

Both are plain, pytree-registered dataclasses (data only, no methods); every
operation is a free function that *takes* a stance/posture. This keeps the data
flat and vmap-friendly and the logic out in the open.

Pure jax; run under ``uv run --extra mjx``.
"""
from __future__ import annotations

import dataclasses
from functools import partial

import jax
import jax.numpy as jnp

from lab.gait_graph.kinematics import (
    JOINT_RANGES,
    LENGTHS,
    SHOULDERS,
    infer_theta,
    tipover_score,
)


def complement(subset, N=6):
    """Legs in ``range(N)`` not in ``subset``: ``(..., K) -> (..., N-K)``.

    The "free" legs of a stance (not currently planted). Vectorized and
    jit-friendly. E.g. ``complement([0, 2, 4]) -> [1, 3, 5]``.
    """
    subset = jnp.asarray(subset)
    K = subset.shape[-1]
    in_subset = jax.nn.one_hot(subset, N).sum(axis=-2) > 0          # (..., N)
    order = jnp.argsort(in_subset, axis=-1, stable=True)            # not-in-subset first
    return order[..., : N - K]


@partial(jax.tree_util.register_dataclass,
         data_fields=["foot_ids", "foot_positions"], meta_fields=[])
@dataclasses.dataclass
class Stance:
    """A set of planted feet.

    foot_ids       : (F,) int    which legs are planted (indices into 0..5).
    foot_positions : (F, 3)      their world foot positions.
    """
    foot_ids: jax.Array
    foot_positions: jax.Array

    def __getitem__(self, index) -> "Stance":
        """Index a batched stance (leading axis) -> a stance."""
        return Stance(self.foot_ids[index], self.foot_positions[index])


@partial(jax.tree_util.register_dataclass,
         data_fields=["body", "theta", "foot_positions"], meta_fields=[])
@dataclasses.dataclass
class Posture:
    """A body pose, joint angles, and the foot positions they realise -- an edge.

    body           : SE3       world body (base) pose.
    theta          : (F, 3)    joint angles [coxa, femur, tibia] per leg.
    foot_positions : (F, 3)    world foot positions the (body, theta) reach; stored
                               so callers (viz, geometry checks) needn't redo FK.
    """
    body: object          # jaxlie SE3 (itself a pytree)
    theta: jax.Array
    foot_positions: jax.Array

    def __getitem__(self, index) -> "Posture":
        """Index a batched posture (leading axis) -> a posture."""
        return Posture(self.body[index], self.theta[index], self.foot_positions[index])


def infer_posture(body, stance: Stance, joint_ranges=JOINT_RANGES):
    """Joint angles that plant ``stance``'s feet under body pose ``body``.

    Runs leg IK for each planted foot, then picks a knee branch: **prefer
    elbow-down**, fall back to elbow-up, and mark the leg invalid if neither
    branch is within ``joint_ranges``.

    Returns ``(valid, theta)`` with ``valid`` a per-leg ``(F,)`` bool (reachable
    AND some branch in-limits) and ``theta`` the chosen ``(F, 3)`` angles. The
    caller reduces ``valid`` (e.g. ``.all()``) to judge the whole stance.
    """
    shoulders = body @ SHOULDERS[stance.foot_ids]                   # SE3 (F,)
    reachable, theta = jax.vmap(infer_theta, (0, 0, None))(          # (F,), (F,2,3)
        shoulders, stance.foot_positions, LENGTHS)

    lo, hi = joint_ranges[:, 0], joint_ranges[:, 1]                  # (3,), (3,)
    in_limits = jnp.all((theta >= lo) & (theta <= hi), axis=-1)      # (F, 2) [down, up]
    down_ok, up_ok = in_limits[:, 0], in_limits[:, 1]               # (F,)

    theta = jnp.where(down_ok[:, None], theta[:, 0], theta[:, 1])    # (F, 3) prefer down
    valid = reachable & (down_ok | up_ok)                           # (F,)
    return valid, theta


def stability_scorer(posture: Posture, stance: Stance):
    """Tip-over score of ``posture``'s body over ``stance``'s support polygon.

    The support feet are given explicitly by ``stance`` (rather than inferred
    from the posture) since which feet bear load is a choice, not a geometric
    fact. CoM proxy is the base origin. Flat-ground heuristic; scalar in [0, 1].
    """
    return tipover_score(posture.body.translation(), stance.foot_positions[:, :2])


def transition_stability(posture: Posture, old_stance: Stance, new_stance: Stance, tau=0.05):
    """Stability of a foot-swap: the body must stand on BOTH supports.

    A transition lifts the free feet and plants the new ones; the body has to be
    statically stable on the old footprint (before) and the new one (after).
    Returns ``(stb_old, stb_new, ok)`` -- both raw scores plus ``ok = both > tau``
    (raw scores kept so callers see margins, not just the boolean).
    """
    stb_old = stability_scorer(posture, old_stance)
    stb_new = stability_scorer(posture, new_stance)
    return stb_old, stb_new, (stb_old > tau) & (stb_new > tau)
