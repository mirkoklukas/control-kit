"""Sampler variants: propose -> score -> select.

Hexapod copy of :mod:`.samplers`, on the proposals of :mod:`.stance_3dof`.

Every sampler shares the proposals of :mod:`.stance` (the "prior": uniform draws
among valid plants) and differs only in how it picks one. All proposals are
scored (cheap, and it lets the playground report the score of the pick); a
selector turns ``(key, valid, scores)`` into an index. Add a variant by adding a
selector to :data:`SELECTORS`.

:meth:`Samplers.leg_toward` is the odd one out: it needs a target point, so it is
its own method rather than a registry entry.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from controlkit.kinematics import candidates

from .scoring import Scorer
from .stance_3dof import Kit, propose_leg, propose_stances


def select_prior(key, valid, scores):
    """Uniform among the valid proposals (i.e. a plain draw from the prior)."""
    return candidates.sample(key, jnp.zeros(valid.shape), valid)


def select_best(key, valid, scores):
    """The highest-scoring valid proposal."""
    return candidates.best(scores["score"], valid)


SELECTORS = {"prior": select_prior, "best": select_best}


class Samplers:
    """The full-stance and single-leg samplers, jitted once per selector.

    Args:
        kit: robot, terrain, config and foothold pool.
        scorer: scores the proposals.
    """

    def __init__(self, kit: Kit, scorer: Scorer):
        self.names = list(SELECTORS)
        args = (kit.robot, kit.scene, kit.cfg, kit.footholds)
        L = kit.robot.num_legs

        def pick(select, key, valid, postures, stances):
            n = valid.shape[0]
            scores = scorer.score_fn(postures, jnp.ones((n, L), bool))
            j = select(key, valid, scores)
            info = {k: v[j] for k, v in scores.items()} | dict(valid=valid)
            return valid[j], postures[j], stances[j], info

        def full(select, key, cand, body):
            k_prop, k_pick = jax.random.split(key)
            oks, postures, stances = propose_stances(k_prop, *args, cand, body)
            return pick(select, k_pick, oks, postures, stances)

        def leg(select, key, cand, posture, stance, i):
            k_prop, k_pick = jax.random.split(key)
            valid, postures, stances = propose_leg(k_prop, *args, cand, posture, stance, i)
            return pick(select, k_pick, valid, postures, stances)

        def leg_toward(key, cand, posture, stance, i, point, sigma):
            k_prop, k_pick = jax.random.split(key)
            valid, postures, stances = propose_leg(k_prop, *args, cand, posture, stance, i)
            d2 = jnp.sum((kit.footholds.position[cand.idx[i]] - point) ** 2, axis=-1)
            toward = lambda k, v, sc: candidates.sample(k, -d2 / sigma**2, v)
            return pick(toward, k_pick, valid, postures, stances)

        self._full = {n: jax.jit(lambda *a, s=s: full(s, *a)) for n, s in SELECTORS.items()}
        self._leg = {n: jax.jit(lambda *a, s=s: leg(s, *a)) for n, s in SELECTORS.items()}
        self._leg_toward = jax.jit(leg_toward)

    def full(self, name, key, cand, body):
        """Full stance at ``body``. Returns ``(ok, posture, stance, info)``;
        ``info`` has the pick's score components and the (T,) ``valid`` mask."""
        return self._full[name](key, cand, body)

    def leg(self, name, key, cand, posture, stance, i):
        """Re-plant leg ``i``. Returns ``(ok, posture, stance, info)``; ``info``
        has the pick's score components and the (K,) ``valid`` mask over leg
        ``i``'s candidates ``cand.idx[i]``."""
        return self._leg[name](key, cand, posture, stance, i)

    def leg_toward(self, key, cand, posture, stance, i, point, sigma):
        """Re-plant leg ``i``, drawn among the valid candidates with weight
        ``exp(-d^2 / sigma^2)``, ``d`` the candidate's distance to ``point``.
        Returns ``(ok, posture, stance, info)`` as :meth:`leg`."""
        return self._leg_toward(key, cand, posture, stance, i, jnp.asarray(point),
                                jnp.asarray(sigma))
