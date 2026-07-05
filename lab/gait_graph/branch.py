"""branch -- expand one stance into its valid, stable, on-task successors.

We stand on ``cur_stance`` at ``posture`` (``last_stance`` is the previous stance,
the swing legs' footholds). Given the legs to plant next (``next_ids``), we sample
``B`` witness bodies and ``S`` candidate stances (foot layouts), then on the full
``(B, S)`` grid infer joint angles (reach + limits) and transition stability.
Three roles are kept separate:

  - **score** (node value): per-stance ``task_scorer`` -- the canonical-body
    advance of the stance move. Does NOT depend on the witness body.
  - **gate** (edge feasibility): reachable AND ``min(stb_old, stb_new) > tau``.
  - **posture** (edge to execute): among a stance's feasible witnesses, the
    *safest* one (max ``min(stb_old, stb_new)``).

``branch`` reduces the grid over witness bodies -- one supporting posture per
surviving next_stance -- which is what the graph / rollout wants. ``_branch_raw``
keeps every valid ``(body, stance)`` pair instead.

The heavy grid stays fixed-shape and vmapped; only the final assembly goes
ragged (variable ``K``), so ``branch`` is eager, not jitted. Making it jittable
(static ``S`` + a mask) is the point-4 upgrade for batched / scanned search.

Pure jax; run under ``uv run --extra mjx``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from lab.gait_graph.sample import body_sampler, foot_sampler
from lab.gait_graph.stance import Posture, Stance, infer_posture, transition_stability
from lab.gait_graph.task import Task, body_neutral_score, task_scorer

ALL_IDS = jnp.arange(6)


def _grid(key, last_stance, cur_stance, posture, next_ids, task, B, S, tau):
    """Build the fixed-shape ``(B, S)`` grid that both branch variants reduce.

    We stand on ``cur_stance`` at ``posture``; ``last_stance`` is the previous
    stance (the swing legs' footholds, for ``foot_sampler``). ``n = len(next_ids)``.

    Flow:
      1. sample ``B`` witness bodies around the current body, and ``S`` candidate
         foot layouts (swing legs striding forward from ``last_stance``);
      2. at every (witness body, layout) cell, infer joint angles (reach + limits
         over all 6 feet) and transition stability (stable on the old support
         ``cur_stance`` AND the new one);
      3. score each candidate next stance by its canonical-body advance -- a
         per-stance quantity, independent of the witness body.

    Returns ``(bodies, layouts, next_stances, ok, theta, stb, score)``:
      bodies       : SE3 (B,)       witness bodies (sampled around ``cur_body``).
      layouts      : (S, 6, 3)      full 6-foot layout per candidate (the feet a
                                    cell's posture realises).
      next_stances : Stance (S,)    candidate next stances (the ``next_ids`` feet).
      ok           : (B, S) bool    feasibility gate: all 6 legs reachable +
                                    in-limits AND stable on both supports.
      theta        : (B, S, 6, 3)   per-cell joint angles (all legs).
      stb          : (B, S)         min(stb_old, stb_new): swap safety, used to
                                    pick the posture (not to score the stance).
      score        : (S,)           per-stance task score (canonical-body advance).
    """
    cur_body = posture.body
    key_b, key_s = jax.random.split(key)
    bodies, _ = body_sampler(key_b, cur_body, N=B)          # (B,); logp unused (no IW yet)
    layouts, _ = foot_sampler(key_s, posture, next_ids, N=S)   # (S, 6, 3)
    next_stances = _stances_at(next_ids, layouts[:, next_ids])   # Stance (S,)

    def cell(body, layout):
        # Reach + limits over ALL 6 feet (old planted + new placed). Assumes
        # next_ids = complement(cur), i.e. every free leg re-plants; for general
        # subsets, reach should cover only old ∪ new.
        full_stance = Stance(ALL_IDS, layout)
        reach, theta = infer_posture(body, full_stance)     # (6,), (6, 3)
        new_stance = Stance(next_ids, layout[next_ids])
        stb_old, stb_new, stb_ok = transition_stability(
            Posture(body, theta, layout), cur_stance, new_stance, tau)
        ok = reach.all() & stb_ok
        return ok, theta, stb_old, stb_new

    # grid[b, s] = cell(bodies[b], layouts[s])
    ok, theta, stb_old, stb_new = jax.vmap(jax.vmap(cell, (None, 0)), (0, None))(bodies, layouts)
    # stb = jnp.minimum(stb_old, stb_new)

    # Per-stance score (canonical-body advance); (1, S) -> (S,). Independent of B.
    score = task_scorer(bodies, stb_old, stb_new, task=task)[0]   # (S,)

    return bodies, layouts, next_stances, ok, theta, (stb_old, stb_new), score


def _stances_at(next_ids, positions):
    """Batch of stances sharing ``next_ids`` at the given ``(K, n, 3)`` positions."""
    foot_ids = jnp.broadcast_to(next_ids, positions.shape[:1] + next_ids.shape)   # (K, n)
    return Stance(foot_ids, positions)


def branch(key, last_stance, cur_stance, posture, next_ids, task: Task, *, B=100, S=100, tau=0.05):
    """Successor stances of ``cur_stance``, one supporting posture each.

    We stand on ``cur_stance`` at ``posture``; ``last_stance`` is the previous
    stance (the swing legs' footholds). For each candidate next-stance ``s`` that
    has at least one feasible witness body, keep the *safest* one (max
    ``min(stb_old, stb_new)``) as the posture; the stance's ``score`` is its
    canonical-body advance. Variable-length output: ``K`` survivors (``K <= S``).

    Returns ``(next_stances, postures, scores)`` -- a batched ``Stance`` (K),
    batched ``Posture`` (K), and ``scores`` (K), the per-stance task score.
    """
    bodies, layouts, next_stances, ok, theta, stbs, score = _grid(
        key, last_stance, cur_stance, posture, next_ids, task, B, S, tau)

    stb_old, stb_new = stbs

    safe = jnp.where(ok, jnp.minimum(stb_old, stb_new), -jnp.inf)                    # (B, S) safest-witness objective
    # best_b = jnp.argmax(safe, axis=0)                     # (S,) safest feasible witness per stance

    body_scores = safe*body_neutral_score(bodies)[:,None]   # (B, S) weight by body height

    # (S,) use most stable as representative witness for each stance
    best_b = jnp.argmax(jnp.where(ok, body_scores, -jnp.inf), axis=0) 

    
    (s,) = jnp.where(ok.any(axis=0))                      # (K,) stances with a witness
    b = best_b[s]                                          # (K,)

    sel_stances = next_stances[s]                          # (K,)
    postures = Posture(bodies[b], theta[b, s], layouts[s])   # SE3 (K,), theta+feet (K, 6, 3)
    scores = score[s]                                      # (K,) canonical-body advance
    return sel_stances, postures, scores


def _branch_raw(key, last_stance, cur_stance, posture, next_ids, task: Task, *, B=100, S=100, tau=0.05):
    """Every valid ``(body, stance)`` transition, unreduced.

    Returns ``(transitions, next_stances, postures, scores)`` where
    ``transitions`` is ``(K, 2)`` of ``(body_id, stance_id)`` and the rest are
    the corresponding ``K`` next_stances / postures / scores. ``scores`` is the
    stance's canonical-body advance (shared across its witnesses). ``branch`` is
    the per-stance reduction of this.
    """
    bodies, next_stances, ok, theta, stb, score = _grid(
        key, last_stance, cur_stance, posture, next_ids, task, B, S, tau)

    b, s = jnp.where(ok)                                   # (K,), (K,)
    transitions = jnp.stack([b, s], axis=-1)               # (K, 2)
    sel_stances = next_stances[s]                          # (K,)
    postures = Posture(bodies[b], theta[b, s])             # (K,)
    scores = score[s]                                      # (K,) per-stance advance
    return transitions, sel_stances, postures, scores
