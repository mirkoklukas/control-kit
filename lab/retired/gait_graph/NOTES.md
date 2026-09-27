# gait_graph — working notes

THESE ARE WORKING NOTES (messy on purpose; prune later). Origin prototype:
`notebooks/04_gait_graph.ipynb`.

## Idea

Plan locomotion as a graph over **stances**.

- **Node** = a stance: which feet are planted (`foot_ids`) and where (`foot_positions`).
- **Edge** = a transition, carried by a **posture** (body pose + joint angles) under
  which the swap from one stance to the next is valid.
- Search the graph for a walk that carries the body toward a **task** → a gait.

Dual graph (sometimes easier): nodes = postures, connected if they share a common
stable stance.

## Framing (settled)

- **state** = `(stance, body)` — passed flat, no `State` type (body persists into the
  next state, so it's a state var you *set* via the action, like position in
  kinematic control).
- **action** = `(next_stance, next_body)` — where to step + the witness body pose.
- **transition reward** = `task_scorer × min(stb_old, stb_new)` — task progress gated
  by stability on *both* the old and new support.

## Modules

- `kinematics.py` — single-leg IK + FK, constants (`SHOULDERS`, `LENGTHS`,
  `JOINT_RANGES`, `PLANTED`), flat-ground `tipover_score`. Copied from `lab.ik.core`
  so this package is self-contained (no import back into `ik`).
- `stance.py` — `Stance`, `Posture` (data-only pytree dataclasses), `complement`,
  and the atoms `infer_posture` (reach + limits, prefer elbow-down),
  `stability_scorer`, `transition_stability`.
- `sample.py` — `body_sampler`, `foot_sampler`; goal-agnostic proposals that also
  return a proposal log-prob (for future importance weighting).
- `task.py` — `Task(step, sigma)`, `task_scorer` (velocity/step reward term).
- `branch.py` — `branch` (one supporting posture per surviving next_stance) and
  `_branch_raw` (all valid `(body, layout)` pairs), sharing a fixed `(B,F)` `_grid`.

## Observed behavior (single `branch` from the home tripod)

- Best transition advances the **body** only ~1 cm and the **new feet don't advance**
  (support centroid ~0). Not a bug — falls out of the simple defaults:
  1. **Stability caps forward body motion.** `min(stb_old, stb_new)` is bound by
     `stb_old` (old support is *behind*), so pushing the body toward the `+x` target
     moves the CoM toward the support edge and stability drops. A far-forward body
     scores high on task, ~0 on stability; the product picks a compromise. Real
     progress comes from *alternating* supports over a rollout, not one branch.
  2. **Nothing rewards advancing the feet.** (RESOLVED — see "Score the next stance
     by its canonical-body advance" below.) `task_scorer` now scores the stance's
     canonical-body advance, so forward stances score highest and the rollout picks
     them. Goal-conditioned `foot_sampler` (task hook plumbed, unused) is still a
     complementary win: it makes forward candidates *exist* in the sample.

## Open threads / TODO

- **Score the next stance by its canonical-body advance (DONE).** The old
  `score = max_b task(b)·min(stb_old,stb_new)` masked forward stances (a forward
  stance needs a forward body, but a forward body has low `stb_old` → `min` kills
  the pairing). Now three roles are **separated** in `branch`:
    - **score** (node value): per-stance `task_scorer` = canonical-body advance,
      `exp(-‖(c_new - c_old) - step‖²/σ²)` where `c = canonical_xy(stance)` = support
      centroid. Independent of the witness body. Returns `(1, S)`.
    - **gate** (edge feasibility): `reach ∧ min(stb_old,stb_new) > τ`.
    - **posture** (edge to execute): safest feasible witness (`argmax_b min(stb_old,
      stb_new)`).
  Verified: top-scored stances advance feet by ~`step` (best 0.098 vs target 0.10;
  top-20 mean 0.104, bottom-20 0.036). Settled: canonical = **centroid** (closed
  form; `argmax_b stb` is the faithful-but-noisier alt), old ref = **canonical(old)**.
  Refinements: weight score by new-canonical stability (prefer roomy stances);
  one-step proxy, true value = future progress from the rollout (Bellman backup).
- **Stability combine: `min` (SETTLED).** `stb` is a *margin* (normalised tip-over
  angle), so the swap's stability is its worst instant → `min` (safe+safe → safe;
  invariant to splitting into more phases). `product` reads each as an independent
  P(no-tip); over-penalises comfortable swaps and the phases aren't independent.
  Used as the feasibility gate.

- **Geometry feasibility check (NEW).** A function that takes a posture + stance (the
  full kinematic config: all leg segments, body) and checks it against the **world
  geometry** — no leg/body passing *through* the floor, the climbing box, or itself.
  Reach + stability say "kinematically reachable & statically stable"; this adds
  "physically realizable" (collision-free except intended foot contacts). Likely via
  MJX/MuJoCo (set `qpos`, inspect penetration/contacts) or analytic capsule-vs-geom.
  Should slot into `branch` as an extra gate alongside `transition_stability`.
- **Goal-conditioned samplers.** `foot_sampler` / `body_sampler` currently ignore the
  task; bias the proposals toward the goal (importance sampling — that's why we carry
  the log-prob). Main spot: `foot_sampler`'s `shift` / `means`.
- **Stride-based foot means (DONE — it walks).** `foot_sampler(key, last_stance,
  posture, current_stance, ...)`: swing legs = `complement(current_stance)`; each
  strides one `shift` ahead of its **last foothold** (`last_stance.foot_positions`),
  rotated into the world by the body's **yaw** (from `posture`), keeping foothold z.
  Assumes tripod alternation (`complement(current) == last_stance.foot_ids`), which
  the rollout maintains by updating `last_stance, cur_stance = cur_stance, next`.
  `branch(key, last_stance, cur_stance, posture, next_ids, task, ...)` threads it.
  `run_planar` T=20: +1.07 m, y-drift −0.06, centroid marches 0->1.02. (Earlier
  home-layout+body-xy variant got +1.62 m but was a less clean model.) Refinement:
  steer `shift` by `task`; general `next_ids` needs per-leg foothold history.
- **Feet-aware task.** Default `task_scorer` rewards body motion only, so the
  reduction keeps placements that maximise `stb_new` (feet near centre), not forward
  feet. Use the threaded `stances` arg to reward stride / footfall targets.
- **General `next_ids`.** `branch` requires all 6 feet reachable, which assumes
  `next_ids = complement(last)` (every free leg re-plants). For subsets of size 4/5,
  reach should cover only `old ∪ new`.
- **Non-planar stability.** `tipover_score` is flat-ground only; generalise via MJX
  (ties in with the geometry check above).
- **JAX masking / static `F`.** `branch` goes ragged (eager) after the grid. For
  batched beam search or a scanned rollout, make it jittable: static `F` output + a
  mask (argmax-over-`B` is already static). Point-4 upgrade.
- **Stance score aggregation.** Reduction uses `max` over witnesses; `logsumexp`
  (rewards placements many good bodies can reach) is the principled alternative.
- **Importance weighting.** Sampler log-probs are computed but unused; fold into the
  score once proposals become goal-biased.
