# Climb env design

Living note. Goal: an env where the robot climbs (slopes, walls, floor -> wall transitions).
Related: `foot_design.md` (foot + MuJoCo model), `task-and-env-design.md` (loose ideas),
`policy-and-reward-terms.md` (foot model, attached, min support), `notes.md` (test ladder).

## Step 1: magnetic feet in the env (flat floor)

Where we are: the foot (Cardan ankle, N x N pad cells, one `adhere{i}_{k}` per cell) is
built in `mjmodel.py`, but `env.py` keeps adhesion ctrl at 0, has no magnet action, and
trains with `pad_cells=1`. Step 1 wires the magnets in, still on the flat floor.

Decisions (2026-10-01):

- **Action:** +4 magnet commands, one per foot, driving all cells of that foot together.
  Binary: magnet on iff action > 0 (EPM-like on/off).
- **Switching:** configurable delay from command to actual state, default 0 (instant).
  EPM switching time not measured yet.
- **`pad_cells = 3`:** with 1 cell an edge or corner holds the full force. Cost ~9x pad
  contacts; check throughput.
- **Observation:** + magnet state per foot (commanded; also actual once the delay is > 0).
- **planted -> attached:** magnet on, switch finished, all cells in contact, pad flat.
  As in `policy-and-reward-terms.md`. Slip, support and drag terms use it.
- **Costs:** maybe a switching cost (EPM only spends energy when it switches). No cost
  for lifting with the magnet on: 30 N adhesion vs 5 N servos already prevents it.
- **Test task:** the existing crawl on the flat floor, with magnets. Checks that the
  policy learns to switch off before lift-off and on after touchdown.

- **Gravity randomization:** config switch; new gravity direction sampled at each reset.
  The floor stays the world xy-plane; only `m.opt.gravity` changes. Sampling: tilt from the
  floor normal in [0, `gravity_tilt_max_deg`], azimuth uniform in [0, 2 pi).
  - Obs: the current "gravity" term is hard-coded as world -z (`env.py`,
    `R.T @ [0, 0, -1]`), i.e. it is already the (negated) surface normal in the body frame.
    Keep it, named as the surface normal (`R.T @ [0, 0, 1]`), and add the actual gravity
    direction `R.T @ normalize(m.opt.gravity)` (+3). Privileged normal in sim; on hardware
    it could be estimated from the foot positions.
  - Tilt distribution: uniform in angle first; curriculum on `gravity_tilt_max_deg` later
    if needed.
  - Floor-relative terms (pad height, pad flat, orientation vs. surface normal) and the
    velocity command (along the surface) stay as they are.
  - Reset: beyond tilt ~atan(mu) = 27 deg the stand pose slides with magnets off, so
    start with magnets on (at least for large tilts).

Helping the policy find the magnet timing:

- Phase-clock contact-pattern reward from `reward-design-simple.md` (crawl gait shaping),
  extended to the magnet: in desired stance reward magnet on + attached; in desired swing
  reward magnet off + no pad force. Needs `sin/cos(2 pi phi)` in the obs.
- Alternatives: start with magnets on; or a scripted magnet schedule from the clock, then
  hand the magnet over to the policy.

Hover and contact:

- The hovering pad (~1.8 mm, inside the 2 mm margin) should still report contact: the
  margin is what makes adhesion act at all, and contacts inside it are active
  (`foot_design.md`). To verify in `test_foot` when coding, since the comment in
  `env._contact_state` says margin contacts can be inactive (`efc_address < 0`).
- Caveat: with the magnet on, the pad normal force includes the adhesion pull (30 N / 9 =
  3.3 N per cell), so `force > contact_force_min` (1 N) no longer means "loaded". Attached
  should count cells in active contact, not threshold the force.

Open:

- Attached needs *all* cells touching: too strict under uneven load? Compare with the
  current `planted` (force + tilt).
- Termination below 2 attached feet: likely yes, add once gravity is tilted (on the floor
  it isn't needed).

## Implementation (2026-10-01): `lab/rl/climb/`

Self-contained copy of `lab/rl` (see its `README.md`). Built as decided above, plus:

- Obs v3 (122) = v1 + surface normal, gravity, magnet cmd / actual, attached, clock phase.
  The clock phase is always in the obs (fixed size), even at `w_phase = 0`.
- `reset(options={"tilt_deg": ..., "azimuth_deg": ...})` pins the gravity (checks, evals).
- Magnets scripted from the clock (`magnet_mode="clock"`, default since 2026-10-01): on in
  the clock's stance, off in its swing; the policy only learns the legs. Replaced a prior
  (`magnet_init`: magnet output bias +1, ~99.7% on), under which "off" was almost never
  explored. The action keeps 16 entries for a later hand-over (`magnet_mode="policy"`).
- Action-rate cost covers the 12 servo actions only; magnet toggles have `w_switch`.
- `contact_force_min` / `contact_touch_min` dropped: touching = any cell in contact.

Findings:

- Hover: confirmed. Pads resting in the margin give active contacts, 9/9 cells per foot.
  The `efc_address < 0` skip in `_contact_state` doesn't drop them.
- `check_climb`, stand + magnets on: holds at 0 ... 180 deg, all cells attached. Magnets
  off: slides from 30 deg (mu = 0.5), as expected.
- Reset noise (0.05 rad) lifts some pads out of the 2 mm margin; at 120-150 deg they never
  attach and the robot falls off. Fix: settle under normal gravity first, then tilt.
- Smoke run with tilt U[0, 90] and no `magnet_init`: episodes ~3 steps; the untrained
  policy flips magnets at random and trips the < 2 attached termination. With
  `magnet_init = 1`: 125-225 steps after 8k steps.
- Cost: 3 x 3 cells = 144 contacts. Step 1.4 ms (walk env with 1 cell: 0.4 ms), reset
  70-110 ms (the 0.2 s settle). Expect ~3-4x slower training than the walk env. Sphere
  cells (1 contact each instead of 4) would help.

## GPU port (to discuss, 2026-10-02)

MuJoCo Warp supports the adhesion actuator (`TrnType.BODY`); MJX-JAX does not. Pieces:

1. Deps: mujoco / mjx at the mujoco_warp version (>= 3.10), `jax[cuda12]` in `uv.lock`.
   Check adhesion + elliptic cone on Warp.
2. Model: `mjmodel.build` as is, `mjx.put_model(m, impl="warp")`, size `nconmax` /
   `njmax` for 144 pad contacts. Check: per-world gravity (a model field).
3. `env_warp.py`: Playground-style JAX env. Substeps in `lax.scan`, fixed-shape contact
   readout, state arrays (magnets, histories), JAX RNG, `rewards.py` in `jnp`, reset
   from pre-settled states, reward weights as inputs (schedule without recompiling).
4. `train_warp.py`: Brax PPO; runkit records, orbax checkpoints, obs normalization.
   Eval / replay could stay on the CPU env (needs identical obs).
5. Parity test CPU vs Warp env (obs, contacts, attached, rewards); float32 stability.

Open: plan here first, or feasibility check of 1-2 on the GPU box first.

## Later steps (rough)

- Tilted gravity / slope 0 -> 90 -> 180 deg, maybe starting with all magnets on.
- Walk up to and onto a slope (0 -> 90 deg); floor -> wall transitions.
- Random pushes on the body; slower actuators.
