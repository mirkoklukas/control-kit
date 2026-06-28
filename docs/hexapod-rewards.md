# Hexapod locomotion reward

Living inventory of the per-step reward terms for `controlkit.reward`.
Built **term by term**; the `Status` column tracks what is wired into
`compute_reward` today vs. still pending.

## Contract (don't break these)

- **Per step.** `compute_reward` is called once per env step, *after* `mjx.step`.
  The trajectory return is the plain sum of step rewards — no summation,
  discounting, or normalization happens here.
- **Pure JAX.** jit-able and vmap-able over a batch of envs. No Python branching
  on traced values, no Python loops over contacts, no in-place mutation.
- **State is threaded, not mutated.** `foot_state` (and `last_action`, owned by the
  env) go in as arguments and come back out; the env carries them across steps.
- **Ids are static.** `compute_reward` takes a `HexapodIds` (built once at env
  init, closed over at jit time) and builds the concrete index arrays it needs
  from it in the body. No name lookups happen inside `compute_reward` — those run
  in `HexapodIds.__init__`.
- **Heading-invariant.** Base linear/angular velocity are rotated from world into
  the base frame, so the reward does not depend on which way the robot faces.

## Conventions

- Tracking terms are positive (max 1.0, Gaussian kernel, width² = 0.25). Penalties
  are non-negative quantities; the *weight* carries the sign. `total = Σ wᵢ · termᵢ`.
- `terms` dict holds the **unweighted** value of each term (for logging); `total`
  applies the weights.

## Terms

| Term | Kind | Formula | Default w | Status |
|---|---|---|---|---|
| `lin_vel` | reward | `exp(-‖v_xy_cmd − v_xy‖² / 0.25)` | `1.0` | ✅ slice 1 |
| `ang_vel` | reward | `exp(-(yaw_cmd − yaw_rate)² / 0.25)` | `0.5` | ✅ slice 1 |
| `base_height` | reward | `exp(-(z − z_target)² / 0.0025)`, `z_target` = model home height (off if `None`) | `1.0` | ✅ |
| `lin_vel_z` | penalty | `v_z²` (base frame) | `-2.0` | ✅ slice 1 |
| `ang_vel_xy` | penalty | `roll_rate² + pitch_rate²` | `-0.05` | ✅ slice 1 |
| `orientation` | penalty | `‖projected_gravity_xy‖²` | `-0.2` | ✅ slice 1 |
| `action_rate` | penalty | `Σ(aₜ − aₜ₋₁)²` | `-0.01` | ✅ slice 1 |
| `torques` | penalty | `Σ τ²` over leg dofs (`qfrc_actuator`) | `-1e-4` | ✅ slice 2 |
| `dof_acc` | penalty | `Σ q̈²` over leg dofs (`qacc`) | `-2.5e-7` | ✅ slice 2 |
| `collision` | penalty | count of bad-geom contacts (normal force > thresh) | `-1.0` | ✅ slice 3 |
| `feet_air_time` | gait | `Σ_feet (air_at_landing − target) · first_contact` | `1.0` | ✅ slice 3 |

`feet_air_time` is paid **only on the touchdown step** (the `first_contact` mask),
rewarding longer swing phases up to `air_time_target` (default `0.4 s`). It is
**gated** by whether any motion is commanded (linear or yaw, deadband
`move_cmd_eps`), so the robot doesn't step in place while told to stand.

All 10 terms are wired into **`compute_reward_2`** (the full reward);
`compute_reward` is the earlier slices-1-2 version, kept for reference.

**Env-reset note:** seed `FootState.last_contact` from the current foot contacts at
reset (`foot_contacts(data, ids.feet())`), not all-`False`. A zero-init makes every
grounded foot register a spurious touchdown on step 0 (a one-step `feet_air_time`
hit of `n_feet · (dt − target)`).

## Velocity / orientation signals (slice 1)

All derived from the base body (`base`, body id 1, carries the freejoint):

- `R = data.xmat[base].reshape(3,3)` — world ← base rotation.
- `base_lin_vel = Rᵀ · data.cvel[base, 3:6]` — linear part of the spatial velocity,
  world → base. (MuJoCo `cvel` layout is `[angular(3); linear(3)]`.)
- `base_ang_vel = Rᵀ · data.cvel[base, 0:3]` — `[roll, pitch, yaw]` rates.
- `projected_gravity = Rᵀ · world_up` — `xy` is the tilt signal (0 when upright);
  IMU-compatible. `world_up` is passed in (default `+z` unit); `model` is no longer
  an argument to `compute_reward`.

## Model ids (this MJCF)

`HexapodIds(model)` is a per-model id descriptor with **labeled** access. The leg
labels are baked in from the `<replicate>` order
(`FL ML BL BR MR FR` = legs `0..5`); the concrete ids are resolved from the model
by name (`coxa{i}/femur{i}/tibia{i}/foot{i}`), so there are no raw id literals.

```python
ids = HexapodIds(model)
ids.leg_joints("FL")   # (coxa, femur, tibia) ids for the front-left leg
ids.leg_joints()       # all 18 hinge ids, in leg order
ids.foot("BR"); ids.feet()
ids.base; ids.bad_geoms
```

`compute_reward(..., ids=ids, ...)` takes this `HexapodIds` directly (closed over
at jit time) and builds the concrete index arrays it needs from it. The groups:

- **base_body**: the freejoint body (`base`), id `1`.
- **leg_joints**: the 18 hinge joints (`coxa/femur/tibia` × 6), ids `1..18`,
  dof addrs `6..23`, qpos addrs `7..24`.
- **foot_geoms**: named spheres `foot0..foot5` → `[6,10,14,18,22,26]` (the only
  intended ground contacts).
- **bad_geoms**: every geom except the floor and the feet → base mesh, head box,
  and the 18 leg capsules.

## Pending decisions / notes

- MJX contact handling (implemented in `_contact_normal_forces`): contacts live in
  `data._impl.contact` (the bare `data.contact` is deprecated in mujoco 3.9); forces
  come from `data._impl.efc_force`. There is no `mjx.contact_force` here —
  `mjx.support.contact_force` needs a *static* contact id, so we use a fixed-shape
  efc gather (sum of the 4 pyramid rows = normal force, pyramidal cone / condim 3)
  for a vmap-safe per-contact normal force.
- **`efc_address >= 0` is NOT an activity filter here.** This MJX build assigns an
  efc address to *every* buffered geom pair (saw 362/362 "active" in a 6-foot
  stance), so separated pairs would be counted. Activity = **normal force above a
  threshold** instead, for both `foot_contacts` and `bad_body_contacts`. Verified
  the gather against `support.contact_force` (exact match) and `collision` against
  a sunk-base control (0 in stance, 6 when penetrating).
- No stand-still term yet (add later only if the robot fidgets when commanded to
  stop). No termination logic here — that lives in `env.done`.



# Unsorted Notes

$$r_t = R(s_t, a_t, s_{t+1})$$

- TODO (future): compare smoothness/regularization as an **extra reward term**
  (shaping, optimized through the RL return) vs. an **auxiliary loss on the
  network** (e.g. `‖a_t − a_{t-1}‖²` added directly to the actor loss). Same
  expression, different mechanism — worth an ablation.

- Idea: smoothness could instead live in the **policy parametrization** as a prior,
  not in the reward — a delta/incremental action `a_t = a_{t-1} + μ_θ(s_t) + σ⊙ε`
  (Gaussian centered on a residual from `a_{t-1}`). Changes the policy class, not
  the objective; `a_{t-1}` enters the obs either way. Caveats: (1) centering smooths
  only the *mean* — white per-step noise still jitters the sample, so temporally
  correlated exploration (OU / gSDE) is needed to smooth the noise; (2) centering ≠
  small steps unless the delta is bounded (tanh/clip), but clipping the delta gives
  a *hard* action rate limit (sim-to-real win the soft penalty can't). Deterministic
  cousin: fixed exponential action filter `â_t = α·a_t + (1−α)·â_{t−1}` (legged_gym).
  Not exclusive with a small `action_rate` penalty.
