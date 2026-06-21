# MPC / MPPI notes

Working notes for the sampling-based MPC we're building in
[`src/controlkit/mpc.py`](../src/controlkit/mpc.py), rewriting
[`examples/03_mpc_cartpole.py`](../examples/03_mpc_cartpole.py) on top of MJX.
Living doc, prune later.

## What MPC is (receding horizon)

Model Predictive Control, one tick:

1. **Plan** a sequence of `T` future controls that looks good from the current
   state `s0`.
2. **Execute only the first action** `u0`.
3. Step the real system, then **throw the rest away and re-plan** next tick.

Planning over a horizon but committing only the first action is the defining
trick: it keeps correcting as reality drifts from the prediction. Unlike LQR
(fixed linear gain about one equilibrium), MPC uses the full nonlinear dynamics
and re-decides every step, so it handles large excursions like cartpole
swing-up (pole starts hanging down, must pump energy in, then catch + balance).

## MPPI = Model Predictive Path Integral

The flavor we use. Derivative-free: instead of solving an optimization, sample
many control sequences, simulate each, and combine by cost.

- **Model Predictive** — the receding-horizon loop above.
- **Path Integral** — the pedigree: a path-integral formulation of stochastic
  optimal control where the optimal controls are an expectation over noisy
  trajectory "paths," weighted by `exp(-cost/λ)`. The cost-weighted average of
  sampled rollouts approximates that expectation.

Per-tick update, `N` candidates rolled out from `s0`:

$$
\begin{aligned}
C_n &= \textstyle\sum_t \text{cost}(y_t^{(n)}) \quad &&\text{(summed cost of rollout } n) \\
W &= \text{softmax}(-C/\lambda) \quad &&\text{(cheap rollouts dominate)} \\
U^\star &= \textstyle\sum_n W_n\, V_n \quad &&\text{(weighted avg of candidate controls)}
\end{aligned}
$$

`λ` is the temperature: lower = greedier (see selection knob below).

## Our abstract decomposition

We factor the system-agnostic machinery from the system-specific pieces.

**Engine (generic):**

- `make_rollout_sampler(mjx_model, control_func, observation_func)` returns a
  `rollout_model(key, s0, T)` that scans the dynamics: at each step it senses
  `y = observation_func(key, s)`, acts `u = control_func(key, y)`, steps, and
  stacks `(ys, us)` plus the final state. Pure JAX (`lax.scan` over `mjx.step`),
  so it jits and vmaps.
- `make_mppi_planner(rollout_sampler, cost_func, T, N, lam)` returns
  `mppi_planner(key, s0) -> plan` of shape `(T, nu)`: vmap the sampler over `N`
  keys, score each trajectory by summed `cost_func`, softmax-weight, average.

**Supplied by the caller (system-specific):** `observation_func`,
`control_func` (the exploration proposal), `cost_func`. MPPI never learns it's
cartpole.

**Planner vs policy.** Keep the *planner* as the primitive — it returns the full
optimized plan `U★` (shape `(T, nu)`). The receding-horizon *policy* is just its
head: `policy = lambda key, s0: planner(key, s0)[0]`. Returning the full plan is
what later enables warm-starting.

**The nesting (same primitive, two levels).** A planner has the shape of a
`control_func`, so the *real-system* loop is itself a rollout sampler whose
`control_func` is the MPPI policy:

```
outer (real system):  rollout_sampler(dyn, control_func = mppi_policy)
                                  │ each real step calls ↓
inner (planning):      mppi_policy uses rollout_sampler(dyn, control_func = U+noise)
                                  vmapped over N candidates
```

Caveat: MPPI branches the simulator, so it needs the **full state** `s0`
(`mjx.Data`), not a slim observation. So at the outer level the
`observation_func` must be identity (hand the policy the full state). Under
partial observability you'd need a state estimator before the rollouts.

## Warm-starting = conditioning on a reference plan

Cold MPPI re-centers its proposal at **zero** every tick, rediscovering a
coordinated torque profile from scratch. But between ticks the optimal plan
barely changes (system advanced one `dt`), so the previous solution **shifted
one step** is a great guess. Reuse it as the proposal mean.

Framing: a **reference plan `U`** is a conditioning input flowing through all
three layers.

| layer         | cold                     | conditioned on reference                       |
| ------------- | ------------------------ | ---------------------------------------------- |
| control model | `control_func(key, y)`   | `control_func(key, y, u_ref)`, proposal at `u_ref` |
| sampler       | scans over `T` keys      | scans over plan `U`, feeds `U[t]` as `u_ref`   |
| planner       | `(key, s0)`              | `(key, s0, U) -> U★`                            |

Vanilla MPPI conditions by centering the Gaussian: `u ∼ 𝒩(u_ref, σ²)`. The cost
weighting is unchanged; only the proposal mean moves. **Cold is the special case
`U = 0`** — same algorithm, trivial reference.

Per-tick with reference: `V_n = U + ε_n`, roll out, weight, `U★ = Σ W_n V_n`.
Then **shift** to seed next tick:

```python
U_next = jnp.concatenate([U_star[1:], jnp.zeros((1, nu))])   # drop executed step, pad
```

Receding-horizon loop:

```python
U = jnp.zeros((T, nu))                                    # cold initial plan
for _ in range(steps):
    key, sub = jax.random.split(key)
    U = mppi_planner(sub, s0, U)                          # warm-started optimize
    s0 = mjx.step(model, s0.replace(ctrl=U[0]))           # advance real system
    U = jnp.concatenate([U[1:], jnp.zeros((1, nu))])      # shift to seed next tick
```

**Implementation cost (TODO, not done yet):** generalize the sampler so its
scan input is a per-step reference sequence (`U`) paired with the key, and give
`control_func` the extra `u_ref` arg. Then one primitive covers both modes
(closed-loop policy = no reference; MPPI = reference is the warm-start plan).
The reference need not be an additive mean — that's just the vanilla choice; you
could condition a learned/feedback proposal the same way.

## The selection knob: how scored rollouts become the next plan

Given `N` scored candidates, how you collapse them to one plan is a continuum:

- **Argmin (random shooting / "predictive sampling," MJPC's default):** take the
  single lowest-cost rollout's controls. `plan = V[jnp.argmin(C)]`.
- **Softmax average (MPPI):** weight all by `exp(-C/λ)` and average. What we have.
- **Elites (CEM):** keep the top-k, refit a Gaussian (mean *and* covariance),
  resample, often iterate within one tick. CMA-ES adapts the full covariance.

Argmin is literally the `λ→0` limit of MPPI:

$$
\begin{aligned}
\lambda \to 0      &:\quad W \to \text{one-hot at } \arg\min_n C_n \;\Rightarrow\; \text{argmin (greedy)} \\
\lambda \to \infty &:\quad W \to \tfrac1N \;\Rightarrow\; \text{plain mean, cost ignored}
\end{aligned}
$$

Trade-offs:

- **Argmin** uses one rollout's info → higher variance, jittery, one lucky/noisy
  sample drives the action. But robust to the averaging failure below.
- **Weighted average** uses all samples → smoother, lower variance, principled
  (path-integral estimate). Weakness: averaging only makes sense if good plans
  are near each other. With a **multimodal** cost (swing left *or* right, both
  good), the mean of two opposite plans can land in the bad valley between them.
- **CEM elites** are the middle ground: hard-select an elite set, then adapt the
  whole sampling distribution.

## Status

- [x] `rollout` primitive (open-loop, summed cost)
- [x] `make_rollout_sampler` (closed-loop, returns `(ys, us)` + final state)
- [x] `make_mppi_planner` (cold, softmax-weighted, returns full plan)
- [x] receding-horizon driver (`examples/03b_mpc_cartpole.py`)
- [ ] reference-conditioned sampler + warm-started planner
- [ ] optional: argmin / CEM selection variants

**Cold result (03b):** with H=30, N=100, λ=1.0 the cold planner pumps the pole
only to ~1.8 rad from upright -- it does *not* complete the swing-up. Two
reasons, both pointing at warm-start: (1) memoryless re-planning can't commit to
a multi-tick pumping strategy; (2) independent zero-mean per-step noise averages
out, so candidates rarely contain the *sustained* one-directional pushes that
build pendulum energy. The previous-plan nominal is exactly the sustained signal
that's missing. (Runtime note: ~120 s on CPU MJX; this is GPU/TPU-shaped work.)
