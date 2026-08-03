"""Abstract sampling-based MPC (MPPI), framework-agnostic.

Nothing here knows about MuJoCo/MJX or any particular system. The dynamics
enter as an `env_step_func(s, u) -> s'`; the observation, the exploration
proposal (`control_func`), and the cost are all supplied by the caller. See
examples/03b_mpc_cartpole.py for the MJX cartpole wiring.

  make_rollout_sampler : build a (key, s0, T) -> trajectory rollout
  make_mppi_planner    : build a (key, s0)    -> optimized plan of shape (T, nu)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def make_rollout_sampler(
        env_step_func,
        control_func,
        observation_func = lambda key, s: s
    ):

    def rollout_model(key, s0, T):
        """
        Roll out a trajectory or length T starting at `s0`
        using the given controller and observation function.
        """

        # Note: Scan takes a function with signature
        #   > `(carry, input) -> (carry', output)`
        #   and returns a tuple `(carry, stacked outputs)``

        def step(s, key):
            key0, key1 = jax.random.split(key, 2)
            y = observation_func(key0, s)
            u = control_func(key1, y)
            s = env_step_func(s, u)
            return s, (y, u)

        keys = jax.random.split(key, T)
        # scan stacks ys/us over the horizon -> (T, ...); a planner that vmaps
        # this over N candidates then holds (N, T, ...). If T/N ever get large,
        # the memory lever is to accumulate a running cost in the scan carry
        # instead of stacking ys (would need a cost_func passed in here). That's
        # orthogonal to where the planner places its vmap (see make_mppi_planner).
        sT, (ys, us) = jax.lax.scan(step, s0, keys, length=T)
        return sT, (ys, us)

    return rollout_model


def make_mppi_planner(
        rollout_sampler,
        cost_func,
        T,
        N,
        lam=1.0
    ):
    """MPPI planner: (key, s0) -> plan of shape (T, nu).

    Abstract in the task -- it knows nothing about the system. It needs only

      - `rollout_sampler`  a trajectory model from `make_rollout_sampler`,
                              whose control_func injects the exploration noise
                              and whose observation_func produces the `y` that
                              `cost_func` scores;
      - `cost_func`           y -> scalar, the per-observation running cost.

    Each call rolls out `N` noisy candidates from the current state `s0`, scores
    each trajectory by its summed cost, and returns the softmax (cost-weighted)
    average of the candidate control sequences -- the full optimized plan `U` of
    shape (T, nu). The receding-horizon controller is just `U[0]`.

    Note `s0` must be a full simulator state (MPPI branches it), so when the
    plan feeds an outer rollout the outer observation_func must hand the planner
    the full state (identity), not a slim observation.
    """

    def mppi_planner(key, s0):
        keys = jax.random.split(key, N)

        def score(key):
            _, (ys, us) = rollout_sampler(key, s0, T)
            c = jax.vmap(cost_func)(ys).sum()
            return c, us

        # We vmap the fused roll-and-score (`score`) over the N candidates.
        # Under jit this is equivalent to vmapping the rollout first and scoring
        # the batch after -- vmap composes and XLA fuses identically, same FLOPs
        # and same (N, T, ...) intermediates. Fused is chosen for ergonomics:
        # s0/T are closed over, so every candidate branches from the same state
        # with no in_axes to get wrong. Factor out a batched_rollout (the split
        # form) only when you need the trajectories themselves -- for plotting,
        # or argmin/CEM selection instead of this weighted average.
        cs, uss = jax.vmap(score)(keys)          # (N,), (N, T, nu)
        ws = jax.nn.softmax(-cs / lam)          # cheap rollouts dominate
        plan = jnp.einsum("n,ntu->tu", ws, uss)  # (T, nu) weighted-average plan
        return plan

    return mppi_planner
