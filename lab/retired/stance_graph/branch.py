from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import jax
from jaxlie import SE3, SO3

from lab.retired.stance_graph.kinematics import (
    infer_posture,
    stability_scorer,
)
from lab.retired.stance_graph.sample import (
    body_sampler,
    foot_sampler,
)



STABILITY_THRESHHOLD = 0.05
XYZ_DELTA = jnp.array([0.2, 0.2, 0.1])
RPY_DELTA = jnp.deg2rad(jnp.array([10.0, 10.0, 20.0]))


def _neutral_scorer(bodies, stb):

    # We score each body individually, these are the "test" bodies.

    zs = bodies.translation()[...,2]
    ns = bodies.rotation().as_matrix()[...,:,2]

    n_scores = jnp.exp( - (ns[:,0]**2 + ns[:,1]**2/ 0.2**2))
    z_scores = jnp.exp(-0.5 * ((zs - 0.241185) / 0.05)**2)
    task_scores = z_scores * n_scores 


    return task_scores[:,None] * stb


def _task_scorer(bodies, stb):
    """Reward term: for a stance."""

    # We score each body individually, these are the "test" bodies.
    xs = bodies.translation()[...,0]
    zs = bodies.translation()[...,2]
    ns = bodies.rotation().as_matrix()[...,:,2]

    n_scores = jnp.exp( - (ns[:,0]**2 + ns[:,1]**2/ 0.2**2))
    z_scores = jnp.exp(-0.5 * ((zs - 0.241185) / 0.05)**2)
    task_scores = xs * z_scores * n_scores 

    stable = stb > STABILITY_THRESHHOLD
    return task_scores[:,None] * stable


def task_scorer(bodies, stb_old, stb_new):
    t0 = jnp.max(_task_scorer(bodies, stb_old), axis=0)
    t1 = jnp.max(_task_scorer(bodies, stb_new), axis=0)
    return jax.nn.sigmoid(t1 - t0)


def transition_sampler(key, feet, stance, new_stance,  N=100, std=0.1):
    """Plant lifted legs in the foot layout."""
    new_feet = foot_sampler(key, feet, N=N,std=std)
    new_feet = new_feet.at[:,stance,:].set(feet[None,stance,:])  
    return new_feet


@partial(jax.jit, static_argnames=("B", "S", "std", "K"))
def branch(key, b0, f0, stance, new_stance, B=1000, S=1000, std=0.05, K=5, xyz_delta=XYZ_DELTA, rpy_delta=RPY_DELTA):
    _, key = jax.random.split(key)
    bodies = body_sampler(key, b0, N=B, xyz_delta=xyz_delta, rpy_delta=rpy_delta)


    _, key = jax.random.split(key)
    fs1 = transition_sampler(key, f0, stance, new_stance,  N=S, std=std)
    valid0, thetas0_grid, fs0_grid = infer_posture(bodies, f0[None], stance)
    valid1, thetas1_grid, fs1_grid = infer_posture(bodies, fs1, new_stance)
    valid = valid0 & valid1

    stb0 = stability_scorer(bodies, thetas0_grid, fs0_grid, stance)
    stb1 = stability_scorer(bodies, thetas1_grid, fs1_grid, new_stance)
    stb = jnp.minimum(stb0, stb1)
    stable = (stb0 > STABILITY_THRESHHOLD) & (stb1 > STABILITY_THRESHHOLD)

    # Can we reach the next stance with a valid posture and stable stance?
    ok = valid & stable
    reachable = ok.any(axis=0)

    # Score each new stance about how well it achieves the task
    task_scores = task_scorer(bodies, stb0, stb1)
    task_scores.shape, task_scores.min(), task_scores.max()

    # Sort the new stances by their task score, and only keep the reachable ones
    # So the first new stance is the best one that can be reached by at least one body pose.
    order = jnp.argsort(task_scores*reachable)[::-1]

    # Let's pick a good representation for each new stance
    tscores1 = _task_scorer(bodies, stb1)
    # tscores1 = _neutral_scorer(bodies, stb1)
    body_reprs = jnp.argmax(tscores1[:,order]*ok[:,order], axis=0)

    aux = {
        "bodies": bodies,
        "fs1_grid": fs1_grid,
        "thetas1_grid": thetas1_grid,
        "stb0": stb0,
        "stb1": stb1,
        "task_scores": task_scores,
        "reachable": reachable,
        "order": order,
    }

    return ok[:,order], bodies[body_reprs], thetas1_grid[body_reprs, :][:,order], fs1_grid[body_reprs,:][:,order], aux