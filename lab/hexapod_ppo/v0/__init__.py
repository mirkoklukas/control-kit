"""hexapod_ppo v0 -- train the radial hexapod to walk straight in +x with PPO.

A Brax-PPO-on-MJX experiment: ``env.py`` wraps the MJX dynamics + this
experiment's ``reward.py`` (composed from ``controlkit`` term kernels) into a
``brax.envs.base.Env``; ``run.py`` runs ``brax.training.agents.ppo`` as a tracked
runkit experiment. See run.py.
"""
