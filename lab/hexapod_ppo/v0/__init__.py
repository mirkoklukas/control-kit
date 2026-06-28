"""hexapod_ppo v0 -- train the radial hexapod to walk straight in +x with PPO.

A Brax-PPO-on-MJX experiment: ``env.py`` wraps the MJX dynamics + the
``controlkit.reward`` reward into a ``brax.envs.base.Env``; ``__main__.py``
runs ``brax.training.agents.ppo`` as a tracked runkit experiment. See __main__.py.
"""
