"""lab — the experiments layer.

Where the churny, system-specific work happens. Each experiment is a
subpackage with its own ``config.py`` / ``env.py`` / ``__main__.py`` and is
launched with ``python -m lab.<experiment>``.

Experiments depend on ``controlkit``, the shared library (mpc, reward,
viz); the dependency only ever points that one way. Code reused across
experiments moves into ``controlkit``.
"""
