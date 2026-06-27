"""lab — the experiments layer.

Where the churny, system-specific work happens. Each experiment is a
subpackage with its own ``config.py`` / ``env.py`` / ``__main__.py`` and is
launched with ``python -m lab.<experiment>``.

Tiers (most to least churn):
  lab.<experiment>   one runnable experiment
  lab.core           shared experimental helpers (added when a 2nd exp needs them)
  controlkit         the stable, system-agnostic core (graduation target)
"""
