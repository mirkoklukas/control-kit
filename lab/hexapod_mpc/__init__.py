"""Hexapod forward-walking with sampling MPC (MPPI).

Ported from examples/02_mpc_hexapod.py. Split into:
  config.py   the tunable knobs (MPPI hyperparams + task targets) as a dataclass
  env.py      the system-specific wiring (MJX step, observation, proposal, cost)
  __main__.py the record/play CLI -> `python -m lab.hexapod_mpc_exp record|play`
"""
