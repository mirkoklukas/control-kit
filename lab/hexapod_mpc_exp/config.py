"""Config for the hexapod forward-walking MPPI experiment.

All the tunable knobs in one dataclass: the MPPI hyperparameters and the task
targets. The system wiring that consumes these lives in env.py; the CLI in
__main__.py exposes a few of them as --flags. Sim dt = 0.004 s.
"""
from dataclasses import dataclass


@dataclass
class Cfg:
    # --- MPPI hyperparameters (we control every `decimation` sim steps) ---
    decimation: int = 8       # hold each planned target this many sim steps (~31 Hz control)
    horizon: int = 20         # T: planning steps -> 0.64 s lookahead at decimation=8 (a gait cycle)
    samples: int = 128        # N: candidate rollouts per tick
    lam: float = 0.2          # temperature: lower -> greedier (commit harder to the best candidate)
    noise_sigma: float = 0.1  # per-control-step increment integrated into the target (rad)
    steps: int = 150          # control decisions to record (150 * 8 * 0.004 = 4.8 s)
    log_every: int = 10       # print a progress line every this many control steps

    # --- task targets ---
    vx_target: float = 0.3    # desired forward speed (m/s)
    z_nominal: float = 0.20   # standing trunk height (m); the freejoint starts here
    settle: int = 25          # zero-control steps to settle onto the feet before recording
