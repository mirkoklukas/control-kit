"""Quick non-RL check of :class:`.env.ClimbEnv`: hold the `stand` pose under tilted
gravity with the magnets on (or off), and print what the env sees.

Answers: do the hovering pads (inside the contact margin) give *active* contacts, so
``cells_touching`` and ``attached`` work? Does the stand pose hold with the magnets on
up to 180 deg?

    uv run --extra mjx python -m lab.rl.climb.check_climb
    uv run --extra mjx python -m lab.rl.climb.check_climb magnets=0 seconds=1
"""
import sys
from dataclasses import dataclass, field

import numpy as np

from .config import ClimbEnvCfg, MjModelCfg, parse_overrides
from .env import ClimbEnv


@dataclass
class CheckCfg:
    mjmodel: MjModelCfg = field(default_factory=MjModelCfg)
    env: ClimbEnvCfg = field(default_factory=ClimbEnvCfg)


def main(argv):
    cfg, extra = parse_overrides(CheckCfg, argv, {"seconds": 2.0, "magnets": 1})
    on = bool(extra["magnets"])
    cfg.env.magnet_start_on = on
    cfg.env.magnet_mode = "policy"                       # magnets from `hold`, not the clock
    env = ClimbEnv(cfg.mjmodel, cfg.env, seed=0)
    hold = np.zeros(16)
    hold[12:] = 1.0 if on else -1.0                      # servos on `stand`, magnets held
    print(f"pad_cells {cfg.mjmodel.pad_cells}, magnets {'on' if on else 'off'}, "
          f"{extra['seconds']} s per tilt; values at the end\n")
    print(f"{'tilt':>4} | {'cells/foot':<10} | {'attached':<9} | {'pad force (N)':<22} "
          f"| {'drift (mm)':>10} | end")
    for tilt in (0, 30, 60, 90, 120, 150, 180):
        env.reset(options={"tilt_deg": tilt})
        x0 = env.data.qpos[:3].copy()
        end = "held"
        for _ in range(int(extra["seconds"] / env.dt)):
            _, _, term, _, info = env.step(hold)
            if term:
                end = f"terminated at {env.steps * env.dt:.2f} s"
                break
        if end == "held" and info["attached"].sum() < cfg.env.min_attached:
            end = "detached"                             # below min_attached: the env
                                                         # terminates (min_attached > 0)
        drift = 1e3 * np.linalg.norm(env.data.qpos[:3] - x0)
        cells, attached = info["cells_touching"], info["attached"].astype(int)
        print(f"{tilt:>4} | {str(cells):<10} | {str(attached):<9} "
              f"| {np.array2string(info['force'], precision=1):<22} | {drift:>10.1f} | {end}")


if __name__ == "__main__":
    main(sys.argv[1:])
