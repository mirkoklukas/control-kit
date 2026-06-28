"""Smoke test for hexapod_ppo v0 -- does the PPO pipeline wire up?

Runs the experiment with a tiny config (a handful of steps, no real learning)
into a temp run dir, then checks the expected artifacts were produced. Exits 0
if the machinery works end to end, 1 otherwise -- so it's CI-friendly.

    uv run --extra ppo python -m lab.hexapod_ppo.v0.smoke

MJX-bound, so it needs the `ppo` extra and a backend that can run MJX (the GPU
box; CPU works but is slow -- the Apple GPU can't, see docs/gotchas.md).
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

from .run import Cfg, run

# Tiny config: enough to exercise env + ppo.train + the sample-episode callback,
# not enough to learn anything. num_evals=2 guarantees the callback fires.
SMOKE = dict(
    num_envs=8,
    num_timesteps=4096,
    num_eval_envs=8,
    num_evals=2,
    batch_size=8,
    num_minibatches=2,
    unroll_length=10,
    episode_length=100,
)


def main() -> int:
    cfg = Cfg(**SMOKE)
    checks: list[tuple[str, bool]] = []

    def check(name, cond):
        checks.append((name, bool(cond)))

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "run"
        try:
            result = run(cfg, out=str(out))   # full experiment into a temp run dir
        except Exception as e:                 # noqa: BLE001 -- any failure = smoke fail
            print(f"  [FAIL] run() raised: {e!r}")
            print("SMOKE FAIL")
            return 1

        check("run() returned a dict", isinstance(result, dict))
        check("policy.pkl saved", (out / "policy.pkl").is_file())
        check("config.yaml frozen", (out / "config.yaml").is_file())

        episodes = sorted((out / "results").glob("sample_episode_*.npz"))
        check("sample episode recorded", episodes)
        if episodes:
            npz = np.load(episodes[0])
            check("episode has frames", npz["qpos"].shape[0] > 0 and npz["qvel"].shape[0] > 0)

    ok = all(passed for _, passed in checks)
    for name, passed in checks:
        print(f"  [{'ok' if passed else 'FAIL'}] {name}")
    print("SMOKE PASS" if ok else "SMOKE FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
