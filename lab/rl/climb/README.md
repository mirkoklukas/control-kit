# climb

Climbing env for the spider: switchable magnetic feet, gravity tilted per episode.
Self-contained copy of `lab/rl` (2026-10-01); nothing here imports from `lab/rl`.
Design and decisions: `lab/rl/docs/climb-env-design.md`.

## Files

| File | What | vs. `lab/rl` |
|---|---|---|
| `env.py` | `ClimbEnv`: action, observation, reward, termination | from `env.py`; magnets, gravity, attached, phase clock, obs v3 |
| `config.py` | `MjModelCfg` (robot), `ClimbEnvCfg` (env) | `WalkEnvCfg` -> `ClimbEnvCfg`, new fields |
| `train.py` | PPO training + eval (runkit experiment `climb`) | from `test_policy.py` |
| `rewards.py` | reward terms, pure functions | + `magnet_switch`, `phase_match` |
| `mjmodel.py` | builds the MuJoCo model | + `gravity_dir` |
| `poses.py`, `gait.py`, `scheduled_config.py` | keyframes, gait diagrams, reward schedule | unchanged |
| `check_climb.py` | non-RL check: hold `stand` under tilt, print contacts | new |
| `configs/magnets.yaml` | step 1 run: crawl on the flat floor with magnets | from `configs/crawl.yaml` |
| `experiment.toml` | runkit: extras, runs root (`climb/runs`) | |

## Commands (from the repo root)

```bash
uv run --extra mjx python -m lab.rl.climb.check_climb
```

```bash
runkit run lab.rl.climb.train exp:configs/magnets.yaml --tag=magnets
```

```bash
runkit eval lab.rl.climb.train
```

## The env in short

- **Action (16):** 12 servo targets + 4 magnets (on iff > 0, all cells of a foot together,
  optional delay `magnet_delay_s`).
- **Magnets:** `magnet_mode="clock"` (default): scripted from the phase clock (on in stance,
  off in swing), the policy's magnet outputs are ignored. `"policy"`: the policy switches.
- **Observation v3 (122):** v1 with the surface normal and gravity (both in the body
  frame), magnet command / state, attached flags, clock phase.
- **Pad cells:** 3 x 3 spheres by default (`pad_cell_shape`; `"box"` for the old boxes).
- **Attached** (replaces *planted*): magnet on, all pad cells in contact, pad flat.
- **Gravity:** `gravity_random=true` draws tilt ~ U[0, `gravity_tilt_max_deg`] and a random
  azimuth per episode. The floor stays the world xy-plane.
- **Reset:** settle `settle_s` under normal gravity with the magnets on, then tilt.
- **New reward terms:** `w_phase` (feet follow the crawl clock), `w_switch` (per toggle).
  Both default to 0.
- **Termination:** + fewer than `min_attached` (3) feet attached; 0 turns it off.
