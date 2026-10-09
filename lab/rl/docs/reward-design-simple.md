# Crawl Gait RL: Reward Design & Curriculum

Sep 30, 2026 · @Mirko Klukas

## Overview

The target is a crawl gait: one leg in swing at a time, at least 3 feet on the ground, duty factor ≥ 0.75. PPO tends to find trot or bound first, so the reward needs explicit gait shaping on top of velocity tracking.

The reward has four groups: **task** (velocity tracking), **stability** (base motion and height), **gait** (contact pattern), and **regularization** (smoothness and energy). Every term is multiplied by the policy `dt`, as in legged\_gym, so weights don't depend on control frequency.

The gait group is what makes it a crawl: a phase clock in the observation, a contact-pattern reward that follows it, a reward for ≥3 feet in contact, and an air-time target of 0.2–0.3 s.

## Reward terms

Starting weights. Tune by logging each term separately.

| Group          | Term                          | Form                                    | Weight                             |
| -------------- | ----------------------------- | --------------------------------------- | ---------------------------------- |
| Task           | Linear velocity tracking (xy) | exp(−‖v\_cmd − v\_xy‖² / σ), σ = 0.25   | +1.0                               |
| Task           | Yaw rate tracking             | exp(−(ω\_cmd − ω\_z)² / σ)              | +0.5                               |
| Stability      | Vertical velocity             | v\_z²                                   | −2.0                               |
| Stability      | Roll/pitch rate               | ‖ω\_xy‖²                                | −0.05                              |
| Stability      | Orientation                   | ‖g\_proj,xy‖² (projected gravity)       | −1 to −5                           |
| Stability      | Base height                   | (h − h\_target)²                        | −10 to −30                         |
| Gait           | Contact pattern match         | phase-based, see crawl section          | +0.5 to +1.0                       |
| Gait           | ≥3 feet in contact            | 1\[n\_contact ≥ 3\]                     | +0.2 to +0.5                       |
| Gait           | Feet air time                 | Σ (t\_air − t\_target) · first\_contact | +0.5 to +1.0 (t\_target 0.2–0.3 s) |
| Gait           | Swing foot clearance          | (z\_foot − z\_target)² · swing\_mask    | −0.5                               |
| Gait           | Foot slip                     | ‖v\_foot,xy‖² · contact                 | −0.1                               |
| Regularization | Torques                       | ‖τ‖²                                    | −1e-5 to −2e-4                     |
| Regularization | Joint accelerations           | ‖q̈‖²                                   | −2.5e-7                            |
| Regularization | Action rate                   | ‖a\_t − a\_{t−1}‖²                      | −0.01                              |
| Regularization | Joint limits                  | soft-limit violation                    | −1 to −10                          |
| Regularization | Body/thigh collision          | count of non-foot contacts              | −1.0                               |
| Termination    | Base contact or flip          | end episode; optional penalty           | −10 to −100                        |

legged\_gym also sets `only_positive_rewards = True`, clipping the total reward at 0 so early penalties don't teach the policy to terminate. For a crawl, lower the air-time target to about 0.2–0.3 s.

## Crawl gait shaping

A periodic phase clock drives the contact pattern. Use the lateral-sequence order LH → LF → RH → RF, i.e. leg phase offsets 0, 0.25, 0.5, 0.75. Each leg swings for 25% of the cycle.

```python
import torch

OFFSETS = torch.tensor([0.25, 0.75, 0.0, 0.5])  # LF, RF, LH, RH (match your foot order)
DUTY = 0.75                                      # stance fraction

def re(phase, contacts, foot_forces, foot_vels):
    # phase: (N,) in [0,1); foot_forces/foot_vels: (N,4) magnitudes
    leg_phase = (phase.unsqueeze(1) + OFFSETS) % 1.0
    desired_stance = leg_phase < DUTY               # (N,4)

    # penalize force during swing, foot velocity during stance
    swing_force = (~desired_stance) * (1 - torch.exp(-foot_forces**2 / 100.0))
    stance_vel  = desired_stance   * (1 - torch.exp(-foot_vels**2 / 2.0))
    return -(swing_force + stance_vel).sum(dim=1)
```

- Add `sin(2πφ)` and `cos(2πφ)` to the observation, or the policy can't follow the clock.
- Cycle period: about 0.8–1.5 s, slower than a trot.
- Simpler alternative: `+ Σ (contact == desired_stance)`. Easier to tune, less smooth gait.

## legged\_gym curriculum

Only two things are scheduled: terrain difficulty (per env) and the command range (global, off by default). Reward weights, domain randomization and noise stay fixed.

### Terrain grid

- 10 rows (difficulty levels) × 20 columns (terrain types), 8 × 8 m tiles.
- Row i has difficulty d = i / 10, so d runs 0 → 0.9.
- Default type mix `[0.1, 0.1, 0.35, 0.25, 0.2]`: smooth slope, rough slope, stairs up, stairs down, discrete obstacles.
- Each env keeps its column (type) for the whole run. Start level is random in 0–5.

| Terrain type | Scales with difficulty d |
| --- | --- |
| Smooth slope | slope = 0.4·d; half the tiles downhill |
| Rough slope | same slope + ±5 cm uniform noise |
| Stairs | step height = 0.05 + 0.18·d m, 0.31 m step width |
| Discrete obstacles | height = 0.05 + 0.2·d m, 20 blocks of 1–2 m |
| Stepping stones, gaps, pits | implemented, unused unless added to the proportions |

### Promotion rule (per env, at reset)

1. **Level up** if distance from origin > terrain\_length / 2 (4 m).
2. **Level down** if distance < 0.5 · |cmd| · episode\_length\_s (20 s).
3. Robots that clear the top level go to a random level.

### Command curriculum (off by default)

Checked once per episode length. If the mean `tracking_lin_vel` reward exceeds 80% of its maximum, `lin_vel_x` widens by ±0.5 m/s, capped at `max_curriculum` (1.0). Lateral velocity and yaw are not scheduled.

### Fixed for the whole run

| Item | Setting |
| --- | --- |
| Reward weights | constant; total clipped at 0 |
| Friction | \[0.5, 1.25\], 64 buckets, sampled once at env creation |
| Pushes | up to 1 m/s every 15 s |
| Base mass randomization | available, off |
| Commands | resampled every 10 s; speeds < 0.2 m/s set to 0 |
| Observation noise | constant |
| PPO learning rate | adaptive on KL (target 0.01), not a task curriculum |

## Adaptations for the crawl robot

The biggest issue is the level-up rule: a crawler commanded below 0.2 m/s can't cover 4 m in 20 s, so it never gets promoted.

1. **Scale the level-up threshold by the command:** `distance > 0.8 · |cmd| · T`, or shorten `terrain_length`.
2. **Rescale terrain constants** to leg length. Step and obstacle heights up to \~0.23 m are sized for ANYmal.
3. **Add a regularization curriculum** (not in legged\_gym): ramp penalty weights in `compute_reward()`, e.g. `scale_k = base_k · min(1, iter / N)` for torque, action-rate and gait terms.
4. **Keep commands low:** `max_curriculum` ≈ 0.3 m/s. High commands push the policy toward trot.
5. **Start small:** velocity tracking, orientation and termination first; add gait terms once it moves. Too many early penalties teach standing still.
6. **Watch for exploits:** belly-dragging (base height + collision terms) and knee-walking (thigh contact penalty).
7. **Log every term separately;** rebalance the dominant one before adding new terms.

## Sources

- [legged\_gym repository](https://github.com/leggedrobotics/legged_gym)
- [legged\_robot.py](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot.py): curriculum, commands, reward functions
- [legged\_robot\_config.py](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot_config.py): default config values
- [terrain.py](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/utils/terrain.py): terrain grid and difficulty scaling
- Margolis & Agrawal, "Walk These Ways" (2022): phase-based contact schedule with gait parameters as inputs
