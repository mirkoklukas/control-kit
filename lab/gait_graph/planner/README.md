# planner

A stance planner on top of `lab/gait_graph`: search a path of stances (four feet planted)
towards a goal, one leg moving per step, each step checked to hold. Later: execute it
(an MPC between stances), and refine it as a trajectory optimization.

Design and decisions: `docs/planner-design.md`. Status (2026-10-06): the pieces for one
step exist (stance sampling for the climb robot, the hold check); `plan_step` and the search
are next. v0: flat ground, move along +x, fixed crawl order.

## Files

| file | what |
|---|---|
| `docs/planner-design.md` | the design: graph views (dual: nodes are four-foot stances), the four-phase step, notation, the choice order, the hold check (derivation, three ways to check it, vs. `controlkit/forces.py`), actuator torques, the reachability map |
| `docs/trajopt-nlp.md` | side topic: motion planning as a nonlinear program (TOWR-like formulation for the floor-to-wall transition) |
| `statics.py` | statics of a planted posture: `statics` (full model, one `mjx.forward`, optional actuated-dof selection), `point_mass_statics` (centre of mass + feet only); `min_norm` / `least_torque` foot forces and torques; the score LPs (adhesion, friction pyramid, torque bounds), solved with `qpax`: `hold_margin` (s*, the largest gravity factor that still holds), `disturbance_margin` (the largest push / twist on the body, worst of 12 directions), `foot_forces_lp` (the most central foot forces under the real gravity; solve with HiGHS, qpax is unreliable on it) |
| `test_statics.py` | checks of `statics.py` on the `gait_graph` robot: vs. `forces.stance_forces`, vs. HiGHS (scipy), vs. the triangle test, point mass vs. full model |
| `climb_model/` | the climb robot, copied from `lab/rl/climb`: `config.py` (`MjModelCfg`), `mjmodel.py` (the MuJoCo model, `make_robot`), `poses.py` (keyframes, IK helpers) |
| `climb_cfg.py` | `ClimbCfg`: `gait_graph`'s stance / sampling settings with the climb robot's dimensions (`climb_cfg(mj)`); the foot is the ankle pivot, 11 mm above the pad face |
| `stance.py` | stances, witness postures and moves (`Kit`: sample a stance, lift / move / re-plant a leg, checks) for the climb robot: a copy of `gait_graph/stance_3dof.py` with the square trunk |
| `climb_statics.py` | `ClimbModel`: the climb model set up for `statics.py` (no adhesion actuators for MJX), posture -> `qpos` with the pads laid flat, torques on the 12 servos only |
| `experiments/torques.py` | runkit experiment `planner_torques`: min-norm vs. least-torque torques of sampled stances (`gait_graph` robot) |
| `experiments/scores.py` | runkit experiment `planner_scores`: score functions on sampled and resampled postures of the climb robot, floor and wall, full stance and left front leg lifted; saves replays with the scores and foot forces |
| `view.py` | viewer for the scores replays: up / down = good / bad list, left / right = previous / next, `,` / `.` = 10 back / ahead; foot force arrows (green push, red pull) and friction cones with adhesion at the contact points, the wall as a wireframe |
| `experiments/experiment.toml` | runkit settings for `experiments/` (runs in `experiments/runs/`, prints its own output) |

Two robots, for now: `test_statics.py` and `experiments/torques.py` use `gait_graph`'s own
robot (4-DOF legs, sphere feet), from before the switch; everything new is for the climb
robot (`climb_*`, `stance.py`).

## Run (from `lab/gait_graph/planner`)

```bash
uv run --extra mjx python -m lab.gait_graph.planner.test_statics
```

```bash
uv run runkit run lab.gait_graph.planner.experiments.scores
```

```bash
uv run --extra mjx python -m lab.gait_graph.planner.view experiments/runs/planner_scores/<run>/out/wall_lifted_samples.npz
```

runkit needs the module name: `runkit run experiments/scores.py` fails on the
package-relative imports.
