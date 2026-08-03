# control-kit — repo design & organization

A MuJoCo playground for control algorithms (classical control, MPC, RL), targeting
a hexapod / radial-legged robot. This document maps the repo's **content and
organization**.

Everything gitignored is out of scope here (`runs/`, `scratch/`, `.venv/`,
`MUJOCO_LOG.TXT`, checkpoints, `*_pool.npz`, …); see
[Out of scope](#out-of-scope).

---

## The core idea

Two layers, dependency pointing one way only:

```
lab/<experiment>   ──imports──▶   controlkit          runkit
 (churny, per-system)             (shared library)    (run harness, separate repo)
```

- **`controlkit`** (`src/`) — the reusable library: kinematics, MPC, rewards,
  forces, model generation, visualization, CLI. Stable, tested-by-use, no
  experiment-specific logic.
- **`lab/`** — the experiments built on `controlkit`. Code is allowed to be
  messy here; a helper that outgrows one experiment graduates *up* into
  `controlkit`. Experiments never import each other and the library never
  imports `lab`.
- **`runkit`** — the run harness (immutable run dirs, provenance). Lives in its
  own repo, consumed here as a dependency.

---

## Top-level layout

| Path              | Role |
|-------------------|------|
| `src/controlkit/` | The shared library + the `ctk` CLI entry point. |
| `lab/`            | Experiments built on `controlkit` (see [Experiments](#experiments-lab)). |
| `models/`         | MuJoCo MJCF models. |
| `notebooks/`      | Exploration notebooks; `notebooks/staged/` holds the ones being cleaned up. |
| `tools/`          | Browser-based helpers (`model_generator`, `model_viz`) — MJCF authoring/viewing over HTTP. |
| `bootstrap-kit/`  | Scripts + notes to bring up a remote GPU box (assume host-specific config; adapt to your own). |
| `README.md`       | Setup (uv + `mjx`/`gpu` extras), layout, how to run experiments. |
| `pyproject.toml`  | Package metadata, dependency extras, `ctk` script, ruff config. |
| `runs/`, `scratch/` | Gitignored — experiment outputs and prototyping. |

---

## The library — `src/controlkit/`

Packaged as `controlkit` (see `pyproject.toml` → `[tool.hatch.build.targets.wheel]`,
which ships both `src/controlkit` and `lab`). CLI entry point `ctk =
controlkit.cli:app`.

### Kinematics — `controlkit/kinematics/`
Layered bottom-up, each layer ignorant of the one above (pure JAX):

| Module          | Responsibility |
|-----------------|----------------|
| `chain.py`      | Forward kinematics for a serial hinge chain, any DOF. |
| `planar.py`     | The two-link planar subproblem every leg type bottoms out in. |
| `candidates.py` | Gate / score / select among the IK branches a solver returns. |
| `metrics.py`    | Posture distance and nearest-neighbour queries. |
| `collision.py`  | Fast approximate self-collision, for filtering postures. |
| `leg.py`        | `Leg` base: a hinge chain (offsets/axes/tool) + its FK. |
| `leg4dof.py`    | `Leg4DOF`: hip-yaw / hip-roll / hip-pitch / knee-pitch leg with analytic IK + sampling. |
| `robot.py`      | `Robot`: a body of identical legs; FK/IK, feet/shoulders, MJCF export, pose helpers (`radial_mounts`, `sample_pose`). |
| `types.py`      | Shared types: `Foothold`, `Support`, `Posture`. |

### Control / dynamics
| Module        | Responsibility |
|---------------|----------------|
| `mpc.py`      | Abstract sampling-based MPC (MPPI), framework-agnostic (`make_rollout_sampler`, `make_mppi_planner`). |
| `rewards.py`  | Reward-term kernels for MJX locomotion (pure JAX). |
| `forces.py`   | Static stance forces: joint torques + foot reaction forces for a planted posture. |
| `hexapod.py`  | Hexapod-specific helpers. |
| `se3.py`      | SE(3) utilities. |
| `utils.py`    | Small JAX/MJX helpers shared across experiments. |

### Models & I/O
| Module          | Responsibility |
|-----------------|----------------|
| `modelgen.py`   | Hexapod MJCF generator, exposed as `ctk model-gen`. |
| `joystick.py`   | Xbox Adaptive Joystick support (macOS, USB-C). |

### Visualization
| Module            | Responsibility |
|-------------------|----------------|
| `viz.py`          | Replay a saved `.npz` state trajectory in the MuJoCo passive viewer (`ctk viz` / `ctk play`). |
| `mujoco_viz.py`   | Minimal MuJoCo-viewer view of a `Robot` (`ctk posture`). |
| `rerun_viz/`      | Rerun-based robot viz from body pose + per-leg angles (`colors`, `draw`, `robot`, `session`, `shapes`). |
| `ui.py`           | Shared terminal UI helpers. |

### CLI — `cli.py`
Umbrella Typer app mounting subcommands: `viz` (replay/plot `.npz`), `model-gen`,
`posture`, `play`, and `pull` (rsync run dirs from a remote host).

---

## Experiments — `lab/`

Recommended (not enforced) shape per experiment: `config.py` (knobs) · `env.py`
(system wiring) · `run.py` (`python -m lab.<exp>.run [config] key=val …`) ·
`viz.py`. Outputs land in gitignored `runs/`.

### `lab/controller/` — interactive posture control
Sculpt a **target posture** with a PlayStation DualSense and have the simulated
robot drive itself onto it. Runs entirely local on the Mac (pad on USB + passive
viewer need a display), one real-time loop:

```
DualSense → edit target Posture → policy → ctrl → mj_step → passive viewer
```

| File         | Role |
|--------------|------|
| `design.md`  | The experiment design: loop, robot build, control scheme, milestones (M0–M2). |
| `remote.md`  | Design sketch for a GPU-offloaded planner (local input+render, remote MJX MPPI over UDP). |
| `config.py`  | Knobs: robot dims, rest pose, edit speeds, control rate, MPPI params. |
| `robot.py`   | Concrete `Robot` build + `rest_posture()` + actuated-MJCF helper. |
| `input.py`   | DualSense driver via **hidapi raw-report parsing** (not pygame/SDL — SDL's pump needs the main thread, which `mjpython` owns). |
| `target.py`  | Mode state machine; apply controller state to the target `Posture` (via IK). |
| `policy.py`  | M1 servo tracking; M2 MPPI regulator (same interface). |
| `run.py`     | The real-time loop (needs `mjpython`). |
| `scene.xml`  | The MuJoCo scene for the loop. |

Status: M1a/M1b done (fly the mocap target; welded-feet MPPI chases it); M2
(unweld + balance + mode switching) open.

### `lab/hexapod_ppo/v1/` — PPO locomotion
Train the radial hexapod to **walk straight in +x** with PPO (Brax on MJX). Real
runs on a GPU box (`--extra vm`); output to immutable runkit run dirs under `runs/`.

| File / dir     | Role |
|----------------|------|
| `config.py`, `config.yaml` | Training config (dataclass + preset). |
| `env.py`       | MJX environment wiring + reward. |
| `run.py`       | Entry point: `python -m lab.hexapod_ppo.v1.run exp:config.yaml --tag=…`. |
| `_compat.py`   | Version-shim glue. |
| `models/`      | `hexapod.xml`, `hexapod2.xml`. |
| `README.md`    | Run command, env prefixes (JAX cache/logging), GPU-only caveat. |

---

## Notebooks — `notebooks/staged/`

`staged/` is the holding area for notebooks being promoted toward clean/published
form (the rest of `notebooks/` is exploratory and gitignored).

- `10_stable_postures.ipynb` — sampling/selecting stable postures (JAX), built on
  `controlkit.kinematics`.

---

## Supporting directories

- **`models/`** — MJCF scenes. `hexapod.xml` is used by `lab/hexapod_ppo/v1`;
  `weld0.xml` illustrates the foot-weld equality referenced in `robot.py`.
- **`tools/`** — `model_generator/` (browser MJCF generator + server) and
  `model_viz/` (browser MJCF viewer + server).
- **`bootstrap-kit/`** — remote GPU box bringup: `bootstrap.sh`, `prep.sh`,
  `pull.sh`, `machine-facts.sh`, `goals.yaml`, plus notes. Host/git-host specifics
  are baked in; adapt them to your own infra.

---

## Packaging & environments

Managed with `uv`; one `pyproject.toml` yields a CPU env on the laptop and a GPU
env on a cloud box via extras:

| Extra   | Contents / purpose |
|---------|--------------------|
| `mjx`   | `mujoco-mjx`, `warp-lang`, `jaxlie`, `rerun-sdk` — the JAX/MJX stack. |
| `ppo`   | `mjx` + `brax` — PPO training. |
| `gpu`   | `jax[cuda12]` (Linux-only marker; no-op on macOS). |
| `vm`    | `gpu` + `ppo` — full GPU training stack for the cloud box. |
| `sb3`   | `stable-baselines3` (Torch) — kept separate to avoid CUDA-wheel clashes with jax. |
| `dev`   | `ipython`, `ruff`. |

---

## Out of scope

Not part of this tree: everything gitignored (`runs/`, `scratch/`, `.venv/`,
`MUJOCO_LOG.TXT`, `*_pool.npz`, checkpoints), the exploratory notebooks under
`notebooks/` outside `staged/`, and editor state. Earlier / exploratory
experiments and full history live on the private `dev` branch, not on `main`.
