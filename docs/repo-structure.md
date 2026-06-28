# Repo structure

Orientation map of `control-kit` for anyone (human or agent) landing in the repo
cold. What each directory is for, where to add things, and the conventions that
tie it together. Living doc; update it when the layout changes.

`control-kit` is a MuJoCo playground for control: classical control, sampling
MPC (MPPI), and RL, with a radial **hexapod** as the main robot platform.
Python is managed with [uv](https://docs.astral.sh/uv/); MuJoCo runs through the
JAX backend (MJX) for the GPU sims.

The core of the repo is **`src/controlkit/`** — the reusable control algorithms.
Everything else exists to support it: `examples/` exercise it, `models/` feed it,
`src/runkit/` runs experiments built on it, and `assets/` holds bundled non-code
material.

## Top-level map

```
control-kit/
├── pyproject.toml        project + deps; defines the `runkit` console script
├── uv.lock               pinned env (hashed into run provenance)
├── run.context.yaml      the "context" half of a run spec (where/how runs stage)
├── CLAUDE.md             project instructions for Claude (notes, learning, style)
├── README.md             setup + examples overview (note: example table is stale)
│
├── src/                  the two installable python packages
│   ├── controlkit/       ★ reusable, framework-agnostic control pieces (the core)
│   └── runkit/           the experiment runner (decorator + CLI + provenance)
│
├── examples/             runnable scripts that exercise controlkit
├── models/               MuJoCo MJCF model files (.xml)
├── assets/               bundled non-code material (not part of the python project)
│   └── bootstrap-kit/    agent-driven cloud-box setup kit
├── tools/model_viz/      standalone WebGL MJCF viewer (three.js, no build)
├── docs/                 working notes (this file lives here)
├── runs/                 run outputs written by runkit (gitignored)
│
├── .claude/              Claude Code config (launch.json, settings.local.json)
└── .obsidian/            this repo doubles as an Obsidian vault for the notes
```

`src/` holds exactly the two installable packages: **controlkit** (the core) and
**runkit** (see `[tool.hatch.build.targets.wheel]` in `pyproject.toml`).
Anything that is not python source lives outside `src/`: the agent-driven
box-setup kit is a bundled asset under `assets/bootstrap-kit/`, not a package.

## `src/controlkit/` — the shared library

Code reused across experiments. Not a purist "framework-agnostic core" — it's
just the one tier above `lab/`, holding whatever experiments share. The
dependency points one way: `lab/` imports from here, never the reverse.

- `mpc.py` — sampling MPC (MPPI), genuinely framework-agnostic; dynamics,
  observation, proposal, and cost all enter as caller-supplied functions. Two
  builders:
  - `make_rollout_sampler(env_step, control, observation)` → a
    `(key, s0, T) → trajectory` rollout (a `jax.lax.scan`).
  - `make_mppi_planner(rollout_sampler, cost, T, N, lam)` → a `(key, s0) → plan`
    of shape `(T, nu)`; the receding-horizon control is `plan[0]`.
- `reward.py` — hexapod-specific reward terms (see
  [`docs/hexapod-rewards.md`](hexapod-rewards.md)).
- `viz.py` — replay/plot saved `.npz` trajectories in the MuJoCo passive viewer
  (imports mujoco; local-only, `mjpython` on macOS).

Related notes: [`docs/mpc.md`](mpc.md).

## `src/runkit/` — reproducible experiment runs

A small framework for launching experiments into immutable, provenance-stamped
run directories. The console script `runkit` is declared in `pyproject.toml`
(`runkit = "runkit.cli:app"`).

Core idea (the **run spec**) has two strictly disjoint halves:

- **config** = the experiment params (`cfg`, a dataclass). Set with bare
  `key=value` tokens and/or a config yaml. This is *what* to run.
- **context** = where/how the attempt is staged (`RunContext`). Set with
  `--flags` and/or a context yaml (`run.context.yaml`). This is *how* it's run.

Files:

| file | role |
|------|------|
| `__init__.py` | public API: `experiment`, `RunContext`, `init_run`, `main`, … |
| `runs.py` | the `@experiment(name=...)` decorator; builds the run dir, captures provenance, the dirty-git gate, marks completed/failed |
| `main.py` | argv-parsing core shared by `python script.py …` and `runkit run …` |
| `cli.py` | the `runkit` Typer app; `runkit run <script.py>` imports the script and delegates to `main` |
| `config.py` | pure argv/dataclass helpers (`split_argv`, `parse_overrides`, `deep_merge`, `build_cfg`); no IO |
| `context.py` | locate/load `run.context.yaml` |
| `provenance.py` | software axis (git SHAs, packages, lockfile hash) + hardware axis (platform, GPU) |
| `design.md` | the design rationale + open questions for the run-spec model |

An experiment script (see `examples/04_runkit_experiment.py`) defines a `cfg`
dataclass and a `run(cfg, ctx)` function decorated with `@experiment(name=...)`,
then calls `main(run)` under `__main__`.

## `examples/` — runnable scripts

The entry points you actually launch. MJX has no viewer and only flies on
CUDA/TPU, so the MPC examples split into a headless **record** (run on the GPU
box, save an `.npz`) and a local **play** (replay in the viewer via `mjpython`).
See [`examples/README.md`](../examples/README.md).

| file | what it is |
|------|------------|
| `00_minimal.py` | smallest MJX rollout (random policy); scratch/sanity |
| `01_mpc_cartpole.py` | cartpole swing-up with MPPI; record/play split |
| `02_mpc_hexapod.py` | hexapod forward walking with MPPI (position-servo proposal) |
| `03_forces_hexapod.py` | live viewer: tripod stance, contact-force arrows, femur-torque plot (local only) |
| `04_runkit_experiment.py` | mock experiment showing the `runkit` `@experiment` flow |

Note: the example table in the root `README.md` lists older filenames
(`01_viewer.py`, `02_lqr_cartpole.py`, …) that no longer exist; trust this list
and the actual directory.

## `models/` — MuJoCo models

MJCF `.xml` files loaded by the examples, tools, and docs.

- `cartpole.xml` — slider cart + hinge pole (single force actuator).
- `hexapod.xml` — current radial hexapod: hexagonal trunk, 6 × 3-DOF legs
  (coxa → femur → tibia), 18 position-servo actuators. The canonical model.
- `hexapod0.xml` — earlier hexapod iteration kept for reference.

Notes: [`docs/hexapod-model.md`](hexapod-model.md),
[`docs/mujoco-concepts.md`](mujoco-concepts.md),
[`docs/mujoco-forces.md`](mujoco-forces.md).

## `tools/model_viz/` — MJCF viewer

A standalone, no-build WebGL viewer (three.js from CDN) for posing a model's
kinematics in the browser. `serve.py` serves it over http (needed so the browser
can `fetch` the model); `mjcf_viewer.html` is the page. Kinematic only, no
physics. Also wired as a Claude Code launch config in `.claude/launch.json`.

## `assets/` — bundled non-code material

Material the repo ships but that is not part of the python project (nothing here
is imported or installed). Today this is just the box-setup kit.

### `assets/bootstrap-kit/` — fresh-box setup (agent-driven)

A self-contained kit for standing up a cloud GPU box. It is copied to a fresh
instance and run there; an agent reads `CLAUDE.md` + `goals.yaml` and builds a
replay-safe `bootstrap.sh`.

- `CLAUDE.md` — the stable contract for the setup agent (read-only input).
- `ENVIRONMENT.md` — per-machine context/rules the user edits freely.
- `goals.yaml` — the definition of done: tasks with a `pass` command as the oracle.
- `bootstrap.sh` — the only thing that touches the machine (idempotent).
- `prep.sh` — pre-step the agent doesn't own.
- `LOG.md` / `BLOCKERS.md` — dead-ends and hard walls recorded during a run.
- `README.md` — one-time Claude Code login + how to launch the bootstrap.

## `docs/` — working notes

Living, sometimes-messy notes (per `CLAUDE.md`: keep more pieces, prune later).
The repo is also an Obsidian vault (`.obsidian/`), so these render as linked
notes. Current set:

- `mpc.md` — MPC/MPPI theory backing `controlkit/mpc.py`.
- `mujoco-concepts.md`, `mujoco-forces.md` — MuJoCo model/state, reading forces.
- `hexapod-model.md` — measured facts about `hexapod.xml`.
- `mjx-minimal-example.md` — minimal MJX cartpole notes.
- `cloud-setup.md` — Lambda GPU box notes.
- `gotchas.md` — verified footguns (e.g. MJX can't run on Apple Metal GPU).
- `mkl-notes.md`, `todos.md` — scratch + task list.
- `repo-structure.md` — this file.

## `runs/` — run outputs (gitignored)

Where `runkit` writes each run. Layout produced by `runs.py`:

```
runs/<YYYY-MM-DD>/<HH-MM-SS>_<name>[_<tag>]/
├── config.yaml       the frozen cfg (reproducible params)
├── provenance.json   software axis: git SHAs, packages, lockfile hash, command
├── env.json          hardware axis: platform, host, GPU
├── status.json       running → completed / failed, with timestamps
├── checkpoints/      model/policy checkpoints
├── logs/
└── results/
```

`run.context.yaml` at the repo root is the matching **context** half: it points
`runkit` at this `runs/` root, the `uv.lock` to hash, and the repos to track for
provenance. Load it explicitly with `--context=run.context.yaml` (runkit does
not auto-discover it).

## Where to put new things

- A new **control algorithm**, reusable and system-agnostic → `src/controlkit/`.
- A new **robot/system** model → `models/` (+ a notes file in `docs/`).
- A new **runnable demo / experiment** → `examples/` (numbered prefix).
- Changes to **how runs launch/record provenance** → `src/runkit/`.
- A new **working note** → `docs/` (link it from related notes).
- Anything about **standing up a cloud box** → `assets/bootstrap-kit/`.
