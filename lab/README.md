# lab

The **experiments layer** of control-kit — where the churny, system-specific
work happens.

`controlkit` is the shared **library** (mpc, reward, viz); `runkit` is
the **run harness**; `lab` is the actual experiments built on them. Code is
expected to be messy here, and to move *up* into the library over time:

```
lab/<experiment>   →   controlkit
 (one experiment)      (shared across experiments)
```

A helper that outgrows a single experiment moves into `controlkit`. The
dependency only ever points one way: experiments import from `controlkit`,
never the reverse.

## Shape of an experiment (Recommended, not enforced)

Each experiment is a subpackage, launched from the CLI:

```
lab/<experiment>/
  config.py     # the knobs (a dataclass) — the "what"
  env.py        # system-specific wiring
  run.py        # the experiment body + launcher  →  python -m lab.<experiment>.run
```

Nesting is fine for grouping or versioning (e.g. `lab/mpc/hexapod/v0`): the leaf
is the runnable experiment, the dirs above it are namespaces that can also hold
code shared within their subtree. New *directory* for new *code*; new *params*
are just CLI overrides, not a new dir.

## Pulling runs from the GPU box (`rsync`)

Real runs happen on a remote CUDA box (MJX needs a GPU); the immutable run dirs
land in `~/control-kit/runs/` there. Pull them down to inspect/replay locally.
From the repo root (`lambda` is the SSH host alias):

```bash
# all runs (immutable + uniquely named, so this only fetches what's new)
rsync -avz --progress lambda:control-kit/runs/ runs/

# just one run
rsync -avz --progress lambda:control-kit/runs/<run-dir>/ runs/<run-dir>/
```

Run dirs are never mutated after creation, so adding `--ignore-existing` lets a
repeat pull skip everything already local without even checksumming it. Output
stays in the gitignored `runs/`, ready for the replay step below.

## Replaying trajectories (`controlkit.viz`)

Experiments that record a state trajectory (the MPC examples, the PPO sample
episodes) save it as an `.npz` carrying `qpos`/`qvel`/`timestep`/`model`. Replay
or plot any of them via the `ctk viz` CLI:

```bash
# replay in the MuJoCo passive viewer (local only; needs mjpython on macOS)
uv run mjpython -m controlkit.viz play <run>/results/sample_episode_<step>.npz

# plot base states (headless-safe; --out to save a PNG)
uv run ctk viz plot <run>/results/sample_episode_*.npz
```

The `play` command opens a window, so on macOS it needs `mjpython` and is
invoked as `-m controlkit.viz` (the `ctk` script runs under plain Python).
`plot` is headless-safe and runs fine under `ctk viz plot`.

## Relationship to runkit

`lab` and `runkit` are two halves of the same activity: `lab` is *what* to run,
`runkit` is *how* a run is staged, tracked, and made reproducible (immutable run
dirs, captured provenance). An experiment plugs into the harness via runkit's
`@experiment` decorator; outputs land in a gitignored `runs/` (ignored at any
depth, including `lab/<experiment>/runs/`).

`runkit` has since moved into its own repo (`../run-kit`, consumed here as an
editable path dependency); see its `design.md` for the run-spec model and roadmap.
