# Control Kit

A MuJoCo playground for control algorithms — classical control, model-predictive
control, and reinforcement learning. The main target robotic platform is a hexapod. 

## System requirements

A few tools live outside the Python env — install once per machine:

- [uv](https://docs.astral.sh/uv/) — manages the Python env (see [Setup](#setup)).
- [glow](https://github.com/charmbracelet/glow) — terminal markdown renderer.

```bash
# macOS (Apple Silicon)
brew install glow

# Linux box (no sudo): drop the release binary into ~/.local/bin, where uv also lives
GLOW_VER=2.0.0
curl -fsSL "https://github.com/charmbracelet/glow/releases/download/v${GLOW_VER}/glow_${GLOW_VER}_Linux_x86_64.tar.gz" \
  | tar -xz -C ~/.local/bin --strip-components=1 --wildcards '*/glow'
```

Verify: `command -v glow`.

## Setup

Requires [uv](https://docs.astral.sh/uv/). The examples use MuJoCo's JAX backend
(MJX), so install the `mjx` extra. On a Linux + NVIDIA box also add the `gpu`
extra to put JAX on the GPU; on macOS leave it off (JAX runs on CPU, which is
fine for development, just slow).

```bash
# macOS / CPU
uv sync --extra mjx

# Linux + NVIDIA GPU
uv sync --extra gpu --extra ppo
uv sync --extra gpu --extra mjx
```

`uv sync` creates `.venv` and installs everything. `.venv` is per-machine
(gitignored), so the *same* repo gives you a CPU env on your laptop and a GPU env
on a cloud box from one `pyproject.toml` — there's no second venv to manage. The
`gpu` extra is gated to Linux, so passing `--extra gpu` on macOS is a harmless
no-op.

Check which backend JAX picked:

```bash
uv run python -c "import jax; print(jax.default_backend(), jax.devices())"
# macOS  -> cpu [CpuDevice(id=0)]
# GPU box -> gpu [CudaDevice(id=0)]
```

On a GPU box you can still force CPU without a separate env:
`JAX_PLATFORMS=cpu uv run python …`.

Quick sanity check (headless, no display needed):

```bash
uv run python examples/01_mpc_cartpole.py record   # MPPI swing-up (cold: pumps but doesn't fully catch yet)
```

## Note on the viewer

The interactive MuJoCo viewer needs a local display, so run with `--render` on your own machine, not in a remote/headless sandbox.

## Model

`models/cartpole.xml` — slider (cart) + hinge (pole). Hinge angle `0` is
upright, `π` is hanging down. Single force actuator on the cart, `ctrlrange
[-20, 20]`.

## Layout

```
src/controlkit/   shared library (mpc, reward, viz); the `ctk` CLI lives here
lab/              experiments built on controlkit (see lab/README.md)
models/           MuJoCo MJCF models
docs/             notes and design docs
bootstrap-kit/    scripts to bootstrap a remote GPU box
examples/         standalone, self-contained scripts
runs/             immutable run dirs: config + checkpoints + results (gitignored)
```

`src/controlkit` is the shared library; `lab` is where the churny,
experiment-specific work happens and depends on it (never the reverse). A helper
that outgrows one experiment moves into `controlkit`.

## Running experiments

Each experiment under `lab/` is a sub-package launched from the CLI:

```bash
uv run python -m lab.<exp>.run [config.yaml] key=value ...   # produce a run
uv run python -m lab.<exp>.viz <run-dir>                     # plot its results
```

Outputs land in a gitignored `runs/` dir. See [`lab/README.md`](lab/README.md)
for the experiment layout, plus pulling runs from the GPU box and replaying
trajectories (`ctk viz`).
