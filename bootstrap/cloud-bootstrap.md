# Cloud bootstrap — run controlkit on an NVIDIA GPU box

This is a **runbook for Claude running *on* a fresh GPU instance**. This doc picks up once you're SSH'd in and starts from a clean OS.

The MJX examples (`examples/00_minimal.py`, `examples/01_mpc_cartpole.py`) are
pure JAX physics. They *run* on a Mac but only on CPU and slowly (see
[gotchas.md](../docs/gotchas.md)); the whole reason for the cloud box is to put that same
device-agnostic code on a real CUDA GPU.

**How to read this doc.** The steps below are *first-run* instructions: run them
interactively on a real instance, **verify each assumption against what's
actually installed**, and adapt where reality differs. As you go you distil the
commands that worked into `bootstrap.sh`. That script is the frozen recipe,
meant to re-run on a *similar* instance later (and to seed a future Docker image).

---

## Two scripts, two jobs

| script | runs on | job |
|---|---|---|
| [`local_bootstrap.sh`](local_bootstrap.sh) | your **laptop** | control plane: provision the box's git (credentials + identity), optionally install Claude, and copy `bootstrap.sh` over. Does **not** run it. |
| [`bootstrap.sh`](bootstrap.sh) | the **box** | data plane: assumes a provisioned box, then does all system + app setup (clone, deps, GPU, examples). No SSH, no secrets, no personal identity — ports straight into a Dockerfile. |

The flow is two phases:

1. **Laptop:** `./bootstrap/local_bootstrap.sh HOST` — sets up everything on the
   box that `bootstrap.sh` needs (git login + identity, Claude if asked) and drops
   `bootstrap.sh` at `~/bootstrap.sh`.
2. **Box:** `ssh HOST`, then `bash ~/bootstrap.sh`.

Keeping `bootstrap.sh` ignorant of who you are (no name/email, no token handling)
is what makes it re-runnable and Docker-ready; all the personal/credential wiring
stays in `local_bootstrap.sh`.

### Driving it: local-Claude over SSH vs Claude-on-the-box

You can let **Claude run the setup from your laptop over SSH** — its Bash tool
runs `ssh "$HOST" '…'` just fine, so it can verify the box, run the steps, read
the output, and fix problems, all remotely. That's the simplest path for the
bootstrap and needs **no Claude on the remote at all**.

The catch: Claude's **command** tool works over SSH, but its **file** tools
(Read/Edit/Grep) only see the *local* filesystem. So:

- **Standing the box up + running the examples** → local-Claude over SSH is ideal.
- **Editing/debugging code *on the box*** → install Claude *on the box* (pass
  `CLAUDE_AUTH=login` or `CLAUDE_AUTH=key` to `local_bootstrap.sh`), where its
  file tools see the box's filesystem directly. Not needed for setup; nice for
  later dev.

Claude on the box has two auth branches (pick one):
- **`CLAUDE_AUTH=login`** (use your subscription) — installs Claude, then *you*
  finish login: `ssh "$HOST"` and run `claude`, which prints a URL to open on your
  laptop and a code to paste back (works over SSH). Easiest if you have a Pro/Max
  plan.
- **`CLAUDE_AUTH=key`** — reads your key from `~/.secrets/anthropic.api` on the
  laptop, copies it to the same path on the box, and exports it. Non-interactive,
  billed pay-as-you-go.

Note: each `ssh "$HOST" 'cmd'` is a fresh shell (no carried-over cwd/env), so
chain commands (`cd repo && …`) or run scripts.

---

## Prerequisites (on your laptop)

1. **`gh` is authenticated** (`gh auth status` is green). The source of the
   GitHub credential — we never type a token by hand.
2. **Your local git identity is set** (`git config user.name` / `user.email`).
   `local_bootstrap.sh` mirrors it onto the box, so nothing is hardcoded.
3. **You can SSH into the instance** (key set up per
   [cloud-setup.md](../docs/cloud-setup.md)). Its address is `$HOST` — a hostname
   from `~/.ssh/config` or `user@ip`, anything `ssh "$HOST"` accepts.
4. *(only for `CLAUDE_AUTH=key`)* your Anthropic key saved at
   **`~/.secrets/anthropic.api`** on the laptop. Get one at the Anthropic Console
   (`console.anthropic.com` → Settings → API keys); billed pay-as-you-go,
   separate from a Pro/Max subscription. Skip this if you'd rather use
   `CLAUDE_AUTH=login`.

### Phase 1 — provision from the laptop

```bash
./bootstrap/local_bootstrap.sh "$HOST"                         # provision + copy bootstrap.sh; no Claude
CLAUDE_AUTH=login ./bootstrap/local_bootstrap.sh "$HOST"       # + Claude, you log in on the box (subscription)
CLAUDE_AUTH=key   ./bootstrap/local_bootstrap.sh "$HOST"       # + Claude, key from ~/.secrets/anthropic.api
```

This configures the box's git **credentials** (a persistent `credential.helper
store` primed from `gh auth token`) and **identity** (mirrored from your laptop),
so `clone`/`pull`/`commit`/`push` work for the life of the instance — then copies
`bootstrap.sh` to `~/bootstrap.sh`. The token is streamed over SSH stdin; it never
appears in this repo, in either script, or in chat.

### Phase 2 — run on the box

```bash
ssh "$HOST"
bash ~/bootstrap.sh
```

> Security: `~/.git-credentials` and (for `CLAUDE_AUTH=key`)
> `~/.secrets/anthropic.api` hold secrets in plaintext on the box. The gh token
> carries your gh scopes (typically full repo read/write) and the API key bills
> your account, so treat the instance as trusted; revoke (gh / Console) if it's
> ever exposed. Fine for a personal dev box; revisit before sharing one.

---

## Your mission (read first)

You are setting up this repo from scratch on the instance and **getting the
examples to run on the GPU**. As you go, you are also writing the recipe down so
the next person can do it in one shot.

**Deliverables**
1. A working `.venv` where `jax.devices()` shows a `CudaDevice`, and both
   examples run to completion.
2. **`bootstrap/bootstrap.sh`**: every command that actually worked, in order,
   idempotent enough to re-run. A v0 already exists — *correct it to match
   reality* as you go.
3. A short **Recipe / changelog** section appended to the bottom of this doc:
   what the environment turned out to be, and every deviation from the steps
   below (driver versions, packages you had to add, errors and their fixes).

**Operating rules**
- Work **one step at a time** and verify each before moving on. Don't paste a
  whole script and hope.
- Put in `bootstrap.sh` **only commands you actually ran and saw succeed.** If a
  command failed and you fixed it, the *fixed* version goes in the script and the
  *failure* goes in the changelog.
- Prefer non-interactive, re-runnable commands (guard clones/installs with
  existence checks). Don't bake secrets (tokens, keys) into the script.
- If something here is wrong for this box, trust the box and update the doc.

---

## Target environment (assumptions to verify)

These are the **starting assumptions** for the first run. Step 0 verifies them
against the actual box; if they hold, `bootstrap.sh` may assume them too (it
targets an instance like this one). If they don't, adapt and record it.

| | Assumed | Why it matters |
|---|---|---|
| Instance | Lambda Cloud GPU (reference: A10, ~`$1.29/h`) | — |
| Image | **Lambda Stack, Ubuntu 22.04** | ships NVIDIA driver + CUDA, Docker, etc. |
| **NVIDIA driver** | **present** (`nvidia-smi` works) | **required**; JAX needs the driver |
| CUDA toolkit | not required | JAX's `cuda12` wheels bundle CUDA + cuDNN |
| GPU | one NVIDIA card, CUDA 12 capable | the workload |
| Shell | bash, sudo available | apt + uv install |

**On the NVIDIA driver:** this is the one hard requirement and the one thing the
script does *not* install (a driver install typically needs a reboot, and inside
Docker the driver comes from the host). We assume the image already has it
(Lambda Stack does). Step 0 fails loudly if `nvidia-smi` is missing; on first run,
if you're on a bare image without a driver, install/repair it manually, note it
in the changelog, and keep the assumption "driver preinstalled" in `bootstrap.sh`.

---

## What the code needs (so you know what "done" means)

| Example | Stack | Needs a GPU? | Notes |
|---|---|---|---|
| `examples/00_minimal.py` | MuJoCo + MJX (JAX) | yes for speed | batched rollout; prints control + `site_xpos` shapes |
| `examples/01_mpc_cartpole.py` | MuJoCo + MJX (JAX), `controlkit.mpc` | yes for speed | MPPI swing-up; prints timing + `SWUNG UP` |

- The base deps also pull `gymnasium[mujoco]`, `stable-baselines3`, and (via SB3)
  a **CPU** PyTorch. There is **no RL example in the tree right now** (the old
  `04_rl_cartpole.py` is deleted), so PyTorch/GPU-RL is **out of scope** for this
  bootstrap. Don't chase a CUDA PyTorch unless an RL example comes back.
- The interactive MuJoCo **viewer (`--render`) is local-only** — it needs a
  display and on macOS `mjpython`. **Do not run `--render` on the headless box.**
  Run the examples in their default headless (print-a-result) mode.
- Offscreen *rendering* (frames/video without a window) is possible but **not
  needed by these examples** — see the optional EGL step.

---

## Steps

Run everything from the **repo root** (example `00` uses the relative path
`models/cartpole.xml`).

### 0. Sanity-check the box

```bash
nvidia-smi                        # GPU present? note Driver Version + CUDA Version
lsb_release -a; uname -m          # distro + arch (expect Ubuntu 22.04, x86_64)
nproc; free -h; df -h /           # cores, RAM, disk
```

- **No `nvidia-smi` / no GPU listed** → the driver isn't up (or this isn't a GPU
  box). On Lambda Stack it should just work; otherwise stop and fix the driver
  before anything else.
- Note the **Driver Version**. JAX's `cuda12` wheels bundle CUDA + cuDNN, so you
  do *not* need a system CUDA toolkit, but the driver must be new enough for
  CUDA 12 (≈ `≥ 525`, comfortably satisfied by Lambda Stack).

### 1. System packages

Almost everything ships with Lambda Stack. Make sure git + curl exist:

```bash
sudo apt-get update -y
sudo apt-get install -y git curl
```

(Optional, only if you later want **offscreen** MuJoCo rendering — see step 8:
`sudo apt-get install -y libegl1 libgles2`.)

### 2. Install uv

The repo is uv-managed. If `uv` isn't already on the box:

```bash
command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh
# make it available in this shell:
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

### 3. Clone

Git is **already provisioned** by `local_bootstrap.sh` (Phase 1): a persistent
`credential.helper store` primed from your gh token, plus your identity mirrored
from the laptop. So `clone` / `pull` / `commit` / `push` just work — for this repo
and any other on your account — and `bootstrap.sh` itself needs no token, name, or
email. It only needs the repo URL.

```bash
test -f ~/.git-credentials || { echo "git not provisioned — run local_bootstrap.sh from your laptop first"; exit 1; }
[ -d control-kit ] || git clone https://github.com/mirkoklukas/control-kit.git control-kit
cd control-kit
```

> Verify push works once if you like: `git commit --allow-empty -m test &&
> git push && git reset --hard HEAD~1`.

### 4. Python deps (CPU baseline first)

`uv sync` creates `.venv` and installs the locked deps. uv will fetch a suitable
Python automatically (`requires-python >=3.10`). The `mjx` extra adds
`mujoco-mjx` + JAX:

```bash
uv sync --extra mjx
uv run python -c "import mujoco, jax; from mujoco import mjx; \
    print('mujoco', mujoco.__version__, '| jax', jax.__version__, '| backend', jax.default_backend())"
```

Expected: `mujoco 3.9.0 | jax 0.10.2 | backend cpu` (jax version may differ if
uv picks Python 3.10, which resolves to jax 0.6.2 — that's fine, the next step
keys off whatever version is installed).

> The harmless lines `Failed to import warp: No module named 'warp'` on import
> are expected — MJX probes for the optional NVIDIA Warp backend and falls back
> to pure JAX. Silence them later via the optional Warp step if you like.

### 5. Make JAX use the GPU

The base `mjx` extra installs a **CPU** JAX. The repo carries a `gpu` extra that
adds the CUDA PJRT plugin + bundled NVIDIA libs on top (same jax version, Linux
only). So the GPU install is just the `--extra gpu` you already ran in step 4:

```bash
uv sync --extra mjx --extra gpu
```

If step 4 only synced `mjx`, re-run the line above now.

> The `gpu` extra is `jax[cuda12]` gated by `sys_platform == 'linux'`, with no
> version pin — it inherits whatever jax `mjx` resolved, so there's never a
> jax/jaxlib skew. On macOS the marker makes it a no-op.
>
> **Fallback** (only if the extra ever fails to resolve on this box): install the
> matching cuda plugin imperatively, then record why in the changelog —
> ```bash
> JAX_VER=$(uv run python -c 'import jax; print(jax.__version__)')
> uv pip install "jax[cuda12]==${JAX_VER}"
> ```
> Note a later `uv sync` will undo the `uv pip install` form; the extra is the
> durable path.

### 6. Verify the GPU is actually live

```bash
uv run python -c "import jax; print(jax.default_backend()); print(jax.devices())"
```

Expected:
```
gpu
[CudaDevice(id=0)]
```

If you see `cpu` / `[CpuDevice(id=0)]` or an error, debug before running examples:
- `Unable to initialize backend 'cuda'` → driver too old or CUDA libs not found.
  Recheck `nvidia-smi` driver version; confirm the `jax-cuda12-plugin` and
  `nvidia-*` wheels installed in step 5 (`uv pip list | grep -E 'jax|nvidia'`).
- Still CPU after a clean install → make sure you're invoking through the venv
  (`uv run …`), not a system Python.

### 7. Run the examples

```bash
uv run python examples/00_minimal.py
uv run python examples/01_mpc_cartpole.py
```

- `00_minimal.py` should print a control array and `site_xpos` shapes without
  error.
- `01_mpc_cartpole.py` should print something like
  `250 steps in <t>s (H=30, N=100, lam=1.0)` and then `SWUNG UP`. On CPU this is
  ~100 s+; on the GPU it should be markedly faster after the first compile.
- **First call is slow** (JAX JIT compile); that's included in the timing. Run
  twice if you want a warm number.
- Sanity-check the GPU is doing the work: in a second shell run
  `watch -n0.5 nvidia-smi` during `01` and confirm utilization/memory move.

### 8. Optional extras (only if needed)

- **Silence the Warp import lines / try the Warp backend:**
  `uv pip install warp-lang` (sizable; purely cosmetic unless you want MJX's Warp
  path).
- **JAX GPU memory:** JAX preallocates ~75% of VRAM by default. For the tiny
  cartpole it doesn't matter; if you run several processes, set
  `export XLA_PYTHON_CLIENT_PREALLOCATE=false`.
- **Offscreen rendering (no window):** `sudo apt-get install -y libegl1 libgles2`
  then `export MUJOCO_GL=egl`. Only needed if you write a script that renders
  frames; the current examples don't.
- **Persistent storage / GH for next time:** if the instance has an attached
  volume, clone the repo and point the uv cache there (`export
  UV_CACHE_DIR=/<volume>/uv-cache`) so a re-launched box warm-starts. (Tracked in
  [todos.md](../docs/todos.md).)

---

## Acceptance criteria

You're done when, from a fresh shell:

```bash
cd control-kit
export PATH="$HOME/.local/bin:$PATH"
uv run python -c "import jax; assert jax.default_backend()=='gpu', jax.devices(); print('GPU OK', jax.devices())"
uv run python examples/00_minimal.py
uv run python examples/01_mpc_cartpole.py     # prints SWUNG UP
```

all succeed, **and** on a *brand-new* identical box the only steps are the two
phases — `./bootstrap/local_bootstrap.sh "$HOST"` from the laptop, then
`ssh "$HOST"` + `bash ~/bootstrap.sh` — with no other fixups.

---

## Known gotchas (watch for these)

- **CPU jaxlib in the lock.** Without step 5 everything imports fine but runs on
  CPU. Always verify `jax.default_backend() == 'gpu'` (step 6).
- **jax/jaxlib version skew.** Pin the cuda extra to the *installed* jax version
  (step 5), not "latest".
- **`--render` does nothing useful headless.** Viewer needs a display; it's a
  local-machine thing. See [gotchas.md](../docs/gotchas.md).
- **`Failed to import warp` on every run.** Harmless; from a hard-coded `print`
  in MJX. Install `warp-lang` to remove, or ignore.
- **RK4 integrator.** `models/cartpole.xml` uses `integrator="RK4"`. The old
  warning was that MJX rejects RK4, but on `mujoco-mjx 3.9.0` it loads and steps
  fine (verified). `00_minimal.py` still overrides to `implicitfast`; `01` runs
  on RK4 as-is. If a future MJX bump reintroduces the error, override
  `model.opt.integrator` after load (as `00_minimal.py` does).
- **First-run latency.** JIT compile dominates the first call; don't read it as
  "GPU is slow."

---

## Out of scope (don't get pulled in)

- **PyTorch GPU / RL.** No RL example exists in the tree now. SB3 pulls CPU
  torch; leave it.
- **ROS2, full viz pipeline.** Future wants in [todos.md](../docs/todos.md), not needed
  to run these examples.

---

## The scripts

The recipe lives in two real files next to this doc in `bootstrap/`, not inline
here (so they don't drift). Both carry a documented header (purpose, prereqs,
usage, assumptions) and per-step comments — read the top of each before running.

- **[`bootstrap.sh`](bootstrap.sh)** — the box-side setup, mapping 1:1 to
  steps 0–7 above. It's **v0**: the steps above are written so you *verify it
  against a real box on the first run and correct it in place*. Config
  (`REPO_URL`, workdir) is at the top; no secrets, no personal identity.
- **[`local_bootstrap.sh`](local_bootstrap.sh)** — the laptop-side wrapper:
  checks prereqs, provisions the box's git (credentials + identity), optionally
  installs Claude (`CLAUDE_AUTH=login` or `CLAUDE_AUTH=key`), and copies
  `bootstrap.sh` to `~/bootstrap.sh`. It does **not** run it.

Phase 1, from the repo root (by hand or via local-Claude over SSH):

```bash
./bootstrap/local_bootstrap.sh "$HOST"
```

Phase 2, on the box:

```bash
ssh "$HOST"
bash ~/bootstrap.sh
# re-runs later, from the clone: bash ~/control-kit/bootstrap/bootstrap.sh
```

> `bootstrap.sh` clones `REPO_URL`, which defaults to
> `https://github.com/mirkoklukas/control-kit.git` (private). Override `REPO_URL`
> for a fork or a different repo.

---

## Recipe / changelog

> Fill this in as you go: the actual box specs, every deviation, every error and
> its fix. This is the payoff of running the bootstrap by hand once.

- _(box)_ instance type / GPU / driver version / CUDA version from `nvidia-smi`:
- _(python)_ Python and jax versions uv resolved:
- _(deviations)_ extra packages, changed commands, anything the steps above got wrong:
- _(errors → fixes)_:
- _(timings)_ `01_mpc_cartpole.py` first run vs. warm:
