# MJX minimal example

Scratch notes for a minimal MuJoCo XLA (MJX) cartpole. The point of MJX is
running *many* sims in parallel on GPU/TPU, all in JAX (so it's also
differentiable). Same MJCF model file as classic MuJoCo.

## Dependencies

MJX is **not** in the base `mujoco` wheel — it's a separate PyPI package,
`mujoco-mjx`, that provides the `mujoco.mjx` import and pulls in JAX. Pin it to
the same version as `mujoco` (we're on 3.9.0).

```bash
uv add "mujoco-mjx==3.9.0"
```

Resolved against our current venv (mujoco 3.9.0, numpy, scipy already present),
this adds 6 packages:

| package      | version | why |
|--------------|---------|-----|
| `mujoco-mjx` | 3.9.0   | the MJX physics in JAX (`from mujoco import mjx`) |
| `jax`        | 0.10.2  | array/jit/vmap backend |
| `jaxlib`     | 0.10.2  | compiled XLA runtime (**CPU build on macOS**) |
| `ml-dtypes`  | 0.5.4   | jax dep (bfloat16 etc.) |
| `opt-einsum` | 3.4.0   | jax dep |
| `trimesh`    | 4.12.2  | mesh handling for collision geometry |

For `pyproject.toml`, keep it optional so the base install stays light:

```toml
[project.optional-dependencies]
mjx = ["mujoco-mjx==3.9.0"]
```

### GPU/TPU note — important on this machine
The `jaxlib` above is the **CPU** build (we're on macOS/darwin). MJX on CPU
works but is *slow* and gives no parallelism win — the whole point of MJX is
accelerators. Real speedup needs one of:
- **Linux + NVIDIA**: `uv add "jax[cuda12]"` (pulls CUDA wheels)
- **TPU**: `jax[tpu]`
- Apple-Silicon GPU via `jax-metal` is experimental and **not reliable for
  MJX** — treat the Mac as CPU-only for development.

## Other caveat

- **RK4 not supported.** `models/cartpole.xml` has `integrator="RK4"`; MJX only
  does Euler / implicitfast. Either edit the XML or override after loading
  (shown below).

## Single sim

```python
import mujoco
from mujoco import mjx
import jax

model = mujoco.MjModel.from_xml_path("models/cartpole.xml")
model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST  # RK4 -> implicit

mx = mjx.put_model(model)        # model onto device
dx = mjx.make_data(mx)           # data onto device

step = jax.jit(mjx.step)
dx = step(mx, dx)                # one physics step
print(dx.qpos)                   # DeviceArray
```

## Batched / parallel (the actual reason to use MJX)

```python
import jax, jax.numpy as jnp

N = 4096
rng = jax.random.split(jax.random.PRNGKey(0), N)

def make_one(key):
    dx = mjx.make_data(mx)
    theta0 = jax.random.uniform(key, minval=-0.1, maxval=0.1)  # small tilt
    return dx.replace(qpos=dx.qpos.at[1].set(theta0))

batch = jax.vmap(make_one)(rng)          # N independent sims, leading axis N

@jax.jit
@jax.vmap                                 # maps over batch + ctrl, mx is closed over
def step(dx, ctrl):
    return mjx.step(mx, dx.replace(ctrl=ctrl))

ctrl = jnp.zeros((N, model.nu))
for _ in range(100):
    batch = step(batch, ctrl)             # 4096 sims advance together
print(batch.qpos.shape)                   # (N, nq)
```

## Getting state back to classic MuJoCo (for the viewer)

MJX has no viewer. Pull one sim back to a classic `MjData` and render that.
Signature is `mjx.get_data(model, mjx_data)` where `model` is the classic
`MjModel` — verified on mujoco 3.9.0:

```python
one  = jax.tree_util.tree_map(lambda x: x[0], batch)  # pick sim 0 out of the batch
data = mjx.get_data(model, one)                        # -> classic MjData
# ... hand `data` to mujoco.viewer as usual
```

## Verified

Ran here on mujoco 3.9.0 / jax 0.10.2 (CPU, macOS): single step, batched
`vmap` over 256 sims (`qpos` shape `(256, 2)`), and the `get_data` roundtrip all
work. On import you'll see harmless `Failed to import warp` lines — MJX can use
NVIDIA Warp as an optional backend and falls back to pure JAX without it.

These come from a hard-coded `print()` in `mujoco/mjx/warp/__init__.py`, so a
logging level or warnings filter won't suppress them. To make them go away,
install Warp so the import succeeds:

```bash
uv add warp-lang
```

Not worth it on a CPU/macOS setup (sizable dep, no benefit here) — the lines are
purely informational. Don't edit the `.venv` package file; it's overwritten on
reinstall.

## Open questions / TODO
- which cartpole features (rail/floor contacts, slider limit) MJX actually
  enforces vs approximates — sim runs clean, haven't checked fidelity
- benchmark N-env throughput CPU vs GPU before porting PPO (`examples/04`)
