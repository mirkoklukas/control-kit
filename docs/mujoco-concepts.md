# 1. Model and state

MuJoCo splits a simulation into two objects:

- **`MjModel`** (call it `model`) — the *constant* description of the system:
  what exists and its fixed properties (bodies, joints, geoms, actuators,
  masses, sizes, global options). Compiled once; never changes while simulating.
- **`MjData`** (call it `data`) — the *live state* of one running copy:
  positions, velocities, time, control inputs, and everything derived from them.
  This is what advances each step.

Rule of thumb: ask **`model`** "what exists, and what are its fixed properties?";
ask **`data`** "what's happening right now?". One `model` can back many
independent `data` instances.

## From XML to model to data

The system is written in an XML dialect called **MJCF**. Two calls turn it into
objects:

```python
import mujoco
model = mujoco.MjModel.from_xml_path("models/cartpole.xml")  # XML   -> model  (compile)
data  = mujoco.MjData(model)                                 # model -> data   (allocate state)
```

- `from_xml_path` runs the MuJoCo **compiler**: parses the XML, numbers every
  element, computes derived constants (inertias, coordinate layout), fills `model`.
- `MjData(model)` allocates the state arrays, sized to fit *that* model.
- `data` always needs a `model`; a `model` needs no `data`.

MJCF is MuJoCo-native, but **URDF (the ROS interchange format) is not a problem**:
`from_xml_path` loads a `.urdf` directly. URDF can't express some MuJoCo-only
features (closed loops, tendons, a full world), so the usual move is to import the
URDF once, then `mj_saveLastXML` to get an MJCF you refine.

## The example, piece by piece

`models/cartpole.xml`:

```xml
<mujoco model="cartpole">                                <!-- XML root -->
  <option timestep="0.01" integrator="RK4"/>             <!-- global physics options -->
  <default>                                              <!-- default attrs for elements below -->
    <joint damping="0.05"/>
    <geom rgba="0.7 0.7 0.7 1"/>
  </default>

  <worldbody>                                            <!-- root of the body tree -->
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="4 4 0.1"/>     <!-- world-fixed geom -->
    <geom name="rail"  type="capsule" fromto="-2 0 0.6 2 0 0.6" size="0.02"/>

    <body name="cart" pos="0 0 0.6">                     <!-- moving body -->
      <joint name="slider" type="slide" axis="1 0 0" range="-1.8 1.8"/>
      <geom  name="cart" type="box" size="0.12 0.08 0.06" mass="1.0"/>
      <body name="pole" pos="0 0 0">                     <!-- child of cart -->
        <joint name="hinge" type="hinge" axis="0 1 0"/>
        <geom  name="pole" type="capsule" fromto="0 0 0 0 0 0.6" mass="0.1"/>
      </body>
    </body>
  </worldbody>

  <actuator>                                             <!-- how you drive it -->
    <motor name="slide" joint="slider" gear="1" ctrlrange="-20 20"/>
  </actuator>
</mujoco>
```

**XML shape.** `<mujoco>` is the root. Directly below it sit *section* tags, each
appearing **at most once**: `<option>`, `<default>`, `<worldbody>`, `<actuator>`.
A section is a container — you put *many entries inside one section*, never many
sections. (`<default>` just supplies default attribute values to elements below
it; it creates no runtime object.)

**The body tree** lives inside `<worldbody>`, which is the root of the
*kinematic tree* and is itself **body 0, named `world`** — auto-created, you
never write it. Nested `<body>` tags are parent/child:

```
world (body 0, fixed)
└── cart  (body 1)   slides along x
    └── pole (body 2)   hinges, relative to cart
```

A geom placed directly under `<worldbody>` (floor, rail) is welded to the world.
**A body moves only if it has a `<joint>`** — a jointless body is rigidly fixed
to its parent.

### What the compiler put on `model` (fixed facts)

| on `model`                                        | value | meaning                                         |
| ------------------------------------------------- | ----- | ----------------------------------------------- |
| `model.nbody`                                     | 3     | bodies incl. `world`                            |
| `model.njnt`                                      | 2     | joints: `slider`, `hinge`                       |
| `model.ngeom`                                     | 4     | geoms: `floor`, `rail`, `cart`, `pole`          |
| `model.nu`                                        | 1     | actuators: `slide`                              |
| `model.nq`                                        | 2     | position coordinates                            |
| `model.nv`                                        | 2     | velocity coordinates (degrees of freedom)       |
| `model.opt`                                       | —     | the `<option>` values (timestep, integrator, …) |
| `model.actuator_gear`, `model.actuator_ctrlrange` | —     | fixed actuator wiring                           |

### What lives on `data` (live numbers)

| on `data` | shape | meaning |
|---|---|---|
| `data.qpos` | `(nq,)=(2,)` | generalized **positions** |
| `data.qvel` | `(nv,)=(2,)` | generalized **velocities** |
| `data.ctrl` | `(nu,)=(1,)` | control input(s) |
| `data.time` | scalar | simulation time |
| `data.xpos`, … | derived | Cartesian poses etc., recomputed from state |

They don't overlap: `model` has no `qpos`; `data` has no `nbody`.

**Lookup by name.** The arrays above are indexed by integer id, but you rarely
track indices by hand. Names live on `model` (not `data`): a name maps to an id
via `mujoco.mj_name2id(model, mjOBJ_JOINT, "hinge")`, then to an array slot via
the model's address arrays (`model.jnt_qposadr`, …). The shortcut is the
per-kind **named accessor**, which returns a *view into `data`*:

```python
data.joint("hinge").qpos     # the pole-angle slice of qpos
data.body("pole").xpos       # the pole's Cartesian position (after mj_forward)
data.site("tip").xpos        # a site's world position; .xmat for its 3x3 orientation
data.actuator("slide").ctrl  # this actuator's control
```

Each object kind (`joint`, `body`, `geom`, `site`, `actuator`, `sensor`) has its
own id space, hence its own accessor. Writing through the view writes straight
into the underlying array, e.g. `data.joint("hinge").qpos[:] = 0.2`.

**Derived quantities need a forward pass.** Cartesian fields like `xpos`,
`site_xpos`, `xmat` are *computed from* `qpos`; they're stale/zero until you call
`mj_forward` (or `mj_step`). Set `qpos`, then `mj_forward`, then read `xpos`.

**MJX has no named accessors.** `dx.site("tip")` does not exist — that's a
classic-MuJoCo convenience. In MJX you map name → integer id once on the classic
`model`, then index the array on the MJX `Data`:

```python
# bodies / sites: index Cartesian arrays by id
body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pole")
tip_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip")
dx = mjx.forward(mx, dx)               # compute kinematics first
dx.xpos[body_id]                       # body world position
dx.site_xpos[tip_id]                   # site world position [x, y, z]

# joints: index qpos/qvel by ADDRESS, not by joint id
jid  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hinge")
qadr = mx.jnt_qposadr[jid]             # slot in qpos
vadr = mx.jnt_dofadr[jid]              # slot in qvel
dx.qpos[qadr], dx.qvel[vadr]
```

A joint's `qpos`/`qvel` slot is **not** its joint id: a free joint eats 7 `qpos`
/ 6 `qvel`, so id ≠ address in general — always go through `jnt_qposadr` /
`jnt_dofadr` (slice a range, `dx.qpos[qadr:qadr+7]`, for multi-DOF joints). All
ids are identical in `mx` (MJX keeps the compiler's numbering); compute them
*outside* any jitted/`scan` function — they're Python ints, not traced values.

### Two index mappings to remember

- **`qpos` / `qvel` are indexed by joint** (declaration order). Each 1-DOF joint
  owns one slot:
  - `qpos = [slider, hinge] = [cart_x, pole_angle]`
  - `qvel = [slider, hinge] = [cart_vel, pole_angvel]`

  Not every joint is one slot — a free-floating body's joint takes 7 `qpos` / 6
  `qvel` (xyz + orientation quaternion).
- **`ctrl` is indexed by actuator** (declaration order). `ctrl[0]` drives the
  `slide` motor; the motor converts it to joint force via `gear`
  (`force = gear × ctrl`), clamped to `ctrlrange`.

### What one `mj_step` changes

`data_{t+1} = mj_step(model, data_t)`. Splitting `data` by who writes it:

- **You set:** `ctrl` (and the initial `qpos`/`qvel`). `mj_step` reads these; it
  does not overwrite them.
- **`mj_step` integrates the true state:** `time`, `qpos`, `qvel`.
- **`mj_step` recomputes everything derived:** `qacc`, `xpos`, contact forces, …
- **`model` is never modified by stepping.**

# 2. Controlling the model: actuators

You drive the model by writing **`data.ctrl`** (length `nu`) before each
`mj_step`. Each actuator turns its scalar `ctrl` into a force/torque, maps it
through its **transmission** (`gear`, onto a joint/tendon/site), and accumulates
the result into `data.qfrc_actuator`. `ctrl` is clamped to `ctrlrange`, the
output force to `forcerange`.

The general law is `force = gain·input + bias`. The named types are just presets
for `gain`/`bias`; below, `q`, `q̇` are the actuator's transmission coordinate
(joint angle/velocity for a joint actuator):

| `<actuator>` type | `ctrl` means        | force produced              | params      |
| ----------------- | ------------------- | --------------------------- | ----------- |
| `motor`           | force/torque        | `gear·ctrl`                 | `gear`      |
| `position`        | target position     | `kp·(ctrl − q) − kv·q̇`     | `kp`, `kv`  |
| `velocity`        | target velocity     | `kv·(ctrl − q̇)`            | `kv`        |
| `intvelocity`     | target rate         | position servo on ∫ctrl     | `kp`, `kv`  |
| `damper`          | damping (≥0)        | `−kv·ctrl·q̇`               | `kv`        |
| `general`         | —                   | set `gain`/`bias`/`dyn` yourself | many   |

`motor` is open-loop torque (what `examples/02` LQR commands); `position` /
`velocity` are the built-in PD/P servos. `muscle` and `adhesion` exist too, and
some types carry an internal **activation** state in `data.act` (size `na`).

To bypass actuators entirely, inject force directly: `data.qfrc_applied`
(joint-space) or `data.xfrc_applied` (Cartesian wrench per body).

# 3. Derivatives (Jacobians)

You can differentiate the dynamics. The natural object is the **one-step
transition Jacobian** of `data_{t+1} = step(model, data_t)`, with state
`x = [qpos; qvel]` (size `2·nv`) and control `u = ctrl` (size `nu`):

- `A = ∂x'/∂x`  (`2nv × 2nv`)
- `B = ∂x'/∂u`  (`2nv × nu`)  ← the controls→state Jacobian

Two ways to get it:

- **Classic, finite differences:** `mujoco.mjd_transitionFD(model, data, eps,
  flg_centered, A, B, C, D)` fills `A`, `B` (and `C`, `D` = sensor-output
  derivatives). Double precision, reliable. Needs Euler/implicit, **not RK4**.
  This is exactly what `examples/02` (LQR) uses.
- **MJX, exact autodiff:** `jax.jacfwd(f)(u)` with
  `f(u) = mjx.step(mx, dx.replace(ctrl=u))`. Use **`jacfwd`** (forward mode) —
  reverse mode (`jax.jacobian`) fails on the solver's `while_loop`. Caveat: MJX
  is **float32**; enable `jax.config.update("jax_enable_x64", True)` for
  accuracy, and never finite-difference it with a tiny `eps` (rounding noise).

**Gotcha we hit:** `B` for the cart came out ~100× smaller than the naive `dt/m`
estimate. Cause — the cart geom overlapped the rail capsule, so they sat in
permanent contact (`data.ncon = 3`) and friction resisted the slide. That tiny
`B` is also why the LQR gain blew up (small `B` ⇒ huge gain) — a derivative is
only as meaningful as the dynamics it linearizes. Fixed by making the rail
non-colliding (`contype="0" conaffinity="0"`); `B` recovers to ~`dt/m` and LQR
now balances. (Quick diagnosis trick: toggle all contacts off with
`model.opt.disableflags |= mjtDisableBit.mjDSBL_CONTACT`.)
