# Reading forces in MuJoCo

Where the different "forces" live, with concrete, runnable examples. MuJoCo
exposes several distinct quantities, so the real question is always *which* force
at *which* level: joint, body, site, or contact.

All snippets below run against [`models/hexapod.xml`](../models/hexapod.xml) and
the printed values are from the robot standing at rest (mass 1.76 kg, weight
17.27 N), MuJoCo 3.9.0. Read forces *after* a `mj_forward`/`mj_step`, never
before: the `cfrc_*` and contact forces are filled in by the constraint solver
and `mj_rnePostConstraint`, which `mj_step` runs.

## Setup

```python
import mujoco, numpy as np
m = mujoco.MjModel.from_xml_path("models/hexapod.xml")
d = mujoco.MjData(m)
for _ in range(300):           # settle onto the feet
    mujoco.mj_step(m, d)

def jid(n): return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
def bid(n): return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
def gid(n): return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
```

## 1. Joint level: generalized (joint-space) forces

All of shape `(nv,)`, indexed by a joint's DOF address `m.jnt_dofadr[j]`. They
decompose the dynamics:

| field | meaning |
|-------|---------|
| `d.qfrc_actuator` | from actuators |
| `d.qfrc_bias` | gravity + Coriolis/centrifugal |
| `d.qfrc_passive` | springs, damping, friction loss, fluid |
| `d.qfrc_constraint` | contacts, limits, equalities mapped to joints |
| `d.qfrc_applied` | what you inject in joint space |

These are the terms of the equation of motion, `M*qacc + qfrc_bias =
qfrc_passive + qfrc_actuator + qfrc_applied + qfrc_constraint`. At rest
(`qacc = 0`, `qvel = 0`) they balance, which is why the standing-pose torques add
up. Each, in turn:

- **`qfrc_actuator`** is the generalized force produced by *all* actuators, mapped
  from actuator space onto the DOFs through the transmission (moment arms). For
  our `position` servos that's `kp*(target - q) - kv*qvel` per joint, so the
  femur's +0.309 N*m is the servo holding the leg up. It's an *output*, recomputed
  every forward step from the current state and `d.ctrl`.
- **`qfrc_bias`** is the bias force `C(q,qvel)*qvel + g(q)`: gravity plus
  Coriolis/centrifugal. It sits on the *left* of the equation of motion, so the
  actuators have to produce it just to stay still. At rest (`qvel = 0`) it is pure
  gravity, and for a single DOF it is the gravity moment of everything *outboard*
  of that joint, so the femur's -0.086 N*m is the leg's own weight
  (femur+tibia+foot), not the trunk.
- **`qfrc_passive`** is the internal force that needs no control: joint/tendon
  springs (`stiffness`/`springref`), `damping`, dry `frictionloss`, and any
  fluid/viscous drag. Here it is ~0.001 N*m because stiffness is 0 and the robot
  is nearly still; the damping (0.4) and frictionloss (0.02) only bite once the
  joints actually move.
- **`qfrc_constraint`** is the solver's reaction force from contacts, joint/tendon
  limits, and equality constraints, projected onto the DOFs by their Jacobians.
  This is how an *external* contact enters joint space: the femur's -0.397 N*m is
  the foot's ground reaction (its 1/6 share of body weight) seen at the lift axis.
  It is zero whenever nothing is in contact or pushing on a limit.
- **`qfrc_applied`** is a generalized force *you* write in (an input, default
  zero), the joint-space twin of `xfrc_applied`. Use it to torque a DOF directly
  without defining an actuator, e.g. a scripted disturbance or a feedforward term.

```python
adr = m.jnt_dofadr[jid("femur_FL")]
print(d.qfrc_actuator[adr], d.qfrc_bias[adr], d.qfrc_passive[adr])
# +0.309  -0.086  +0.001   (N*m: servo torque, gravity/bias, passive)
```

A hinge is one scalar at `adr`; the free joint occupies 6 entries (`[lin(3),
ang(3)]`) starting at its `dofadr`.

**Per-actuator torque** (`(nu,)`, the scalar each actuator produces) and an
**effort penalty** for a cost:

```python
a = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "femur_FL")
print(d.actuator_force[a])          # +0.309 N*m

effort = np.sum(d.actuator_force**2)
print(effort)                        # 0.622  -> add w_effort * effort to a cost
```

For **torque limits**, compare `d.actuator_force` against the actuator
`forcerange` (here +-15 N*m, from the `hexapod` class).

## 2. Body level: Cartesian 6D wrenches

Shape `(nbody, 6)`. Each row is a spatial **wrench** `[moment(3), force(3)]` in
world axes, with the moment taken about the **world origin** (not the body CoM).
They are only filled when `mj_rnePostConstraint` runs, which the pipeline does
automatically *only* if the model has an acceleration-stage sensor (force,
torque, accelerometer, ...); otherwise call it yourself (below).

| field | meaning |
|-------|---------|
| `d.cfrc_ext` | net **external** wrench (contacts + applied) |
| `d.cfrc_int` | **reaction** transmitted through the body's joint (body <-> parent) |
| `d.xfrc_applied` | external wrench *you* set on the body (an input) |

- **`cfrc_ext`** is the net external wrench acting on the body: contact forces
  plus anything in `xfrc_applied`, but **not** gravity (gravity is a body force,
  carried in `qfrc_bias`). At rest its per-body z-force sums to the weight, the
  ground holding the robot up.
- **`cfrc_int`** is the internal reaction wrench transmitted across the body's own
  joint, i.e. what the parent pushes/twists the body with. It is essentially the
  6D "joint reaction load" at that connection (near equal-and-opposite to the
  external load the body carries).
- **`xfrc_applied`** is an *input* you set, a Cartesian wrench applied at the body
  CoM in the world frame. This is the body-level twin of `qfrc_applied`; use it to
  push or twist a body directly (drag, thruster, scripted disturbance).

```python
mujoco.mj_rnePostConstraint(m, d)   # required to fill cfrc_ext / cfrc_int
b = bid("tibia_FL")
print(d.cfrc_ext[b])    # [ 0.468 -0.811  0.  -0.103 -0.06  2.878]  = [moment | force]
print(d.cfrc_int[b])    # [-0.385  0.666 -0.   0.103  0.06 -2.346]

# total ground reaction = sum of external z-force over all bodies
print(d.cfrc_ext[:, 5].sum())   # +17.27 N  == weight (floor holds it up)
```

The `tibia_FL` force part `[-0.103, -0.06, 2.878]` is that foot's ground reaction
(section 4); the moment part is its moment about the origin, `p x F`. The per-body
z-forces sum to the weight, which confirms the layout.

> **Ordering gotcha.** `cfrc_ext`/`cfrc_int` are `[torque(3), force(3)]`
> (rotational first). But `d.xfrc_applied` is the opposite, `[force(3),
> torque(3)]`, in the world frame at the body CoM. So to push a body forward:
> `d.xfrc_applied[b, 0:3] = [fx, fy, fz]` (force in the first three).

## 3. Site level: force/torque sensors

Sites have no force field by default; attach sensors and read `d.sensordata`:

```xml
<sensor>
  <force  name="ft_force_FL"  site="foot_FL"/>   <!-- 3D force,  site frame -->
  <torque name="ft_torque_FL" site="foot_FL"/>   <!-- 3D torque, site frame -->
  <touch  name="touch_FL"     site="foot_FL"/>   <!-- scalar normal force in a zone -->
</sensor>
```

```python
def sensor(name):
    s = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, name)
    a, n = m.sensor_adr[s], m.sensor_dim[s]
    return d.sensordata[a:a + n]

print(sensor("ft_force_FL"))    # [ 0.083 -0.    -2.348] N
```

This is an **F/T load cell at the site**: it reports the force transmitted
between that body and its parent, in the **site frame**, so it is the ground
reaction minus the outboard link's own weight (2.35 vs the raw 2.88 contact
force) which is exactly what a real sensor mounted there would read.

The `touch` sensor is a single scalar (normal force of contacts inside the
site's *sphere*). It reads 0 here because `foot_FL`'s site radius is tiny
(0.012); enlarge the site to cover the contact patch to use it.

## 4. Contact level: per-contact force

`mj_contactForce` gives one contact's force in the **contact frame**
`[normal, tangent1, tangent2, (torsion, roll)]`:

```python
g = gid("foot_FL")
for i in range(d.ncon):
    c = d.contact[i]
    if g in (c.geom1, c.geom2):
        f = np.zeros(6); mujoco.mj_contactForce(m, d, i, f)
        print(f[:3])     # [ 2.878 -0.06   0.103]  (normal, t1, t2)
```

Usually you want the **net world-frame reaction on a foot**, summed over its
contacts. `contact.frame` is a 3x3 (rows = contact axes in world), and
`mj_contactForce` returns the force on `geom2` from `geom1`:

```python
def geom_force_world(m, d, geom_id):
    """Net contact force on geom_id, in world coordinates (N)."""
    total = np.zeros(3)
    for i in range(d.ncon):
        c = d.contact[i]
        if geom_id not in (c.geom1, c.geom2):
            continue
        f = np.zeros(6); mujoco.mj_contactForce(m, d, i, f)
        fw = c.frame.reshape(3, 3).T @ f[:3]           # contact frame -> world
        total += fw if geom_id == c.geom2 else -fw     # force ON our geom
    return total

for nm in ["foot_FL", "foot_ML", "foot_BL", "foot_BR", "foot_MR", "foot_FR"]:
    print(nm, geom_force_world(m, d, gid(nm)))
# foot_FL [-0.104 -0.06   2.878]   ... each foot ~ +2.878 N up
# the six Fz sum to +17.27 N (= weight); use this for a contact-schedule or
# force-distribution cost, or to detect which feet are in stance (Fz > eps).
```

## Caveats

- **Timing.** Valid only after `mj_forward`/`mj_step`; reading before stepping
  gives stale/zero values. The `cfrc_*` are the special case: they come from
  `mj_rnePostConstraint`, which the pipeline runs **only** if an acceleration-stage
  sensor needs it, so without such a sensor you must call it yourself (section 2).
- **Frames & ordering.** `cfrc_*` are `[torque, force]`; `xfrc_applied` is
  `[force, torque]`; sensor and contact forces are in the *site* and *contact*
  frames respectively, not world. Rotate as needed (section 4).
- **MJX** (for the MPPI rollouts in [`examples/02_mpc_hexapod.py`](../examples/02_mpc_hexapod.py)):
  joint-space `qfrc_*`, `actuator_force`, and contact forces (`efc`) are
  available inside the differentiable rollout, but `cfrc_ext`/`cfrc_int` and
  several sensors have only partial support. If a cost needs body wrenches or
  F/T sensors, confirm they exist for your MJX version, or compute them on the
  classic-MuJoCo side at playback.
