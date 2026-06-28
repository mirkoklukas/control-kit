# The MuJoCo data model

How state is laid out in `MjModel` / `MjData`, and how a name turns into an array
index. Counts in parentheses are for [`models/hexapod.xml`](../models/hexapod.xml)
(a floating base plus 18 actuated leg joints), MuJoCo 3.9.0. See also
[`mujoco-forces.md`](mujoco-forces.md) for the force quantities specifically.

## The one principle

`MjData` is a bag of flat arrays. Every array is indexed by exactly one
**dimension**, a count fixed by the model (`nbody`, `nv`, `nu`, ...). A *name*
resolves, through the model, to an *id*, and that id is an index into one of these
dimensions.

So the question is never "does X carry data", it is "which dimension is X, and
which arrays live on that dimension".

## Names on the model, state on the data

Three-part split:

- **Names** live on `MjModel` only (`mj_name2id`, or `m.actuator('coxa0').id`).
  `MjData` has no name table of its own.
- **Static properties** (mass, size, ranges, gains) live on `MjModel`.
- **Runtime state** (poses, qpos, forces) lives on `MjData`.

`d.body('base')` is just sugar: it asks the model for the id, then indexes the
data's flat arrays.

### Resolving a name to an index

Two stages, and the resulting index is per object type:

```
semantic label  --(our convention)-->  MJCF name  --(model)-->  array index
("femur","ML")          "femur1"        m.actuator("femur1").id -> 4
```

Stage 2 is baked in at compile time. Stage 1 (our label to the MJCF name) is the
only part we own, and it depends on how the XML was authored: the `<replicate>`
in `hexapod.xml` suffixes the copy index, giving `femur0..femur5`.

## The MjData table

For the five everyday namespaces: resolve `id` from the model, then index the
data array with it.

| namespace    | id from model         | access (using `id`)                                                                                                          |
| ------------ | --------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| **body**     | `m.body(name).id`     | `d.xpos[id]`, `d.xquat[id]`, `d.xmat[id]`, `d.cvel[id]`, `d.subtree_com[id]`, `d.cfrc_ext[id]`                               |
| **joint**    | `m.joint(name).id`    | `d.xanchor[id]`, `d.xaxis[id]`, `d.qpos[m.jnt_qposadr[id]]`, `d.qvel[m.jnt_dofadr[id]]`, `d.qfrc_actuator[m.jnt_dofadr[id]]` |
| **actuator** | `m.actuator(name).id` | `d.ctrl[id]`, `d.actuator_force[id]`, `d.actuator_length[id]`, `d.actuator_velocity[id]`                                     |
| **geom**     | `m.geom(name).id`     | `d.geom_xpos[id]`, `d.geom_xmat[id]`                                                                                         |
| **site**     | `m.site(name).id`     | `d.site_xpos[id]`, `d.site_xmat[id]`                                                                                         |

**Joint is the exception.** Its `id` indexes the world-frame anchor/axis directly
(`d.xanchor[id]`, `d.xaxis[id]`), but the configuration state (`qpos`/`qvel`) is
*not* at `id`; it goes through `m.jnt_qposadr[id]` / `m.jnt_dofadr[id]` first.
Every other namespace's `id` indexes its arrays directly.

## Every dimension

Curated to physical fields. Pure solver scratch (`*_awake`, `iLDiagInv`, `map_*`,
`moment_rowadr`, `efc` sparsity helpers, ...) is omitted.

| dimension | id is | arrays |
|---|---|---|
| `nq` (25) | a position coordinate | `qpos` |
| `nv` (24) | a **dof** (velocity coordinate) | `qvel`, `qacc`, `qacc_smooth`; generalized forces `qfrc_actuator`, `qfrc_bias`, `qfrc_passive`, `qfrc_constraint`, `qfrc_applied`, `qfrc_smooth`, `qfrc_spring`, `qfrc_damper`, `qfrc_gravcomp`, `qfrc_fluid`, `qfrc_inverse`; `cdof`, `cdof_dot` |
| `na` (0) | an actuator activation | `act`, `act_dot` (stateful actuators only) |
| `nu` (18) | an **actuator** | `ctrl`; `actuator_length`, `actuator_velocity`, `actuator_moment`, `actuator_force` |
| `nbody` (20) | a **body** | pose `xpos`, `xquat`, `xmat`; inertial frame `xipos`, `ximat`; spatial motion `cvel`, `cacc`; spatial force `xfrc_applied`, `cfrc_ext`, `cfrc_int`; inertia `cinert`, `crb`; subtree `subtree_com`, `subtree_linvel`, `subtree_angmom` |
| `njnt` (19) | a **joint** | `xanchor`, `xaxis` (world-frame anchor/axis only) |
| `ngeom` (27) | a **geom** | `geom_xpos`, `geom_xmat` |
| `nsite` (6) | a **site** | `site_xpos`, `site_xmat` |
| `ncam` / `nlight` | a camera / light | `cam_xpos`, `cam_xmat` / `light_xpos`, `light_xdir` |
| `ntendon` (0) | a tendon | `ten_length`, `ten_velocity` |
| `nsensordata` (0) | a sensor output slot | `sensordata` |
| `ncon` (dynamic) | a **contact** | `contact[k]` structs; force via `mj_contactForce(m, d, k, out)` |
| `nefc` (dynamic) | a constraint row | `efc_force`, `efc_J`, `efc_pos`, ... |
| scalar | n/a | `time`, `energy` (`[potential, kinetic]`) |

Anything not on this table is solver bookkeeping.

## State vs inputs vs derived

Only `qpos` (`nq`) + `qvel` (`nv`) + `act` (`na`) is independent state. Everything
else in the tables is **derived**, recomputed from that state on every
`mj_forward` / `mj_step`. By role:

- **You set (inputs):** `ctrl`, `qfrc_applied`, `xfrc_applied`, `mocap_pos`/`mocap_quat`
- **The integrator advances (state):** `qpos`, `qvel`, `act`, `time`
- **The pipeline computes (derived):** all the `x*`, `c*`, `geom_*`, `site_*`,
  `actuator_*`, `qfrc_*`, `qacc`, contacts, constraints

The pipeline, in order:

```
state: qpos, qvel
  -> forward kinematics  -> poses: xpos, geom_xpos, site_xpos, xanchor
  -> velocities          -> cvel, actuator_length / actuator_velocity
  -> forces              -> actuator_force, qfrc_passive / qfrc_bias, contacts -> efc_force
  -> forward dynamics    -> solve  M qacc = sum of qfrc  ->  qacc
  -> integrate           -> new qvel, qpos
```

Read derived quantities *after* a `mj_forward` / `mj_step`, never before.

## Two coordinate systems

Each motion quantity exists in a **generalized** (dof-indexed) form and a
**Cartesian** (body-indexed) form:

| quantity | generalized (`nv`, via `jnt_dofadr`) | Cartesian (`nbody`) |
|---|---|---|
| velocity | `qvel` | `cvel` |
| acceleration | `qacc` | `cacc` |
| force | `qfrc_*` | `cfrc_ext`, `cfrc_int`, `xfrc_applied` |

Summary of where motion data exists at all:

- **forces:** actuator (scalar `actuator_force`), body (Cartesian wrench), dof
  (generalized `qfrc_*`). Not `njnt`, geom, or site.
- **accelerations:** body (`cacc`), dof (`qacc`).

The **pose-only** namespaces (geom, site, the joint anchor, camera, light) carry
just `xpos`/`xmat`. They have no velocity, acceleration, or force of their own;
for those, go through the parent body or compute on demand with
`mj_objectVelocity` / `mj_objectAcceleration` / `mj_contactForce`.

## Degrees of freedom

DOFs come only from **joints**. `nv` is the sum of the per-joint dof widths, which
are fixed by joint type:

| joint type | qpos width | dof width |
|---|---|---|
| free | 7 (3 pos + 4 quat) | 6 (3 lin + 3 ang) |
| ball | 4 (quat) | 3 |
| slide | 1 | 1 |
| hinge | 1 | 1 |

For the hexapod: `nv` = 6 (the free `root` joint on `base`) + 18 (3 hinges x 6
legs) = 24, and `nq` = 7 + 18 = 25.

The address arrays are computed by the compiler and frozen per model:

$$\text{jnt\_qposadr}[j] = \sum_{k<j} \text{qpos\_width}(\text{type}_k)$$

and likewise `jnt_dofadr` over the dof widths. `jnt_qposadr[j]` indexes into
`qpos`; `jnt_dofadr[j]` indexes into `qvel` / `qacc` / `qfrc_*`.

The free joint is exactly why `nq != nv`: a quaternion needs 4 numbers to store
but only 3 to differentiate, so the base takes 7 position slots and 6 velocity
slots. After the root, the qpos and qvel addresses of every later joint differ by
1.

Reverse maps (on the model): `dof_jntid[v]` is the joint owning dof `v`, and
`dof_bodyid[v]` is the body that dof moves (the joint's child body). A body's dof
count is the sum over the joints attaching it to its parent: 0 if welded (for
example `world`), 6 for `base` via its free joint, 1 per leg segment.

## Base frame

The **base** is the root link of the kinematic tree, the body whose pose is not
set by any joint angle but is itself part of the state. For the hexapod that is
`base` (it carries the free joint). The **base frame** is the coordinate frame
attached to that body (`d.body('base').xpos`, `d.body('base').xquat`).

Floating-base systems split the generalized coordinates into base plus joints:

```
qpos = [ base pose (7: pos + quat) | joint angles (18) ]   = qpos[0:7]  +  qpos[7:]
qvel = [ base twist (6: v + omega) | joint rates   (18) ]   = qvel[0:6]  +  qvel[6:]
```

The base frame is not the CoM. The base frame is fixed to the body geometry
(`xpos`, our hexagon-center body origin); the body's own center of mass is `xipos`;
the whole-robot center of mass is `d.body('base').subtree_com` and moves as the
legs swing.
