# Controller

Interactive posture control: sculpt a **target posture** with a PlayStation
DualSense, and have the simulated robot drive itself onto that target.

## Goal

Control a robot in a MuJoCo simulation with a DualSense controller: define a
target posture with the stick/buttons, then use a policy to reach it. The policy
starts as plain position-servo tracking (to prove the loop), then upgrades to an
MPPI regulator so the motion is dynamically feasible and balanced.

## The loop

Everything runs **locally on the Mac**, in one real-time loop:

```
DualSense ──▶ edit target Posture ──▶ policy ──▶ ctrl ──▶ mj_step ──▶ passive viewer
   input        (mode + sticks)      (servo→MPPI)         (mujoco)     (mjpython)
```

Local is forced, not chosen: the controller is plugged into the Mac and the
passive viewer needs a display, while the GPU box is headless. So the whole
interactive loop is CPU/local. That's fine — a static-target regulator is cheap,
and MPPI rollouts on CPU (via `mujoco.rollout`, threaded) are viable at a modest
control rate. The MJX path (reuse `controlkit.mpc`) stays a variant for offline
or GPU work, but the interactive loop does not depend on it.

## The robot

A 4-leg, 4-DOF radial robot, built from the kinematics library:

```python
from controlkit.kinematics.robot import Robot, radial_mounts
from controlkit.kinematics.leg4dof import Leg4DOF

leg = Leg4DOF.from_lengths(
    jnp.array([0.05, 0.05, 0.2, 0.3]),
    jnp.deg2rad(jnp.array([[-90, 90], [-180, 180], [-150, 150], [-150, 150]])),
)
robot = Robot(mounts=radial_mounts(4, radius=0.20), leg=leg)
```

Flat ground, starting from a sensible **rest posture**: body at a nominal height,
feet planted radially outward, elbow-down. Build it by IK — place each foot at a
radial foothold and solve (`Leg4DOF.ik_from_foot_and_yaw`), reusing the same
kinematics the editing pipeline uses.

## What we reuse vs. what we write

Reuse from `controlkit`:

- `Robot` / `Leg4DOF` — FK, IK (`ik_from_foot_and_yaw`), `to_mjcf`/`to_mujoco`,
  `feet`, `shoulders`, self-collision, `Posture` type.
- `mujoco_viz.show` / `launch_passive` pattern — the passive viewer + macOS
  `mjpython` relaunch (see `controlkit/mujoco_viz.py`).
- `controlkit.mpc` (`make_rollout_sampler` / `make_mppi_planner`) — for the MJX
  variant; the CPU MPPI is a small separate loop over `mujoco.rollout`.

DualSense driver (`input.py`): **hidapi raw-report parsing**, not pygame/SDL. SDL's
event pump must run on the process main thread on macOS, but the MuJoCo viewer
under `mjpython` owns the main thread and runs our loop on a worker thread — so
`pygame.event.pump()` crashes there. hidapi reads a device handle on any thread,
so it coexists with the viewer (and `hidapi` is already a dependency).

Resolved: **actuators are now in `Robot.to_mjcf`** (opt-in). `to_mjcf(actuators=True,
kp=…, joint_damping=…)` emits one `<position>` servo per leg joint (`ctrlrange` =
joint limits) plus optional joint damping; default off keeps the pure force/viz
model. `to_mjcf(actuators=True, weld_feet=True)` is exactly the welded-M1 model.

## Control scheme

*(M1 uses only body mode with feet welded — MPC handles the leg joints, so the IK
note below does not apply there. The full scheme lands in M2, once unwelded.)*

**Modes.** One target subsystem is active at a time:

```
Body  ─▶ Leg 0 ─▶ Leg 1 ─▶ Leg 2 ─▶ Leg 3 ─▶ (Body …)
```

- **R1** switches the active mode directly (Body → Leg 0 → … → Leg 3 → Body); the
  sticks always edit the current mode. Single press, no separate commit step.
- **L1** reserved (e.g. hold-to-edit deadman, or snap-target-to-current pose).
  R1/L1 are digital bumpers — clean press edges; the analog L2/R2 stay free for
  continuous use later (e.g. hold-to-scale edit speed).

**Editing is rate-based**: stick deflection is a *velocity* on the target,
integrated each tick, so holding a stick slews the target and releasing it holds.

**Body mode** (feet stay planted; the body moves over them):

- left stick → body **x / y**, along the current heading (yaw)
- right stick → **pitch / yaw**
- R2 / L2 triggers → body **±z** (proportional)
- roll → TBD (spare buttons / shoulder combo)

A body pose is only reachable *through the planted legs* (the base is a free
joint — nothing actuates it directly). So editing the body target means: keep the
current footholds fixed, and IK each planted leg to hold its foot while the body
moves to the target → target `thetas`. This is exactly the stance/posture IK the
library already does.

**Leg mode** (the selected leg swings): edit that leg's **foot target** in the
body frame with the sticks, IK the leg to it → that leg's `thetas`. *(Open:
Cartesian-foot vs. raw joint editing — see below.)*

The edited target — body `SE3` + per-leg `thetas` — is a `Posture`; the policy's
job is to make the sim's posture match it.

## First testbed: welded feet, body-pose target (M1)

Start with the four feet **welded to the ground** and let MPC do the rest — no
IK. Welding turns the robot into a parallel mechanism (pinned feet + leg joints +
free body), so the body moves *only* through the leg joints, and MPC absorbs the
inverse kinematics: it searches the 16 servo targets and the body pose falls out.
Welds also remove the two hardest parts up front — no balance/tipover, no
stepping — leaving the pure controller→target→MPC→sim loop.

- **Model:** `to_mjcf(actuators=True, weld_feet=True)`; activate all four
  `weld_foot{i}` equalities at start (feet pinned at the rest stance).
- **Target:** a **mocap body** (`<body mocap="true">` box). The controller writes
  `data.mocap_pos` / `mocap_quat`; it is kinematic, just a movable target the MPC
  reads. Body mode's sticks move it (xy, pitch/yaw, ±z). No mode cycling yet.
- **MPC:** cost = body-pose error to the mocap target (position weighted over
  orientation at first); controls = the 16 leg servo targets; rollouts via
  `mujoco.rollout` on CPU. Out-of-reach targets degrade gracefully (MPC gets as
  close as the welds allow) — no IK-failure handling needed.
- **DOF check:** 6 (body) + 16 (joints) − 4×3 (point welds) = 10 internal DOF.
  Mobile enough to servo the body, rigid enough to hold.

## Policy: target → ctrl

- **M1 — MPPI regulator, welded feet** (above): the planner drives the servos to
  bring the welded body onto the mocap target. Tests the whole loop with MPC in
  from the start.
- **M2 — unweld: full posture + balance.** Release the welds, add balance terms
  (upright, height, contact / no-tip) and per-leg targets, so reachability, joint
  limits, and balance are respected instead of assumed. Cost shape follows
  `lab/hexapod_mpc/env.py`; rollouts via `mujoco.rollout` on CPU (or MJX +
  `controlkit.mpc` on the GPU box).

## Proposed files (lab experiment shape)

```
lab/controller/
  design.md
  config.py      # knobs: robot dims, rest pose, edit speeds, control rate, MPPI params
  robot.py       # the concrete Robot build + rest_posture() + actuated-MJCF helper
  input.py       # DualSense (hidapi) → normalized ControllerState (axes, buttons, edges)
  target.py      # mode state machine + apply ControllerState to the target Posture (IK)
  policy.py      # M1 servo tracking; M2 MPPI regulator (same interface)
  run.py         # the real-time loop: passive viewer + input + policy  (needs mjpython)
```

## Milestones

- **M0** — pygame reads the DualSense; print axes / buttons / trigger edges.
- **M1a** — fly the mocap target box with the controller; robot static at rest,
  no physics. Tune the control feel (rates / signs / limits). *(done)*
- **M1b** — welded feet + MPPI drives the servos to chase the box in the viewer.
  *(done)* CPU MPPI on `mujoco.rollout` (~3–6 ms/plan). Gotcha: the planner resets
  each rollout's `eq_active` from `model.eq_active0`, so the welds must be set
  **on the model** (`eq_active0=1`), not just `data.eq_active`, or it rolls out an
  unwelded robot and diverges. Cost = body pose error + velocity + rest-regularizer.
- **M2** — unweld; add balance terms + per-leg targets and R1 mode switching.

## Open questions

- **Leg-mode editing.** Cartesian foot target + IK (intuitive, matches body mode)
  vs. raw per-joint editing (direct, no IK failure modes)?
- **Roll control** in body mode — which input.
- **Reachability feedback.** When an edit drives the target out of reach (IK
  fails / self-collision), what does the user see — clamp to the last valid
  target, rumble, a color change?
- **Actuator model** — servo gains (kp/damping) that track without oscillating;
  whether this helper belongs in `controlkit`.
```
