# Planner design

Living note. A stance planner on top of `lab/gait_graph`: search a path of stances
towards a goal, then execute it (an MPC between stances, later).

v0 (2026-10-05): flat ground, move the body along +x; the existing `gait_graph`
quadruped (4-DOF legs, sphere feet); fixed crawl order; a runkit experiment.

## The graph: two views

From `docs/projects/staged/gait-graph.md`. A crawl moves one leg at a time, so the
robot alternates between four feet planted and three feet planted (one leg in swing).

- **Primal:** a node is a **tripod stance** (three feet planted, one leg free). Two
  consecutive tripods are connected by a moment with all four feet planted: the
  four-foot stance is the edge.
- **Dual:** a node is a **fully planted stance** (four feet). Two nodes are connected
  if they differ in one leg's foothold, i.e. by one swing. The tripod during that
  swing must hold.

The two are equivalent; they differ in where things live. In the primal view the
critical check (holding on three feet) belongs to the node and the body shift to the
edge. In the dual view both belong to the edge.

**Decision: the dual view.** Nodes are the states the robot rests in between steps
(all feet down), and `gait_graph` already uses it ("graph type A": nodes share three
of four footholds).

## Notation

| symbol | meaning |
|---|---|
| S = (f_0, f_1, f_2, f_3) | the current stance (a node): leg j's foothold f_j |
| i | the swing leg |
| T = S \ {i} | the swing tripod: the three stance legs at their footholds, {(j, f_j) : j != i} |
| f' | leg i's new foothold |
| S' | the next stance: S with f_i replaced by f' |
| n_j | the surface normal at foothold f_j |
| g | gravity (vector); what makes a wall different from the floor |
| B | the body pose of S (from its witness posture) |
| B_lift | the body pose at lift-off (end of phase 1) |
| B_plant | the body pose at touchdown (end of phase 3); also the body pose of S' |
| q | joint angles; a posture is (body pose, q) |
| P | a posture (body pose, q) |
| hold(T, P, g) | the tripod T holds the robot in posture P under gravity g (see "The hold check") |
| m | the required hold margin |

## One step (an edge), from S to S'

Swing leg i, from the fixed crawl order. Four phases:

| # | phase | feet down | what must hold |
|---|---|---|---|
| 1 | **shift for lifting**: B -> B_lift | 4 (S) | all four feet of S reachable along the way; hold(T) at B_lift |
| 2 | **lift** leg i at B_lift | 3 (T) | hold(T) |
| 3 | **shift for planting**: B_lift -> B_plant, leg i swinging | 3 (T) | hold(T) along the whole path; feet of T reachable; leg i hits nothing |
| 4 | **plant** leg i on f' at B_plant | 4 (S') | S' a valid stance at B_plant: node S' |

"hold(T)" short for hold(T, P, g) with margin >= m, at that posture.

Notes:

- Two body poses per step: B_lift and B_plant. Both set the hold margin and how far the
  step goes.
- Phase 3 may be zero (B_plant = B_lift): shift -> lift -> plant. Then f' must be
  reachable from B_lift, and the hold is checked at one pose only. With a phase-3 shift,
  strides can be longer, but hold(T) must hold along the path, checked at points along it
  (joint torques depend on the posture nonlinearly, so the two ends are not enough).
- The body moves with four feet down only in phase 1. A B_lift good for leg i can leave
  a poor start for the next leg: that is what the search is for.
- Phases 1+2 and 3+4 may be combined in execution (the foot lifts / lands as the body
  arrives). The planner checks the same poses and paths either way.
- Exists in `gait_graph/stance.py`: reach, joint limits, ankle angle, collisions, foot
  spacing. New: the hold check, the shift path checks.

## Choosing a step: the order

Per step the planner chooses B_lift, B_plant and f' (i is fixed by the crawl order).
Body first, then the foothold:

1. **i**: next in the crawl order.
2. **B_lift**: a pose where hold(T) and all four feet of S are reachable. Does not depend
   on f'.
3. **B_plant**: a pose where hold(T) and the feet of T are reachable, as far forward as
   possible.
4. **f'**: among leg i's footholds reachable from B_plant, one ahead in x that makes S' a
   valid stance at B_plant.

B_lift and B_plant are linked through the swing tripod T: both must hold on the same
three feet, and so must the path between them.

Why body first, not foothold first:

- f' is reachable by construction: its candidates are the footholds within reach at
  B_plant. Foothold first would have to search a B_plant that reaches a given f' (an
  inverse problem).
- It is what `gait_graph` already does: `resample_leg` re-plants one leg at the current
  body pose, with candidates from `candidates_at`. Step 4 is that call at B_plant.
- Progress comes from B_plant (pushed forward within hold(T) and the reach of T); step 4
  then picks the best foothold there.

## The hold check: $\mathrm{hold}(T, P, g)$

Does the tripod $T$ hold the robot in posture $P$ under gravity $g$ (one leg lifted)? A force
condition, not a geometric one: on a wall or overhang the feet pull, and the centre of
mass can be anywhere relative to the feet. (Without magnets on flat ground it reduces to
"centre of mass above the support triangle".)

**Condition.** There are foot forces $F_j$, $(j, f_j) \in T$, with

1. equilibrium: $\sum_j F_j + M g = 0$, and the moments balance ($M$: robot mass);
2. adhesion: $F_j \cdot n_j \ge -A$ (a foot pulls at most $A$);
3. friction with adhesion: $\lVert F_{j,t} \rVert \le \mu \, (F_j \cdot n_j + A)$ ($F_{j,t}$: tangential part);
4. servo torques: the joint torques that carry $F_j$ (and the legs' weight) within
   $\tau_{\max}$.

**Margin.** E.g. the largest factor on $g$ that still has a solution, or the smallest
slack in 2-4. The planner asks for margin $\ge m$.

### Three ways to check it

From fast and idealized to slow and realistic:

| | check | gives | needs |
|---|---|---|---|
| a | **force optimization**: solve 1-4 for $F_j$ (1-3 a second-order cone problem in $F_j$; 4 linear through the leg Jacobians) | holds or not, margin; no dynamics | the posture, the footholds and normals, $g$, the mass model |
| b | **simulation, welded feet**: weld the three feet to the world, settle the posture under $g$, read the weld forces and the servo torques, compare to the limits in 2-4 | the demand on each foot and servo, with full servo compliance; cannot fall | any robot (sphere feet too) |
| c | **simulation, patches**: a small plate under each of the three footholds (oriented by $n_j$), the magnets attach to it; then lift leg $i$, as in `lab/rl/climb/magnet_test.py` | held / fell, slip; peel, the release shock, servo sag included | the climb robot (magnetic pads) |

The patches make (c) independent of the scene: it needs only $T$ (positions and normals),
$P$ and $g$. For v0 (the `gait_graph` robot, sphere feet) only (a) and (b) apply.

### (a) The force optimization in detail

**Setup.** Tripod $T$: feet $j = 1, 2, 3$ at contact points $p_j$ (world), outward unit
surface normals $n_j$. Posture $P$ fixes the centre of mass $c$ and the total mass $M$;
gravity $g$. Unknowns: the contact forces $F_j \in \mathbb{R}^3$ (what surface and magnet
exert on foot $j$), 9 numbers.

Assumptions: the servos hold $P$, so the robot is one rigid body; a foot transmits a force,
no moment (sphere feet; for the pad approximately, the passive ankle carries almost no
moment).

**1. Equilibrium.** Internal forces (joints, servos) cancel in action-reaction pairs;
the external ones are gravity and the $F_j$. At rest, net force and net moment vanish.
Gravity's moment about any point $o$ is $M (c - o) \times g$ (from $M c = \sum_k m_k r_k$),
so about $o = c$ it drops out:

$$
\begin{aligned}
\sum_{j=1}^{3} F_j + M g &= 0 \\
\sum_{j=1}^{3} (p_j - c) \times F_j &= 0
\end{aligned}
$$

(A force $F$ at $p$ has the moment $(p - o) \times F$ about $o$: it is what changes the
angular momentum, $\dot L_o = \sum_k (r_k - o) \times F_k$.) In matrix form, with
$[v]_\times$ the cross-product matrix:

$$
\begin{aligned}
G \, F &= w, \qquad F = (F_1, F_2, F_3) \in \mathbb{R}^9 \\
G &= \begin{bmatrix} I_3 & I_3 & I_3 \\ [p_1 - c]_\times & [p_2 - c]_\times & [p_3 - c]_\times \end{bmatrix}, \qquad w = \begin{bmatrix} -M g \\ 0 \end{bmatrix}
\end{aligned}
$$

6 equations, 9 unknowns: for non-collinear feet $G$ has rank 6 and the solutions are

$$
F = F_0 + N z, \qquad z \in \mathbb{R}^3
$$

$F_0$ one particular solution (e.g. the minimum-norm one), $N$ spanning the null space of
$G$: the **internal forces** (feet squeezing towards / pulling apart from each other),
which change no net force or moment. The check searches over $z$: the stance holds if
*some* $z$ keeps every foot within its limits.

Flat-ground sanity check: $g = (0, 0, -g_0)$ and vertical $F_j = (0, 0, N_j)$ give
$c_{xy} = \sum_j N_j \, p_{j,xy} / \sum_j N_j$, the normal-force-weighted mean of the feet.
With $N_j \ge 0$ (no pulling) that is inside the support triangle: the old triangle
condition is the special case $A = 0$.

**2. Per-foot limits.** Split $F_j$ into normal and tangential parts:

$$
\begin{aligned}
N_j &= n_j \cdot F_j \\
F_{j,t} &= F_j - N_j \, n_j
\end{aligned}
$$

The environment acts on the foot through the magnet, a pull $-A \, n_j$ (on, pad flat and
in contact), and the surface contact $C_j$: normal part $R_j = n_j \cdot C_j \ge 0$ (can only
push), Coulomb friction $\lVert C_{j,t} \rVert \le \mu R_j$. With $F_j = C_j - A \, n_j$:

$$
\begin{aligned}
N_j &\ge -A && \text{(adhesion: the foot pulls at most } A) \\
\lVert F_{j,t} \rVert &\le \mu \, (N_j + A) && \text{(friction with adhesion)}
\end{aligned}
$$

Friction scales with the pressure the magnet adds, $R_j = N_j + A$; a foot hanging fully
on its magnet ($N_j = -A$) takes no shear. The first condition follows from the second, so
both are one second-order cone, the ordinary friction cone $K_\mu$ (half-angle
$\arctan \mu$ about $n_j$) with its tip moved from $0$ to $-A \, n_j$:

$$
F_j + A \, n_j \in K_\mu
$$

**3. Servo torques.** Leg $j$ hangs from the held body: joint angles $q_j \in \mathbb{R}^d$,
foot at $p_j(q_j)$, Jacobian $J_j = \partial p_j / \partial q_j$ ($3 \times d$), link centres
of mass $c_k$ with Jacobians $J_{c,k}$. Virtual work, for every small joint motion
$\delta q_j$:

$$
\tau_j \cdot \delta q_j + F_j \cdot J_j \, \delta q_j + \sum_k m_k \, g \cdot J_{c,k} \, \delta q_j = 0
$$

so

$$
\begin{aligned}
\tau_j &= -J_j^\top F_j - \sum_k J_{c,k}^\top m_k \, g \\
-\tau_{\max} &\le \tau_{j,k} \le \tau_{\max} \quad \text{for every joint } k \text{ of every stance leg } j
\end{aligned}
$$

Linear in $F_j$ at a fixed $P$: $2d$ inequalities per leg. The lifted leg $i$ carries only
its own weight ($\tau_i = -\sum_k J_{c,k}^\top m_k \, g$), a fixed check. The torques are
why $z$ matters twice: squeezing the feet helps friction but loads the joints.

All together: a second-order cone program in $F$ (linear equilibrium, three shifted cones,
linear torque bounds). Convex: a global answer, fast.

**Caveats.**

- $A$ is taken constant: a real magnet loses force with the air gap, and an edge contact
  holds less (in our model the cells not in contact pull nothing). The check assumes
  flat, fully touching pads.
- $\mu$ and $A$ equal for all feet here; different surfaces would give per-foot values.
- Soft servos: position servos with stiffness $k_p$ (10 N m/rad in the climb model) deflect
  by $\tau / k_p$ (2 N m: ~0.2 rad), so the real posture sags away from $P$ (the ~4 cm in the
  magnet test) and with it the geometry. The check is exact for stiff joints; for ours,
  iterate (torques -> deflection -> new posture) or leave it to the simulation checks.

**Implemented** in `statics.py` (2026-10-05): the margin as the largest gravity factor
$s^*$, an LP with friction pyramids, solved with `qpax`.

### Actuator torques for a given stance and body pose

Given a stance and a body pose, IK gives $q$, so the posture $P$ is fixed. The torques
follow from the foot forces (section 3 above), but the foot forces are not unique:
$F = F_0 + N z$ (3-dimensional for a tripod, 6 for four feet). So the torques are a family,
affine in $z$:

$$
\tau(z) = \tau_0 + K z, \qquad K = -\operatorname{blockdiag}(J_j^\top) \, N
$$

with $\tau_0$ the torques for $F_0$. Three ways to pick one, by question:

1. **Minimum-norm forces** ($z = 0$): `forces.stance_forces` returns these torques. A guess,
   reasonable for symmetric stances on the floor, without physical justification.
2. **Least torque**: the $z$ minimizing $\lVert \tau(z) \rVert^2$, closed form
   $z^* = -K^+ \tau_0$. With the cone constraints and the largest $|\tau|$ as objective, it is the cone program of (a).
   Answers "can it be held within the servo limits at all, and with what margin": the
   planner's question.
3. **What the real robot does**: with the feet held, the distribution is set by the
   compliance (each servo's deflection at stiffness $k_p$). The welded-feet simulation (b)
   gives it, servo sag included.

With position servos $z$ is not chosen directly: internal forces come from offsets between
servo targets and actual angles (pre-load). A controller could shape them (e.g. squeeze
the feet by commanding a slightly narrower stance); by default you get what the
compliance gives. So (2) says what is achievable, (3) what you get without trying.

**For the planner:** (2), the least achievable peak torque, as part of the hold margin.

### (a) vs. `controlkit/forces.py`

`forces.py` solves the same equations; it lacks the limits and the search over $z$.

Same:

- Equilibrium: `stance_forces` solves the base rows of $\sum_i J_{p,i}^\top f_i = $
  `qfrc_bias`. In MuJoCo's generalized coordinates the six free-joint rows are the net force
  and moment on the robot, and `qfrc_bias` at rest is gravity with the legs' mass
  included: its `M` is our $G$.
- Joint torques: its joint rows, $\tau = $ `qfrc_bias_joint` $- \sum_i J_{p,i,\mathrm{joint}}^\top f_i$,
  are our $\tau_j$ (leg weights from `qfrc_bias`).
- Everything from one `mjx.forward` (centre of mass, Jacobians, mass model); batched and
  differentiable.

Different:

| | `forces.py` | the hold check |
|---|---|---|
| which solution | one: the minimum-norm least squares $F_0$ (zero internal forces) | all $F_0 + N z$, searched for one within the limits |
| limits | none; reports forces and torques | adhesion, friction with adhesion, $\lvert\tau\rvert \le \tau_{\max}$ |
| verdict | "negative z-force: would tip" (floor normal = z, no pulling) | holds if some $z$ meets every limit; any normal, pulling feet |
| margin | none in this sense (`stance_sigma_min`, `stance_sensitivity`: degeneracy, sensitivity) | largest gravity factor, or smallest slack |
| gravity | the model's (`opt.gravity`) | any $g$ (`qfrc_bias` is linear in $g$ at rest) |

The minimum-norm forces can break a limit although other internal forces would hold, and
without limits there is no "holds" on a wall. So: build the hold check **on top of**
`forces.py`: take $G$ (the base rows), the joint-torque map and the gravity term from the
same `mjx.forward`, add the cone constraints and the optimization over $z$.

### Later: distill into a classifier

Learn $\mathrm{hold}(T, P, g)$, or better its margin, from examples labelled by (a), (b) or (c).

- Why: speed (thousands of candidates per step, batched in JAX); gradients (push B_lift,
  B_plant towards stable poses instead of only filtering); labelled by (c), it also
  learns the dynamic effects (a) misses.
- Risks: errors near the boundary are dangerous on a wall (a false "holds" is a fall), so
  use it as a conservative filter and verify the final plan with (a) or in simulation;
  coverage of postures, foothold geometries and gravity directions; the input
  representation (e.g. the feet of T and their normals in the body frame, g in the body
  frame, q, which leg is lifted).
- Order: build the exact check first (labels and the reference), distill when speed or
  the dynamic effects call for it.

## Reachability map (2026-10-05)

Is a foothold reachable? Today (`gait_graph/stance.py`): a ball of radius = total leg
length as prefilter (`candidates_at`), then per (foothold, leg) pair up to `reach_samples`
IK attempts (the 4-DOF leg is redundant: sampled hip yaws) with limit, ankle and collision
checks (`plant_table`); the expensive part. A reach box / ellipsoid is cheap but loose (the
real region has a hole near the shoulder, depends on joint limits, and the ankle must fit
the surface normal), and the planner needs the joint angles anyway (collisions, torques,
the full hold check). So: a precomputed **reachability map** as a tighter prefilter, IK
only on what passes. (In the trajectory NLP, a box stays the right choice: linear.)

- **Frame**: the shoulder frame. The legs are identical: one map, each leg looked up in its
  own shoulder frame; body pose changes cost nothing.
- **Per voxel**: not just "reachable", but which surface orientations work there. A bitmask
  over a set of surface-normal directions (64-128, evenly over the sphere, e.g. a Fibonacci
  sphere): bit k set if some valid leg configuration puts the foot in this voxel with the
  ankle angle to normal k within its limits.
  - Plus, per (voxel, normal), an ankle margin (how far within the ankle limits the best
    configuration is) rather than the ankle angles: the angles vary within a voxel, the
    margin is useful for scoring.
  - Optionally the joint angles of the best configuration, as a seed for the exact IK.
- **Building it**: forward kinematics over a dense grid of joint angles (within limits by
  construction). Each sample: foot position (its voxel) and tibia direction; normal k is
  admissible there if the angle between tibia and normal k is within the ankle limit.
  Leg-vs-body collisions can be included (the body is fixed in the shoulder frame); terrain
  collisions cannot. One-time, saved to a file.
- **Size**: `gait_graph` leg (0.6 m reach) at 2 cm: 60^3 = 216k voxels x 128 bits ~ 3.5 MB;
  the climb leg (0.4 m) at 1 cm similar.
- **Use**:
  - the prefilter in `candidates_at`, replacing the ball: foothold into the shoulder frame,
    look up voxel and normal bin (a gather in JAX: batched, cheap);
  - exact IK only on the survivors, for the witness posture;
  - a reach score: a distance transform of the map (how deep inside the workspace a
    foothold is) for scoring: prefer footholds that leave room for the next body shift.
- **Coarseness**: boundary voxels are partly reachable. For a prefilter, mark them
  optimistically (never reject a reachable foothold; the IK catches false positives).
- **Open**: which robot first, the `gait_graph` robot (v0; 4-DOF leg, redundant: more
  orientations per voxel) or the climb robot (3-DOF leg, pad ankle limits).

## Side note: motion planning as a nonlinear program

Moved to `trajopt-nlp.md` (trajectory optimization: what an NLP solver can take over, and a
formulation for the floor-to-wall transition).
