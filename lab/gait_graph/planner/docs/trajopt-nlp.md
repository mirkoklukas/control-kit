# Trajectory optimization as an NLP

Living note. Could an NLP solver plan the motion (climbing, or locomotion in general) if
the problem is formulated right? Companion to `planner-design.md` (the stance graph).

## What an NLP solver can take over (2026-10-05)

Largely yes: **trajectory optimization**, well studied for legged robots. The question is
which parts the NLP solver gets.

1. **Contact sequence given** (which foot moves when, onto which surface): optimize body
   poses, joint angles, foot positions and contact forces over time, subject to dynamics
   (full or centroidal), reach, friction / adhesion cones, torque limits. General NLP
   solvers (IPOPT, SNOPT, via CasADi) or DDP-style ones (Crocoddyl). Works well: the
   combinatorial part is fixed.
2. **Footholds as continuous variables** on known surfaces: TOWR (Winkler et al.)
   optimizes body motion, foothold positions and step timing together for a given gait
   pattern. Close to what we need.
3. **Contact sequence too** ("contact-implicit", Posa et al. 2014): contacts as
   complementarity constraints, the solver discovers when and where to step. In practice
   very nonconvex, sensitive to the initial guess, local minima.

Hard part: the **discrete choices** (which surface, which leg next, how many steps).
Gradient-based solvers do not make those well; graph search or mixed-integer formulations
(footstep planning over convex regions, Deits & Tedrake) do.

**Our case**, near-static climbing: a quasi-static version suffices and is much smaller. K
stances, per stance the body pose, footholds on their surfaces and the foot forces;
constraints: reach, static equilibrium, adhesion cones, torque limits (the hold check
becomes a constraint). A mid-sized NLP, doable in JAX: `mjx` for the gradients, a
penalty / augmented-Lagrangian method with L-BFGS, or IPOPT via `cyipopt`.

**How it would fit:** the graph search keeps the discrete structure (sequence, surfaces,
rough footholds); an NLP refines its plan (footholds and body poses for more margin, fewer
steps). The same split as planner -> MPC, one level up; the NLP needs the search's plan as
its initial guess anyway.

## A formulation for the floor-to-wall transition (2026-10-05)

Close to TOWR (Winkler, Bellicoso, Hutter, Buchli, RA-L 2018, "Gait and Trajectory
Optimization for Legged Systems through Phase-based End-Effector Parameterization"; project
page: https://www.alex-winkler.com/gato):

- **Variables as splines**: body position and orientation (polynomial splines over time);
  each foot's position (constant while in contact); contact forces (zero in the air);
  the phase durations.
- **Constraints**, at points along time: reach, dynamics or statics, terrain, friction
  (pyramids), solved with IPOPT; for a given number of steps it finds body motion,
  footholds and timing from a rough initial guess, typically well under a second.

What changes for us:

1. **Adhesion**: shift the friction cones by A (as in the hold check); magnet timing
   follows the contact phases.
2. **Heavy legs** (36% of the mass): a single rigid body puts all mass in the body.
   Approximate the centre of mass from body and feet instead, each leg's mass between its
   shoulder and its foot:

   $$
   c(t) \approx \frac{1}{M}\Big( m_b \, b(t) + m_\ell \sum_j \big( \alpha \, s_j(t) + (1 - \alpha) \, f_j(t) \big) \Big)
   $$

   (b body position, s_j shoulders, f_j feet, alpha fitted to the robot). Linear in the
   variables.
3. **Reach: a box.** TOWR's range-of-motion constraint is a box in the **base frame**,
   $\lvert R_B(t)^\top (f_j(t) - b(t)) - \bar p_j \rvert \le \bar b_j$ ($\bar p_j$ nominal foot
   position, $\bar b_j$ half-size per axis), checked at sampled times in stance and swing:
   it rotates with the body, linear in positions, nonlinear in the orientation (analytic
   Jacobian through the Euler angles). (Corrected 2026-10-08: an earlier version of this
   note called TOWR's box world-aligned.)
   - Simplification for us: a **world-aligned** box, f_j - b(t) in box_j: linear, convex.
     Exact only while the body orientation is roughly fixed; on the floor-to-wall
     transition the body pitches up to 90 deg and the reachable region turns with it. Keep
     it linear by orienting each box by a pre-planned body orientation (from the graph
     search, or a fixed floor-to-wall schedule; then not optimized), or by a smaller,
     conservative box valid over a phase's range of orientations.
   - Alternative (2026-10-08): replace the box by a signed distance to the actual reach set
     (from `lab/kinematics/reach-grid-limits.ipynb`), same base-frame input:
     $\mathrm{sdf}\big(R_B^\top (f_j - b) - s_j\big) \ge$ margin. Needs a smooth (C1)
     interpolant whose gradient is its true derivative, and sensible values outside the
     reach set and outside the grid.
     - Derivatives (2026-10-09): nothing hand-written. The lookup is JAX (array gather +
       interpolation weights), so `jax.jacfwd` of the inequality rows differentiates
       through it, the rotation and the body pose, as `refine.py` already does for the
       other rows. Catmull-Rom is C1 only, so use a quasi-Newton (L-BFGS) Hessian, not
       the exact one.
     - Findings (2026-10-09, `lab/kinematics/reach-grid-limits.ipynb`, resolution
       sweep 1 cm / 5 mm / 2 mm, projection NLP onto the reach set with SLSQP):
       - The grid comes from a yes/no mask per vertex, so the zero level is only good to
         about h (h: voxel size) and sits slightly *outside* the true boundary:
         projections land ~h/2 outside. Use **margin >= h**: with sdf >= h every
         projection was truly reachable (exact IK), with sdf >= h/2 only ~75%, although
         random points with sdf > h/2 all were (the solver ends on the boundary, where
         the bias is largest). Margins h, 2h, 3h cost no extra iterations.
       - Code: `lab/kinematics/reach_sdf.py` (`make_reach_sdf`, `ReachSDF.sdf`).
       - Catmull-Rom (cubic): ~17 SLSQP iterations at every h. Trilinear: 42 / 59 / 76;
         its gradient jumps at every cell face, more faces at finer h. Prefer Catmull-Rom.
       - Cost: 1 cm 2.8 MB, 0.3 s to build; 5 mm 19 MB, 1.5 s; 2 mm 274 MB, 19 s.
         Evaluation ~0.03 us per point at every h for local queries (an NLP's); random
         queries over a large grid are slower only from cache misses.
       - 5 mm looks like the sweet spot (5 mm margin).
4. **Surfaces: one height map along a tilted direction.** Seen along v at 45 deg between
   the floor's and the wall's normal, the inside corner (L shape) is single-valued: every
   point u of the plane perpendicular to v sees one surface point. "Foot on the surface"
   becomes h(u_j) = d_j, one constraint for both surfaces, no assignment to choose.
   - The corner is a kink: round it (a few cm) for gradients; keep feet out of a band
     around the crease.
   - The surface normal (and with it the friction cone) jumps at the corner; the rounded
     corner turns it quickly but smoothly.
   - Only where the surface is a graph along v: not for an outside corner, an opening, or an
     overhang along v (those need the surface assignment again).
5. **No fixed gait pattern (partly)**: phase-based parameterization. Each foot has
   alternating stance / swing phases with their durations as variables; timing and the
   order of the feet come out of the optimization. Still fixed: the number of phases per
   foot (steps each foot takes) and whether it starts in contact. "At least three feet
   attached" is not automatic (the number of feet down at time t is not smooth in the
   durations); on a wall the equilibrium constraints usually force enough feet down; for
   a strict crawl, fix the order.
6. **Quasi-static or dynamic**: slow climbing allows dropping the dynamics (static
   equilibrium at each time point): smaller, better behaved. Dynamics for faster gaits on
   the floor.
7. **Initial guess**: a local optimum near the start; the graph search's plan is the natural
   initial guess.

Together: splines for body and feet, phase durations as variables, reach boxes (TOWR's
base-frame boxes, or world-aligned ones oriented by a planned body orientation), one tilted height map with a rounded corner,
shifted friction cones, the centre of mass from body and feet. With the sequence and surfaces
from the graph search, it would replace the hand-made sampling of B_lift, B_plant, f' by a
joint optimization and give smooth trajectories for the MPC.

Tools: CasADi + IPOPT (easiest start); a JAX model with `cyipopt` (reuses `statics.py`);
TOWR itself (C++) as a reference implementation (project page:
https://www.alex-winkler.com/gato).

## A quasi-static keyframe NLP, full kinematics (2026-10-07)

The "our case" version above, worked out: no time, no dynamics, a fixed contact order
(crawl), a sequence of isolated static postures. Unlike TOWR, the legs are modelled, so
reach needs no boxes, and the statics are the full model's (`statics.md`).

**Keyframes.** $k = 1, \dots, K$; per step four: B_lift (four feet down), lifted, B_plant
(swing leg still up), planted. The planted set $P_k$ at each keyframe is fixed by the
crawl order. Each foot's **contact segment** (from planting to the next lift) has one
foothold, shared by every keyframe in that segment.

**Terrain as a height function.** The surface as a graph over a domain
$\Omega \subset \mathbb{R}^2$ along a projection direction (for floor and wall the tilted
one of item 4 above):

$$
\begin{aligned}
s(u) &= \big(u,\ h(u)\big) && \text{surface point (in the projection frame)} \\
n(u) &= \frac{(-\nabla h(u),\ 1)}{\lVert(-\nabla h(u),\ 1)\rVert} && \text{unit normal}
\end{aligned}
$$

A foothold is a point $u \in \Omega$ of the domain, not a 3D point: two variables, on the
terrain by construction, no planting constraint (the alternative, a 3D foot position with
the constraint "normal coordinate $= h$(tangential coordinates)", has three variables and
one equality).

**Variables.**

| variable | per | size |
|---|---|---|
| $B_k$ body pose | keyframe | 6 |
| $\theta_k$ servo angles | keyframe | 12 |
| $F_k$ foot forces of $P_k$ | keyframe | $3 \lvert P_k \rvert$ |
| $u_c$ foothold | contact segment | 2 |
| $F_k^e$ disturbance forces (below) | keyframe, direction $e$ | $3 \lvert P_k \rvert$ each |

**Constraints**, per keyframe $k$, $q_k = (B_k, \theta_k)$, $u_j$ the foothold of foot
$j$'s current segment, $r$ the pivot's offset from the pad face:

$$
\begin{aligned}
& x_j(q_k) = s(u_j) + r\, n(u_j), \quad j \in P_k && \text{the leg reaches its foothold} \\
& \theta_{\min} \le \theta_k \le \theta_{\max}, \quad u_j \in \Omega && \text{joint limits, domain} \\
& \angle\big(\text{tibia}_j(q_k),\ n(u_j)\big) \le \alpha_{\max} && \text{ankle limit} \\
& G(q_k)\, F_k = b_b(q_k) && \text{equilibrium} \\
& \tau_k = b_a(q_k) - J_a(q_k)^\top F_k, \quad \lvert \tau_k \rvert \le \tau_{\max} && \text{servo torques} \\
& F_{k,j} + A\, n(u_j) \in K_{\mu_p}\big(n(u_j)\big) && \text{adhesion-shifted friction pyramid} \\
& d\big(\text{links}(q_k),\ \text{terrain}\big) \ge 0, \quad d\big(\text{links}(q_k),\ \text{body}\big) \ge 0 && \text{clearance} \\
& \text{swing foot of a lifted keyframe: } \ge \ell \text{ above the surface} && \text{lift height}
\end{aligned}
$$

- **Reach is exact.** A foothold is reachable iff some $\theta$ within limits satisfies
  the kinematics equality; with $\theta$ a variable there is no reachability function to
  build. For 3-DOF legs the forward kinematics are cheap and smooth.
- **Stability as constraints.** The disturbance gate ("resists a push or twist of
  $\lambda_{\min}$ in each of the 12 directions", `dist >= dist_min`) as scenario
  constraints: for each direction $e$, forces $F_k^e$ with
  $G(q_k) F_k^e = b_b(q_k) + \lambda_{\min} e$, within the cones and $\tau_{\max}$. Linear in
  the $F_k^e$; exact; no min / max to differentiate.
- **Transitions.** The robot moves between keyframes (the body shifts with the feet held,
  the swing leg's path). Add a few interior knots per shift (the sampling planner checks
  2), interpolated, with the same constraints: the quasi-static path holds too.
- **Clearance** against planes or a height map is cheap (link points vs. $h$); between
  links and body, capsule distances.

**Objective.**

$$
\min \sum_k \lVert \tau_k \rVert^2 \;-\; w_{\text{prog}} \cdot \text{progress} \;+\; w_{\text{smooth}} \sum_k \lVert q_{k+1} - q_k \rVert^2
$$

(progress: the last body pose along the walking direction; or fix the number of steps and
a target and drop the term).

**Size.** 8 steps x 4 keyframes x (6 + 12 + 12) is about 1000 variables, about 13000 with
the 12 disturbance scenarios per keyframe. Small for IPOPT.

**Notes.**

- **The internal forces are chosen** (they are inside $F_k$): realizable only with
  feedforward of the planned torques, $\theta^{\text{t}} = \theta + \tau / k_p$
  (`statics.md`, section 5). Without it the robot gets the least-torque internal forces
  instead.
- **Nonconvex**: $J(q)^\top F$ is bilinear, the kinematics and $n(u)$ nonlinear. A local
  optimum near the initial guess; the `plan_step` walk is a ready one (body poses, joint
  angles, footholds; forces from `least_torque_qp`).
- **Requirements on $h$**: $C^1$ at least (gradients, and normals that do not jump:
  round the corners); single-valued along the projection direction (no overhangs, outside
  corners). Holes and gaps become steep or masked regions: the optimizer can get stuck on
  the wrong side, so which region a foot goes to still comes from the graph search (the
  warm start).
- **Not modelled**: the pad as a patch (centre-of-pressure limit), compliance and sag
  beyond the feedforward, MuJoCo's soft contacts. Check the result in simulation
  (`execute.py`).

**Tools.** JAX for the kinematics, Jacobians and statics (`statics.py`, `climb_statics.py`);
`cyipopt` for IPOPT, gradients and Hessians of the Lagrangian from JAX. A first prototype:
two steps on the floor ($h = 0$), warm-started from a `walk` run.

## Planning / refining NLP over transfers: from B to B' (2026-10-08)

The tripod-graph view (`planner-design.md`, "the tripod view, transfers and swings"), as an
NLP (nonlinear program). One transfer $(T, S, T')$: the planting leg lands from
$B_{\text{plant}}$ (closing $S$), the body moves on the four feet of $S$ to $B_{\text{lift}}$,
the lifting leg lifts (opening $T'$); the two transfer legs stay planted throughout.

**Given**: the start body $B$, the goal body $B'$, the gait order, the number of transfers
$N$ (and, optionally, the start footholds).

**Variables**

| item | per | size |
|---|---|---|
| feet of the first tripod $T$ | once | 3 x 3 (free; can be fixed to the current footholds later) |
| $B_{\text{plant}}$, $B_{\text{lift}}$ | transfer | 6 + 6 |
| planting foot (the planting leg's new foothold) | transfer | 3 |
| joint angles $\theta$ at each body (option a) | transfer | 12 + 12 |

Every foot is a free 3D point, on the terrain by the constraint $z = h(x, y)$ (TOWR style:
iterates may leave the surface, IPOPT drives the violation to zero). In a sequence the
transfer feet and the lifting foot of a transfer are earlier transfers' planting feet, and
$B$ of transfer $k$ is $B_{\text{lift}}$ of transfer $k - 1$. Size: $9 + 15 N$, or
$9 + 39 N$ with explicit joint angles.

**Joint angles**: (a) explicit, with forward-kinematics equalities
$x_j(B, \theta) = f_j + r\, n(f_j)$ ($f_j$ the foothold, $r$ the pivot height above the pad
face): general, infeasible iterates allowed; or (b) eliminated, $\theta = \mathrm{IK}(B, f)$
with the closed-form 3-DOF inverse kinematics on a fixed branch: fewer variables, smooth
only away from the reach boundary.

**Constraints**, per body:

- kinematics: the planted feet reached (option a: the equalities above); joint limits; the
  ankle cone (tibia within $\alpha_{\max}$ of the normal); clearance (signed distances
  $\ge d_{\min}$: legs and body vs. terrain, legs vs. body);
- stability, the fast model as smooth constraints (below), for the supporting tripod:
  $T$ at $B$ and $B_{\text{plant}}$, $T'$ at $B_{\text{lift}}$; the node path
  $B \to B_{\text{plant}}$ at interpolated knots;
- the end: $B_{\text{lift}}$ of the last transfer $= B'$ (or a cost on the distance).

**Stability as smooth constraints.** The least-torque split (the servos' own internal
forces, `statics.md` section 5) solves

$$
F^\star(q) = \arg\min_F \lVert b_a(q) - J_a(q)^\top F \rVert^2 \quad \text{s.t.} \quad G(q)\, F = b_b(q)
$$

whose KKT (Karush-Kuhn-Tucker) conditions, with a multiplier $\mu \in \mathbb{R}^6$ for the
six equilibrium equations, are one linear system:

$$
\begin{bmatrix} J_a J_a^\top & G^\top \\ G & 0 \end{bmatrix}
\begin{bmatrix} F^\star \\ \mu \end{bmatrix}
=
\begin{bmatrix} J_a b_a \\ b_b \end{bmatrix}
$$

All entries are smooth in $q$, so $F^\star(q)$ is smooth wherever the matrix is invertible
($G$ of rank 6: feet not on a line; no kinematic singularity changing the rank of the
torque map). The pushed splits $F^e$ (gravity plus a push $\lambda e$, 12 directions)
solve the same system with $b_b + \lambda e$: $F^e = F^\star + H(q)\, \lambda e$, one
factorization for all. Then, per case (undisturbed + 12 pushes) and planted foot, the
pyramid rows (linear in $F$) and $|\tau_k| \le \tau_{\max}$ (linear) are smooth
inequalities, about 660 per body for a tripod; or the round-cone margin
$d = (\mu a - t) / \sqrt{1 + \mu^2} \ge m$ with the tangential size $t$ smoothed as
$\sqrt{t^2 + \epsilon^2}$. No min over feet or cases: each condition is its own
constraint. (Not the LP margins: continuous in $q$ but only piecewise smooth. For chosen
internal forces instead, the force sets per case become variables: bilinear, smooth,
optimistic.)

**Objective**: $\sum \tau^2$ (effort) over the bodies; smoothness between consecutive
bodies; (progress, if $B'$ is a cost rather than a constraint).

**The number of transfers $N$** is discrete: an outer loop, $N = 1, 2, \dots$, the smallest
feasible (or the best by cost); with the gait order it fixes which leg plants when.
**Warm start**: the sampling planner's sequence for the same $B \to B'$, or bodies
interpolated from $B$ to $B'$ with nominal feet (an infeasible start is fine for IPOPT).

**First implementation**: a single transfer.

**Single transfer, first results (2026-10-08)** (`refine.py`, `experiments/refine.py`, SLSQP
with JAX gradients; floor, no adhesion, push 0.1 x weight, the first crawl transfer 1 -> 0,
warm-started from `plan_transfer`):

- B_lift free (a trust region of 3 cm / 0.15 rad around the warm start): feasible, the
  independent checks pass; sum tau^2 4.67 -> 3.19 (-32 %), the push margins traded down
  (0.55 / 0.32 -> 0.36 / 0.14 N: the objective is effort only, the rows ask margin >= 0);
  hit the iteration limit (200, ~37 s, ~0.18 s per iteration, mostly the constraint
  Jacobian through the statics).
- B_lift fixed to B' = B + 6 cm along +x (same height and sideways position): infeasible
  (violations ~5 cm in reach, ~0.05 in the stability rows). The fixed pose is too strict:
  supporting the next tripod without adhesion needs the body to shift sideways and drop.
  Fine for now; better later: only a progress condition (x of B_lift - x of B >= advance),
  the joint angles at B_lift initialized by IK.

## TOWR-style transfer, v0 (2026-10-09)

`towr_transfer.py`, `experiments/towr_transfer.py`. One transfer $(T, S, T')$ from a fixed
body B: bodies B (fixed, on T), B_plant (on T, reaches S), B_lift (on T', reaches S), B'
(on T', reaches T'). Closer to TOWR than the v0 refine above: **no joint angles**.

- Variables (57): B_plant, B_lift, B' (position + roll / pitch / yaw), the new foothold f
  (3D, on the floor by $f_z = 0$), the supporting feet's **forces** at all four bodies (3 x 3
  each).
- Reach: per body and foot, $\mathrm{sdf}(s_j(B)^{-1}(f_j + r\,n_j)) \ge$ margin (reach grid,
  5 mm, margin 5 mm; `lab/kinematics/reach_sdf.py`).
- Equilibrium: $G(c, \text{feet}) F = b_b$ per body, $c$ from `com_approx`: a leg's mass at
  $\alpha$ shoulder + $(1 - \alpha)$ foot, $\alpha = 0.525$ fit against the kinematic centre of
  mass (median error 6 mm over random postures). Legs in the air: leg i at B at its old
  foothold, leg i' at B' tucked (a fixed point in its shoulder frame).
- Limits: friction pyramid, no adhesion; each supporting foot's normal force $\ge$ 0.1 x its
  share of the weight; f at least 8 cm from the other footholds; body box corners >= 2 cm
  above the floor.
- Objective: max $x(B') - x(B)$ + small regularizers.
- Verification afterwards with joint angles: exact IK for every foot a body must reach, the
  real centre of mass from those angles, an exact equilibrium LP with it.

**First result** (floor, no adhesion, gravity only, transfer 1 -> 0, B 5 cm towards T's
centroid): progress **17.9 cm**; SLSQP 1000 iterations in 4 s (iteration limit, but
feasible: violations ~1e-6); every body verified (IK reaches all, holds with the true
centre of mass; centre-of-mass error 2-13 mm). Without the normal-force margin the optimum
put the whole weight of B' on one foot (a knife edge that failed with the true centre of
mass); without the foot separation f went under the body.

Open: roll / pitch sit at their bounds (+-20 deg; tilting extends the reach); B' still
leans on one foot (normal forces 0.9 / 0.9 / 25.6 N, the margin at its bound); no pushes,
no ankle cone, no leg collisions, no paths between the bodies; SLSQP does not report
convergence.

## Pitfalls seen so far (2026-10-09)

From the TOWR-style transfer runs (sdf and joint-angle variants). The optimizer finds every
gap in the model:

- **Collisions are not modelled**: self-collisions (legs crossing each other, the planting
  leg reaching past its neighbour, legs vs. body) and collisions with the ground (links,
  knees through the floor). Checked nowhere in the NLP; the solutions use it.
- **Knife-edge equilibrium**: maximizing progress puts the centre of mass over one foot (all
  the weight on it, zero margin). Needed a minimum normal force per foot, better a push
  margin.
- **Tilting and dropping the body** to extend the reach: roll / pitch at their bounds
  (+-20 deg), the body down to 8-10 cm. On the floor an exploit; for climbing possibly a
  feature later (pitching the body up at the floor-to-wall transition, lowering it to reach
  a far foothold), so bound or penalize it per scenario rather than forbid it.
- **The bounds decide the optimum**: progress = the translation bound (25 cm); the answer
  is the box, not the physics.
- **Footholds in odd places**: the new foothold under the body, next to another foot, or
  crossing in front of a neighbour; two feet side by side. Foot separation alone does not
  prevent it.
- **Feasible forces vs. the servos' forces**: with the foot forces as variables, the NLP
  proves *some* forces hold the stance; the least-torque split (what the servos settle at)
  can still fail (full push score negative) while the min-norm push margin is met.
- **A fixed B is part of the problem**: placed by a heuristic it was near a support edge
  (one foot at 3.7 N), over the incenter its ankles broke the 45 deg cone; nothing checks
  a fixed pose.
- **Approximations bite at the boundary**: the reach grid's zero level is ~h/2 outside the
  true boundary (use margin >= h); the joint-angle-free centre of mass is off by 6-18 mm;
  a leg in the air needs some position for the centre of mass.
- **The ankle cone sits at its bound** (45 deg) whenever reach is tight.
- **Solver**: SLSQP never reports convergence (always the iteration limit); ~19 ms per
  iteration, mostly the JAX callbacks (the objective gradient recomputing the whole model,
  ~5.5 line-search evaluations per iteration). A GPU does not help (small sequential calls:
  77 s on an A10 vs. 19 s on the laptop CPU).

## References

Papers:

- **[Winkler 2018]** A. W. Winkler, C. D. Bellicoso, M. Hutter, J. Buchli, "Gait and
  Trajectory Optimization for Legged Systems through Phase-based End-Effector
  Parameterization", IEEE Robotics and Automation Letters (RA-L), 2018. TOWR. Project
  page: https://www.alex-winkler.com/gato; code: https://github.com/ethz-adrl/towr
- **[Posa 2014]** M. Posa, C. Cantu, R. Tedrake, "A direct method for trajectory
  optimization of rigid bodies through contact", International Journal of Robotics
  Research, 2014. Contact-implicit trajectory optimization.
- **[Deits 2014]** R. Deits, R. Tedrake, "Footstep planning on uneven terrain with
  mixed-integer convex optimization", IEEE-RAS Humanoids, 2014. Footsteps over convex
  regions.

Solvers and tools:

- **IPOPT**: interior-point NLP solver, https://github.com/coin-or/Ipopt
- **cyipopt**: Python interface to IPOPT, https://github.com/mechmotum/cyipopt
- **CasADi**: symbolic modelling and automatic differentiation for NLPs,
  https://web.casadi.org
- **SNOPT**: SQP solver for large sparse NLPs (commercial), https://ccom.ucsd.edu/~optimizers/
- **Crocoddyl**: DDP-style optimal control for legged robots,
  https://github.com/loco-3d/crocoddyl
