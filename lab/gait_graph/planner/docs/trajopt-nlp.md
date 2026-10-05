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
3. **Reach: a world-aligned box placed relative to the body**, f_j - b(t) in box_j: linear,
   convex, easier than a box rotating with the body. Exact only while the body orientation
   is roughly fixed; on the floor-to-wall transition the body pitches up to 90 deg and the
   reachable region turns with it. Keep it linear by orienting each box by a pre-planned
   body orientation (from the graph search, or a fixed floor-to-wall schedule; then not
   optimized), or by a smaller, conservative box valid over a phase's range of
   orientations.
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

Together: splines for body and feet, phase durations as variables, world-aligned reach boxes
oriented by a planned body orientation, one tilted height map with a rounded corner,
shifted friction cones, the centre of mass from body and feet. With the sequence and surfaces
from the graph search, it would replace the hand-made sampling of B_lift, B_plant, f' by a
joint optimization and give smooth trajectories for the MPC.

Tools: CasADi + IPOPT (easiest start); a JAX model with `cyipopt` (reuses `statics.py`);
TOWR itself (C++) as a reference implementation (project page:
https://www.alex-winkler.com/gato).
