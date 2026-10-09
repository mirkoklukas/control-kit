# Policy and Reward Terms

Sep 23, 2026 · @Mirko Klukas

## References

- Hwangbo et al., [Learning Agile and Dynamic Motor Skills for Legged Robots](https://arxiv.org/abs/1901.08652), Science Robotics, 2019. Basis for the reward structure and policy setup.
- Joonho Lee et al., [Learning Quadrupedal Locomotion over Challenging Terrain](https://arxiv.org/abs/2010.11251), arXiv, 2020.
- Schulman et al., [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477), ICML, 2015.
- Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347), arXiv, 2017.
- [Walk These Ways: Tuning Robot Control for Generalization with Multiplicity of Behavior](https://arxiv.org/pdf/2212.03238)
- Learning To Walk in Minutes Using Massively Parallel Deep Reinforcement Learning, [legged gym (project page)](https://leggedrobotics.github.io/legged_gym/), [Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning](https://arxiv.org/pdf/2109.11978), [https://proceedings.mlr.press/v164/rudin22a.html](https://proceedings.mlr.press/v164/rudin22a.html)

## Policy

The controller is a small MLP that maps sensor readings and a velocity command to joint position targets and magnet commands, following the paper's design.

**Architecture:** 2 hidden layers (256, 128), tanh activation. Bounded activations keep actions sane in states not seen during training.

**Inputs:**

- gravity direction in the body frame (IMU)
- base linear and angular velocity (state estimator)
- joint positions and velocities
- joint state history: position error and velocity at t − 0.01 s and t − 0.02 s
- previous action
- velocity command in the surface plane: forward, lateral, yaw
- ankle hinge angles, 2 per foot (encoders on the passive ankle)
- per-foot attached state (Hall sensors); in the floor env: planted flags
- foot positions relative to the base, body frame (forward kinematics from the encoders)
- pad heights above the surface (floor env). Not measured on the robot (needs the body height or a terrain estimate): a candidate for the privileged critic inputs instead

**In the floor env** (`WalkEnvCfg.obs`, layouts in the `env.py` docstring): `v1` (101, default) follows Hwangbo et al. 2019 ([arXiv:1901.08652](https://arxiv.org/abs/1901.08652)): gravity, base height, base velocities, joint positions (relative to `stand`) and velocities, joint history (position error and velocity one and two policy steps back, t − 0.02 s and t − 0.04 s at 50 Hz, instead of the paper's t − 0.01 s and t − 0.02 s), previous action, command, plus foot heights above the floor (new, not in the paper). `v2` (76) replaces base height and joint history with ankle angles, body-frame foot positions and planted flags.

**Privileged critic inputs (sim only):** adhesion fraction α per foot, pad contact forces, load margin per foot. Asymmetric actor–critic: the critic sees these, the actor only what the real robot measures.

**Outputs:**

- 12 joint position targets, converted to torque by the actuator model in sim (PD with delay and limits, later an actuator network) and by the motor drivers on hardware
- 4 magnet on/off commands, one per foot

**Rate:** 50–100 Hz for the policy; the actuator model and physics run at the sim time step.

**Policy optimization candidates:**

- TRPO: used in the paper. Hard KL-divergence constraint per update, stable, needs little tuning, but complex and sample-inefficient.
- PPO: clipped surrogate objective, plain first-order optimization with several epochs per batch. Standard for legged locomotion today and available in rsl\_rl and Brax/MuJoCo Playground.

## Reward

The reward is a weighted sum of terms, following the structure of Hwangbo et al. (2019): task terms define what to achieve, regularization terms define how, and adhesion terms keep the robot on the wall. All weights are per-step and multiplied by the time step Δt. Weights marked "start" are initial guesses to tune, not paper values.

Convention: rewards are positive and added, costs are positive and subtracted. Only the two tracking terms are rewards; everything else is a cost.

$$
r_t = \sum_i w_i \, R_i \; - \; \sum_j w_j \, C_j
$$

Tracking errors pass through a bounded logistic kernel instead of a squared norm, so a large early error never makes falling the cheapest option. It is normalized to 1 at zero error, so a tracking weight is the maximum reward per second:

$$
K(x) = \frac{4}{e^{x} + 2 + e^{-x}} = \operatorname{sech}^2\!\left(\frac{x}{2}\right), \quad K \in (0, 1], \text{ maximal at } x = 0
$$

Hwangbo et al. use the same kernel without the factor 4 (peak 0.25). Their tracking weights (10 and 6) therefore correspond to 2.5 and 1.5 here.

Regularization terms are scaled by a curriculum factor k\_c that starts at 0.3 and grows as k\_c ← k\_c^0.997 per iteration. Task and adhesion-safety terms are never scaled, so safety dominates from the first iteration.

## Task terms (rewards)

The policy tracks a velocity command in the wall plane, expressed in the body frame, so the same command works on floor, wall and ceiling.

### Linear velocity tracking

$$
R_{lin} = K\left(4 \cdot \lVert v_{xy} - \hat{v}_{xy} \rVert\right)
$$

**Weight:** +5.0 (reward) · not curriculum-scaled · source: paper's 10 is 2.5 with our normalized kernel; 2.5 was below the gait costs (runs settled on standing), so doubled. In the env the sharpness is 40, not 4: at a 0.1 m/s command, standing still then earns ~7% of the max instead of ~99%.

v\_xy is the base velocity projected onto the surface tangent plane, expressed in the body frame; v̂\_xy is the command. The factor 4 sharpens the kernel: an error of 0.25 m/s already drops the reward from 1 to about 0.79.

**Setup needed:**

- Surface frame in sim: take the normal n̂ of the surface under the robot (the wall geom, or the average contact normal of attached feet) and project velocities onto the plane orthogonal to it.
- Command ranges: a crawl gait is slow, so start narrow, for example forward ±0.3 m/s, lateral ±0.15 m/s (start values). Widen once the policy tracks reliably.
- Sample zero commands in 10–20% of episodes so the robot learns to hold still on the wall.

### Yaw rate tracking

$$
R_{yaw} = K\left(\lvert \omega_n - \hat{\omega}_n \rvert\right)
$$

**Weight:** +3.0 (reward, start) · not curriculum-scaled · source: paper, raised (the paper's 6 is 1.5 with the normalized kernel; doubled to hold the heading better)

ω\_n is the base angular velocity about the surface normal n̂, not about world z. On a wall, "turning" means rotating in the wall plane.

**In the env:** the error is scaled by a sharpness of 10 (rad/s)⁻¹: R_yaw = K(10 · |ω_n − ω̂_n|). Without it (sharpness 1) the kernel is flat: a 0.3 rad/s error still keeps 98% of the reward, so the heading drifted freely. At 10, 0.1 rad/s keeps ~79% and 0.3 rad/s ~17%.

**Setup needed:** start with a yaw command range of ±0.5 rad/s (start value).

## Regularization terms (costs)

All are costs. These shape how the robot moves. All are multiplied by the curriculum factor k\_c.

### Torque

$$
C_{\tau} = \lVert \tau \rVert^2
$$

**Weight:** 0.005 (cost) · curriculum-scaled · source: paper

Penalizes energy use. On a wall the robot needs holding torque just to stay put, so this term is never zero. If it pushes the body away from the wall to relax the joints, lower it: the body-distance and adhesion-margin terms matter more.

**Setup needed:** if hip and knee motors differ in rating, normalize each joint by its rated torque.

### Joint speed

$$
C_{\dot{\phi}} = \lVert \dot{\phi} \rVert^2
$$

**Weight:** 0.03 (cost) · curriculum-scaled · source: paper

Discourages thrashing and fast leg swings. No extra setup.

### Smoothness

$$
C_{smooth} = \lVert \tau_{t-1} - \tau_t \rVert^2
$$

**Weight:** 0.5 (cost) · curriculum-scaled · source: paper

Penalizes abrupt torque changes, which excite vibrations and can jolt a foot loose. A common alternative in newer work is the action rate ‖a\_t − a\_t−1‖² on the position targets. Either works; no extra setup.

### Foot clearance

$$
C_{clear} = \sum_{i \in \text{swing}} (\hat{h} - h_i)^2 \, \lVert v_{t,i} \rVert
$$

**Weight:** 0.1 (cost) · curriculum-scaled · source: paper, adapted

h\_i is the height of foot i above the surface along n̂, and v\_t,i its speed tangential to the surface. The speed factor means the cost applies only while the foot travels, not while it lifts or lowers in place.

**Setup needed:**

- Pick ĥ from your leg geometry and expected surface features (welds, bolt heads): start at 5 cm.
- Compute h\_i in sim with a ray cast from the foot along −n̂ (MuJoCo `mj_ray`).

### Hard clearance

$$
C_{hard\_clear} = \sum_{i \,:\, h_i < h_{min}} \max\left(0,\; \lVert v_{t,i} \rVert - v_{tol}\right)
$$

**Weight:** 0 (cost; 1 in `configs/crawl.yaml`) · h\_min = `hard_clear_height` 1 cm, v\_tol = `hard_clear_vtol` 0.01 m/s · scheduled like slip and drag (from 0.4 of full weight) · source: new

A foot below h\_min must not move along the surface (beyond v\_tol, for settling noise), whatever its contact state: planted, touching, or skimming just above the surface. "Hard" is only the name: a step in height, no cost at or above h\_min.

**Why:** the other foot terms leave a gap. Slip covers planted feet and drag touching ones; a foot hovering a few mm up touches nothing, and C\_clear near h = 0 is only ĥ² ‖v\_t‖ (0.36 ‖v\_t‖ at ĥ = 6 cm, weight 100). So a foot could skim along the floor almost for free.

**Overlap:** all feet count, so a sliding planted foot pays slip and this term, a touching one drag and this term.

### Foot slip

$$
C_{slip} = \sum_{i \in \text{attached}} \left( \lVert v_{t,i} \rVert + r_{pad} \, \lvert \omega_{n,i} \rvert \right)
$$

**Weight:** 10 (cost, start) · curriculum-scaled · source: paper uses 2.0, raised

An attached pad should not move. v\_t,i is its sliding velocity along the surface; ω\_n,i its rotation about the surface normal. The Cardan ankle cannot yaw, so any leg twist goes straight into the pad, resisted only by friction. r\_pad = pad\_size / 2 converts the rotation into an edge speed so both parts have the same units.

### Foot drag

$$
C_{drag} = \sum_{i \in \text{contact} \setminus \text{attached}} \lVert v_{t,i} \rVert
$$

**Weight:** 2 (cost, start) · curriculum-scaled · source: new

A foot that touches the surface without being attached (magnet off, still switching, or only an edge or corner in contact) should not move along it. v\_t,i is its speed along the surface. This covers the gap between the other foot terms: slip applies only to attached feet, clearance only to feet in the air, so a foot scraping along the surface — at lift-off, or a swinging foot hanging too low — is otherwise free. Dragging wears the pad and, on steel, the paint, and on a wall it drags the pad's edge across welds and bolt heads.

**Setup needed:**

- Contact per foot: any pad cell in contact with the surface (in sim, from the contact list; on hardware, the Hall sensors or the ankle encoders).
- Before magnets and attachment are modelled (floor, magnets off), "attached" can be approximated by load: a foot carrying more than a small normal force counts as planted, and the drag term then applies to feet in light contact below that threshold, while slip applies above it.

### Orientation

$$
C_{orient} = \lVert \hat{n} - z_{body} \rVert
$$

**Weight:** 0.4 (cost) · curriculum-scaled · source: paper, adapted

The paper kept the body level with gravity. Here the body's z-axis should align with the surface normal n̂, so the body stays parallel to the wall.

**Setup needed:**

- Define n̂ on curved surfaces as the average contact normal of the attached feet.
- Transitions (floor to wall, wall to ceiling) have two normals. Use the normal under the majority of attached feet, or blend them by foot count, so the target rotates smoothly during the transition.

### Body distance

$$
C_{dist} = (\hat{d} - d)^2
$$

**Weight:** 1.0 (cost, start) · curriculum-scaled · source: new

d is the distance from the base to the surface along n̂. On a wall, gravity acts at the center of mass, and the peel moment on the feet grows with d. A lower body means less pull-off force on the upper feet.

**Setup needed:**

- Choose d̂ as the lowest height that still clears the body and knees over expected surface features. Check in CAD.
- Compute d in sim with a ray cast from the base along −n̂.

## Adhesion terms (costs)

All are costs, none are curriculum-scaled, and most need bench measurements before the weights mean anything.

### Foot model

Each foot has a passive Cardan ankle (two hinges, weak centering springs, hard stops at ±q\_max) and a square magnetic pad. In sim, the pad is split into N × N cells, N\_c = N² per foot, each with its own adhesion actuator of gain A / N\_c. All cells of a foot share one magnet command m\_i ∈ {0, 1}: the cells are a sim device, the real foot has one magnet. Current design: N = 3 (`pad_cells`).

### Definition: adhesion fraction

$$
\alpha_i = \frac{n_{contact,i}}{N_c}, \qquad A_{eff,i} = \alpha_i \, m_i \, A
$$

n\_contact,i is the number of cells of foot i in contact. A flat pad has α = 1, an edge 1/N, a corner 1/N². The foot's effective holding force scales with α.

**Setup needed:** on hardware α is not measured directly; 2–3 Hall sensors across the pad can approximate it. Until then α is a privileged (critic-only) quantity.

### Definition: attached foot

$$
\text{attached}_i = m_i = 1 \,\wedge\, \text{switch complete} \,\wedge\, \alpha_i = 1
$$

A foot is attached when its magnet is on, the switch has finished, and all N\_c cells are in contact. A foot on an edge or corner is not attached.

**Setup needed:** measure the EPM switching time on the bench and model it in sim.

### Minimum support

$$
C_{support} = \max\left(0,\; n_{min} - n_{planted}\right), \qquad n_{min} = 3
$$

**Weight:** 5 (cost; 2 in `configs/crawl.yaml`) · scheduled: 0 until 500k training steps, then linearly to full over 2M · source: new

At least 3 feet attached with full adhesion (α = 1) at all times. With 4 legs this forces a crawl gait on the wall: one leg moves at a time, and the next foot must attach before another detaches. Dropping below 2 attached feet terminates the episode.

**In the env (floor, magnets off):** a foot counts if it is *planted*: normal force above `contact_force_min` (1 N) and pad face within `pad_tilt_max_deg` (3°) of the contact normal. On the wall this becomes "attached".

**Graded, not 0/1:** one foot short costs 1, two cost 2. The original indicator treated a near-crawl (occasionally one foot short) like a trot (often two short), so it gave no gradient towards the crawl; weighted at 10–20 it made not stepping at all the cheapest option.

**Ramped in:** at full strength from the start the policy never learns to step. It is off until the policy walks, then ramped in, so that a trot turns into a crawl rather than into standing. The run metric `support` (fraction of steps with ≥ n_min feet planted) shows whether it works: 1 for a crawl.

### Load margin

$$
F_{rel}(\theta, A_{eff}) = \frac{\mu \, A_{eff}}{\sin\theta + \mu\cos\theta}, \qquad C_{load} = \sum_i \max\left(0,\; \frac{\lVert F_i \rVert}{F_{rel}(\theta_i, A_{eff,i})} - 0.5\right)^2
$$

**Weight:** 5 (cost, start) · source: new, replaces separate pull-off and shear margins

F\_i is the force the leg transmits to the pad through the pivot; θ\_i is its angle from the surface normal. F\_rel is the release force from Coulomb friction with adhesion: A\_eff for a straight pull (θ = 0), μ · A\_eff for pure shear (θ = 90°). Below 50% of F\_rel there is no cost; above it the cost grows quadratically. A pad on an edge has only 1/N of the capacity, so loading a badly landed foot is penalized automatically.

**Computing F\_i in MuJoCo:** call `mj_rnePostConstraint` and read the linear part of the pad body's `cfrc_int` (force from the parent, the tibia).

**Setup needed: bench test.**

1. Mount one foot on a load cell pulling against a steel test plate; a lead screw or lever gives slow, controlled pulling.
2. Measure the release force at several pull angles (0°, 30°, 60°, 90°). This gives A and μ, and checks the F\_rel formula.
3. Repeat for surface conditions: shims of 0–2 mm for paint, plate thickness, painted, rusty, dirty. About 5 pulls per condition.
4. Use a conservative value (minimum or 10th percentile). Set the per-cell adhesion gain to A / N\_c and the pad geom `friction` to μ; randomize both per episode over the measured spread.

### Full-face contact

$$
C_{face} = \sum_{i \,:\, m_i = 1,\; t - t_{td,i} > t_{settle}} (1 - \alpha_i)
$$

**Weight:** 2 (cost, start) · source: new

Applies to every foot with its magnet on, after a settling time t\_settle ≈ 50 ms from touchdown t\_td,i so the limp ankle can rotate the pad flat. This teaches clean landings directly, instead of only penalizing their consequences.

### Ankle range in stance

$$
C_{ankle} = \sum_{i \in \text{attached}} \;\sum_{j \in \{a,b\}} \max\left(0,\; \lvert q_{ij} \rvert - q_{safe}\right)^2
$$

**Weight:** 5 (cost, start) · source: new

q\_ij are the two ankle hinge angles of foot i. With the pad flat they equal the tibia's tilt from the surface normal, so this is the foot-placement term: the leg must keep the tibia well inside the stops. At a stop the tibia levers the pad directly, which peels it.

**Setup needed:** set q\_safe to about 30° for the current q\_max = 45° (`ankle_range_deg`).

### Ankle stops in swing

$$
C_{stop} = \sum_{i \in \text{swing}} \;\sum_{j \in \{a,b\}} \mathbb{1}\left[\, \lvert q_{ij} \rvert > q_{max} - \epsilon \,\right]
$$

**Weight:** 0.5 (cost, start) · source: new

A limp ankle on a light leg can swing dynamically into its stops (seen in `test_foot`). A pad that hits a stop mid-air arrives tilted at touchdown.

**Setup needed:** ε ≈ 2°. The hardware-side fix is tuning `ankle_stiffness` and `ankle_damping`.

### Detach before lift

$$
C_{lift} = \sum_i m_i \cdot \max(0,\; v_{n,i})
$$

**Weight:** 2 (cost, start) · source: new

v\_n,i is the pad velocity along n̂, positive away from the surface. The cost applies when a pad moves away while its magnet is still on. The policy learns: switch off, wait for the switch to complete, then lift. This needs the EPM switching time modeled in sim.

### Soft touchdown

$$
C_{touch} = \sum_{i \in \text{touchdown}} \lVert v_{pad,i} \rVert^2
$$

**Weight:** 0.5 (cost, start) · source: new

Applied at the step a pad makes contact. Penalizing the full velocity covers both hard landings, which bounce the pad off, and tangential approach, which scrapes the pad in on an edge.

**Setup needed:** find on the bench how attachment reliability drops with approach speed.

### Magnet switching

$$
C_{switch} = \sum_i \lvert m_{i,t} - m_{i,t-1} \rvert
$$

**Weight:** 0.1 (cost, start) · source: new

Each EPM switch costs a current pulse and adds wear. The cost also prevents chattering the magnets on and off.

**Setup needed:** get pulse energy and switching time from the datasheet or bench.

## Termination conditions (costs)

Each condition ends the episode with a one-time cost. The paper used 1 for its conditions; falling off a wall should cost far more.

### Detachment

$$
n_{attached} < 2
$$

**Cost:** 50 (start) · source: new

With fewer than 2 attached feet the robot is falling or about to. Minimum support already penalizes dropping below 3; this is the hard stop.

### Fell off

$$
d > 0.3\,\text{m}
$$

**Cost:** 50 (start) · source: new

The body is more than 0.3 m from the surface. Catches falls the attachment check misses, for example a foot that reads as attached through a sim glitch.

### Body contact

$$
\text{contact}(\text{base}, \text{surface})
$$

**Cost:** 1 · source: paper

The base touching the surface ends the episode.

**Setup needed:** exclude intended contacts, such as a protective skid plate, if the design has one.

### Joint limits

$$
\phi_j \notin [\phi_{j,min},\, \phi_{j,max}]
$$

**Cost:** 1 · source: paper

**Setup needed:** set the limits slightly inside the mechanical hard stops, so the policy never trains against them.

## Tuning notes

- Tune weights relative to the termination costs; only ratios matter.
- Add a terrain curriculum on top of the reward curriculum: floor first, then inclines from 0° to 90°, then overhangs and ceiling.
- Randomize adhesion force A and friction μ per foot, and occasionally zero one foot's adhesion, so the policy learns to recover from a failed attach.
- Set A and μ from the load-margin bench test, not from datasheets.
