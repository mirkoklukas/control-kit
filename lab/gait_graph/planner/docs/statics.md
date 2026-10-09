# Statics: equilibrium, Jacobians and internal forces

Living note. What `statics.py` computes and why: the equilibrium of a robot standing on
$p$ planted feet, written with the foot Jacobians; why it leaves the foot forces partly
free (the internal forces); and what physically decides them. The 2D example in
`mechanics-2d.md` is the smallest case of everything here ($p = 2$, one internal force
$s$).

## 1. Notation

**Robot and coordinates** (MuJoCo's, as in `statics.py`).

| symbol                             | meaning                                                                                                                                             |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| $q \in \mathbb{R}^{n_q}$           | configuration $q = (\text{base pose}, \theta)$: the base pose as $p_{\text{base}}$ and the quaternion of $R_{\text{base}}$ (free joint, 7 numbers), then the joint angles; $n_q = 7 + n_\theta = n_v + 1$ |
| $v \in \mathbb{R}^{n_v}$           | generalized velocity, $n_v = 6 + n_\theta$: base linear velocity (world frame) and angular velocity (body frame), then the joint rates $\dot\theta$ |
| $\theta \in \mathbb{R}^{n_\theta}$ | joint angles: the 12 servos (3 per leg) and the 8 passive ankle angles (in `qpos` leg by leg, each leg's ankle angles after its servos); $\theta_j$: leg $j$'s |
| $\mathcal{A}$                      | the actuated joints (the 12 servos); $n_a = 12$                                                                                                     |
| $\tau \in \mathbb{R}^{n_a}$        | servo torques                                                                                                                                       |
| $g \in \mathbb{R}^3$               | gravity (world); on a wall still $(0, 0, -g_0)$ in world, the robot is rotated                                                                      |

**Feet.**

| symbol | meaning |
|---|---|
| $P$ | the planted feet, $p = \lvert P \rvert$ (3: a tripod, 4: full stance) |
| $x_j(q) \in \mathbb{R}^3$ | foot point $j$: the ankle pivot (body `foot{j}`), world frame |
| $F_j \in \mathbb{R}^3$ | the force the surface (and magnet) exerts on foot $j$, world frame |
| $F = (F_j)_{j \in P} \in \mathbb{R}^{3p}$ | all planted feet's forces, stacked |
| $n_j$ | unit surface normal at foot $j$, pointing out of the surface |

**Jacobians.**

| symbol | size | meaning |
|---|---|---|
| $J_j(q)$ | $3 \times n_v$ | foot point Jacobian: $\dot x_j = J_j v$ |
| $J_{j,b}$ | $3 \times 6$ | its base columns |
| $J_{j,a}$ | $3 \times n_a$ | its servo columns |
| $G = [J_{j,b}^\top]_{j \in P}$ | $6 \times 3p$ | the grasp map: foot forces to the net wrench on the robot |
| $J_a = [J_{j,a}]_{j \in P}$ stacked | $3p \times n_a$ | foot forces to servo torques (through $J_a^\top$) |

$J_j$ is the Jacobian of foot $j$'s forward kinematics, the map from configuration to
the world position of its ankle pivot,

$$
x_j(q) = p_{\text{base}} + R_{\text{base}}\, (x_{\text{base}})_j(\theta_j)
$$

($x_j$: foot $j$ in the world frame; $p_{\text{base}}, R_{\text{base}}$: the base's world
position and rotation, the body pose; $(x_{\text{base}})_j$: foot $j$ in the base frame, a
function of leg $j$'s joint angles only, the leg's forward kinematics from its mount).
$J_j(q)$ is the differential of $x_j$ at $q$,
a linear map between tangent spaces,

$$
J_j(q) = dx_j(q) : T_q \mathcal{Q} \to \mathbb{R}^3, \qquad \dot x_j = J_j\, v
$$

written as a $3 \times n_v$ matrix in MuJoCo's basis of $T_q \mathcal{Q}$, the
$v$-coordinates. (Not the $3 \times n_q$ matrix $\partial x_j / \partial q$: the four
quaternion components are not a basis of the 3-dimensional tangent space of the
rotations.) Its columns: for the base translation and the joints the partial derivatives
($I_3$, $\partial x_j / \partial \theta_k$); for the base rotation, how the point moves when
the base turns about axis $e_k$, $e_k \times (x_j - p_{\text{base}})$ (in MuJoCo $e_k$ is a
body axis). In code: `mjx.jac(mx, d, d.xpos[fid], fid)`, the point
Jacobian of the pivot `xpos[fid]` on body `foot{j}`.

**Gravity terms.** $b(q) \in \mathbb{R}^{n_v}$: MuJoCo's `qfrc_bias` at rest ($v = 0$),
i.e. minus gravity's generalized force. Split as $b_b$ (6 base rows) and $b_a$ (servo
rows).

## 2. Generalized forces and the Jacobian transpose

A force $F$ applied at a point $x(q)$ of the robot does, under a small motion $\delta q$
(velocity $v$ over $\delta t$), the work $F^\top \delta x = F^\top J \, v \, \delta t$. So
its effect on the generalized coordinates is the generalized force

$$
f = J^\top F \in \mathbb{R}^{n_v}
$$

one entry per coordinate: the "push" along each degree of freedom. Each row of $J^\top F$
has a plain reading:

- **Base translation rows** ($J_{b}$'s first 3 columns are $I_3$: moving the base moves
  every point equally): the force itself, $F$.
- **Base rotation rows** (a rotation $\delta\omega$ moves $x$ by $\delta\omega \times
  (x - o)$, $o$ the base origin): the moment $(x - o) \times F$. In MuJoCo's free joint
  the angular velocity is in the body frame, so these rows come out rotated into the
  body frame: the same three equations in other coordinates.
- **Joint rows** (joint $k$ moves $x$ by $\partial x / \partial \theta_k$): the torque the
  force exerts about that joint, $(\partial x / \partial \theta_k)^\top F$. Zero for
  joints that do not move the point (the other legs' joints, joints further out).

So for one foot the base block is

$$
J_{j,b}^\top F_j = \begin{bmatrix} F_j \\ (x_j - o) \times F_j \end{bmatrix}
\quad \text{(world frame)}
$$

and $J_{j,a}^\top F_j$ is non-zero only in leg $j$'s three servo rows.

## 3. Equilibrium

**Equations of motion** (MuJoCo's, contact forces written explicitly):

$$
M(q)\, \dot v + b(q, v) = \begin{bmatrix} 0 \\ \tau_{\text{all}} \end{bmatrix} + \sum_{j \in P} J_j^\top F_j
$$

The base is not actuated: its 6 rows get no actuator force. At rest ($v = 0$,
$\dot v = 0$) $b$ is only gravity, and the equation splits into its base rows and its
joint rows.

**Base rows: the robot as a whole.**

$$
\sum_{j \in P} J_{j,b}^\top F_j = b_b
\qquad \Longleftrightarrow \qquad
G F = b_b
$$

Six equations: the feet's net force cancels the weight, $\sum_j F_j = -M g$, and their
net moment cancels gravity's, $\sum_j (x_j - o) \times F_j = -M (c - o) \times g$ ($c$:
the centre of mass). The joints do not appear: the joint forces are internal to the
robot and cancel in action-reaction pairs. Only the geometry (where the feet and the CoM
are) enters.

**Servo rows: one equation per servo.**

$$
\tau = b_a - J_a^\top F
$$

Each servo holds what gravity puts on it ($b_a$: the weight of the links beyond it)
minus what the foot forces put on it. Given $F$, the torques follow; they add no
constraint on $F$ (each servo can produce any torque, up to $\tau_{\max}$).

**Passive ankles.** Their rows would read $0 = b_{\text{ankle}} - J_{\text{ankle}}^\top F$.
The foot point is the ankle pivot, which ankle rotation does not move:
$J_{\text{ankle}} = 0$. What remains is the pad's own weight about the pivot, which we
drop (`statics(..., joint_dofs=servo_dofs)` keeps only the servo rows). This is why a
pinned foot transmits a force but no moment.

## 4. Internal forces

$G$ is $6 \times 3p$ and, for feet not on a line, has rank 6. So the base rows fix the
foot forces only up to the null space of $G$:

$$
F = F_0 + N z, \qquad G N = 0, \qquad z \in \mathbb{R}^{3p - 6}
$$

- $F_0$: one particular solution; in code the minimum-norm one (`min_norm`, $z = 0$).
- $N$: a basis of $\ker G$ ($3p \times (3p - 6)$; in code the right-singular vectors of
  the zero singular values).
- $z$: the **internal forces**. $N z$ is a set of foot forces with zero net force and
  zero net moment: it holds nothing, it only pushes the feet against each other.

| feet $p$ | unknowns $3p$ | equations | internal forces $3p - 6$ |
|---|---|---|---|
| 2 (2D: $2 \cdot 2$ unknowns, 3 equations) | 4 | 3 | 1 ($s$, `mechanics-2d.md`) |
| 3 (tripod) | 9 | 6 | 3 |
| 4 (full stance) | 12 | 6 | 6 |

A concrete basis: one squeeze per pair of feet, equal and opposite forces along the line
through them (zero net force; both on one line, so zero net moment). A tripod has 3
pairs, 3 squeezes. With four feet there are 6 pairs, but on a flat surface only 5 of
them are independent; the sixth internal force is the "twist" pattern of normal forces
$+,-,+,-$ around the feet. On a flat surface a tripod's internal forces are all
tangential, so its normal forces are fixed by equilibrium alone (as $N$ in 2D).

The basis $N$ from the SVD is an arbitrary orthonormal basis of the same space; its
columns need not be these physical squeezes.

**Torques as a function of $z$.** Substituting into the servo rows:

$$
\tau(z) = \tau_0 + K z, \qquad \tau_0 = b_a - J_a^\top F_0, \qquad K = -J_a^\top N
$$

The internal forces leave the robot's balance alone but change every servo torque. In 2D:
$\partial \tau / \partial s = \pm$ the joint's distance from the wall.

## 5. What decides the internal forces

Statics alone does not: the robot is statically indeterminate. With rigid links, rigid
joints and pinned feet, every $z$ is in equilibrium. What decides it is how the robot
deforms.

**Servo compliance.** A position servo is a spring about its target:
$\tau_k = k_k (\theta^{\text{t}}_k - \theta_k)$, stiffness $k_k$ ($k_p = 10$ N m/rad in the
climb model). Let the geometric posture (where the feet are) be $\theta^{\text{g}}$, the
misfit $e = \theta^{\text{t}} - \theta^{\text{g}}$, and the actual posture
$\theta = \theta^{\text{g}} + \delta\theta$ (small; the base may move by $\delta b$). Then

$$
\delta\theta = e - \tau / k \quad \text{(elementwise)}
$$

**Compatibility.** The planted feet do not move:
$J_{j,b}\, \delta b + J_{j,a}\, \delta\theta = 0$ for all $j \in P$. Take any change of the
internal forces, $\Delta F = N \Delta z$, and contract:

$$
\begin{aligned}
0 &= \sum_j \Delta F_j^\top \big( J_{j,b}\, \delta b + J_{j,a}\, \delta\theta \big) \\
  &= (G \Delta F)^\top \delta b + (J_a^\top \Delta F)^\top \delta\theta \\
  &= 0 - \Delta\tau^\top \delta\theta
\end{aligned}
$$

using $G \Delta F = 0$ and $\Delta\tau = -J_a^\top \Delta F = K \Delta z$. So
$\Delta\tau^\top \delta\theta = 0$ for every $\Delta z$, i.e.

$$
K^\top \big( e - \tau(z) / k \big) = 0
$$

These are $3p - 6$ equations for the $3p - 6$ unknowns $z$, the missing equations. They
are the stationarity conditions of

$$
z^\star = \arg\min_z \; \sum_{k \in \mathcal{A}} \frac{\tau_k(z)^2}{2 k_k} - e^\top \tau(z)
$$

(Menabrea's theorem, the principle of least complementary energy).

- **No misfit** ($e = 0$: targets at the planned posture), equal stiffness: $z^\star$
  minimizes $\sum_k \tau_k^2$. The real robot's internal forces are the least-torque
  ones, `least_torque` (no limits). Not a convention: what the springs produce.
- **Misfit** $e \ne 0$ pre-loads the feet: a foot landing a few mm off, the start
  posture snapped to the nearest footholds. Then $z^\star$ shifts.
- **Sag.** Even with $e = 0$ the joints deflect by $\delta\theta = -\tau / k$ (about
  0.2 rad at 2 N m) and the base moves with them. The analysis is linearized about
  $\theta^{\text{g}}$; large sag changes the geometry and with it $G$, $J_a$.

**Choosing them (feedforward).** Command
$\theta^{\text{t}} = \theta^{\text{g}} + \tau^{\text{d}} / k$ for desired torques
$\tau^{\text{d}} = \tau(z^{\text{d}})$. Then $e = \tau^{\text{d}} / k$, the condition
holds at $z = z^{\text{d}}$ with $\delta\theta = 0$: the robot sits exactly at the planned
posture with the chosen internal forces. So any $z$ the planner picks can be realized,
sag removed at the same time.

**Not modelled.** Compliance of the links, the passive ankles' springs, the pads and
MuJoCo's soft contacts also enter compatibility and shift $z^\star$; and the
linearization.

## 6. What the code uses

| function (`statics.py`) | internal forces |
|---|---|
| `statics` | builds $G$ (`J_base`), $J_a$ (`J_joint`), $b_b$, $b_a$ (`bias_base`, `bias_joint`) from one `mjx.forward` |
| `min_norm` | $z = 0$: arbitrary, the minimum-norm $F_0$ (`controlkit.forces`) |
| `least_torque` | $z = -K^+ \tau_0$: least $\sum \tau^2$, no limits; the compliance answer for $e = 0$, equal $k$ |
| `least_torque_qp` | least $\sum \tau^2$ within the cones, adhesion and $\tau_{\max}$ (the `effort` score) |
| `hold_lp`, `disturbance_lp` | $F$ free subject to $G F = \ldots$: the best $z$ for the margin, i.e. assumes the internal forces are chosen (feedforward) |

So the margins are achievable only with feedforward of the matching torques; without it
the robot gets the least-torque $z$ (section 5), and a stance can fail that the margin
says holds.

The point-mass model (`point_mass_statics`) has the same base rows with $o = c$ (the
CoM), $J_{j,b}^\top = [I_3;\ [x_j - c]_\times]$ and $b_b = (-M g, 0)$, and no servo rows.

Related, in `controlkit.forces`: `stance_sigma_min` is the smallest singular value of $G$
(moment rows scaled by a length), pure geometry (no gravity, no limits; 0 iff some body
wrench cannot be resisted at all); `stance_sensitivity` gives the linear response
$\partial F / \partial w = -G^+$ to a body wrench $w$ ($z = 0$). Unlike the disturbance
margin, neither sees the cones, adhesion or $\tau_{\max}$.
