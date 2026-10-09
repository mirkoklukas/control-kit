# Mechanics in 2D: a two-legged rectangle on a wall

Living note. A planar toy version of the hold check (`planner-design.md`, "The hold
check"), small enough to solve by hand: what the feet must do to hold the body on a wall,
and how that depends on the body's distance to the wall and the feet's spread.

## 1. Setup

**The plane.** World coordinates $(x, y)$: the wall is the line $x = 0$, the robot on the
side $x > 0$. The wall's outward normal is $n = (1, 0)$, its tangent $t = (0, 1)$ (up).
Gravity points down the wall: $g = (0, -g_0)$.

**The body.** A rectangle of length $a$ (along the wall) and thickness $b$ (across it),
$a > b$, mass $M$, its centre of mass at its centre
$c = (h, 0)$, $h > b/2$: $h$ is the body's distance to the wall (centre to wall), so
the gap between body and wall is $h - b/2$. The body is held parallel to the wall; its
height on the wall is set to $y_c = 0$.

**The legs.** Two planar legs, 2 DoF each (hip and knee, links $\ell_1, \ell_2$), mounted at
the middle of the body's two short faces: the front (up the wall) and the back,

$$
\begin{aligned}
m_f &= c + \big(0, \tfrac{a}{2}\big) \\
m_b &= c - \big(0, \tfrac{a}{2}\big)
\end{aligned}
$$

Each foot touches the wall at a point, $p_f = (0, y_f)$ and $p_b = (0, y_b)$, with
$y_b < 0 < y_f$.

**The feet are pinned** to the wall: welded in position (they cannot slide or come off),
free to rotate about the contact point. The wall exerts a force on a foot, no moment, so
the force's line of action passes through the foot point. As the climb robot's pad: stuck
to the wall, the leg rotating about it through the passive ankle.

**Unknowns.** The foot forces, what wall (and magnet) exert on each foot:

$$
F_f = (N_f, T_f), \qquad F_b = (N_b, T_b)
$$

$N$ along the normal ($> 0$: the wall pushes, $< 0$: the foot pulls), $T$ along the wall
($> 0$: up). Four numbers.

**Simplifications** (to revisit): massless legs (all mass in the body); the joints hold
the posture rigidly (no servo compliance). No limits at the feet (pinned: they never slip
or come off); "does it hold" is not the question here, the quantities are.

## 2. The quantities, for a fixed posture

The posture (joint angles $q$, so foot positions and body pose) is fixed. What there is to
compute, from the feet inward:

**At each foot** $j \in \{f, b\}$:

- its position $p_j = (0, y_j)$, fixed by the posture;
- the contact force $F_j = (N_j, T_j)$: normal part $N_j$ (push $> 0$, pull $< 0$),
  tangential part $T_j$ (along the wall);
- its direction (angle to the normal, $\arctan(T_j / N_j)$) and magnitude;
- no contact moment (pinned): the line of action passes through $p_j$.

**At each joint** (hip, knee of each leg), given the angles $q$:

- the joint torque $\tau$ the servo applies; for a massless leg
  $\tau_{\mathrm{leg}} = -J(q)^\top F_j$;
- the joint reaction force, carried across the joint by the bearing (massless leg: $F_j$
  passed through, the same at knee and hip);
- the internal link loads: the axial force along each link (compression / tension), the
  shear and bending moment in it;
- the leg Jacobian $J(q)$: foot motion from joint motion, and so the lever arms from foot
  forces to joint torques.

**At the body:**

- its pose and weight: $c$, its orientation, $M g$ acting at $c$;
- the load each leg puts on it at its mount $m_f, m_b$: the hip's reaction force, and the
  hip torque as a moment on the body;
- equilibrium: the mount forces balance the weight, their moments about $c$ (forces and
  hip torques) balance;
- from that: how the weight splits between the legs, and the moment the body must be held
  against.

**What stays free.** 4 unknowns ($N_f, T_f, N_b, T_b$), 3 equilibrium equations (forces
along $x$ and $y$, moment): statically indeterminate, 1 free internal force, the feet
squeezing towards / pulling apart along the line between them, i.e. along the wall.
Everything above depends on that choice, except the body's total load.

## 3. Equilibrium, in terms of the squeeze $s$

Joints frozen, the whole robot is one rigid body held at the two foot points. Forces on it:
$F_f$ at $p_f$, $F_b$ at $p_b$, the weight $M g$ at $c$. In 2D the moment of a force
$F = (F_x, F_y)$ at $p$ about $o$ is $(p - o) \times F = u F_y - v F_x$, $(u, v) = p - o$.
With $p_j - c = (-h, y_j)$, equilibrium (forces along $x$, along $y$, moment about $c$):

$$
\begin{aligned}
N_f + N_b &= 0 \\
T_f + T_b - M g_0 &= 0 \\
-h\,(T_f + T_b) - (y_f N_f + y_b N_b) &= 0
\end{aligned}
$$

With $\Delta = y_f - y_b > 0$ (the feet's spread), the third with the first two gives
$-h M g_0 - N_f \Delta = 0$. The fourth unknown stays free; call it $s$:

$$
\begin{aligned}
N_f &= -\frac{M g_0 h}{\Delta}, & N_b &= +\frac{M g_0 h}{\Delta} \\
T_f &= \tfrac{1}{2} M g_0 + s, & T_b &= \tfrac{1}{2} M g_0 - s
\end{aligned}
$$

- $s$ is the **squeeze**: the pair $(0, s)$ at $p_f$, $(0, -s)$ at $p_b$ has zero net force
  and zero moment ($(p_f - p_b) \times (0, s) = (0, \Delta) \times (0, s) = 0$), so it can be
  added to any solution. $s > 0$: the front foot carries more of the weight.
- The normal forces do not depend on $s$, nor on the legs: the front foot pulls (peels),
  the back foot pushes, $|N| = M g_0 h / \Delta$. Body closer to the wall ($h$ small) or
  feet further apart ($\Delta$ large): less peel.
- Foot torques are zero (pinned).

Which $s$ the robot actually has is not decided by statics: rigid joints, it is whatever
the joint torques produce (section 4). With compliant joints it would follow from the
stiffnesses.

## 4. Joint torques

**Knee position.** Joint angles from the posture: hip at the mount $m_j$, foot at $p_j$,
$d = |p_j - m_j|$ (reachable iff $|\ell_1 - \ell_2| \le d \le \ell_1 + \ell_2$). With
$\alpha$ the direction of $p_j - m_j$ and $\beta = \arccos\big((\ell_1^2 + d^2 - \ell_2^2) / (2 \ell_1 d)\big)$,

$$
k_j = m_j + \ell_1 \big(\cos(\alpha \pm \beta), \sin(\alpha \pm \beta)\big),
$$

the branch with the knee away from the wall (larger $x$).

**Torque at a joint.** Massless leg: the part of the leg between a joint at $r = (x_r, y_r)$
and the foot is in equilibrium under the foot force $F_j$ and the joint torque $\tau$
(applied by the proximal side on the distal one, counterclockwise $> 0$), so
$\tau + (p_j - r) \times F_j = 0$. With $p_j - r = (-x_r, y_j - y_r)$:

$$
\tau = x_r\, T_j + (y_j - y_r)\, N_j
$$

(the same as $\tau_{\mathrm{leg}} = -J(q)^\top F_j$). The lever arm of $T_j$ is the joint's
distance from the wall, that of $N_j$ its height above the foot.

**The four torques**, hips at $m_f = (h, a/2)$, $m_b = (h, -a/2)$, knees at
$k_f = (x_{kf}, y_{kf})$, $k_b = (x_{kb}, y_{kb})$:

$$
\begin{aligned}
\tau_{\mathrm{hip},f} &= h \big(\tfrac{1}{2} M g_0 + s\big) - \big(y_f - \tfrac{a}{2}\big) \frac{M g_0 h}{\Delta} \\
\tau_{\mathrm{knee},f} &= x_{kf} \big(\tfrac{1}{2} M g_0 + s\big) - (y_f - y_{kf}) \frac{M g_0 h}{\Delta} \\
\tau_{\mathrm{hip},b} &= h \big(\tfrac{1}{2} M g_0 - s\big) + \big(y_b + \tfrac{a}{2}\big) \frac{M g_0 h}{\Delta} \\
\tau_{\mathrm{knee},b} &= x_{kb} \big(\tfrac{1}{2} M g_0 - s\big) + (y_b - y_{kb}) \frac{M g_0 h}{\Delta}
\end{aligned}
$$

**Dependence on $s$.** Each torque is affine in $s$, $\tau_k = \tau_k^0 + c_k s$, with
$c = (h,\ x_{kf},\ -h,\ -x_{kb})$: the sensitivity is $\pm$ the joint's distance from the
wall. In the $(\tau_{\mathrm{hip}}, \tau_{\mathrm{knee}})$ plane each leg moves along
$\pm(h, x_k)$ as $s$ varies. The least-torque squeeze (minimum $\sum_k \tau_k^2$):

$$
s^\star = -\frac{\sum_k \tau_k^0 c_k}{\sum_k c_k^2}
$$

**Interactive version:** `mechanics-2d.html` (open in a browser). Sliders for $s$, $h$,
$y_f$, $y_b$, $\ell_1$, $\ell_2$; feet and body draggable; plots of $(N, T)$ per foot and
$(\tau_{\mathrm{hip}}, \tau_{\mathrm{knee}})$ per leg. Parameters: $a = 0.14$, $b = 0.06$
m, $M = 2.8$ kg.

## 5. The same robot on the floor

Gravity now points into the surface. World coordinates $(x, y)$: the floor is the line
$y = 0$, the robot above it, $x$ to the front; $n = (0, 1)$, $t = (1, 0)$,
$g = (0, -g_0)$. Body centre $c = (0, h)$ ($h$: height above the floor), mounts
$m_f = (a/2, h)$, $m_b = (-a/2, h)$, feet pinned at $p_f = (x_f, 0)$, $p_b = (x_b, 0)$,
$x_b < 0 < x_f$, $\Delta = x_f - x_b$. Foot forces $F_j = (T_j, N_j)$ ($T$ along the floor,
$> 0$ to the front; $N$ normal, $> 0$ push). Knees on the branch away from the floor.

**Equilibrium.** With $p_j - c = (x_j, -h)$, the moment of $F_j$ about $c$ is
$x_j N_j + h T_j$:

$$
\begin{aligned}
T_f + T_b &= 0 \\
N_f + N_b - M g_0 &= 0 \\
x_f N_f + x_b N_b + h\,(T_f + T_b) &= 0
\end{aligned}
$$

so

$$
\begin{aligned}
N_f &= -\frac{x_b}{\Delta}\, M g_0, & N_b &= \frac{x_f}{\Delta}\, M g_0 \\
T_f &= s, & T_b &= -s
\end{aligned}
$$

- The lever rule: both feet push, the weight split by the feet's horizontal distances
  to the CoM. Independent of $h$ and of the legs.
- $s$ is again the squeeze along the line between the feet (now horizontal); same sign
  convention as on the wall, $+s$ on the front foot along $t$. $s > 0$: the floor pushes
  the feet apart (the legs pull them together).
- The wall and the floor differ only in the direction of gravity relative to the surface:
  on the wall the weight acts along $t$ and is carried by $T$ (with $N$ the peel couple,
  $\propto h$); on the floor it acts along $-n$ and is carried by $N$ ($h$ drops out).

**Joint torques.** For a joint at $r = (x_r, y_r)$, with $p_j - r = (x_j - x_r, -y_r)$,
$\tau + (p_j - r) \times F_j = 0$ gives

$$
\tau = (x_r - x_j)\, N_j - y_r\, T_j
$$

the lever arm of $N_j$ is the joint's horizontal offset from the foot, that of $T_j$ its
height. The four torques:

$$
\begin{aligned}
\tau_{\mathrm{hip},f} &= \big(\tfrac{a}{2} - x_f\big) \frac{-x_b}{\Delta} M g_0 - h\, s \\
\tau_{\mathrm{knee},f} &= (x_{kf} - x_f) \frac{-x_b}{\Delta} M g_0 - y_{kf}\, s \\
\tau_{\mathrm{hip},b} &= \big(-\tfrac{a}{2} - x_b\big) \frac{x_f}{\Delta} M g_0 + h\, s \\
\tau_{\mathrm{knee},b} &= (x_{kb} - x_b) \frac{x_f}{\Delta} M g_0 + y_{kb}\, s
\end{aligned}
$$

Sensitivities $c = (-h,\ -y_{kf},\ h,\ y_{kb})$: $\mp$ the joint's height. The
least-torque $s^\star$ is the same formula as in section 4. At $s = 0$ the torques do not
depend on $h$ directly, only through the knee position.

**Interactive version:** `mechanics-2d-floor.html`.

## 6. The (gravity) hold margin, in 2D

The planner's first score (`hold_margin`, `planner-design.md`): the largest factor
$\sigma^*$ by which gravity can be scaled with the stance still holding, the internal
force ($s$ here) chosen freely. Here with the limits added back: adhesion $N_j \ge -A$,
2D friction $|T_j| \le \mu (N_j + A)$, joint torques $|\tau| \le \tau_{\max}$.

On the wall, with gravity scaled, $N_f = -N_b = -\sigma M g_0 h / \Delta$ and
$T_f + T_b = \sigma M g_0$:

- summing the two friction rows: $\sigma M g_0 \le \mu (N_f + A) + \mu (N_b + A) = 2 \mu A$;
  the $N$ cancel (the peel is an internal couple), so $h$ drops out;
- the pulled foot's adhesion row: $\sigma M g_0 h / \Delta \le A$.

$$
\sigma^* = \min\Big(\frac{2 \mu A}{M g_0},\ \frac{A \Delta}{M g_0 h},\ \text{torque bound}\Big)
$$

With $A = 40$ N, $\mu = 0.5$, $\Delta = 0.4$ m, $h = 0.1$ m: $1.46$ (friction) vs. $5.8$
(adhesion). Limited by friction, independent of $h$ until $h \approx 0.4$ m. The same
mechanism gives the 3D wall cap $\mu_p \cdot 4A / (M g) = 2.69$.

So the hold margin barely sees $h$, while the joint loads (section 4: $\sum \tau^2$) fall
with it: the margin is set by the one limit reached first, at the best load sharing; the
torques add up all joints at a given $s$.
