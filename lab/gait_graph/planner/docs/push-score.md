# Push score

Living note (started 2026-10-08). Goal: a continuous **push score** per stance, high if
the stance resists pushes well, low if it is fragile; for four feet and for a tripod.
Builds on `statics.md` (equilibrium, Jacobians, gravity terms).

## Approaches (2026-10-08)

| # | approach | output | sees cones, adhesion, $\tau_{\max}$ | cost |
|---|---|---|---|---|
| 1 | `disturbance_check`: 12 LPs (linear programs), each capped at $\lambda_{\min}$ | bool | yes | 12 LPs |
| 2 | `disturbance_margin`: largest push per axis direction, min over the 12 | continuous, not smooth | yes | 12 LPs |
| 3 | `push_check_fast`: linear force response $F + H w$, test the 12 pushes | bool (+ cone margin) | yes | 1 linear solve |
| 4 | ratio test on 3: largest push $\lambda(e)$ along any direction $e$ (section 4 below) | continuous, any direction | yes | 1 solve + one product per direction |
| 5 | **ball radius** $r^*$: largest push withstood in every direction (below) | continuous, one number | yes | about as 4 |
| 5b | 5 with $F$ and $H$ optimized: a second-order cone program (SOCP) | continuous | yes | 1 SOCP |
| 6 | wrench polytope: all body wrenches the stance resists, margin = distance from the origin to its boundary | continuous | yes | expensive (6D polytope projection); its support function is 2 with more directions |
| 7 | singular values of $G$ (`stance_sigma_min`) or of $H$ | continuous, smooth | no | 1 SVD |
| 8 | classical: support polygon, energy margin, force-angle measure (Papadopoulos & Rey 1996) | continuous | gravity / tip-over only | trivial |
| 8b | 8 with adhesion: per tip-over edge, the moment to peel | continuous | adhesion, no slipping | trivial |
| 9 | log-barrier over the limit rows at rest | continuous, smooth, no unit | yes | 1 solve |

Current pick: 5, the ball radius. Derivation below.

## 1. The stance at rest

**Notation.** (`statics.md` writes the foot force as $F_j$; here $f_j$, with $F$ the stack.)

| symbol | meaning |
|---|---|
| $p$ | number of planted feet (3: tripod, 4: full stance) |
| $j = 1, \dots, p$ | a planted foot |
| $n_j \in \mathbb{R}^3$ | unit surface normal at foot $j$, pointing out of the surface |
| $f_j \in \mathbb{R}^3$ | force the surface (with the magnet) exerts on foot $j$ |
| $F = (f_1, \dots, f_p) \in \mathbb{R}^{3p}$ | all foot forces, stacked |
| $E_j \in \mathbb{R}^{3 \times 3p}$ | picks foot $j$ out of $F$: $f_j = E_j F$ ($I_3$ in block $j$, zeros elsewhere) |
| subscript $b$ | **base**: the body's 6 coordinates |
| subscript $a$ | **actuated**: the $n_a = 12$ servo joints (not the passive ankles) |
| $J_b \in \mathbb{R}^{3p \times 6}$ | Jacobian of the foot positions w.r.t. the base coordinates |
| $J_a \in \mathbb{R}^{3p \times n_a}$ | Jacobian of the foot positions w.r.t. the servo angles; $J_{a,r}$ its column $r$ |
| $G = J_b^\top \in \mathbb{R}^{6 \times 3p}$ | foot forces to the net wrench on the body |
| $b_b \in \mathbb{R}^6$, $b_a \in \mathbb{R}^{n_a}$ | gravity terms of the base rows and the servo rows (`statics.md`) |
| $\tau \in \mathbb{R}^{n_a}$ | servo torques |
| $A$ | maximum magnet pull per foot (N) |
| $\mu$ | friction coefficient |
| $K$ | faces of the friction pyramid; $t_1, \dots, t_K$ unit tangents at foot $j$, $\perp n_j$, evenly spaced |
| $\mu_p = \mu \cos(\pi / K)$ | pyramid coefficient (the pyramid lies inside the round cone) |
| $\tau_{\max}$ | maximum servo torque (N m) |

**Equilibrium.** The foot forces balance gravity on the body, and the servos supply the
rest:

$$
\begin{aligned}
G F &= b_b \\
\tau &= b_a - J_a^\top F
\end{aligned}
$$

Six equations for $3p$ unknowns: $F$ is not unique; here it is one fixed solution (which
one: section 2b).

**Limits.** The stance holds if, for every planted foot $j$, every face $k$ and every
servo $r$:

$$
\begin{aligned}
-n_j^\top f_j &\le A && \text{adhesion: the magnet pulls at most } A \\
t_k^\top f_j &\le \mu_p \,(n_j^\top f_j + A) && \text{friction: normal force plus pull} \\
\lvert \tau_r \rvert &\le \tau_{\max} && \text{torque}
\end{aligned}
$$

**One common form.** Each limit is one inequality, linear in $F$: $c^\top F \le d$ with a
fixed $c \in \mathbb{R}^{3p}$ and $d \in \mathbb{R}$. With $f_j = E_j F$, e.g.
$-n_j^\top f_j = (-E_j^\top n_j)^\top F$, and $\tau_r = b_{a,r} - J_{a,r}^\top F$:

| limit | count | $c$ | $d$ |
|---|---|---|---|
| adhesion, foot $j$ | $p$ | $-E_j^\top n_j$ | $A$ |
| friction, foot $j$, face $k$ | $pK$ | $E_j^\top (t_k - \mu_p n_j)$ | $\mu_p A$ |
| $\tau_r \le \tau_{\max}$ | $n_a$ | $-J_{a,r}$ | $\tau_{\max} - b_{a,r}$ |
| $-\tau_r \le \tau_{\max}$ | $n_a$ | $J_{a,r}$ | $\tau_{\max} + b_{a,r}$ |

$m = p(K + 1) + 2 n_a$ inequalities in all (tripod, $K = 8$: $m = 51$). Numbered
$i = 1, \dots, m$: inequality $i$ is $c_i^\top F \le d_i$. Stacked: $C F \le d$
(componentwise), $C \in \mathbb{R}^{m \times 3p}$ with rows $c_i^\top$, $d \in \mathbb{R}^m$.
These are the inequality rows of `hold_lp` at gravity factor 1 (`limit_rows` evaluates
$c_i^\top F - d_i$).

**Slack.** How far limit $i$ is from breaking (N for the foot rows, N m for the torque
rows):

$$
s_i = d_i - c_i^\top F
$$

The stance holds at rest iff $s_i \ge 0$ for every $i$.

## 2. A push on the body

**Notation.**

| symbol | meaning |
|---|---|
| $w \in \mathbb{R}^6$ | the push: extra load the feet must carry; 3 force components, 3 moment components as moment / $L$ (all in N) |
| $L$ | lever length (m); `DIST_LENGTH = 0.1` |
| $S = \mathrm{diag}(I_3, L I_3)$ | turns $w$ into a load in the units of $b_b$ |
| $H \in \mathbb{R}^{3p \times 6}$ | response of the foot forces to the push |
| $h_i \in \mathbb{R}^6$ | sensitivity of limit $i$ to the push |

**Equilibrium under a push.** $G F(w) = b_b + S w$.

**Linear response.** $F(w) = F + H w$, with $F$ the forces at rest. Equilibrium for every
$w$ requires $G H = S$. $H$ is a choice; we use the least-torque response (section 2b). The torques
follow, $\tau(w) = b_a - J_a^\top F(w)$, and are covered by the torque rows.

**Each slack changes linearly.**

$$
\begin{aligned}
s_i(w) &= d_i - c_i^\top (F + H w) \\
&= s_i - h_i^\top w, \qquad h_i := H^\top c_i
\end{aligned}
$$

$h_i$ is the push direction that uses up limit $i$'s slack fastest, $\lVert h_i \rVert$
the rate.

**Each limit tolerates a half-space of pushes**: limit $i$ holds under $w$ iff
$h_i^\top w \le s_i$. For $h_i \ne 0$ its boundary is the plane $h_i^\top w = s_i$, with
normal $h_i$, at signed distance $s_i / \lVert h_i \rVert$ from $w = 0$. For $h_i = 0$ the
push does not affect limit $i$.

**All limits: a polytope of pushes.**

$$
\mathcal{P} = \{\, w \in \mathbb{R}^6 : h_i^\top w \le s_i \ \text{ for all } i \,\}
$$

Convex (an intersection of half-spaces), possibly unbounded; $0 \in \mathcal{P}$ iff the
stance holds at rest.

## 2b. Which foot forces: least torque (2026-10-08)

The score uses the **least-torque** forces: of all $F$ in equilibrium, the one with the
smallest sum of squared servo torques.

**Notation.** $\nu \in \mathbb{R}^6$: Lagrange multiplier of the equilibrium constraint;
$\varepsilon$: a small regularization (`reg = 1e-8`).

**At rest.**

$$
F = \arg\min_F \ \lVert b_a - J_a^\top F \rVert^2 + \varepsilon \lVert F \rVert^2
\quad \text{subject to} \quad G F = b_b
$$

Setting the gradient of the Lagrangian to zero gives one linear system, the KKT
(Karush-Kuhn-Tucker) conditions:

$$
\begin{bmatrix} J_a J_a^\top + \varepsilon I & G^\top \\ G & 0 \end{bmatrix}
\begin{bmatrix} F \\ \nu \end{bmatrix}
=
\begin{bmatrix} J_a b_a \\ b_b \end{bmatrix}
$$

$\varepsilon$ keeps the matrix invertible; it is invertible anyway when the feet are not
on a line and the legs are not at a kinematic singularity.

**Under a push.** The load becomes $b_b + S w$; only the right-hand side changes, linearly
in $w$. So the least-torque forces under the push are $F + H w$, with $H$ from the same
matrix:

$$
\begin{bmatrix} J_a J_a^\top + \varepsilon I & G^\top \\ G & 0 \end{bmatrix}
\begin{bmatrix} H_0 \\ \cdot \end{bmatrix}
=
\begin{bmatrix} 0 \\ I_6 \end{bmatrix},
\qquad H = H_0 S
$$

($G H_0 = I_6$, so $G H = S$.) One factorization gives both $F$ and $H$.

**The limits do not choose $F$.** Adhesion, the friction pyramid and $\tau_{\max}$ play
no part in the solve; they only enter afterwards, through the slacks $s_i$. If the
least-torque $F$ breaks a limit, $r^* < 0$ and `holds` is false, even when another $F$ in
equilibrium would satisfy every limit. The LPs (`hold_lp`, `disturbance_lp`) search over
all $F$ instead, so they can pass where the score fails.

**Code.** `least_torque_kkt(st, planted_idx)` returns $F$, $\tau$, $H_0$ (`H`, for an
unscaled load) and $J_a^\top$ (`M`); `push_slacks` scales it, `H @ S`.

## 3. The push score: the ball radius

**Notation.** $\lVert w \rVert$: Euclidean length of $w$; $B_r = \{ w : \lVert w \rVert \le r \}$.

**Definition.** For a stance that holds at rest,

$$
r^* = \max \{\, r \ge 0 : B_r \subseteq \mathcal{P} \,\}
$$

Every push of size at most $r^*$, in any direction, is withstood; some slightly larger
one is not.

**One limit.** For $h_i \ne 0$: $B_r \subseteq \{ w : h_i^\top w \le s_i \}$ iff
$r \lVert h_i \rVert \le s_i$.
*Proof.* The ball fits iff $\max_{w \in B_r} h_i^\top w \le s_i$. Cauchy-Schwarz:
$h_i^\top w \le \lVert h_i \rVert \lVert w \rVert \le r \lVert h_i \rVert$ on $B_r$, with
equality at $w = r\, h_i / \lVert h_i \rVert \in B_r$. $\square$
For $h_i = 0$ the limit holds for every $r$ (as $s_i \ge 0$).

**All limits.** The ball fits in $\mathcal{P}$ iff it fits in every half-space:

$$
r^* = \min_{i \,:\, h_i \ne 0} \ \frac{s_i}{\lVert h_i \rVert}
$$

The distance from $w = 0$ to the nearest face of $\mathcal{P}$. The minimizing $i^*$ names
the limit that breaks first (a foot's magnet or friction, a servo's torque) and
$h_{i^*} / \lVert h_{i^*} \rVert$ the weakest push direction.

Limits with $h_i = 0$: e.g. a lifted leg's servos (their torque is the leg's own weight,
independent of $F$: $c_i = 0$). They only need $s_i \ge 0$.

## 4. Relation to per-direction pushes

For a direction $e$, $\lVert e \rVert = 1$, the largest $\lambda$ with $\lambda e \in \mathcal{P}$:

$$
\lambda(e) = \min_{i \,:\, h_i^\top e > 0} \ \frac{s_i}{h_i^\top e}
$$

$\mathcal{P}$ is convex and contains 0, so

$$
r^* = \min_{\lVert e \rVert = 1} \lambda(e)
$$

$r^*$ is the worst case over **all** directions; a min over finitely many (the 12 axis
pushes) is $\ge r^*$ and can miss a weak diagonal direction.

## 5. Properties and caveats

- **Cost**: $F$ and $H$ from one linear solve, then $h_i = H^\top c_i$ and one ratio per
  limit. No LP; batches well.
- **Smoothness**: $s_i$, $h_i$ are smooth in the posture, so $r^*$ is continuous and
  piecewise smooth (kinks where $i^*$ switches). Smooth version for the NLP, a soft
  minimum with sharpness $\beta > 0$ ($r_\beta \to r^*$ as $\beta \to \infty$):

  $$
  r_\beta = -\frac{1}{\beta} \log \sum_i \exp\!\Big(-\beta \, \frac{s_i}{\lVert h_i \rVert}\Big)
  $$

- **Failing at rest** (some $s_i < 0$): the formula goes negative, usable to rank
  failures, but it is not the distance from 0 to $\mathcal{P}$, only the worst violated
  limit, normalized. A failing limit with $h_i = 0$ is not captured; check it separately.
- **Depends on $H$**: another response gives other $h_i$ and another $r^*$.

## 2c. Minimum-norm variant: no servos, no MuJoCo (2026-10-09)

The same ball radius, built only from the **base rows** (equilibrium of the whole robot)
and the **foot rows** (adhesion, friction). No servo rows, so no $J_a$, $b_a$: nothing that
needs the MuJoCo statics.

**Notation.** $c$: centre of mass (world); $M$: total mass; $x_j$: planted foot points
(the ankle pivots, world); $g$: gravity.

**Geometry instead of MuJoCo.** The robot as a point mass at $c$ (`point_mass_statics`):
foot $j$'s block of $G$ is $\begin{bmatrix} I \\ [x_j - c]_\times \end{bmatrix}$ (force and
moment about $c$), $b_b = (-M g, 0)$. $c$ from the kinematics (`kinematic_com`): the body's
mass at the body origin, each link's along its link, the pad's at the pivot (`MassModel`,
read once from the MuJoCo model by `ClimbModel.mass_model`). Checked against MuJoCo's centre
of mass on 200 random postures: within 0.6 mm (median 0.4 mm; the pad's mass placed at the
pivot).

**Forces: minimum norm**, the smallest sum of squared foot forces in equilibrium:

$$
\begin{aligned}
F &= G^\top (G G^\top)^{-1} b_b \\
H &= G^\top (G G^\top)^{-1} S, \qquad F(w) = F + H w
\end{aligned}
$$

($G G^\top$ is 6 x 6, invertible unless the feet are on a line; a tiny Tikhonov term keeps
it so.) This is **not** the least-torque split of section 2b, which needs the servo rows: it
is another choice of the internal forces.

**Limits: the foot rows only** (adhesion, friction pyramid). No torque limits: a stance the
servos cannot hold (a far-out foot on a wall) can score well here.

**What it means.**

- `holds` (all slacks >= 0): the minimum-norm forces satisfy adhesion and friction, a
  **sufficient** equilibrium check. Other internal forces might hold where these do not
  (exact: an LP over all $F$, `hold_lp` without torques, much slower).
- $r$: the ball radius against adhesion and friction, for these forces.

**Code.** `statics.min_norm_slacks`, `statics.push_score_min_norm`,
`statics.kinematic_com`, `statics.MassModel`, `ClimbModel.mass_model`.

**Benchmark** (`experiments/bench.py`, floor, laptop CPU; run
`experiments/runs/planner_bench/2026-10-09_18-28-57_f6a6ebe0`). Per candidate, from the
posture (everything included), batch 4096-16384:

| | B_lift (tripod) | B_plant (tripod) | f' (four feet) |
|---|---|---|---|
| push score (MuJoCo statics, least torque) | 4.7-5.1 us | 4.7-5.2 us | 5.1-5.7 us |
| push min-norm | 1.8-2.0 us | 1.9-2.0 us | 1.8-1.9 us |

~2.5x faster, not the ~10x estimated: the kinematic centre of mass and feet (forward
kinematics per leg) and the row Jacobian remain. Agreement on 1024 unique valid candidates
per phase, push at 0.3 x weight:

| | B_lift | B_plant | f' |
|---|---|---|---|
| pass: push score / min-norm | 0.97 / 1.00 | 0.99 / 1.00 | 1.00 / 1.00 |
| min-norm passes, push score fails | 33 | 6 | 0 |
| push score passes, min-norm fails | 0 | 0 | 0 |
| rank correlation of the scores | 0.92 | 0.81 | **-0.22** |
| median score: push score / min-norm (N) | 24.5 / 30.6 | 27.5 / 32.0 | 25.0 / 45.1 |

- Min-norm is more optimistic (no torque limits): it never fails where the push score
  passes; where it passes and the push score fails, likely the torques.
- Tripods: the rankings agree well (0.81-0.92). Four feet: they do not (-0.22): with 6
  internal-force directions the two splits differ a lot, and the torque limits shape the
  full score. As a ranking for four-foot stances, min-norm is no substitute.

## Cost: the statics are the bottleneck (2026-10-09)

Per posture, laptop CPU, batch ~16k-25k (`experiments/bench.py`;
`lab/kinematics/posture-sampling.ipynb`):

| step | per posture | 25,600 postures |
|---|---|---|
| sample body + single leg (reach grid) | ~0.55 us | ~14 ms |
| sample body + full posture (reach grid) | ~1.3 us | ~33 ms |
| push score incl. statics | ~4.7 us | ~120 ms |
| of which statics (`mjx.forward` + foot Jacobians) | ~4.0 us | ~100 ms |
| of which the score itself | ~0.7 us | ~20 ms |

Scoring is ~4x the sampling, and the statics are ~85% of the scoring. Options, roughly from
cheapest to try:

1. **Filter before scoring**: cheap rejects first (collisions, foot spacing, centre of mass
   over the support), statics only for the survivors.
2. **Statics from our own kinematic model**: `controlkit`'s `Robot` is pure JAX with the
   joint frames; foot Jacobians from the chain, gravity terms from the link masses and
   centres of mass. Skips the parts of `mjx.forward` we do not use. Expected a few x
   faster; not measured. Check agreement with the MuJoCo statics.
3. **Point-mass statics** (`point_mass_statics`): very cheap, but no leg masses and no servo
   torques (legs are ~36% of the mass).
4. **GPU**: the pipeline is batched JAX. Measured below.

**Laptop CPU vs. GPU** (2026-10-09, `experiments/score_bench.py`; laptop: Apple CPU, jax
0.10.2; GPU: NVIDIA A10 24 GB, jax 0.10.2 + CUDA 12, mujoco 3.9.0). Per sample, after
compilation; sampling with the 1 cm reach grid, scores on the sampled full postures:

| stage | CPU, n = 16k | GPU, n = 16k | CPU, n = 262k | GPU, n = 262k | speedup at 262k |
|---|---|---|---|---|---|
| leg, fixed body | 0.100 us | 0.024 us | 0.034 us | 0.0020 us | 17x |
| body + leg | 0.60 us | 0.039 us | 0.38 us | 0.021 us | 19x |
| body + posture | 1.41 us | 0.096 us | 1.08 us | 0.068 us | 16x |
| push score (MuJoCo statics) | 4.94 us | 0.56 us | 5.13 us | 0.54 us | 9.5x |
| push min-norm | 1.82 us | 0.054 us | 2.03 us | 0.045 us | 45x |
| hold min-norm | 1.55 us | 0.043 us | 1.78 us | 0.034 us | 53x |

- At n = 1 the GPU is no faster (sub-ms per call either way; the scores are slower on the
  GPU: kernel launches). At 1k it starts to pay off; at 16k it is close to its rate.
- 262k full postures on the GPU: sampling ~18 ms, push score ~141 ms, push min-norm ~12 ms,
  hold min-norm ~9 ms. The MuJoCo statics remain the bottleneck on the GPU too, and gain
  least (~10x); the MuJoCo-free scores gain ~50x.
- Compile times on the GPU are ~3x the CPU's (push score 11-16 s, samplers 1-4 s).
- Runs: laptop `experiments/runs/score_bench/` (2026-10-09), GPU box
  `~/control-kit/lab/gait_graph/planner/experiments/runs/score_bench/2026-10-09_17-28-03_aa3165f2`.

## Open

- Compare $r^*$ with `disturbance_margin` on the stances of `experiments/scores.py`,
  four feet and tripods: same ranking?
- Weighting of forces vs moments ($L$). (No scaling needed between foot rows and torque
  rows: each ratio $s_i / \lVert h_i \rVert$ is a push size in N, whatever the row's unit.)
