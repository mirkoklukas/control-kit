# stance_graph — MJX stance benchmarks

Two runkit experiments that measure, on MJX (meant for an A100):

1. how fast we can **score** a posture (body pose + joint angles) via static
   stance forces, and
2. how fast we can run the parallel **weld-and-settle** dynamics.

A "posture" is a body pose + per-leg joint angles + a planted-foot set on the
climb terrain, sampled as in `notebooks/07_sample_postures.ipynb`.

## Experiments

| module | runkit name | measures | cost |
| --- | --- | --- | --- |
| `benchmark_scoring.py` | `mjx_scoring_bench` | force **scoring** only (`stance_forces_batch`, statics, 0 steps) across batch sizes | fast |
| `benchmark.py` | `mjx_stance_bench` | scoring **+** weld-and-settle throughput (batch × settle steps) | slow (settle sweep) |

Both sample the same postures and save them with their static joint torques +
foot reaction forces (from `mjx_stance`). The forces are what a score would be
built from, so the scoring timing is a proxy for scoring throughput.

## What scoring computes (static stance forces)

`mjx_stance.stance_forces` reads the joint torques and foot reaction forces that
hold a posture (body pose + joint angles, some feet planted) in static
equilibrium. Start from the equation of motion in generalized coordinates `q`
(here `nv = 24`: 6 free-joint base dofs + 18 leg-joint dofs):

$$M(q)\,\ddot q + c(q,\dot q) = \tau + \sum_i J_i(q)^\top f_i,$$

where `c` is the bias force (Coriolis + centrifugal + gravity; MuJoCo's
`qfrc_bias`), `τ` the actuator generalized forces, `f_i` the reaction force at
planted foot `i`, and `J_i = ∂x_i/∂q ∈ ℝ^{3×nv}` its translational Jacobian
(`δx_i = J_i δq`). Setting `q̇ = 0` and `q̈ = 0` drops the inertial and velocity
terms and leaves just the gravity load `g := c(q, 0)`:

$$g = \tau + \sum_i J_i^\top f_i.$$

Split this by dof. The 6 base dofs are unactuated (`τ_base = 0`), so the planted
feet alone must carry the body's gravity wrench — solve those rows for the forces:

$$\sum_i J_{i,\text{base}}^\top\, f_i = g_\text{base}.$$

The joint torques then follow from the remaining rows:

$$\tau = g_\text{joint} - \sum_i J_{i,\text{joint}}^\top\, f_i.$$

So the whole thing is one `mjx.forward` (for `g` and the Jacobians), a small
least-squares on the base rows (minimum-norm when 3+ feet make it
underdetermined), and a matrix–vector product for `τ` — no stepping, and
differentiable. A planted foot whose vertical reaction comes out `≤ 0` would have
to be *pulled* onto the surface: a tip-over on the ground, but fine if that foot
is welded / gripping (e.g. on a wall).

**Convention note.** Here `J = ∂x/∂q` has the standard shape `(3, nv)`
(`δx = J δq`). `mjx.jac` returns its **transpose**, `J^\top` of shape `(nv, 3)`,
which is the handy layout because the generalized force from a foot force is
exactly `J^\top f` — a plain matrix–vector product (`Jp @ f` in the code, no
transpose needed).

**No welds needed.** Scoring never touches the model's equality constraints (the
`wf*` welds in `weld0.xml`): it solves `J^\top f = g_base` algebraically, with the
welds inactive throughout. `stance_forces` gives identical results on a model with
no `<equality>` block — "planted" is purely the `stance` mask (which feet's
Jacobians enter the balance), not a physical weld. The welds matter only for
`mjx_weld`, the dynamics cross-check that activates them and steps physics.

## Run

Scoring only (start here):

```bash
uv run --extra mjx python -m lab.retired.stance_graph.benchmark_scoring \
    batch_sizes='[256,1024,4096,16384,65536]' n_stances=8000 --tag=a100
```

Full sweep (scoring + settle dynamics):

```bash
uv run --extra mjx python -m lab.retired.stance_graph.benchmark \
    batch_sizes='[256,1024,4096,16384]' step_counts='[50,200,800]' \
    n_stances=8000 --tag=a100
```

Any `Cfg` field is a CLI override (`key=value`); list-valued ones take a quoted
literal like `batch_sizes='[256,1024]'`. `--tag` labels the run dir.

## Config knobs

- `batch_sizes` — how many postures run in parallel per timed call (the throughput
  knob). Postures are tiled to fill a batch, so this can exceed the number of
  distinct postures.
- `step_counts` — settle steps for the weld-and-settle sweep (`benchmark.py` only).
  At weld0's `dt` (4 ms, 250 Hz) each step is 4 ms of sim time.
- `n_bodies`, `n_stances` — posture generation: the candidate grid is
  `n_bodies × n_stances`; valid ones become the `K` distinct postures that are
  scored and saved. For no tiling, keep `n_bodies × n_stances ≥` the largest batch.
- `body_xyz`, `body_rpy_deg`, `stance`, `xyz_delta`, `rpy_delta_deg` — the base
  body pose to sample around and the planted legs.
- `repeats` — timed warm runs per cell (the fastest is reported).

## Outputs (in the run dir `results/`)

- `postures.npz` — `body_wxyz_xyz`, `qpos`, `theta`, `feet`, `foot_forces`,
  `joint_torques`, `stance_mask` (the `K` distinct postures), plus the mesh.
- `scoring.npz` — `batch`, `cold_s`, `warm_s`, `postures_per_s`, `us_per_posture`.
- `throughput.npz` — `batch`, `steps`, `cold_s`, `warm_s`, `env_steps_per_s`
  (`benchmark.py` only).
- `summary.txt` — human-readable: sim dt/Hz, array shapes, and the timing tables.

## Notes

- **cold vs warm.** The first call for a given batch/steps pays JIT compilation
  (`cold_s`); a second compiled run right after is timed as `warm_s`. Throughput
  uses the warm run.
- **statics vs dynamics.** Scoring is a single `mjx.forward` + least-squares (no
  stepping); the forces are the exact static equilibrium (foot reactions + joint
  hold torques). The settle sweep is the dynamics cross-check / throughput, and
  does not feed the saved forces.
