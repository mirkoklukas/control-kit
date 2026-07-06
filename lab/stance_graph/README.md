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

## Run

Scoring only (start here):

```bash
uv run --extra mjx python -m lab.stance_graph.benchmark_scoring \
    batch_sizes='[256,1024,4096,16384,65536]' n_stances=8000 --tag=a100
```

Full sweep (scoring + settle dynamics):

```bash
uv run --extra mjx python -m lab.stance_graph.benchmark \
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
