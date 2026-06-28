# Notes

Each note gets its own subsection. Structure `## title [(date)] [#tags]`

## Example Note (28-06-2016) #example

This is just an example.

## Set up tooling for team

- High level project planning (notion?)
- Slack for communication

## runkit `@viz` decorator: rehydrate ctx from a run_dir (28-06-2026) #runkit #viz #todo

TODO (runkit): a decorator, sibling to `@experiment`, that lets an experiment
write `def viz(ctx): ...` while the CLI accepts a **run_dir path** and hands the
function a `ctx` reconstructed from that dir.

- Mirrors `@experiment` + `main(run)`: author-facing it's `ctx` (config,
  provenance, paths in hand); CLI-facing it's just a path you point at any
  finished run (incl. ones rsync-ed down from the GPU box).
- Resolves the `viz(run_dir)` vs `viz(ctx)` choice — you get both.
- Prerequisite primitive: runkit must **rehydrate a read-only RunContext from an
  existing run dir** — no new run dir created, no provenance captured, just load
  the saved config/paths. The decorator is thin once that exists.
- Launch would parallel `run.py`: `python -m lab.<exp>.viz <run_dir>`.
- Open Qs: what does a read-only ctx expose / forbid (writes? `ctx.out`?); does
  `viz` get the frozen config back out of the run dir automatically.

