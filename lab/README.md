# lab

The **experiments layer** of control-kit — where the churny, system-specific
work happens.

The repo separates code by how much it churns: `controlkit` is the stable,
system-agnostic **core**; `runkit` is the **run harness**; `lab` is everything
in between and on top — the actual experiments built on them. Code is expected
to be messy here, and to move *up* over time:

```
lab/<experiment>   →   lab.core   →   controlkit
 (one experiment)     (shared across      (stable, general,
                       experiments)        graduation target)
```

A helper that outgrows a single experiment moves to `lab.core`; a piece that
becomes stable and system-agnostic graduates further, into `controlkit`.

## Shape of an experiment

Each experiment is a subpackage, launched from the CLI:

```
lab/<experiment>/
  config.py     # the knobs (a dataclass) — the "what"
  env.py        # system-specific wiring
  __main__.py   # the entry point  →  python -m lab.<experiment>
```

Nesting is fine for grouping or versioning (e.g. `lab/mpc/hexapod/v0`): the leaf
is the runnable experiment, the dirs above it are namespaces that can also hold
code shared within their subtree. New *directory* for new *code*; new *params*
are just CLI overrides, not a new dir.

## Relationship to runkit

`lab` and `runkit` are two halves of the same activity: `lab` is *what* to run,
`runkit` is *how* a run is staged, tracked, and made reproducible (immutable run
dirs, captured provenance). An experiment plugs into the harness via runkit's
`@experiment` decorator; outputs land in a gitignored `runs/` (ignored at any
depth, including `lab/<experiment>/runs/`).

The two are deliberately close-coupled and may get bundled together down the
line — open question for now (see `src/runkit/design.md`).
