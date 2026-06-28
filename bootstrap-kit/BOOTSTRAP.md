# BOOTSTRAP.md — instructions for the setup agent

Your job: make every task in `goals.yaml` reach its definition of done on this
fresh instance — its `pass` command exits 0 — AND leave behind a `bootstrap.sh`
that reproduces the result on the next fresh box without you.

Each task has a required `pass` (the check that proves it's done) and an
optional `do` (the high-level intent of what to set up). Tasks with a `do` are
something to install/build; tasks without one are pure assertions that must hold
(e.g. a required filesystem layout). Either way, the `pass` command is the only
arbiter of done.

## The core rule

**`bootstrap.sh` is the only thing that touches this machine.**
Never run install commands by hand. For each step: write it into `bootstrap.sh`,
then run `bash bootstrap.sh`, observe, fix the *script*, rerun. The script is the
source of truth from the first command — not a log you assemble at the end. This
is what prevents drift between "what worked" and "what's in the script."

## Environment & extra rules

Per-VM context, preconditions, and extra constraints live in **ENVIRONMENT.md**
— a file the user edits freely per task, machine, or run. Read it before you
start and honor everything in it proactively, not as a box to tick at the end.
This file (BOOTSTRAP.md) is the stable contract; ENVIRONMENT.md is the variable
layer — don't put environment-specific rules here.

How it ties to `goals.yaml`: anything *checkable* belongs there as a `pass`
assertion (verified in the clean-room run) and is the authority on what's
required; the *uncheckable* half — the rule, the reason, machine facts — stays
in ENVIRONMENT.md.

## Per task

1. **If the task has a `do`:** discover the how — read the repo (README,
   pyproject, CI config) or the official docs, figure out the steps yourself,
   and write them into `bootstrap.sh` under that task's section. If it has no
   `do`, it's a pure check — make sure the other tasks' steps produce the state
   it asserts.
2. **Run the script**, then run the task's `pass` command.
3. **Read the real exit code and output.** Don't declare success because you
   think it worked — the `pass` command exiting 0 is the only proof.
4. If it fails, fix the script and rerun. Move on only when `pass` truly exits 0.

## Tests are the oracle — don't touch them

- Do NOT edit any of the bootstrap-kit files: `README.md`, `goals.yaml`,
  `ENVIRONMENT.md`, `BOOTSTRAP.md`. They are read-only input.
- Do NOT edit, skip, mock-out, or weaken any `pass` command or any test file in
  a cloned repo. The test is the spec.
- After a repo's tests pass, confirm you didn't modify them:
  `git -C repos/<name> diff --exit-code` must be clean for test paths.

## bootstrap.sh must be replay-safe

- Start with `set -euo pipefail`.
- **Idempotent**: re-running must not double-clone or choke on existing state
  (`[ -d repo ] || git clone`, `dpkg -s pkg >/dev/null 2>&1 || apt-get install`).
  You'll rerun it many times while building it, and the next box may too.
- Assume human-provided prerequisites (credentials, tokens) already exist; fail
  loudly if they don't — don't try to provision them (see BLOCKERS).

## Two records, two jobs

- **LOG.md** — append as you go. Dead ends only: what you tried, why it
  failed, what you switched to, any non-obvious gotcha. NOT a command transcript,
  NOT routine successes. This is the reasoning the script can't show.
- **BLOCKERS.md** — hard walls you cannot get past: private repo with no
  credentials, missing API key/token, license, sudo you lack. Write an
  actionable entry (what's needed, where, how to verify), then **move on to the
  next task** — do not stall the whole run and do not fake around it.

## Finish

1. Do every task you can. Log blockers for the ones you can't.
2. **Clean-room validation**: the real test of the script is a run from zero.
   In a fresh container, run ONLY `bash bootstrap.sh` (nothing by hand) and
   confirm every unblocked task's `pass` command exits 0. State you
   accumulated during discovery can hide a missing step — this catches it.
3. Report: "<X> of <N> green, <Y> blocked — see BLOCKERS.md", and confirm whether
   the clean-room run passed.
