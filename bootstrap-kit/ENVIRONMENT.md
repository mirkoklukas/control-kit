# ENVIRONMENT.md — context and rules for this machine

Read this before you start and honor it throughout the run.

## Constraints

- **Clone repos to `~/`**. You are likely in `~/bootstrap-kit`
- Moreover **install to local disk (`~/`), not the NFS mount (`/lambda/nfs/workspace`).**  Create venvs on the instance's own disk.
  Why: NFS cripples Python install/import performance — thousands of tiny-file
  ops become network round-trips — and a shared mount breaks the clean-room
  guarantee, since the next "fresh" box would see prior installs.

## Preconditions — must be true before you start

If one is not satisfied, record it in `BLOCKERS.md` and skip the tasks that
depend on it. Don't try to provision it yourself.

- **CUDA toolkit installed and on PATH.** Verify: `nvcc --version` succeeds.

## Machine facts

- NFS mount point: `/lambda/nfs/workspace` — large, read-only data only; never install here.
- Ubuntu 22.04
