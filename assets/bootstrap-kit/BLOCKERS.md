# BLOCKERS.md — human action required

Hard walls the agent could not get past. Nothing dependent on these finishes
until you clear them. This doubles as the "must be true before running
bootstrap.sh on a fresh box" checklist. Clear, then rerun.

---

- [x] **Git credentials for `git@github.com:org/private-repo`** (target `bar`)
      Needs an SSH key with read access, or a PAT in the environment.
      Without it the clone fails and `bar` cannot be installed.
      Verify: `git clone git@github.com:org/private-repo /tmp/_chk` succeeds.

---
Cleared everything above? Run `bash bootstrap.sh`.
