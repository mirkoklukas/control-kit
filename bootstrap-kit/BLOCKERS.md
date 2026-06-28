# BLOCKERS.md — human action required

Hard walls the agent could not get past. Nothing dependent on these finishes
until you clear them. This doubles as the "must be true before running
bootstrap.sh on a fresh box" checklist. Clear, then rerun.

Only APPEND to the List below.

The checked blocker items should have been resolved by the orchestrator (the user likely) already. Append new blockers as unchecked items. DON'T touch the checked blocker items. 

---

## Checklist

- [x] Git credentials for running `git clone https://github.com/mirkoklukas/run-kit.git`. Without it the clone fails and run-kit cannot be installed.