# Instance Bootstrap

## Claude Code setup (one-time per box)

The Claude-driven first run (step 4) needs Claude Code installed and logged in to your
Pro/Max subscription on the box. This is a manual prerequisite — OAuth needs interactive
browser approval and the token is personal, so `prep.sh` deliberately doesn't automate it.

1. **Install** on the box:

   ```bash
   curl -fsSL https://claude.ai/install.sh | bash      # or: npm install -g @anthropic-ai/claude-code
   ```

2. **Ensure no API key is set** — otherwise Claude Code bills the API instead of your plan:

   ```bash
   echo "$ANTHROPIC_API_KEY"   # must be empty
   unset ANTHROPIC_API_KEY
   ```

3. **Log in headlessly.** Run `claude`, choose the subscription account (or `/login`). With no
   browser on the box it prints an `https://claude.ai/...` URL and waits.
4. **Approve on your laptop.** Open that URL in a browser logged into your Pro account, approve,
   and paste the code back into the SSH terminal.
5. **Verify** with `/status` — it should show your Pro/Max plan, not API credits.

Credentials persist in `~/.claude/.credentials.json` (no keychain on headless Linux), so this
is one-time per box. If `/login` doesn't offer the subscription option, run `/logout`,
`claude update`, restart the shell, and retry.

## Start bootstrapping

```
cd ~/bootstrap-kit && claude --permission-mode bypassPermissions 'Read BOOTSTRAP.md and complete the bootstrap.'
```

## Pull the results back

The agent builds `bootstrap.sh` on the box (and writes `LOG.md` / `BLOCKERS.md`
as it goes). When it's done, pull those back into the repo from your laptop:

```bash
./pull.sh <HOST>        # HOST = the same ssh alias / user@ip you used for prep.sh
```

`pull.sh` copies `bootstrap.sh`, `LOG.md`, and `BLOCKERS.md` from `~/bootstrap-kit`
on the box over the local copies (missing ones are skipped). Review with
`git diff` before committing.