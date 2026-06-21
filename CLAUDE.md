# controlkit

A project for learning and experimenting with simulation environments and RL and control in general

## How to work with notes
- Notes are **living documents**. Don't pre-fill or over-structure them.
- It's fine for them to start unstructured and messy.
- Prefer keeping more pieces over fewer — we prune later or make a clean
  version derived from the raw notes.
- Don't impose heavy structure early; let it emerge.

## Learning / derivation sessions
- Go one step at a time; don't dump the whole topic in one turn.
- Pause and ask before advancing to the next step; let the user drive pace and order.
- Stay rigorous and precise (full notation, short explicit proofs) and concise (no preamble,
  no filler, no recap of what they just watched happen).
- Flag caveats, approximations, and dropped terms honestly rather than glossing.
- Consolidate the thread into a living note in `docs/` as you go.

## Formatting
- For multi-line display math, use an `aligned` environment with each row on its own source
  line ended with `\\` (readable in source, renders everywhere). Never separate steps inside a
  `$$` block with bare newlines (no `\\`); avoid `\displaylines`.
- Go easy on em-dashes; a comma, semicolon, parentheses, or a sentence split usually works
  better.
