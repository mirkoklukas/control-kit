

```python
# experiment.py

# could be a dataclass or so
class Config:
	...

@experiment(name="baseline")
def run(cfg:Config, ctx:RunContext)

if __name__ == "__main__":
	something_smart(run)

```


```bash

# square brackets indicate optional
uv run python experiment.py x=1 y=2 --tag=ablation_a --out=/my/run/folder
runkit run experiment.py x=1 y=2 --tag=ablation_a --settings=run.settings.yaml

```

## vocabulary
- **run spec** = the whole thing to launch. two halves:
  - **config**  -> the experiment params (cfg). populated by `key=value` (+ a config yaml).
  - **context** -> where/how it's staged (RunContext). populated by `--flags` (+ a context yaml).
- a config yaml + a context yaml together = a run spec. one file holding both is a TODO.

## convention
- bare `key=value` -> overrides into `config`  (the experiment definition)
- `--flag=value`   -> feeds the `context`      (how this attempt is run)
- two namespaces, strictly disjoint. flags never touch config, k=v never touches context.
- `--config=PATH` / `--context=PATH` name the two yaml halves (config wins via k=v;
  context wins via --flags).

## why split this way
- `cfg` = what to run (seeds, hyperparams). serialized to `config.yaml`. reproducible.
- `ctx` = how this attempt is staged (name, out dir, dirty bypass). shapes the run dir.
- a name belongs to "this attempt", not "this experiment" -> it's a flag, not in cfg.
- consequence: drop `name` from cfg dataclasses. run dir uses `--name` or a default.

## meta flags (draft)
- `--tag=TAG`       -> variant label, appended to run dir name
- `--out=PATH`      -> use this as run dir; if it exists, append timestamp: `{PATH}_{ts}`
- `--settings=PATH` -> load this yaml as ctx defaults (instead of default lookup)
- `--dry-run`       -> resolve cfg, print, exit; no run dir
- `--allow-dirty`   -> bypass uncommitted-changes guard
- `--multirun`      -> later

name lives on the decorator only: `@experiment(name="baseline")`. no CLI override.
if you need a different name, use a different script.

## run dir naming
  {runs_root}/{YYYY-MM-DD}/{HH-MM-SS}_{name}[_{tag}]/

example: runs/2026-06-23/14-32-15_baseline_ablation_a/

## resolution order
cfg:  dataclass defaults -> config.yaml (positional or --config) -> CLI `key=value`. last wins.
ctx:  builtin defaults -> decorator args -> settings yaml -> CLI `--flags`. last wins.
      (no settings auto-discovery; --settings opts in)

## two entry points, one core
`python experiment.py ...` and `runkit run experiment.py ...` should do the same thing.
=> `something_smart(run)` and `runkit run` share the same argv-parsing core
   (probably `runkit.main(run, argv)`).

what does `runkit run` add then?
- discovery: `runkit list <dir>`
- inspection: `runkit show <run_id>`
- sweeps: `--multirun seed=1,2,3`

## open
- precedence when decorator AND context yaml set the same ctx field. current chain
  has yaml > decorator. fine for `allow_dirty`, wrong for `repos_in_dev` (script
  knows its own deps). revisit; for now whatever the chain does is what it does.

## todo: a single run-spec file (config + context in one)
idea: one yaml = a complete run spec, holding both halves as siblings:
    config:  { seed: 7, ... }      # -> cfg
    context: { runs_root: ..., repos_in_dev: ... }   # -> RunContext
this supersedes "config as a key in settings": config is NOT nested under context;
both are nested under the run spec. naming the config file stays a meta/pointer act,
so the disjoint-namespace rule holds (context never sets cfg *values*).

- precedence to decide: CLI (`key=value`, `--flags`) > run-spec file > defaults.
- ARCHITECTURE WRINKLE: today the config yaml is resolved in `main()` (before the
  decorated fn runs), but the context yaml loads *inside* the decorator wrapper. a
  single run-spec file means both halves load in one place -> either main() loads the
  whole spec, or the wrapper owns it. pick one before implementing.
