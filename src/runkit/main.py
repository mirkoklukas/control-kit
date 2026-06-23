"""Entry point shared by `python experiment.py ...` and `runkit run ...`.

Usage in an experiment script:

    from runkit import experiment, RunContext, main

    @experiment(name="hexapod_walk")
    def run(cfg: HexapodConfig, ctx: RunContext): ...

    if __name__ == "__main__":
        main(run)

`main(run)` parses sys.argv (or a provided list), introspects the wrapped
function's `cfg` annotation to know what dataclass to build, and invokes
the decorated function with the resolved cfg + ctx overrides.
"""
import dataclasses
import inspect
import pathlib
import sys

import yaml

from .config import build_cfg, deep_merge, parse_overrides, split_argv
from .runs import CTX_FIELDS


# Meta flags, in display order, with one-line descriptions.
_FLAG_HELP = {
    "tag": "variant label, appended to the run dir name",
    "out": "use this exact run dir (collision-safe: appends timestamp)",
    "runs_root": "root dir for run folders (default: ./runs)",
    "context": "load a context yaml (run.context.yaml): repos, runs_root, dirty gate",
    "allow_dirty": "bypass the uncommitted-changes guard",
    "lockfile": "lockfile hashed into provenance (auto-detected from the venv if unset)",
    "repos_in_dev": "(yaml only) repos to track in provenance",
    "dry_run": "resolve cfg + ctx, print, and exit (no run dir)",
}


def main(run, argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if "--help" in argv or "-h" in argv:
        print(_help_text(run))
        return
    try:
        cfg_tokens, flags, positionals = split_argv(argv)
        config_file = _resolve_config_file(flags.pop("config", None), positionals)
        base = _load_yaml(config_file) if config_file else {}
        overrides = parse_overrides(cfg_tokens)
        merged = deep_merge(base, overrides)
        cfg = build_cfg(_cfg_type(run), merged)
    except (ValueError, TypeError) as e:
        sys.exit(str(e))

    bad_flags = [k for k in flags if k not in CTX_FIELDS and k != "dry_run"]
    if bad_flags:
        sys.exit(f"unknown flag(s): {bad_flags}; allowed: {sorted(CTX_FIELDS)}")

    if flags.pop("dry_run", False):
        resolved = dataclasses.asdict(cfg) if dataclasses.is_dataclass(cfg) else cfg
        print(f"[dry-run] cfg = {resolved}")
        print(f"[dry-run] ctx overrides = {flags}")
        return

    return run(cfg, _ctx_overrides=flags)


def _help_text(run):
    cfg_cls = _cfg_type(run)
    name = getattr(run, "_runkit_name", run.__name__)
    lines = [
        "usage: runkit run <script.py> [config.yaml] [key=value ...] [--flag ...]",
        "       python <script.py> [config.yaml] [key=value ...] [--flag ...]",
        "",
        f"experiment: {name}   (cfg: {cfg_cls.__name__})",
    ]
    doc = (run.__doc__ or "").strip()
    if doc:
        lines += ["", doc.splitlines()[0]]

    lines += ["", "base config (optional; key=value below overrides it):",
              "  <config.yaml>                positional path to a cfg yaml",
              "  --config=PATH                same, explicit flag form"]

    lines += ["", "config overrides  (bare key=value):"]
    for f in dataclasses.fields(cfg_cls):
        kv = f"{f.name}={_default_of(f)!r}"
        lines.append(f"  {kv:<28} {_type_name(f.type)}")

    lines += ["", "meta flags  (--flag, shape the RunContext):"]
    for key, desc in _FLAG_HELP.items():
        flag = "--" + key.replace("_", "-")
        lines.append(f"  {flag:<16} {desc}")
    return "\n".join(lines)


def _default_of(field):
    if field.default is not dataclasses.MISSING:
        return field.default
    if field.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
        return field.default_factory()
    return "(required)"


def _type_name(t):
    return getattr(t, "__name__", str(t))


def _resolve_config_file(flag_value, positionals):
    """Reconcile the two ways to name a base config: `--config PATH` and a
    bare positional path. At most one source; they must not conflict."""
    if len(positionals) > 1:
        raise ValueError(
            f"unexpected extra arguments {positionals}; only one bare config "
            f"path is allowed (use key=value for overrides)")
    positional = positionals[0] if positionals else None
    if flag_value and positional and flag_value != positional:
        raise ValueError(
            f"config given twice: --config={flag_value} and positional "
            f"{positional}")
    return flag_value or positional


def _load_yaml(path):
    p = pathlib.Path(path)
    if not p.is_file():
        raise ValueError(f"config file not found: {p}")
    data = yaml.safe_load(p.read_text())
    if data is not None and not isinstance(data, dict):
        raise ValueError(f"config file {p} must be a mapping, got {type(data).__name__}")
    return data or {}


def _cfg_type(run):
    """Find the dataclass annotated on the wrapped fn's `cfg` parameter."""
    sig = inspect.signature(run)
    if "cfg" not in sig.parameters:
        raise TypeError(
            f"{run.__name__!r} must accept a 'cfg' parameter")
    ann = sig.parameters["cfg"].annotation
    if ann is inspect.Parameter.empty:
        raise TypeError(
            f"{run.__name__}'s cfg parameter needs a type annotation "
            f"(a dataclass) so runkit knows what to build from CLI args")
    return ann
