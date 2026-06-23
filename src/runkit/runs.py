"""The @experiment decorator and the machinery behind it.

A decorated function is `run(cfg, ctx)`. Before the body runs, the decorator:
  1. resolves ctx fields (decorator args -> context yaml -> defaults)
  2. captures provenance and refuses to start on uncommitted code
  3. creates an immutable run dir and freezes config/provenance/env into it
  4. injects a RunContext and stamps completed/failed on exit

ctx fields resolvable in four layers (last wins):
  builtin defaults -> @experiment(...) args -> context yaml -> CLI --flags
There is NO auto-discovery of run.context.yaml: by default you get the
builtin defaults (runs_root = ./runs, no tracked repos). Pass --context=PATH
(or context= to @experiment) to opt into a yaml layer. The CLI layer is
applied by callers (e.g. runkit.main) via _ctx_overrides.
"""
import dataclasses
import datetime
import functools
import json
import pathlib
import sys

import yaml

from .provenance import software_record, hardware_record, find_lockfile


_DEFAULTS = {
    "runs_root": "runs",     # ./runs under cwd; run dirs created here
    "allow_dirty": False,
    "lockfile": None,
    "repos_in_dev": {},
}


@dataclasses.dataclass
class RunContext:
    out: pathlib.Path       # the immutable run dir; everything writes here
    id: str                 # "2026-06-23/14-30-01_baseline_ablation_a"


def _serialize_cfg(cfg):
    if dataclasses.is_dataclass(cfg) and not isinstance(cfg, type):
        return dataclasses.asdict(cfg)
    for attr in ("model_dump", "dict"):
        if hasattr(cfg, attr):
            try:
                return getattr(cfg, attr)()
            except Exception:
                pass
    if hasattr(cfg, "__dict__"):
        return dict(vars(cfg))
    return {"repr": repr(cfg)}


def _write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=str))


def _resolve_out(runs_root, name, tag, out_override):
    """Return the run dir path. If `out_override` is set, use it; on
    collision, append `_{ts}`. Otherwise auto-id under runs_root."""
    if out_override is not None:
        p = pathlib.Path(out_override).resolve()
        if p.exists():
            ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            p = p.parent / f"{p.name}_{ts}"
        return p, str(p)
    now = datetime.datetime.now()
    label = f"{name}_{tag}" if tag else name
    run_id = f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{label}"
    return pathlib.Path(runs_root).resolve() / run_id, run_id


def init_run(cfg, *, name, tag=None, out=None, repos_in_dev,
             runs_root, lockfile, allow_dirty):
    sw, any_dirty = software_record(repos_in_dev, lockfile, " ".join(sys.argv))
    if any_dirty and not allow_dirty:
        dirty = [n for n, r in sw["repos_in_dev"].items() if r["dirty"]]
        sys.exit(f"Refusing to launch: uncommitted changes in {dirty}. "
                 f"Commit, or pass --allow-dirty.")

    out_path, run_id = _resolve_out(runs_root, name, tag, out)
    (out_path / "checkpoints").mkdir(parents=True, exist_ok=False)
    (out_path / "logs").mkdir(parents=True, exist_ok=True)
    (out_path / "results").mkdir(parents=True, exist_ok=True)

    (out_path / "config.yaml").write_text(
        yaml.safe_dump(_serialize_cfg(cfg), sort_keys=False))
    _write_json(out_path / "provenance.json", sw)
    _write_json(out_path / "env.json", hardware_record())
    _write_json(out_path / "status.json",
                {"status": "running", "run_id": run_id,
                 "name": name, "tag": tag})

    return RunContext(out=out_path, id=run_id)


def _mark(out, status, **extra):
    p = pathlib.Path(out) / "status.json"
    data = json.loads(p.read_text()) if p.is_file() else {}
    data.update(
        status=status,
        ended_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        **extra,
    )
    _write_json(p, data)


# ctx fields the decorator and CLI can both set. Order matters only as
# a closed enumeration: any extra kwarg passed to @experiment that isn't
# here is rejected so typos surface early.
CTX_FIELDS = ("repos_in_dev", "runs_root", "lockfile", "allow_dirty",
              "tag", "out", "context")


def experiment(*, name, **decorator_ctx):
    """Mark a function as an experiment entry point.

    Required:
        name: short label for the run dir (e.g. "hexapod_walk").

    Optional ctx fields (override context yaml entries):
        repos_in_dev, runs_root, lockfile, allow_dirty, tag, out, context

    The wrapped function is called as `f(cfg, ctx=RunContext)`. Pass extra
    ctx overrides via a `_ctx_overrides=` kwarg at call time (runkit.main
    does this with CLI --flags).
    """
    bad = [k for k in decorator_ctx if k not in CTX_FIELDS]
    if bad:
        raise TypeError(
            f"@experiment got unknown kwarg(s) {bad}; allowed: {CTX_FIELDS}")

    def decorator(f):
        @functools.wraps(f)
        def wrapper(cfg, *args, _ctx_overrides=None, **kwargs):
            ctx_overrides = _ctx_overrides or {}
            context_path = ctx_overrides.get("context") \
                or decorator_ctx.get("context")
            # No auto-discovery: builtin defaults unless --context is given.
            context_yaml = {}
            if context_path:
                context_yaml = yaml.safe_load(
                    pathlib.Path(context_path).read_text()) or {}

            # Resolve each ctx field with the documented precedence:
            # CLI --flag > context yaml > @experiment(...) > builtin defaults
            def pick(field, default=None):
                if field in ctx_overrides:
                    return ctx_overrides[field]
                if field in context_yaml:
                    return context_yaml[field]
                if field in decorator_ctx:
                    return decorator_ctx[field]
                return default

            repos = pick("repos_in_dev", _DEFAULTS["repos_in_dev"])
            root  = pick("runs_root",    _DEFAULTS["runs_root"])
            dirty = pick("allow_dirty",  _DEFAULTS["allow_dirty"])
            lock  = pick("lockfile",     _DEFAULTS["lockfile"])
            tag   = pick("tag")
            out   = pick("out")

            # No explicit lockfile? auto-detect the one defining this venv.
            if lock is None:
                found = find_lockfile()
                lock = str(found) if found else None

            ctx = init_run(cfg, name=name, tag=tag, out=out,
                           repos_in_dev=repos, runs_root=root,
                           lockfile=lock, allow_dirty=dirty)
            try:
                result = f(cfg, *args, ctx=ctx, **kwargs)
                _mark(ctx.out, "completed")
                return result
            except BaseException as e:
                _mark(ctx.out, "failed", error=repr(e))
                raise
        wrapper._runkit_name = name          # for `--help` and introspection
        wrapper._runkit_ctx = decorator_ctx
        return wrapper
    return decorator
