"""(Copied from ``lab/rl/scheduled_config.py``, unchanged.)

Config values that follow a schedule over training: :class:`ScheduledConfig`.

A config carries a ``_schedule`` section that mirrors its structure: under the path of
each scheduled value sits that value's schedule. The config's own values are the
full-strength values; ``ScheduledConfig(cfg)(step)`` returns a new config with every
scheduled value changed for that step. Pure: the config it was built from is not
modified, and the same step always gives the same config.

    _schedule = {
        "enabled": True,                                   # all on / off
        "env": {
            "w_support": {"op": "scale", "kind": "linear", "start": 1e6, "length": 2e6},
            "cmd_vx":    {"op": "lerp", "from": 0.05, "kind": "linear", "length": 3e6},
        },
    }

A mapping with ``op`` or ``kind`` is a schedule; any other mapping is a path step.

Operations (the schedule's value s(step) is in [0, 1]):
- ``scale``: ``value * s``            -- a fraction of the full-strength value
- ``lerp``:  ``from + (value - from) * s`` -- from a start value to the full one

Schedules (``kind``), all functions of the step:
- ``linear``    (``start`` = 0, ``length``): 0 until ``start``, then linearly to 1
- ``geometric`` (``x0``, ``rate``, ``per`` = 1): ``x0 ** (rate ** (step / per))`` --
  Hwangbo et al.'s curriculum factor; ``per`` = steps per exponent unit (one PPO
  iteration reproduces the per-iteration update)
- ``step``      (``at``): 0 before ``at``, 1 from ``at`` on

Each schedule may carry ``enabled`` (default true); a disabled one, or ``None`` in its
place, leaves the value at full strength. ``_schedule.enabled = false`` disables all.

Writing a default in code: :func:`schedule` returns a dataclass field (like runkit's
``random_seed()``), built from :func:`scale` / :func:`lerp` around :func:`linear` /
:func:`geometric` / :func:`step`. Dotted keys are expanded into the nested form:

    _schedule: dict = schedule({
        "env.w_torque": scale(geometric(x0=0.4, rate=0.997, per=12288)),
        "env.w_support": scale(linear(start=1_000_000, length=2_000_000)),
    })

The helpers only build the plain dict above; ``config.yaml``, CLI overrides and
:class:`ScheduledConfig` see the same data either way.

See ``docs/scheduled-config-proposal.md``.
"""
import copy
import dataclasses

SCHEDULE_KEYS = ("op", "kind")      # a mapping with one of these is a schedule, not a path


# ------------------------------------------------------------------ schedules: step -> [0, 1]
def _linear(step, start=0, length=1):
    return min(max((step - start) / max(length, 1e-12), 0.0), 1.0)


def _geometric(step, x0, rate, per=1):
    return x0 ** (rate ** (step / per))


def _step(step, at):
    return 1.0 if step >= at else 0.0


KINDS = {"linear": _linear, "geometric": _geometric, "step": _step}
OPS = {
    "scale": lambda value, s, p: value * s,
    "lerp": lambda value, s, p: p["from"] + (value - p["from"]) * s,
}
OP_PARAMS = {"scale": (), "lerp": ("from",)}     # operation parameters, not schedule ones


def deep_merge(base: dict, over: dict) -> dict:
    """Recursively merge ``over`` into ``base`` (``over`` wins); returns a new dict."""
    out = dict(base)
    for k, v in over.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


# ------------------------------------------------------------------ writing a schedule in code
def linear(start=0, length=1) -> dict:
    """0 until ``start``, then linearly to 1 over ``length`` steps."""
    return {"kind": "linear", "start": start, "length": length}


def geometric(x0, rate, per=1) -> dict:
    """``x0 ** (rate ** (step / per))`` -- Hwangbo et al.'s curriculum factor."""
    return {"kind": "geometric", "x0": x0, "rate": rate, "per": per}


def step(at) -> dict:
    """0 before ``at``, 1 from ``at`` on."""
    return {"kind": "step", "at": at}


def scale(sched: dict) -> dict:
    """``value * s(step)``: a fraction of the full-strength value."""
    return {"op": "scale", **sched}


def lerp(from_, sched: dict) -> dict:
    """``from_ + (value - from_) * s(step)``: from ``from_`` to the full-strength value."""
    return {"op": "lerp", "from": from_, **sched}


def _expand(spec: dict) -> dict:
    """Dotted keys (``"env.w_support"``) -> the nested form; entries merged per path."""
    out = {}
    for key, v in spec.items():
        *path, leaf = key.split(".")
        node = out
        for k in path:
            node = node.setdefault(k, {})
        node[leaf] = deep_merge(node[leaf], v) if isinstance(node.get(leaf), dict) else v
    return out


def schedule(spec: dict, *, enabled: bool = True):
    """A dataclass field holding a ``_schedule`` default (each instance gets a fresh copy).

    Args:
        spec: path -> schedule entry, e.g. ``{"env.w_support": scale(linear(...))}``.
            Dotted keys are expanded into the nested form; nested dicts work as well.
        enabled: the section-wide switch.

    Returns:
        A ``dataclasses.field`` with a ``default_factory``.
    """
    default = {"enabled": enabled, **_expand(spec)}
    return dataclasses.field(default_factory=lambda: copy.deepcopy(default))


@dataclasses.dataclass(frozen=True)
class Entry:
    """One scheduled value: where it is, how it changes."""
    path: tuple[str, ...]            # e.g. ("env", "w_support")
    op: str
    kind: str
    params: dict                     # schedule parameters (start, length, x0, ...)
    op_params: dict                  # operation parameters (from, ...)

    def factor(self, step) -> float:
        """The schedule's value s(step) in [0, 1]."""
        return float(KINDS[self.kind](step, **self.params))

    def apply(self, value, step):
        return OPS[self.op](value, self.factor(step), self.op_params)


def _parse(spec, prefix=()) -> list[Entry]:
    """The active entries of a ``_schedule`` (sub)section."""
    entries = []
    for key, v in (spec or {}).items():
        if key == "enabled" or v is None:
            continue
        path = (*prefix, key)
        if not isinstance(v, dict):
            raise ValueError(f"_schedule.{'.'.join(path)}: expected a mapping, got {v!r}")
        if any(k in v for k in SCHEDULE_KEYS):
            if not v.get("enabled", True):
                continue
            op, kind = v.get("op", "scale"), v.get("kind")
            if op not in OPS:
                raise ValueError(f"_schedule.{'.'.join(path)}: unknown op {op!r} ({list(OPS)})")
            if kind not in KINDS:
                raise ValueError(f"_schedule.{'.'.join(path)}: unknown kind {kind!r} ({list(KINDS)})")
            rest = {k: x for k, x in v.items() if k not in ("op", "kind", "enabled")}
            op_params = {k: rest.pop(k) for k in OP_PARAMS[op] if k in rest}
            missing = [k for k in OP_PARAMS[op] if k not in op_params]
            if missing:
                raise ValueError(f"_schedule.{'.'.join(path)}: op {op!r} needs {missing}")
            entries.append(Entry(path, op, kind, rest, op_params))
        else:
            entries.extend(_parse(v, path))
    return entries


def _get(cfg, path):
    for k in path:
        cfg = getattr(cfg, k)
    return cfg


def _replace(cfg, path, value):
    """A copy of ``cfg`` with the value at ``path`` replaced (nested dataclasses)."""
    if len(path) == 1:
        return dataclasses.replace(cfg, **{path[0]: value})
    return dataclasses.replace(cfg, **{path[0]: _replace(getattr(cfg, path[0]), path[1:], value)})


class ScheduledConfig:
    """A config whose values follow the schedules in its ``_schedule`` section.

    Args:
        cfg: a dataclass config with a ``_schedule`` field (a nested dict mirroring
            the config). Its values are the full-strength values.
        spec: the schedule section to use instead of ``cfg._schedule`` (used by
            :meth:`sub`).

    Raises:
        ValueError: an unknown op / kind, a missing op parameter, a path that is not
            a field of ``cfg``, or a scheduled value that is not a number -- at
            construction, not mid-training.
    """

    def __init__(self, cfg, spec: dict | None = None):
        self.cfg = cfg
        spec = getattr(cfg, "_schedule", None) if spec is None else spec
        enabled = (spec or {}).get("enabled", True)
        self.entries = _parse(spec) if enabled else []
        for e in self.entries:                          # validate against the config
            try:
                v = _get(cfg, e.path)
            except AttributeError:
                raise ValueError(f"_schedule.{'.'.join(e.path)}: not a field of the config") from None
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError(f"_schedule.{'.'.join(e.path)}: scheduled value must be a number, got {v!r}")

    def __call__(self, step):
        """A new config with every scheduled value at ``step`` (``self.cfg`` unchanged)."""
        cfg = self.cfg
        for e in self.entries:
            cfg = _replace(cfg, e.path, e.apply(_get(self.cfg, e.path), step))
        return cfg

    def values(self, step) -> dict:
        """Each schedule's factor s(step), by dotted path (``"env.w_support"``), for logging."""
        return {".".join(e.path): e.factor(step) for e in self.entries}

    def sub(self, name: str) -> "ScheduledConfig":
        """The schedules of sub-config ``name``, over that sub-config (keys relative to it)."""
        spec = (getattr(self.cfg, "_schedule", None) or {})
        enabled = spec.get("enabled", True)
        return ScheduledConfig(getattr(self.cfg, name),
                               {"enabled": enabled, **(spec.get(name) or {})})
