"""argv parsing and dataclass instantiation.

Two namespaces, strictly disjoint:
  - bare `key=value`  -> cfg overrides    (the experiment definition)
  - `--flag[=value]`  -> ctx overrides    (how this attempt is run)

Pure functions; no IO, no globals. Tested in isolation.
"""
import dataclasses
import typing


def split_argv(tokens):
    """Split tokens into (cfg_overrides, ctx_flags, positionals).

    cfg_overrides: list of "key=value" strings (the cfg layer).
    ctx_flags: dict of {flag_name: value} parsed from --flag / --flag=value /
               --flag value pairs. Bare --flag (no value) becomes True.
    positionals: bare tokens with no leading dash and no '=' (e.g. a config
                 file path). Callers decide what they mean.
    """
    cfg, flags, positionals = [], {}, []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t.startswith("--"):
            body = t[2:]
            if "=" in body:
                k, v = body.split("=", 1)
                flags[k.replace("-", "_")] = _coerce(v)
            elif i + 1 < len(tokens) and not tokens[i + 1].startswith("--") \
                    and "=" not in tokens[i + 1]:
                flags[body.replace("-", "_")] = _coerce(tokens[i + 1])
                i += 1
            else:
                flags[body.replace("-", "_")] = True
        elif "=" in t:
            cfg.append(t)
        else:
            positionals.append(t)
        i += 1
    return cfg, flags, positionals


def deep_merge(base, over):
    """Recursively merge `over` into `base`; `over` wins. Returns a new dict."""
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def parse_overrides(tokens):
    """Parse ["a=1", "b.c=hi"] -> {"a": 1, "b": {"c": "hi"}}.

    Dotted keys become nested dicts. Values are type-coerced.
    """
    out = {}
    for t in tokens:
        if "=" not in t:
            raise ValueError(f"override {t!r} must be key=value")
        key, raw = t.split("=", 1)
        _set_dotted(out, key.split("."), _coerce(raw))
    return out


def _set_dotted(d, path, value):
    for p in path[:-1]:
        d = d.setdefault(p, {})
        if not isinstance(d, dict):
            raise ValueError(f"override path collides with scalar at {p!r}")
    d[path[-1]] = value


def _coerce(s):
    """Cheap YAML-ish scalar coercion. No external dep."""
    if not isinstance(s, str):
        return s
    if s in ("true", "True"): return True
    if s in ("false", "False"): return False
    if s in ("null", "None", "~"): return None
    try: return int(s)
    except ValueError: pass
    try: return float(s)
    except ValueError: pass
    return s


def build_cfg(cls, overrides):
    """Instantiate dataclass `cls` with `overrides` applied to defaults.

    Supports nested dataclasses: a dict value is recursively passed to the
    field's annotated dataclass type.
    """
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass")
    field_types = typing.get_type_hints(cls)
    kwargs = {}
    for k, v in overrides.items():
        if k not in field_types:
            raise ValueError(
                f"unknown field {k!r} for {cls.__name__}; "
                f"known: {sorted(field_types)}")
        t = field_types[k]
        if isinstance(v, dict) and dataclasses.is_dataclass(t):
            kwargs[k] = build_cfg(t, v)
        else:
            kwargs[k] = v
    return cls(**kwargs)
