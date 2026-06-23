"""Locate and load a context yaml (run.context.yaml).

The context yaml carries the staging half of a run: runs_root, repos_in_dev,
allow_dirty, lockfile. (The config half — the experiment params — is a
separate `config` yaml.) Discovery bubbles up from `start` through parents
until a file is found, but runkit does NOT auto-discover by default; pass
--context=PATH explicitly.
"""
import pathlib
import yaml

CONTEXT_FILENAME = "run.context.yaml"


def find_context(start=None):
    """Return the path to the nearest run.context.yaml at or above `start`."""
    start = pathlib.Path(start or pathlib.Path.cwd()).resolve()
    for d in (start, *start.parents):
        candidate = d / CONTEXT_FILENAME
        if candidate.is_file():
            return candidate
    return None


def load_context(start=None):
    """Return (context_dict, context_path). Empty dict if none found."""
    path = find_context(start)
    if path is None:
        return {}, None
    data = yaml.safe_load(path.read_text()) or {}
    return data, path
