"""Capture the three reproduction axes: software and hardware.

(Parameters come from the config object and are frozen separately.)
"""
import datetime
import hashlib
import importlib.metadata
import pathlib
import platform
import subprocess
import sys


def _git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, text=True,
    ).stdout.strip()


def repo_state(path):
    """Commit + dirty flag for one git working tree."""
    path = pathlib.Path(path)
    inside = _git(path, "rev-parse", "--is-inside-work-tree")
    if inside != "true":
        raise ValueError(f"not a git repo: {path}")
    return {
        "path": str(path),
        "sha": _git(path, "rev-parse", "HEAD"),
        "dirty": bool(_git(path, "status", "--porcelain")),
    }


def collect_repos(repos_in_dev):
    """Map {name: path} -> {name: state}; also report if any is dirty."""
    states, any_dirty = {}, False
    for name, path in repos_in_dev.items():
        p = pathlib.Path(path)
        if not p.is_dir():
            raise ValueError(f"repo path for {name!r} does not exist: {p}")
        states[name] = repo_state(p)
        any_dirty |= states[name]["dirty"]
    return states, any_dirty


def installed_packages():
    """The realized package set of the live interpreter (ground truth)."""
    return {
        d.metadata["Name"]: d.version
        for d in importlib.metadata.distributions()
    }


def find_lockfile(start=None, filename="uv.lock"):
    """Best-effort discovery of the lockfile defining the running environment.

    Anchors, in order:
      1. the project root under uv's default layout: the parent of the venv
         (`sys.prefix`), since uv puts `.venv` next to `pyproject.toml`/`uv.lock`.
         This ties the lockfile to the env that is actually executing, not cwd.
      2. the cwd (or `start`), bubbling up through parents.
    Returns the first existing path, or None.
    """
    anchors = [pathlib.Path(sys.prefix).parent,
               pathlib.Path(start or pathlib.Path.cwd())]
    seen = set()
    for anchor in anchors:
        anchor = anchor.resolve()
        for d in (anchor, *anchor.parents):
            if d in seen:
                continue
            seen.add(d)
            cand = d / filename
            if cand.is_file():
                return cand
    return None


def lockfile_info(lockfile):
    if not lockfile:
        return None
    p = pathlib.Path(lockfile)
    if not p.is_file():
        return {"path": str(p), "present": False}
    return {
        "path": str(p),
        "present": True,
        "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
    }


def software_record(repos_in_dev, lockfile, command):
    """SOFTWARE axis: what actually executed."""
    repos, any_dirty = collect_repos(repos_in_dev)
    record = {
        "command": command,
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "venv": sys.prefix,
        "repos_in_dev": repos,
        "lockfile": lockfile_info(lockfile),
        "packages": installed_packages(),
    }
    return record, any_dirty


def hardware_record():
    """HARDWARE axis: what it ran on (diagnostic, not reproduction-critical)."""
    record = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hostname": platform.node(),
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    try:
        gpu = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        if gpu:
            record["gpu"] = gpu
    except Exception:
        pass  # no GPU / no nvidia-smi — fine
    return record
