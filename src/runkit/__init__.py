"""runkit — lightweight, reproducible experiment runs.

Decorate with @experiment(name=...); the run dir, provenance capture, and
dirty-git gate are handled behind the curtain. By default runs land under
./runs with no tracked repos; pass --context=PATH (or context= to
@experiment) to load a run.context.yaml layer.

Vocabulary: a *run spec* has two halves — *config* (the experiment params,
key=value) and *context* (where/how it's staged, --flags).
"""
from .runs import experiment, RunContext, init_run
from .context import load_context, find_context
from .main import main

__all__ = ["experiment", "RunContext", "init_run", "main",
           "load_context", "find_context"]
