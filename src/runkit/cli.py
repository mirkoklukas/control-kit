"""The `runkit` console script.

Usage:
    runkit run <script.py> [key=value ...] [--flag ...]

Imports the script, finds the @experiment-decorated function (named `run`
by convention), and delegates to runkit.main with the remaining argv.
"""
import importlib.util
import pathlib
import sys

import typer

from .main import main as run_main

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  pretty_exceptions_enable=False)


@app.callback()
def _root():
    """runkit — lightweight, reproducible experiment runs."""


@app.command(
    name="run",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True,
                      "help_option_names": []},
)
def run_cmd(ctx: typer.Context, script: str = typer.Argument(...)):
    """Run an @experiment-decorated function from a script."""
    fn = _load_run(pathlib.Path(script))
    run_main(fn, argv=ctx.args)


def _load_run(script):
    if not script.is_file():
        sys.exit(f"script not found: {script}")
    spec = importlib.util.spec_from_file_location("_runkit_user_script", script)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    fn = getattr(mod, "run", None)
    if fn is None:
        sys.exit(f"{script}: no `run` function found at module level")
    return fn


if __name__ == "__main__":
    app()
