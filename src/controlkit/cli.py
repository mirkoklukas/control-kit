"""controlkit CLI (`ctk`).

Umbrella Typer app that mounts the package's subcommands: `viz` (replay/plot
saved .npz trajectories) and `model-gen` (hexapod model generator); grows as
more land.
"""
import subprocess
from pathlib import Path
from typing import Optional

import typer

from controlkit.modelgen import app as modelgen_app
from controlkit.viz import app as viz_app
from controlkit.viz import play as _play

_ROOT = Path(__file__).resolve().parents[2]  # src/controlkit/cli.py -> repo root

app = typer.Typer(add_completion=False, no_args_is_help=True, help="controlkit CLI.")
app.command("play")(_play)
app.add_typer(viz_app, name="viz")
app.add_typer(modelgen_app, name="model-gen")


@app.command("pull")
def pull_runs(
    host: str = typer.Argument(..., help="Remote host (ssh alias or user@host)."),
    root: Optional[Path] = typer.Argument(
        None, help="Local destination root (default: repo root)."
    ),
) -> None:
    """rsync the remote `control-kit/runs/` tree into `<root>/runs/`."""
    root = root or _ROOT
    cmd = ["rsync", "-avz", "--progress", f"{host}:control-kit/runs/", f"{root}/runs/"]
    typer.echo(" ".join(cmd))
    raise typer.Exit(subprocess.call(cmd))


if __name__ == "__main__":
    app()
