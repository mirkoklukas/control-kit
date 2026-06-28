"""controlkit CLI (`ctk`).

Umbrella Typer app that mounts the package's subcommands: `viz` (replay/plot
saved .npz trajectories) and `model-gen` (hexapod model generator); grows as
more land.
"""
import typer

from controlkit.modelgen import app as modelgen_app
from controlkit.viz import app as viz_app

app = typer.Typer(add_completion=False, no_args_is_help=True, help="controlkit CLI.")
app.add_typer(viz_app, name="viz")
app.add_typer(modelgen_app, name="model-gen")

if __name__ == "__main__":
    app()
