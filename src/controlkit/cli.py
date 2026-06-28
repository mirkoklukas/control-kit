"""controlkit CLI (`ctk`).

Umbrella Typer app that mounts the package's subcommands. Currently just
`viz` (replay/plot saved .npz trajectories); grows as more land.
"""
import typer

from controlkit.viz import app as viz_app

app = typer.Typer(add_completion=False, no_args_is_help=True, help="controlkit CLI.")
app.add_typer(viz_app, name="viz")

if __name__ == "__main__":
    app()
