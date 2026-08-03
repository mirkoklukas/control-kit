"""Hexapod model generator, exposed under `ctk model-gen`.

The implementation lives in tools/model_generator/ (a standalone web tool plus a
YAML->MJCF generator); this module just wires it into the CLI. `serve` opens the
interactive web tool; `build` turns a YAML spec into an XML model.
"""
import subprocess
import sys
from pathlib import Path

import typer

_ROOT = Path(__file__).resolve().parents[2]   # src/controlkit/modelgen.py -> repo root
_TOOL = _ROOT / "tools" / "model_generator"

app = typer.Typer(
    add_completion=False, no_args_is_help=True,
    help="Design a hexapod and emit a MuJoCo MJCF. `serve` opens the interactive "
         "web tool (three.js preview + save); `build` turns a YAML spec into XML.",
)


@app.command()
def serve(open_browser: bool = typer.Option(True, "--open/--no-open", help="open a browser tab")):
    """Launch the interactive web tool at http://localhost:8001 (Ctrl-C to stop)."""
    args = [sys.executable, str(_TOOL / "serve.py")]
    if not open_browser:
        args.append("--no-open")
    subprocess.run(args, cwd=_ROOT)


@app.command()
def build(
    config: str = typer.Argument(..., help="path to the YAML spec"),
    out: str = typer.Option(None, "-o", "--out", help="output .xml (default: spec path with .xml)"),
):
    """Generate an MJCF model from a YAML spec."""
    sys.path.insert(0, str(_TOOL))
    import generate

    cfg = generate.load_config(config)
    out = out or str(Path(config).with_suffix(".xml"))
    print(f"wrote {generate.write_model(cfg, out)}")
