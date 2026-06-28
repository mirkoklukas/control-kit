"""Visualize a finished mock_exp run: the toy reward curve.

Sibling entry point to ``run.py`` -- ``run.py`` produces a run, this renders one
that already exists. Takes a plain ``run_dir`` path (works on any finished run,
including one rsync-ed down from the GPU box), reads the per-checkpoint rewards
this experiment saved, and saves a reward-vs-step plot back into the run dir.

    uv run python -m lab.mock_exp.viz runs/<run-dir>

(The eventual runkit `@viz` decorator -- see docs/notes.md -- would let this be
`viz(ctx)` with the CLI rehydrating a read-only ctx from the run_dir. For now the
experiment reads its own artifact layout directly.)
"""
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # always headless; we save, never show
import matplotlib.pyplot as plt  # noqa: E402
import typer  # noqa: E402


def viz(run_dir: Path) -> None:
    """Plot the toy reward curve from a mock_exp run dir; save it into the dir."""
    run_dir = Path(run_dir)
    ckpts = sorted(
        run_dir.glob("checkpoints/step_*.pkl"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    if not ckpts:
        raise SystemExit(f"no checkpoints under {run_dir}/checkpoints/")
    pts = [pickle.loads(p.read_bytes()) for p in ckpts]
    steps = [d["step"] for d in pts]
    rewards = [d["reward"] for d in pts]

    final_path = run_dir / "results" / "final.pkl"
    final = pickle.loads(final_path.read_bytes()) if final_path.exists() else None

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, rewards, marker="o")
    ax.set(title=f"mock_exp reward — {run_dir.name}", xlabel="step", ylabel="reward")
    ax.grid(alpha=0.3)
    if final is not None:
        seed = final.get("cfg", {}).get("seed")
        ax.axhline(final["reward"], color="r", ls="--", lw=1,
                   label=f"final={final['reward']:.4f}"
                         + (f" (seed={seed})" if seed is not None else ""))
        ax.legend(loc="best")

    out = run_dir / "results" / "reward.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"saved plot -> {out}")


if __name__ == "__main__":
    typer.run(viz)
