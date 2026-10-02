"""Magnet test: does a three-foot stance hold, with one leg lifted, under any gravity?

Test ladder step 3 (``lab/rl/docs/notes.md``): the worst case of a crawl on a wall. For
every gravity direction on a grid, the robot stands with three magnets on and lifts the
fourth foot; the trial passes if the stance holds for ``hold_s``.

    uv run runkit run lab.rl.climb.magnet_test                       # full grid
    uv run runkit run lab.rl.climb.magnet_test mjmodel.adhesion_gain=20 --tag=a20
    uv run runkit run lab.rl.climb.magnet_test tilt_step=30 azimuth_step=45   # coarse
    uv run --extra mjx ctk play lab/rl/climb/runs/magnet_test/latest/out/worst.npz

One trial (the env's model, :class:`.env.ClimbEnv`, driven by servo targets directly):

1. Reset: ``stand``, all magnets on, settled under normal gravity (no joint noise).
2. Ramp ``ramp_s``: gravity tilts from the floor normal to ``tilt`` towards ``azimuth``.
3. Lift ``lift_s``: the lifted leg's magnet off, its foot raised ``lift_height`` (IK,
   :func:`.poses.leg_angles`), the servo targets moving there linearly.
4. Hold ``hold_s``.

Failure, checked every control step from the ramp on (the first one ends the trial):

- ``detached``: fewer than 3 stance feet attached (magnet on, all cells, pad flat);
- ``slipped``: a stance pad moved more than ``slip_max`` from where it settled;
- ``trunk``: the trunk touches the floor.

Gravity grid: tilt from the floor normal in [0, 180] by ``tilt_step`` (0 floor, 90 wall,
180 ceiling); azimuth (direction of gravity's in-floor part, about z from +x) in [0, 360)
by ``azimuth_step``; at tilt 0 and 180 the azimuth does not matter (one trial each). One
lifted leg covers all four: the robot is 4-fold symmetric, so lifting leg i at azimuth
phi is lifting leg 0 at phi - 90 i. Leg 0 (LF) sits at azimuth 45 deg: at azimuth 45,
gravity pulls the robot towards the lifted leg.

A trial runs to the end even after its first failure, to tell a stance that hangs on
(a pad tipped past ``pad_tilt_max_deg`` or lost a cell) from one that falls off (base
moved more than ``fall_drift``, or the trunk on the floor).

Printed: a grid (rows tilt, columns azimuth) of each trial's outcome, the margins of the
held ones (stance-pad slip, peak servo torque / limit) and the worst trial. Saved in the
run's ``out/``: ``trials.jsonl`` (one row per trial), ``grid.png`` and ``worst.npz`` /
``.xml`` (a replay of the worst trial for ``ctk play``).
"""
import dataclasses
import json
import math
import sys
import time

import mujoco
import numpy as np

from runkit import Experiment, RunContext

from .config import ClimbEnvCfg, MjModelCfg
from .env import ClimbEnv
from .mjmodel import build, gravity_dir, make_robot
from .poses import leg_angles

OUTCOME_CHAR = {"held": ".", "slipped": "s", "detached": "d", "trunk": "t"}


@dataclasses.dataclass
class MagnetTestCfg:
    """The model, and the test protocol and grid."""
    mjmodel: MjModelCfg = dataclasses.field(default_factory=MjModelCfg)
    lifted_leg: int = 0             # 0 LF, 1 LB, 2 RB, 3 RF (one is enough: symmetry)
    lift_height: float = 0.03       # foot raised this far (m)
    settle_s: float = 0.2           # reset: settle under normal gravity (s)
    ramp_s: float = 1.0             # gravity tilts to the trial's direction (s)
    lift_s: float = 0.5             # magnet off, foot raised (s)
    hold_s: float = 2.0             # then held (s)
    tilt_step: float = 15.0         # grid step in tilt (deg)
    azimuth_step: float = 15.0      # grid step in azimuth (deg)
    slip_max: float = 0.005         # a stance pad moving more than this has slipped (m)
    min_attached: int = 3           # fewer stance feet attached: detached
    fall_drift: float = 0.05        # base moved more than this (or trunk down): fell (m)


def grid(cfg: MagnetTestCfg) -> list[tuple[float, float]]:
    """The trials' ``(tilt, azimuth)`` in degrees; one azimuth at tilt 0 and 180."""
    tilts = np.arange(0.0, 180.0 + 1e-9, cfg.tilt_step)
    azims = np.arange(0.0, 360.0 - 1e-9, cfg.azimuth_step)
    return [(t, a) for t in tilts for a in (azims if 0 < t < 180 else [0.0])]


class Trial:
    """One gravity direction: the robot, the lift targets and the measurements.

    Args:
        env: the climb env (its model, data and contact readout are reused).
        cfg: the test config.
        servo_lift: (12,) servo targets with the lifted leg raised.
    """

    def __init__(self, env: ClimbEnv, cfg: MagnetTestCfg, servo_lift: np.ndarray):
        self.env, self.cfg, self.servo_lift = env, cfg, servo_lift
        self.stance = [i for i in range(4) if i != cfg.lifted_leg]
        self.n_sub = env.n_sub                     # sim steps per check (one control step)

    def run(self, tilt: float, azim: float, log: bool = False) -> dict:
        """Run the trial.

        Args:
            tilt, azim: gravity direction (deg).
            log: also keep ``qpos`` per control step (for the replay).

        The trial runs to the end even after a failure, to see whether the robot
        hangs on (a pad tipped or lost a cell) or falls off.

        Returns:
            Row: ``tilt``, ``azimuth``; the first failure: ``outcome`` (held / slipped
            / detached / trunk), ``phase`` and ``t`` (s, into that phase), ``foot`` (the
            stance foot that failed first) and ``why`` (its cells in contact, pad tilt
            in deg, magnet); ``fell`` (base drift > ``fall_drift`` or trunk down, by
            the end), ``drift_mm`` (final base drift), ``slip_mm`` (max stance-pad slip),
            ``torque`` (peak servo |torque| / limit), ``attached_min`` (fewest stance feet
            attached); with ``log``, also ``qpos``.
        """
        env, cfg, m, d = self.env, self.cfg, self.env.model, self.env.data
        env.reset(options={"tilt_deg": 0.0})       # settled, magnets on, normal gravity
        servo0 = env.servo0.copy()
        pad0 = d.xpos[env.pads[self.stance]].copy()
        base0 = d.qpos[:3].copy()
        tau_max = m.actuator_forcerange[:12, 1].max()
        row = {"tilt": float(tilt), "azimuth": float(azim), "outcome": "held",
               "phase": None, "t": None, "foot": None, "why": None, "fell": False,
               "drift_mm": 0.0, "slip_mm": 0.0, "torque": 0.0,
               "attached_min": len(self.stance)}
        qpos = []

        phases = (("ramp", cfg.ramp_s), ("lift", cfg.lift_s), ("hold", cfg.hold_s))
        for phase, duration in phases:
            n = max(1, round(duration / env.dt))
            for k in range(1, n + 1):
                f = k / n                                        # progress in the phase
                if phase == "ramp":
                    m.opt.gravity[:] = gravity_dir(f * tilt, azim)
                elif phase == "lift":
                    if k == 1:                                   # lifted leg's magnet off
                        env.mag_on[cfg.lifted_leg] = False
                        d.ctrl[env.adh[cfg.lifted_leg]] = 0.0
                    d.ctrl[:12] = (1 - f) * servo0 + f * self.servo_lift
                for _ in range(self.n_sub):
                    mujoco.mj_step(m, d)
                if log:
                    qpos.append(d.qpos.copy())
                # --- checks
                _, cells_touching, trunk = env._contact_state()
                att = env._attached(cells_touching)[self.stance]
                slips = np.linalg.norm(d.xpos[env.pads[self.stance]] - pad0, axis=1)
                row["slip_mm"] = max(row["slip_mm"], 1e3 * float(slips.max()))
                row["torque"] = max(row["torque"],
                                    float(np.abs(d.actuator_force[:12]).max() / tau_max))
                row["attached_min"] = min(row["attached_min"], int(att.sum()))
                row["fell"] |= bool(trunk)
                fail = ("trunk" if trunk else "detached" if att.sum() < cfg.min_attached
                        else "slipped" if slips.max() > cfg.slip_max else None)
                if fail and row["outcome"] == "held":           # the first failure
                    j = int(np.argmin(att)) if fail == "detached" else int(np.argmax(slips))
                    i = self.stance[j]
                    tilt_pad = math.degrees(math.acos(np.clip(-d.xmat[env.pads[i]][6], -1, 1)))
                    row.update(outcome=fail, phase=phase, t=round(k * env.dt, 3), foot=i,
                               why={"cells_touching": int(cells_touching[i]),
                                    "pad_tilt_deg": round(tilt_pad, 1),
                                    "magnet": bool(env.mag_on[i])})
        drift = float(np.linalg.norm(d.qpos[:3] - base0))
        row["fell"] |= drift > cfg.fall_drift
        row["drift_mm"] = round(1e3 * drift, 1)
        row["slip_mm"], row["torque"] = round(row["slip_mm"], 2), round(row["torque"], 3)
        if log:
            row["qpos"] = np.array(qpos)
        return row


def print_grid(rows: list[dict], cfg: MagnetTestCfg) -> None:
    """Rows = tilt, columns = azimuth; one character per trial (see the legend)."""
    azims = sorted({r["azimuth"] for r in rows if 0 < r["tilt"] < 180})
    by = {(r["tilt"], r["azimuth"]): r for r in rows}
    head = [" "] * (3 * len(azims) + 3)                 # 3 chars per azimuth column
    for j, a in enumerate(azims):
        if a % 90 == 0:
            label = f"{a:.0f}"
            head[3 * j + 1: 3 * j + 1 + len(label)] = label
    head = "".join(head).rstrip()
    print(f"\n  outcome per gravity direction (lifted leg {cfg.lifted_leg}, at azimuth "
          f"{45 + 90 * cfg.lifted_leg} deg)")
    print(f"  {'tilt':>4}  azimuth -> {head}")
    for tilt in sorted({r["tilt"] for r in rows}):
        cells = ([by[(tilt, a)] for a in azims] if 0 < tilt < 180
                 else [by[(tilt, 0.0)]] * len(azims))
        line = "".join(f" {_char(c)} " for c in cells)
        held = [c for c in cells if c["outcome"] == "held"]
        worst = (f"slip <= {max(c['slip_mm'] for c in held):4.1f} mm, torque <= "
                 f"{max(c['torque'] for c in held):.2f}" if held else "")
        print(f"  {tilt:>4.0f}  {' ' * 11}{line}  {worst}")
    print("  legend: . held   s slipped   d detached (< 3 stance feet attached)   "
          "t trunk on the floor;\n          upper case: fell off by the end (S, D, T); "
          "torque = peak servo |torque| / limit")


def _char(row: dict) -> str:
    """The grid character of a trial: its first failure, upper case if it fell."""
    c = OUTCOME_CHAR[row["outcome"]]
    return c.upper() if row["fell"] else c


def plot_grid(rows: list[dict], path) -> None:
    """Heatmap of the outcome, held trials shaded by their stance-pad slip."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tilts = sorted({r["tilt"] for r in rows})
    azims = sorted({r["azimuth"] for r in rows if 0 < r["tilt"] < 180}) or [0.0]
    by = {(r["tilt"], r["azimuth"]): r for r in rows}
    z = np.full((len(tilts), len(azims)), np.nan)
    for i, t in enumerate(tilts):
        for j, a in enumerate(azims):
            r = by.get((t, a)) or by[(t, 0.0)]
            z[i, j] = r["slip_mm"] if r["outcome"] == "held" else -1.0
    fig, ax = plt.subplots(figsize=(9, 5))
    held = np.ma.masked_less(z, 0)
    im = ax.imshow(held, origin="lower", aspect="auto", cmap="viridis",
                   extent=(azims[0], azims[-1], tilts[0], tilts[-1]))
    ax.imshow(np.ma.masked_greater_equal(z, 0), origin="lower", aspect="auto",
              cmap="Reds_r", vmin=-1, vmax=0, extent=(azims[0], azims[-1], tilts[0], tilts[-1]))
    fig.colorbar(im, ax=ax, label="held: max stance-pad slip (mm)")
    ax.set(xlabel="gravity azimuth (deg)", ylabel="gravity tilt from floor normal (deg)",
           title="three-foot hold, one leg lifted (red: failed)")
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


exp = Experiment("magnet_test")


@exp.run
def run(cfg: MagnetTestCfg, ctx: RunContext) -> dict:
    """Run every gravity direction of the grid; print the grid and the worst trial.

    Returns:
        Summary: ``trials``, ``held`` (fraction), counts per outcome, ``worst`` (the
        failed trial at the smallest tilt, else the held one with the most slip),
        ``slip_mm_max`` / ``torque_max`` over the held trials.
    """
    env_cfg = ClimbEnvCfg(magnet_mode="policy", magnet_start_on=True, reset_noise=0.0,
                          settle_s=cfg.settle_s, gravity_random=False)
    env = ClimbEnv(cfg.mjmodel, env_cfg, seed=0)
    lift = np.zeros(4)
    lift[cfg.lifted_leg] = cfg.lift_height
    ok, thetas = leg_angles(make_robot(cfg.mjmodel), cfg.mjmodel, cfg.mjmodel.stand_height, lift)
    if not bool(np.asarray(ok).all()):
        raise SystemExit(f"lift_height={cfg.lift_height}: foot target unreachable")
    trial = Trial(env, cfg, np.asarray(thetas).reshape(-1))

    trials = grid(cfg)
    print(f"magnet test: {len(trials)} gravity directions, leg {cfg.lifted_leg} lifted "
          f"{100 * cfg.lift_height:.0f} cm, adhesion {cfg.mjmodel.adhesion_gain:.0f} N/foot, "
          f"servo limit {cfg.mjmodel.forcerange[1]:.0f} N m, "
          f"{cfg.mjmodel.pad_cell_shape} cells; ramp {cfg.ramp_s} s, lift {cfg.lift_s} s, "
          f"hold {cfg.hold_s} s")
    ctx.progress(0, total=len(trials))
    rows, t0 = [], time.perf_counter()
    with open(ctx.out / "trials.jsonl", "w") as f:
        for k, (tilt, azim) in enumerate(trials):
            row = trial.run(tilt, azim)
            rows.append(row)
            f.write(json.dumps(row) + "\n")
            ctx.progress(k + 1)
    print(f"  ({time.perf_counter() - t0:.0f} s)")

    print_grid(rows, cfg)
    failed = [r for r in rows if r["outcome"] != "held"]
    held = [r for r in rows if r["outcome"] == "held"]
    # worst: a fall at the smallest tilt, else any failure there, else the most slip
    worst = (min(failed, key=lambda r: (not r["fell"], r["tilt"], r["azimuth"])) if failed
             else max(held, key=lambda r: r["slip_mm"]))
    print(f"\n  worst: tilt {worst['tilt']:.0f}, azimuth {worst['azimuth']:.0f} -> "
          f"{worst['outcome']}" + (f" in {worst['phase']} at {worst['t']} s, foot "
                                   f"{worst['foot']} {worst['why']}" if worst["phase"] else "")
          + f"; {'fell' if worst['fell'] else 'hung on'} (drift {worst['drift_mm']} mm), "
          f"slip {worst['slip_mm']} mm, torque {worst['torque']}, "
          f"stance attached >= {worst['attached_min']}")

    # replay of the worst trial (gravity in the xml: the trial's final direction)
    rep = trial.run(worst["tilt"], worst["azimuth"], log=True)
    spec, _ = build(cfg.mjmodel, write=False)
    spec.option.gravity = gravity_dir(worst["tilt"], worst["azimuth"])
    xml = ctx.out / "worst.xml"
    xml.write_text(spec.to_xml())
    np.savez(ctx.out / "worst.npz", qpos=rep["qpos"], timestep=env.dt, model=str(xml))
    plot_grid(rows, ctx.out / "grid.png")
    print(f"  saved {ctx.out}/: trials.jsonl, grid.png, worst.npz (+ .xml)")

    counts = {o: sum(r["outcome"] == o for r in rows) for o in OUTCOME_CHAR}
    return {"trials": len(rows), "held": round(counts["held"] / len(rows), 3), **counts,
            "fell": sum(r["fell"] for r in rows),
            "worst": {k: worst[k] for k in ("tilt", "azimuth", "outcome", "phase", "t")},
            "slip_mm_max": max((r["slip_mm"] for r in held), default=None),
            "torque_max": max((r["torque"] for r in held), default=None)}


if __name__ == "__main__":
    exp.main(sys.argv[1:])
