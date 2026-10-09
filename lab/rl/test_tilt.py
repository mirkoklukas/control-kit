"""Build the model, run one test, print a summary, save npz + model XML.

    uv run --extra mjx python -m lab.rl.test_tilt                          # standup (default)
    uv run --extra mjx python -m lab.rl.test_tilt test=hold tilt_deg=90    # hold on a wall
    uv run --extra mjx python -m lab.rl.test_tilt test=sweep tilt_deg=180  # 0 -> 180 deg
    uv run --extra mjx ctk play runs/rl/test_tilt_hold-90.npz        # replay (mjpython)

Config: :class:`TiltCfg` -- which test, the gravity tilt, the adhesion ctrl, the
model (``mjmodel.*``, :class:`.config.MjModelCfg`) and the scripted-motion timing
(``scripted.*``, :class:`.config.ScriptedCfg`). E.g. ``test=hold tilt_deg=90
mjmodel.kp=30 mjmodel.ankle_range_deg=45``.

Output: ``runs/rl/test_tilt_<test>-<tilt_deg>.npz`` and a matching ``.xml`` (the model the
run used), which ``ctk play`` picks up.
"""
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import MjModelCfg, ScriptedCfg, parse_overrides
from .mjmodel import build
from .scripted import hold, standup

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runs" / "rl"
TESTS = ("standup", "hold", "sweep")


@dataclass
class TiltCfg:
    """Config of the tilt test."""
    mjmodel: MjModelCfg = field(default_factory=MjModelCfg)
    scripted: ScriptedCfg = field(default_factory=ScriptedCfg)
    test: str = "standup"           # standup | hold | sweep
    tilt_deg: float = 0.0           # gravity tilt about world y: 0 floor, 90 wall, 180 ceiling
    adhesion: float | None = None   # adhesion ctrl in [0, 1]; None: 0 for standup, else 1


def summary(log: dict, tcfg: TiltCfg) -> None:
    """Print what matters for sizing: torques vs limit, pad loads, slip margin."""
    cfg, test = tcfg.mjmodel, tcfg.test
    n = cfg.num_legs
    tau = np.abs(log["torque"]).reshape(len(log["time"]), n, 3)
    lim = cfg.forcerange[1]
    F = log["pad_force"]                                # (T, n, 3), floor normal = +z
    normal, tang = F[..., 2], np.linalg.norm(F[..., :2], axis=-1)
    loaded = normal > 1e-3
    ratio = np.where(loaded, tang / np.where(loaded, normal, 1.0), np.nan)
    tail = slice(-max(1, int(0.5 / (log["time"][1] - log["time"][0]))), None)  # last 0.5 s

    print(f"test          : {test}, tilt {tcfg.tilt_deg} deg, adhesion ctrl "
          f"{log['ctrl'][0, 3 * n:].max():.2f} x {cfg.adhesion_gain} N")
    print(f"servos        : forcerange +/-{lim} N m, kp {cfg.kp}, kv {cfg.kv}")
    for k, name in enumerate(("hip-yaw", "hip-pitch", "knee")):
        sat = (tau[..., k] >= 0.99 * lim).mean()
        print(f"  {name:9s} |tau| max {tau[..., k].max():5.2f}  end-mean "
              f"{tau[tail][:, :, k].mean():5.2f}  saturated {100 * sat:4.1f}% of steps")
    print(f"pad normal (N): end-mean per foot {np.round(normal[tail].mean(0), 1)}  "
          f"min over run {normal.min():.1f}")
    print(f"pad shear  (N): end-mean per foot {np.round(tang[tail].mean(0), 1)}")
    print(f"shear/normal  : max {np.nanmax(ratio) if loaded.any() else float('nan'):.2f} "
          f"(mu {cfg.pad_friction})  unloaded-foot steps {100 * (~loaded).mean():.1f}%")
    print(f"ankle (deg)   : |max| {np.degrees(np.abs(log['ankle']).max()):.1f} "
          f"(stop {cfg.ankle_range_deg})")
    if "height_target" in log:
        stand = log["height_target"] >= log["height_target"].max() - 1e-9
        h = log["qpos"][:, 2]
        print(f"body height   : target {cfg.stand_height:.3f}  reached "
              f"{h[stand].mean():.3f} +/- {h[stand].std():.4f}  final {h[-1]:.3f}")
    drift = np.linalg.norm(log["qpos"][-1, :3] - log["qpos"][0, :3])
    print(f"base drift    : {1000 * drift:.1f} mm (start -> end)")


def main(argv: list[str]) -> None:
    tcfg, _ = parse_overrides(TiltCfg, argv, {})
    if tcfg.test not in TESTS:
        raise SystemExit(f"test must be one of {TESTS}")
    adhesion = tcfg.adhesion
    if adhesion is None:
        adhesion = 0.0 if tcfg.test == "standup" else 1.0
    spec, model = build(tcfg.mjmodel, tilt_deg=tcfg.tilt_deg)
    if tcfg.test == "standup":
        log = standup(model, tcfg.mjmodel, tcfg.scripted, adhesion=adhesion)
    else:
        log = hold(model, tcfg.mjmodel, tcfg.scripted, tilt_deg=tcfg.tilt_deg,
                   adhesion=adhesion, sweep=(tcfg.test == "sweep"))
    summary(log, tcfg)

    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / f"test_tilt_{tcfg.test}-{tcfg.tilt_deg:g}"
    stem.with_suffix(".xml").write_text(spec.to_xml())
    np.savez(stem.with_suffix(".npz"), **log, timestep=model.opt.timestep,
             model=str(stem.with_suffix(".xml").relative_to(REPO)))
    print(f"saved {stem.with_suffix('.npz').relative_to(REPO)} (+ .xml)")


if __name__ == "__main__":
    main(sys.argv[1:])
