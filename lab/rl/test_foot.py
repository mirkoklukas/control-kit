"""Foot pull test: single-foot fixtures in a row, each pulled until it lets go.

    uv run --extra mjx python -m lab.rl.test_foot                 # all cases
    uv run --extra mjx python -m lab.rl.test_foot mjmodel.adhesion_gain=40 pull_rate=5
    uv run --extra mjx ctk play runs/rl/test_foot.npz             # replay

What one fixture is
-------------------
A tibia (length ``leg_lengths[-1]``) rigidly attached to a *carriage* at its top
end, carrying the real foot from :func:`.mjmodel._add_foot` at its bottom end (Cardan
ankle with weak centering springs and +/-``ankle_range_deg`` hard stops, a pad of
N x N cells with one ``adhesion`` actuator each). The floor plays the wall. Gravity
is off, so the pull is the only load. Two carriage types:

- **locked** (blue): the carriage slides freely along world x/y/z but cannot
  rotate. The tibia keeps the lean it was attached at -- a leg whose joints hold
  the tibia's angle perfectly stiff.
- **rigid** (purple): locked carriage, and the ankle held at fixed angles by joint
  equalities -- the pad keeps a set orientation (flat, on an edge, on a corner).
- **free** (orange): the carriage is a free body. The tibia can also rotate, so
  with the weak ankle springs it swings into line with the pull, until the ankle
  hits its stop; past that the stop levers the pad (peel).

Protocol
--------
1. Settle ``pull_settle`` s: adhesion on (ctrl 1), no pull.
2. Pull: a force at the carriage (the tibia's top end), in the case's direction,
   ramping at ``pull_rate`` N/s up to ``pull_max``.
3. Release: when the pad has moved ``pull_detach`` from where it settled, the force
   at that moment is the *release force*, and that fixture's pull switches off
   (the pad may then re-stick if it only slid, or float away if it lifted off).

Geometry: fixtures are lined up along +x. The tibia lean and the pull direction
are angles in the **y-z plane**, measured from the surface normal (+z): pull 0 is
straight off the wall, +90 is shear along +y, -90 shear along -y. A lean of
``tilt`` puts the tibia's foot end towards +y (top end towards -y), so shear +90
pulls the way the leg leans and -90 against it.

Sections (cases are grouped; a gap along x between sections, a bigger one between
the two carriage types):

- **locked / upright** -- pull 0, 30, 60, 90 on an upright tibia: strength vs pull
  direction. Expect pull-off at the adhesion force A, and slip at
  ``mu A / (sin t + mu cos t)`` for pull angle t (Coulomb with adhesion).
- **locked / lean 30** -- tibia attached at 30 deg (the ankle bends 30 so the pad is
  flat); pull 0, shear with and against the lean. Does a leaning leg change
  strength? (Only through the pivot height -- the ankle takes the lean.)
- **free / upright** -- pull 0, 20, 40 at the top of a free leg. The leg swings
  into line with the pull, so the force goes through the ankle pivot; compare with
  the locked pulls at the same angle.
- **free / start leaned** -- lean 20 / 40: the free leg starts leaned (ankle bent,
  pad flat), then is pulled straight off.

- **rigid / flat, edge, corner** -- pad held flat (16 contacts), on an edge (tilted
  20 deg about one axis) or on a corner (20 deg about both); pull 0 and shear 90.
  Checks the MuJoCo adhesion flaw: if the edge and corner hold ~30 N like the
  flat pad, adhesion ignores contact area, and a policy could land sloppily for free.

Everything stays within the ankle range. Past the stop, the tibia levers the pad
through the stop (peel); and a pad starting on an edge is not flattened by MuJoCo's
adhesion (force only at contact points) -- see docs/notes.md, docs/foot_design.md.

Everything is logged for ``ctk play``: the pull is saved as ``qfrc_applied``
(re-applied in replay, so contact forces are right) and drawn as red arrows
(``force_pos``/``force_vec``, at 4x the contact-force scale).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import NamedTuple

import mujoco
import numpy as np

from .config import MjModelCfg, parse_overrides
from .mjmodel import _add_foot, adhesion_actuators, pad_cell_bodies
from .poses import ankle_angles

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runs" / "rl"

COLORS = {"locked": (0.04, 0.76, 1.0), "free": (1.0, 0.55, 0.1), "rigid": (0.6, 0.3, 0.9)}


@dataclass
class FootTestCfg:
    """Config of the foot pull test: the model (``mjmodel.*``) plus the pull protocol."""
    mjmodel: MjModelCfg = field(default_factory=MjModelCfg)
    pull_rate: float = 10.0         # force ramp (N/s)
    pull_max: float = 60.0          # ramp stops here (N)
    pull_settle: float = 0.5        # adhesion on, no pull, before the ramp (s)
    pull_detach: float = 0.005      # pad displacement counted as detached / slipped (m)
    fixture_spacing: float = 0.15   # fixtures side by side (m)


class Case(NamedTuple):
    section: str      # layout group (gap between sections)
    name: str
    carriage: str     # "locked" | "free" | "rigid"
    pull: float       # pull direction from the surface normal, y-z plane (deg)
    tilt: float       # tibia lean from the surface normal, y-z plane (deg)
    pad_tilt: tuple = (0.0, 0.0)   # rigid only: ankle a/b held at these angles (deg)


# All cases stay within the ankle range (+/-45 deg by default): tibia leans and, for
# the free leg, pull angles (it swings into line with the pull) are below the stop.
CASES = (
    # locked carriage, upright tibia: how strength depends on pull direction
    Case("locked / upright", "pull 0", "locked", 0.0, 0.0),
    Case("locked / upright", "pull 30", "locked", 30.0, 0.0),
    Case("locked / upright", "pull 60", "locked", 60.0, 0.0),
    Case("locked / upright", "shear 90", "locked", 90.0, 0.0),
    # locked carriage, tibia leaning 30 deg (ankle bends to keep the pad flat)
    Case("locked / lean 30", "pull 0", "locked", 0.0, 30.0),
    Case("locked / lean 30", "shear with lean", "locked", 90.0, 30.0),
    Case("locked / lean 30", "shear against lean", "locked", -90.0, 30.0),
    # free leg, force at the tibia top: leg aligns with the pull (below the stop)
    Case("free / upright", "pull 0", "free", 0.0, 0.0),
    Case("free / upright", "pull 20", "free", 20.0, 0.0),
    Case("free / upright", "pull 40", "free", 40.0, 0.0),
    # free leg, starting leaned (below the stop), pulled straight off
    Case("free / start leaned", "lean 20, pull 0", "free", 0.0, 20.0),
    Case("free / start leaned", "lean 40, pull 0", "free", 0.0, 40.0),
    # rigid: locked carriage + ankle held fixed -> pad kept flat / on an edge / on a
    # corner. Checks MuJoCo's adhesion: full force regardless of contact count?
    Case("rigid / flat, edge, corner", "flat, pull 0", "rigid", 0.0, 0.0, (0.0, 0.0)),
    Case("rigid / flat, edge, corner", "edge, pull 0", "rigid", 0.0, 0.0, (0.0, 20.0)),
    Case("rigid / flat, edge, corner", "corner, pull 0", "rigid", 0.0, 0.0, (20.0, 20.0)),
    Case("rigid / flat, edge, corner", "edge, shear 90", "rigid", 90.0, 0.0, (0.0, 20.0)),
    Case("rigid / flat, edge, corner", "corner, shear 90", "rigid", 90.0, 0.0, (20.0, 20.0)),
)


def _x_positions(cfg: FootTestCfg, cases) -> list[float]:
    """Fixture x positions: ``fixture_spacing`` apart, +1 spacing between sections,
    +2 between carriage types."""
    xs, x = [], 0.0
    for j, c in enumerate(cases):
        if j:
            prev = cases[j - 1]
            x += cfg.fixture_spacing * (1 + (c.section != prev.section)
                                        + (c.carriage != prev.carriage))
        xs.append(x)
    return xs


def _quat(axis, angle):
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, np.asarray(axis, float), angle)
    return q


def build(cfg: FootTestCfg, cases=CASES):
    """One fixture per case, in a row along x. Returns ``(spec, model)``.

    Names per fixture ``j``: carriage ``carriage{j}`` (locked / rigid: slides
    ``slide{j}_x/y/z``; free: freejoint ``free{j}``), tibia ``leg{j}_2``, foot
    ``foot{j}`` / ``pad{j}`` (cells ``pad{j}_c{k}``) / ``ankle{j}_a|b`` (as in the
    robot), actuators ``adhere{j}_{k}``; rigid fixtures add joint equalities pinning
    ``ankle{j}_a|b``.
    """
    spec = mujoco.MjSpec()
    spec.compiler.degree = False    # MjSpec defaults to degrees; _add_foot passes radians
    opt = spec.option
    opt.timestep = cfg.mjmodel.timestep
    opt.integrator = getattr(mujoco.mjtIntegrator, f"mjINT_{cfg.mjmodel.integrator.upper()}")
    opt.cone, opt.impratio, opt.gravity = mujoco.mjtCone.mjCONE_ELLIPTIC, 10.0, [0, 0, 0]

    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                            size=[0, 0, 0.05], rgba=[0.25, 0.3, 0.35, 1],
                            friction=[cfg.mjmodel.pad_friction, 0.005, 0.0001])
    spec.worldbody.add_light(pos=[0, 0, 2], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8])

    L, r = cfg.mjmodel.leg_lengths[-1], cfg.mjmodel.link_radius
    z_pivot = cfg.mjmodel.pivot_height + cfg.mjmodel.pad_thickness
    xs = _x_positions(cfg, cases)
    for j, c in enumerate(cases):
        phi = math.radians(c.tilt)
        rgb = COLORS[c.carriage]
        # tibia direction (top -> foot): (0, sin phi, -cos phi); pivot at (x_j, 0, z_pivot)
        car = spec.worldbody.add_body(
            name=f"carriage{j}", pos=[xs[j], -L * math.sin(phi), z_pivot + L * math.cos(phi)])
        if c.carriage in ("locked", "rigid"):
            for ax, v in zip("xyz", np.eye(3)):
                car.add_joint(name=f"slide{j}_{ax}", type=mujoco.mjtJoint.mjJNT_SLIDE,
                              axis=v, damping=0.5)
        else:
            car.add_freejoint(name=f"free{j}")
        # carriage marker (where the pull acts): visual only
        car.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, 0.02, 0.02],
                     rgba=[*rgb, 1.0], mass=0, contype=0, conaffinity=0)
        # body x-axis -> tibia direction: Ry(90) takes x to -z, then Rx(phi) leans it to +y
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, _quat([1, 0, 0], phi), _quat([0, 1, 0], math.pi / 2))
        tib = car.add_body(name=f"leg{j}_2", quat=q)
        tib.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[0, 0, 0, L, 0, 0],
                     size=[r, 0, 0], mass=cfg.mjmodel.mass_links[-1],
                     rgba=[*rgb, 0.35])                      # see-through: arrows inside
        foot = tib.add_body(name=f"foot{j}", pos=[L, 0, 0])
        foot.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.005, 0, 0])  # replaced
        _add_foot(spec, cfg.mjmodel, j)
        if c.carriage == "rigid":            # hold the ankle: q = pad_tilt
            for s_, q in zip("ab", c.pad_tilt):
                eq = spec.add_equality(type=mujoco.mjtEq.mjEQ_JOINT, name1=f"ankle{j}_{s_}")
                eq.data[:5] = [math.radians(q), 0, 0, 0, 0]
                eq.solref = [2 * cfg.mjmodel.timestep, 1.0]   # stiffest stable: pad must not yield
    return spec, spec.compile()


def _pad_corners(model, data, j):
    """World corners of pad ``j``'s boxes."""
    pts = []
    for g in np.nonzero(np.isin(model.geom_bodyid, pad_cell_bodies(model, j)))[0]:
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX:
            continue
        R = data.geom_xmat[g].reshape(3, 3)
        for s in np.array(np.meshgrid(*[[-1, 1]] * 3)).T.reshape(-1, 3):
            pts.append(data.geom_xpos[g] + R @ (s * model.geom_size[g]))
    return np.array(pts)


def initial_state(model, cfg: FootTestCfg, cases) -> mujoco.MjData:
    """Ankles laid flat (clipped to the stops), each pad's lowest corner at its resting
    height: 0.9 * ``adhesion_margin``. The contact rests ~at the margin (see notes:
    margin inflates the surface); starting at z = 0 instead is a 2 mm overlap that
    kicks the pad off on the first step -- on a leaning tibia it flips the pad."""
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    lim = math.radians(cfg.mjmodel.ankle_range_deg)
    for j, c in enumerate(cases):
        a = (np.radians(c.pad_tilt) if c.carriage == "rigid"
             else np.clip(ankle_angles(model, data, j), -lim, lim))
        data.qpos[model.joint(f"ankle{j}_a").qposadr[0]] = a[0]
        data.qpos[model.joint(f"ankle{j}_b").qposadr[0]] = a[1]
    mujoco.mj_kinematics(model, data)
    for j, c in enumerate(cases):
        dz = 0.9 * cfg.mjmodel.adhesion_margin - _pad_corners(model, data, j)[:, 2].min()
        if c.carriage in ("locked", "rigid"):
            data.qpos[model.joint(f"slide{j}_z").qposadr[0]] += dz
        else:
            data.qpos[model.joint(f"free{j}").qposadr[0] + 2] += dz
    mujoco.mj_forward(model, data)
    return data


def run(model, cfg: FootTestCfg, cases=CASES) -> dict:
    """Settle, then ramp every fixture's pull; stop each one's pull once it lets go.

    Returns:
        Log dict: ``time``, ``qpos``, ``qvel``, ``ctrl``, ``qfrc_applied`` (T, nv,
        the pulls as generalized forces), ``pull`` (T, n) force magnitude,
        ``force_pos`` / ``force_vec`` (T, n, 3) the pulls as world forces at the
        carriages, ``release`` (n,) force at release (nan: held to ``pull_max``),
        ``moved`` (n, 3) pad displacement at release / end, ``contacts0`` (n,) pad
        contacts after settling, ``tilt0`` / ``tilt_rel`` (n,) tibia lean from the
        normal after settling / at release (deg).
    """
    n, dt = len(cases), model.opt.timestep
    data = initial_state(model, cfg, cases)
    pads = np.array([model.body(f"pad{j}").id for j in range(n)])
    cars = np.array([model.body(f"carriage{j}").id for j in range(n)])
    tibs = np.array([model.body(f"leg{j}_2").id for j in range(n)])
    adhere = np.concatenate([adhesion_actuators(model, j) for j in range(n)])
    cells = [set(pad_cell_bodies(model, j)) for j in range(n)]
    dirs = np.array([[0.0, math.sin(math.radians(c.pull)), math.cos(math.radians(c.pull))]
                     for c in cases])

    def tibia_tilt():
        # angle between the tibia (body x-axis, top -> foot) and -z
        return np.degrees(np.arccos(np.clip(-data.xmat[tibs][:, 6], -1, 1)))

    data.ctrl[adhere] = 1.0
    for _ in range(int(cfg.pull_settle / dt)):
        mujoco.mj_step(model, data)
    x0 = data.xpos[pads].copy()
    tilt0, tilt_rel = tibia_tilt(), np.full(n, np.nan)
    contacts0 = np.array([sum(1 for c in data.contact[:data.ncon]
                              if {model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]} & cells[j])
                          for j in range(n)])

    release, moved = np.full(n, np.nan), np.zeros((n, 3))
    active = np.ones(n, bool)
    log = {k: [] for k in ("time", "qpos", "qvel", "ctrl", "qfrc_applied", "pull",
                           "force_pos", "force_vec")}
    steps = int((cfg.pull_max / cfg.pull_rate + 1.0) / dt)
    for k in range(steps):
        f = min(cfg.pull_rate * k * dt, cfg.pull_max)
        mag = np.where(active, f, 0.0)
        fvec = mag[:, None] * dirs
        data.qfrc_applied[:] = 0.0
        for j in range(n):
            if mag[j] > 0:
                mujoco.mj_applyFT(model, data, fvec[j], np.zeros(3), data.xpos[cars[j]],
                                  cars[j], data.qfrc_applied)
        log["force_pos"].append(data.xpos[cars].copy())    # where it was applied
        mujoco.mj_step(model, data)
        d = data.xpos[pads] - x0
        let_go = active & (np.linalg.norm(d, axis=1) > cfg.pull_detach)
        release[let_go], moved[let_go] = f, d[let_go]
        tilt_rel[let_go] = tibia_tilt()[let_go]
        active &= ~let_go
        for key, v in (("time", data.time), ("qpos", data.qpos), ("qvel", data.qvel),
                       ("ctrl", data.ctrl), ("qfrc_applied", data.qfrc_applied),
                       ("pull", mag), ("force_vec", fvec)):
            log[key].append(np.copy(v))
    moved[active] = (data.xpos[pads] - x0)[active]
    out = {k: np.asarray(v) for k, v in log.items()}
    out.update(release=release, moved=moved, contacts0=contacts0, tilt0=tilt0,
               tilt_rel=tilt_rel)
    return out


def summary(log: dict, cfg: FootTestCfg, cases=CASES) -> None:
    mu_f = cfg.mjmodel.pad_friction * cfg.mjmodel.adhesion_gain
    print(f"adhesion {cfg.mjmodel.adhesion_gain} N, mu {cfg.mjmodel.pad_friction} (mu*F = {mu_f:.1f} N), "
          f"ankle stop {cfg.mjmodel.ankle_range_deg} deg, pivot {1000 * cfg.mjmodel.pivot_height:.0f} mm, "
          f"ramp {cfg.pull_rate} N/s to {cfg.pull_max} N")
    section = None
    for j, c in enumerate(cases):
        if c.section != section:
            section = c.section
            print(f"  [{section}]")
            print(f"    {'case':24s} {'contacts':>8s} {'tibia tilt':>12s} {'release (N)':>12s}  how")
        rel, d = log["release"][j], log["moved"][j]
        how = ("held" if np.isnan(rel) else
               "pull-off" if abs(d[2]) >= np.linalg.norm(d[:2]) else "slip")
        rs = f">{cfg.pull_max:.0f}" if np.isnan(rel) else f"{rel:.1f}"
        tilt = f"{log['tilt0'][j]:.0f}->{log['tilt_rel'][j]:.0f}" if not np.isnan(rel) \
            else f"{log['tilt0'][j]:.0f}"
        print(f"    {c.name:24s} {log['contacts0'][j]:8d} {tilt:>12s} {rs:>12s}  {how}")


def main(argv: list[str]) -> None:
    cfg, _ = parse_overrides(FootTestCfg, argv, {})
    spec, model = build(cfg)
    log = run(model, cfg)
    summary(log, cfg)
    OUT.mkdir(parents=True, exist_ok=True)
    xml = OUT / "test_foot.xml"
    xml.write_text(spec.to_xml())
    np.savez(OUT / "test_foot.npz", **log, timestep=model.opt.timestep,
             model=str(xml.relative_to(REPO)),
             force_scale=4 * model.vis.map.force,   # pull arrows 4x the contact-force scale
             cases=np.array([f"{c.section}: {c.name}" for c in CASES]))
    print("saved runs/rl/test_foot.npz (+ .xml)")


if __name__ == "__main__":
    main(sys.argv[1:])
