"""Execute a planned walk in MuJoCo: play the plan open-loop on the full climb model.

Reads a plan written by :mod:`.walk` (``walk.npz``: the key postures ``key_qpos``, the
magnets on at each ``key_magnets``, the scene) and simulates it with the climb robot's
full model (position servos, passive ankles, magnetic pads as adhesion actuators; the
wall's friction = the pads' ``pad_friction``, as the planner assumes):

1. **Start**: the first key posture, all magnets on; settle ``settle_s``.
2. **Phases**: between consecutive key postures (S -> B_lift -> leg lifted -> B_plant ->
   pressed -> S'; "pressed": the swing foot slightly past the surface, to press its pad
   flat), the 12 servo targets move linearly over ``phase_s``. The body is not driven: it
   follows from the servos with the feet held. The magnets of each phase's *end* posture
   are on during it: the swing leg's switches off as its lift begins, back on as its
   foot comes down (adhesion acts only in contact).
3. **Measured** at the end of each phase, against the plan: the base position error, the
   stance feet's slip (pad displacement since the phase began, feet that stay down) and
   cells in contact, the swing foot's landing error. ``fell``: base error > ``fall_m``.

    uv run runkit run lab.gait_graph.planner.experiments.execute walk=experiments/runs/planner_walk/<run>/out/walk.npz
    uv run --extra mjx ctk play experiments/runs/planner_execute/latest/out/exec.npz

Prints a row per step (the worst over its four phases). Saves ``out/exec.npz`` (the
simulated motion, ``record_hz`` frames per second) for ``ctk play``, with a live plot of
the base error.
"""
import dataclasses
import sys
from pathlib import Path

import mujoco
import numpy as np

from runkit import Experiment, RunContext

REPO = Path(__file__).resolve().parents[4]


@dataclasses.dataclass
class ExecuteCfg:
    """Which plan, and how to play it."""
    walk: str = ""                  # the plan: a walk.npz (relative to the working directory)
    phase_s: float = 0.8            # seconds per phase (servo targets move linearly)
    settle_s: float = 0.5           # seconds at the start posture before the first phase
    record_hz: float = 50.0         # replay frames per second
    fall_m: float = 0.10            # base error counted as a fall (m)
    kp: float = 0.0                 # servo stiffness override (N m / rad); 0: the model's


exp = Experiment("planner_execute")


@exp.run
def run(cfg: ExecuteCfg, ctx: RunContext) -> dict:
    """Simulate the plan; print a row per step; save the replay.

    Returns:
        ``steps`` executed, ``fell`` (the first phase where the base error exceeded
        ``fall_m``, or None), ``base_err_max`` (m), ``slip_max`` (m).
    """
    if not cfg.walk:
        raise SystemExit("walk=...: the walk.npz of a planner_walk run")
    plan = np.load(cfg.walk)
    kq, kmag = plan["key_qpos"], plan["key_magnets"]
    phases = [str(x) for x in plan["key_phase"]]
    model_path = Path(str(plan["model"]))
    model_path = model_path if model_path.is_absolute() else REPO / model_path
    m = mujoco.MjModel.from_xml_path(str(model_path))
    d = mujoco.MjData(m)
    L = kmag.shape[1]
    if cfg.kp:                                                    # position servos: kp
        for a in range(3 * L):
            m.actuator_gainprm[a, 0], m.actuator_biasprm[a, 1] = cfg.kp, -cfg.kp

    # servo targets: the leg joints' qpos entries, in actuator order (leg-major)
    leg_qadr = np.array([m.joint(f"leg{i}_j{k}").qposadr[0] for i in range(L) for k in range(3)])
    servo_act = np.array([m.actuator(f"leg{i}_j{k}").id for i in range(L) for k in range(3)])
    adh = [np.array([a for a in range(m.nu) if m.actuator(a).name.startswith(f"adhere{i}_")])
           for i in range(L)]
    pads = np.array([m.body(f"pad{i}").id for i in range(L)])
    cell_leg = {m.body(b).id: i for i in range(L) for b in range(m.nbody)
                if m.body(b).name.startswith(f"pad{i}_c")}
    # planned pad positions per key posture (for the landing error)
    dk = mujoco.MjData(m)
    planned_pads = []
    for q in kq:
        dk.qpos[:] = q
        mujoco.mj_kinematics(m, dk)
        planned_pads.append(dk.xpos[pads].copy())

    def set_ctrl(q_leg, magnets):
        d.ctrl[servo_act] = q_leg
        for i in range(L):
            d.ctrl[adh[i]] = 1.0 if magnets[i] else 0.0

    def cells_touching():
        n = np.zeros(L, int)
        seen = set()
        for c in d.contact[: d.ncon]:
            for g in (c.geom1, c.geom2):
                b = m.geom_bodyid[g]
                if b in cell_leg and b not in seen:
                    seen.add(b)
                    n[cell_leg[b]] += 1
        return n

    frames, err_plot = [], []
    every = max(1, round(1.0 / (cfg.record_hz * m.opt.timestep)))
    step_count = [0]

    def sim(n_steps, q_from, q_to, magnets, q_base_plan):
        for k in range(n_steps):
            a = (k + 1) / n_steps
            set_ctrl((1 - a) * q_from + a * q_to, magnets)
            mujoco.mj_step(m, d)
            step_count[0] += 1
            if step_count[0] % every == 0:
                frames.append(d.qpos.copy())
                err_plot.append(np.linalg.norm(d.qpos[:3] - q_base_plan))

    # 1. start, settle
    d.qpos[:] = kq[0]
    mujoco.mj_forward(m, d)
    q0 = kq[0][leg_qadr]
    sim(round(cfg.settle_s / m.opt.timestep), q0, q0, kmag[0], kq[0][:3])

    # 2. phases
    n_phase = round(cfg.phase_s / m.opt.timestep)
    rows, fell = [], None
    base_err_max = slip_max = 0.0
    for a in range(len(kq) - 1):
        pad0 = d.xpos[pads].copy()
        sim(n_phase, kq[a][leg_qadr], kq[a + 1][leg_qadr], kmag[a + 1], kq[a + 1][:3])
        base_err = float(np.linalg.norm(d.qpos[:3] - kq[a + 1][:3]))
        stay = kmag[a] & kmag[a + 1]                              # feet down the whole phase
        slip = float(np.linalg.norm(d.xpos[pads] - pad0, axis=1)[stay].max()) if stay.any() else 0.0
        cells = cells_touching()
        # the swing foot's landing, at the end of the step ("planted"): it was off at B_plant
        landed = (~kmag[a - 1] if phases[a + 1] == "planted" and a >= 1
                  else np.zeros(L, bool))
        land_err = (float(np.linalg.norm(d.xpos[pads][landed] - planned_pads[a + 1][landed], axis=1).max())
                    if landed.any() else float("nan"))
        base_err_max, slip_max = max(base_err_max, base_err), max(slip_max, slip)
        rows.append(dict(phase=a + 1, name=phases[a + 1], base_err=base_err, slip=slip,
                         min_cells=int(cells[kmag[a + 1]].min()), land_err=land_err))
        if fell is None and base_err > cfg.fall_m:
            fell = f"phase {a + 1} ({phases[a + 1]})"

    # print: one row per step (its phases: from one B_lift to the next), the worst of each
    starts = [k for k, r in enumerate(rows) if r["name"] == "B_lift"] + [len(rows)]
    print(f"planner_execute: {len(kq) - 1} phases ({len(starts) - 1} steps) of {cfg.walk}; "
          f"{cfg.phase_s} s per phase; wall friction "
          f"{m.geom_friction[[g for g in range(m.ngeom) if m.geom(g).name.startswith('terrain')]][:, 0].tolist() or '-'}")
    print(f"  {'step':>4} | {'base err (mm)':>13} | {'slip (mm)':>9} | {'min cells down':>14} "
          f"| {'landing err (mm)':>16}")
    for s_, (a0, a1) in enumerate(zip(starts[:-1], starts[1:])):
        rs = rows[a0: a1]
        land = [r["land_err"] for r in rs if not np.isnan(r["land_err"])]
        print(f"  {s_:>4} | {1e3 * max(r['base_err'] for r in rs):13.1f} "
              f"| {1e3 * max(r['slip'] for r in rs):9.1f} | {min(r['min_cells'] for r in rs):14d} "
              f"| {1e3 * max(land) if land else float('nan'):16.1f}")
    print(f"  servo kp {m.actuator_gainprm[0, 0]:.0f} N m/rad; max base error "
          f"{1e3 * base_err_max:.0f} mm, max slip {1e3 * slip_max:.0f} mm"
          + (f"; FELL at {fell}" if fell else "; held"))

    np.savez(ctx.out / "exec.npz", qpos=np.array(frames), qvel=np.zeros((len(frames), m.nv)),
             timestep=1.0 / cfg.record_hz, model=str(plan["model"]),
             plot=np.array(err_plot)[:, None] * 1e3, plot_labels=np.array(["base error (mm)"]),
             plot_title="executed vs. planned")
    print(f"  saved {ctx.out}/exec.npz ({len(frames)} frames)")
    return {"steps": len(starts) - 1, "fell": fell, "base_err_max": round(base_err_max, 4),
            "slip_max": round(slip_max, 4)}


if __name__ == "__main__":
    exp.main(sys.argv[1:])
