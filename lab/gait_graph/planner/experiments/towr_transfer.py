"""One TOWR-style transfer on the floor (:mod:`..towr_transfer`): from a fixed body B, as far
in +x as possible.

``formulation``: ``angles`` (joint angles as variables, forward-kinematics equalities, exact
ankle cone and centre of mass: :func:`..towr_transfer.solve_angles`) or ``sdf`` (no joint
angles, reach by the reach grid's signed distance: :func:`..towr_transfer.solve`).

Climb robot, flat floor, no adhesion, gravity only. Start stance: each foot 0.2 m beyond its
mount; the planting leg ``i`` in the air at B. B is fixed (``b_place``; default level at
stand height over the origin). With ``formulation=angles`` T's footholds are variables too:
the start footholds only warm-start them. Transfer i -> i' (crawl 1, 0, 2, 3: the first
transfer is 1 -> 0).

Visualized in rerun (``viz``): a timeline over the four bodies (B, B_plant, B_lift, B'),
each with the posture (IK angles), the NLP's foot forces (arrows from the contact points,
1 cm per N, labelled), the footholds, the approximate and
the true centre of mass (and its floor projection), the support triangle; the body trail
static. ``viz=save``: ``out/towr_transfer.rrd`` in the run dir (``rerun <file>``);
``viz=connect``: a running viewer on port 8812; ``viz=none``.

Then verified with joint angles: every foot each body must reach, by exact IK (joint limits,
body box); the real centre of mass (:func:`..statics.kinematic_com` from the IK angles) and
an exact equilibrium LP with it, per body.

    uv run --extra mjx runkit run lab.gait_graph.planner.experiments.towr_transfer
"""
import dataclasses
import sys

import jax
import jax.numpy as jnp
import numpy as np
import rerun as rr

from runkit import Experiment, RunContext

from controlkit.kinematics.types import Posture
from controlkit.se3 import SE3
import controlkit.rerun_viz as rrv

from ....kinematics.reach_sdf import Keepout, branches_fn, make_reach_sdf
from ..climb_model.config import MjModelCfg
from ..climb_model.mjmodel import make_robot
from ..climb_statics import ClimbModel
from ..scoring import G
from ..statics import kinematic_com, push_score, push_score_min_norm
from ..towr_transfer import Transfer, TowrCfg, com_approx, hold_exact, solve, solve_angles


@dataclasses.dataclass
class TowrTransferCfg:
    """The robot, the start, the NLP."""
    mjmodel: MjModelCfg = dataclasses.field(default_factory=MjModelCfg)
    nlp: TowrCfg = dataclasses.field(default_factory=TowrCfg)
    h: float = 0.005                # reach grid voxel size (m); nlp.margin should be >= h
    foot_out: float = 0.2           # start footholds: this far beyond each mount (m)
    b_place: str = "origin"         # B (fixed): "origin": level at stand height over the
                                    # origin | "incenter": centre of mass over T's start
                                    # incenter | "shift": above b_shift x T's start centroid
    b_shift: float = 0.5            # for b_place="shift"
    i: int = 1                      # planting leg
    i_prime: int = 0                # lifting leg
    viz: str = "save"               # save (out/towr_transfer.rrd) | connect (port 8812) | none
    formulation: str = "angles"     # angles | sdf


def _ik_thetas(robot, branches, body, targets, normals=None):
    """Per leg: a valid IK branch reaching its target (world), or None; with ``normals``,
    the valid branch whose foot-to-knee direction is closest to the normal (the ankle)."""
    sh = robot.shoulders(body)
    out = []
    for j in range(robot.num_legs):
        ok, th = branches(sh[j].inverse().apply(jnp.asarray(targets[j], jnp.float32)))
        ok, th = np.asarray(ok), np.asarray(th)
        if not ok.any():
            out.append(None)
            continue
        score = np.zeros(len(ok))
        if normals is not None:
            v = np.asarray(jax.vmap(lambda t: sh[j].rotation().apply(robot.leg.contact_vector(t)))(
                jnp.asarray(th)))
            score = v @ normals[j]
        out.append(th[int(np.argmax(np.where(ok, score, -np.inf)))])
    return out


FORCE_SCALE = 0.01   # force arrows: m per N (the whole weight, ~27 N, is 27 cm)


def _visualize(robot, drawn, footholds, normals, cfg, ctx):
    """Log the four bodies on a rerun timeline (see the module docstring)."""
    if cfg.viz == "save":
        ctx.out.mkdir(parents=True, exist_ok=True)
        path = ctx.out / "towr_transfer.rrd"
        rrv.init_viewer("towr_transfer", save=path)
        print(f"  rerun: {path}")
    else:
        rrv.init_viewer("towr_transfer", connect=8812)
    rr.log("floor", rr.Boxes3D(centers=[[0.0, 0.0, -0.003]], half_sizes=[[0.5, 0.5, 0.003]],
                               colors=[(70, 70, 70)], fill_mode="solid"), static=True)
    trail = np.stack([np.asarray(d["body"].translation()) for d in drawn.values()])
    rr.log("trail", rr.LineStrips3D([trail], colors=[(250, 200, 40)], radii=0.002), static=True)
    rr.log("trail/points", rr.Points3D(trail, radii=0.006, colors=[(250, 200, 40)],
                                       labels=list(drawn)), static=True)
    rrv.log_contacts("footholds/start", footholds, normals, color=(120, 120, 120))
    L = footholds.shape[0]
    for k, (name, d) in enumerate(drawn.items()):
        rrv.set_time(k, timeline="body")
        rrv.log_posture("robot", robot, Posture(d["body"], jnp.asarray(np.stack(d["thetas"]), jnp.float32)))
        sup = d["tgt"][d["support"]] - d["foot_r"] * normals[d["support"]]   # pivots -> footholds
        F = np.asarray(d["F"])                                               # surface on foot (N)
        rr.log("forces", rr.Arrows3D(origins=sup, vectors=F * FORCE_SCALE, radii=0.004,
                                     colors=[(40, 160, 255)],
                                     labels=[f"{np.linalg.norm(f):.1f} N" for f in F]))
        rrv.log_contacts("footholds/support", sup, normals[d["support"]], color=(60, 200, 90))
        tri = np.concatenate([sup, sup[:1]])
        rr.log("support", rr.LineStrips3D([tri * [1, 1, 0] + [0, 0, 0.001]], colors=[(60, 200, 90)],
                                          radii=0.002))
        coms = [d["com_apx"]] + ([d["com_true"]] if d["com_true"] is not None else [])
        rr.log("com", rr.Points3D(coms, radii=0.01, colors=[(255, 140, 0), (230, 40, 40)][:len(coms)],
                                  labels=["CoM approx", "CoM true"][:len(coms)]))
        c = coms[-1]
        rr.log("com/floor", rr.LineStrips3D([[c, [c[0], c[1], 0.001]]], colors=[(230, 40, 40)],
                                            radii=0.0015))
        rr.log("label", rr.TextDocument(f"{name}: x = {float(d['body'].translation()[0]):.3f} m"))


exp = Experiment("towr_transfer")


@exp.run
def run(cfg: TowrTransferCfg, ctx: RunContext) -> dict:
    """Solve and verify one transfer; see the module docstring.

    Returns:
        The solution summary and the verification per body.
    """
    mj, nlp = cfg.mjmodel, cfg.nlp
    robot = make_robot(mj)
    cm = ClimbModel(mj)
    mm = cm.mass_model()
    L = mj.num_legs
    m_leg = float(jnp.sum(mm.links) + mm.foot)
    nlp = dataclasses.replace(nlp, foot_r=mj.pivot_height + mj.pad_thickness,
                              tau_max=float(mj.forcerange[1]))

    body_center, body_half = robot.mount_box(half_height=mj.body_half_height)
    keepout = Keepout(robot.mounts[0].inverse(), body_center, body_half)
    sdf = make_reach_sdf(robot.leg, cfg.h, keepout=keepout)
    branches = branches_fn(robot.leg, keepout=keepout)

    mount_xy = np.asarray(robot.mounts.translation())[:, :2]
    feet_xy = mount_xy * (1 + cfg.foot_out / np.linalg.norm(mount_xy, axis=1, keepdims=True))
    footholds = np.concatenate([feet_xy, np.zeros((L, 1))], 1)
    normals = np.tile([0.0, 0.0, 1.0], (L, 1))
    T = [j for j in range(L) if j != cfg.i]
    cxy = np.zeros(2) if cfg.b_place == "origin" else cfg.b_shift * footholds[T, :2].mean(0)
    B = SE3(jnp.asarray([1.0, 0, 0, 0, cxy[0], cxy[1], mj.stand_height]))
    if cfg.b_place == "incenter":
        P = footholds[T, :2]
        a = np.array([np.linalg.norm(P[(k + 1) % 3] - P[(k + 2) % 3]) for k in range(3)])
        inc = (a[:, None] * P).sum(0) / a.sum()                  # incenter: side-length weights
        tg0 = footholds + nlp.foot_r * normals
        for _ in range(5):                                       # body shift until CoM over it
            th = np.stack(_ik_thetas(robot, branches, B, tg0, normals))
            com = np.asarray(kinematic_com(robot, Posture(B, jnp.asarray(th, jnp.float32)), mm))
            cxy = cxy + (inc - com[:2])
            B = SE3(jnp.asarray([1.0, 0, 0, 0, cxy[0], cxy[1], mj.stand_height]))
        print(f"  B: centre of mass {np.round(com[:2], 4)} over T's incenter {np.round(inc, 4)}")
    tr = Transfer(B, jnp.asarray(footholds, jnp.float32), jnp.asarray(normals, jnp.float32),
                  cfg.i, cfg.i_prime)
    print(f"towr_transfer ({cfg.formulation}): legs {cfg.i} -> {cfg.i_prime}, "
          f"B at {np.round(np.asarray(B.translation()), 3)}, push >= {nlp.push_frac:g} x weight "
          f"= {nlp.push_frac * cm.mass * 9.81:.1f} N")

    if cfg.formulation == "angles":
        theta_B = _ik_thetas(robot, branches, B, footholds + nlp.foot_r * normals, normals)
        assert all(t is not None for t in theta_B), "B does not reach its footholds"
        theta_B = np.stack(theta_B)
        r = solve_angles(robot, mm, G, tr, nlp, theta_B, mj.ankle_range_deg)
    else:
        r = solve(robot, sdf, mm, G, tr, nlp)
    print(f"  {r['message']} | success {r['success']} | {r['nit']} iterations, {r['t']:.1f} s "
          f"(+ {r['t_compile']:.1f} s compile) | eq viol {r['eq_viol']:.1e}, ineq viol {r['ineq_viol']:.1e}")
    print(f"  progress x(B') - x(B) = {r['progress'] * 100:.1f} cm | f = {np.round(r['f'], 3)} "
          f"(old {np.round(footholds[cfg.i], 3)})")

    # --- verification with joint angles
    foot_r = nlp.foot_r
    S = footholds.copy(); S[cfg.i] = r["f"]; S[cfg.i, 2] = 0.0
    if cfg.formulation == "angles":                        # T's footholds were variables too
        S = np.asarray(r["P"]).copy(); S[:, 2] = 0.0
        print(f"  footholds {np.round(S, 3).tolist()}\n  (start    {np.round(footholds, 3).tolist()})")
    tgt_S = S + foot_r * normals
    tgt_0 = (S if cfg.formulation == "angles" else footholds) + foot_r * normals
    bodies = {"B": (B, tgt_0, T, T), "B_plant": (r["B_plant"], tgt_S, list(range(L)), r["T"]),
              "B_lift": (r["B_lift"], tgt_S, list(range(L)), r["Tp"]),
              "B'": (r["B_end"], tgt_S, r["Tp"], r["Tp"])}
    report, drawn = {}, {}
    for k, (name, (body, tgt, reach_legs, support)) in enumerate(bodies.items()):
        if name == "B'":                                   # leg i' in the air, tucked
            tgt = tgt.copy()
            tgt[cfg.i_prime] = np.asarray(robot.shoulders(body)[cfg.i_prime].apply(
                jnp.asarray(nlp.air_foot, jnp.float32)))
        if cfg.formulation == "angles":                    # the NLP's own angles
            th = list(r["thetas"][k])
        else:
            th = _ik_thetas(robot, branches, body, tgt, normals)
        reach_ok = all(th[j] is not None for j in reach_legs)
        sh = robot.shoulders(body)
        n_w = normals.copy(); n_w[cfg.i] = [0.0, 0.0, 1.0]
        if reach_ok and reach_legs:
            fk = np.stack([np.asarray(sh[j].apply(robot.leg.forward(jnp.asarray(th[j])).translation()[-1]))
                           for j in reach_legs])
            fk_err = float(np.abs(fk - tgt[reach_legs]).max()) * 1e3
            v = np.stack([np.asarray(sh[j].rotation().apply(robot.leg.contact_vector(jnp.asarray(th[j]))))
                          for j in reach_legs])
            ankle = float(np.degrees(np.arccos(np.clip(np.sum(v * n_w[reach_legs], -1), -1, 1))).max())
            lim = np.asarray(robot.leg.limits)
            in_lim = all(np.all((th[j] >= lim[:, 0] - 1e-4) & (th[j] <= lim[:, 1] + 1e-4)) for j in range(L))
        else:
            fk_err, ankle, in_lim = np.nan, np.nan, False
        if all(t is not None for t in th):
            thetas = jnp.asarray(np.stack(th), jnp.float32)
            com_true = np.asarray(kinematic_com(robot, Posture(body, thetas), mm))
        else:
            com_true = None
        sh = np.asarray(robot.shoulders(body).translation())
        com_apx = np.asarray(com_approx(body, jnp.asarray(sh), jnp.asarray(tgt, jnp.float32),
                                        mm.body, m_leg, nlp.alpha))
        hold, fn = (hold_exact(com_true, tgt[support], normals[support], cm.mass, G, nlp.adhesion,
                               nlp.mu) if com_true is not None else (False, None))
        r_mn = r_full = np.nan
        if com_true is not None:                                 # push scores, both kinds
            thetas_j = jnp.asarray(np.stack(th), jnp.float32)
            sh_b = robot.shoulders(body)
            fk_sup = np.stack([np.asarray(sh_b[j].apply(robot.leg.forward(thetas_j[j]).translation()[-1]))
                               for j in support])
            n_sup = n_w[support]
            r_mn = float(push_score_min_norm(jnp.asarray(com_true), jnp.asarray(fk_sup, jnp.float32),
                                             jnp.asarray(n_sup, jnp.float32), cm.mass, G,
                                             nlp.adhesion, nlp.mu)[0])
            n_all = jnp.asarray(n_w, jnp.float32)
            st = cm.statics(cm.qpos(Posture(body, thetas_j), n_all), G)
            r_full = float(push_score(st, tuple(support), n_all[jnp.array(support)], nlp.adhesion,
                                      nlp.mu, float(mj.forcerange[1]))[0])
        v = np.asarray(body.translation())
        rpy = np.rad2deg(np.asarray(body.rotation().as_rpy_radians()))
        err = np.linalg.norm(com_true - com_apx) * 1e3 if com_true is not None else np.nan
        drawn[name] = dict(body=body, thetas=[t if t is not None else np.zeros(3) for t in th],
                           support=support, tgt=tgt, F=r["F"][k], com_true=com_true, com_apx=com_apx,
                           foot_r=foot_r)
        report[name] = dict(xyz=v.tolist(), rpy_deg=rpy.tolist(), reach_ik=reach_ok,
                            fk_err_mm=fk_err, ankle_max_deg=ankle, in_limits=bool(in_lim),
                            push_min_norm=r_mn, push_full=r_full,
                            tau_max_abs=(float(np.abs(r["tau"][k]).max()) if "tau" in r else np.nan),
                            tau_sq=(float(np.sum(r["tau"][k] ** 2)) if "tau" in r else np.nan),
                            com_err_mm=float(err), hold_true_com=bool(hold),
                            normal_forces=None if fn is None else np.round(fn, 2).tolist(),
                            nlp_normal_forces=np.round(r["F"][k][:, 2], 2).tolist())
        print(f"  {name:8s} xyz {np.round(v, 3)} rpy {np.round(rpy, 1)} | reaches "
              f"{'all' if reach_ok else 'NOT all'} (FK err {fk_err:.2f} mm) | ankle max {ankle:.1f} deg | "
              f"limits {in_lim} | holds: {hold} | push min-norm {r_mn:5.1f} N, full {r_full:5.1f} N | "
              f"|tau| max {report[name]['tau_max_abs']:.2f} N m, sum tau^2 {report[name]['tau_sq']:.1f} | "
              f"NLP normal forces {report[name]['nlp_normal_forces']}")
    if cfg.viz != "none":
        _visualize(robot, drawn, footholds, normals, cfg, ctx)
    return dict(progress=r["progress"], f=r["f"].tolist(), success=r["success"], nit=r["nit"],
                t=r["t"], eq_viol=r["eq_viol"], ineq_viol=r["ineq_viol"], bodies=report)


if __name__ == "__main__":
    exp.main(sys.argv[1:])
