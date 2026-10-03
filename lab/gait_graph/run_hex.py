"""Gait-graph playground with the hexapod (6x3-DOF radial robot).

The playground of :mod:`.run` with the robot, config and stance moves swapped
(:mod:`.config_hex`, :mod:`.robot_hex`, :mod:`.stance_3dof`, :mod:`.samplers_hex`),
a per-leg **place** mode, and its own button layout (:class:`HexPlayground`).

    uv run --extra mjx mjpython -m lab.gait_graph.run_hex

Controls:
    Triangle          switch mode: explore <-> step
    Options           reset (clears posture and path; the saved list is kept)
    D-pad left/right  select the leg: explore full -> 0 -> ... -> 5 -> full,
                      step none -> 0 -> ... -> 5 -> none (the selected leg is lifted;
                      leaving it puts it back down unless it was placed)
    Square            cycle the leg mode: random -> aim -> place -> random
                      (kept when switching legs)
    Cross             act on the selected leg in its mode (explore with no leg
                      selected: sample a full stance at the ghost)
                        random  re-plant it on a foothold drawn by the sampler
                        aim     re-plant it, drawn with weight exp(-d^2 / sigma^2) to
                                the orange sphere; L1 / R1 shrink / grow sigma
                        place   put the foot at the blue sphere, *not* planted; it
                                stays in the air until the leg is planted again
                                (going into step puts it back on its foothold)
    D-pad up/down     cycle the sampler: prior / best

  explore: sticks / R2 / L2 move the ghost; Circle saves the posture (placed legs
  in the air).
  step: sticks / R2 / L2 move the body with its feet held; Circle commits the
  stance as the next node; Create jumps back to the last committed node.
  aim / place: the sticks move the sphere instead (camera frame, see :mod:`.run`).

On exit, the saved postures and the committed path are written to
``runs/hex_<timestamp>.jsonl`` next to this file (see :func:`write_jsonl`).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from controlkit.mujoco_viz import to_qpos

from . import run, terrain
from .config_hex import HexCfg
from .robot import RGBA_BAD, RGBA_MATERIAL, RGBA_SELECTED, leg_geoms
from .robot_hex import make_robot, scene_xml
from .run import LIFT_HEIGHT, Playground, _draw, _fails, _marker_sig
from .samplers_hex import Samplers
from .scoring import Scorer
from .stance_3dof import Kit
from .target import Point, Target

# Path-marker colour per leg. Playground.markers reads run.LEG_RGBA (4 colours),
# so it is replaced with 6 here.
LEG_RGBA = np.array([[0.95, 0.35, 0.2, 1], [0.95, 0.8, 0.2, 1], [0.3, 0.8, 0.35, 1],
                     [0.2, 0.8, 0.85, 1], [0.3, 0.55, 0.95, 1], [0.75, 0.4, 0.95, 1]])
run.LEG_RGBA = LEG_RGBA
RGBA_PLACE = (0.2, 0.6, 1.0, 1.0)


class HexPlayground(Playground):
    """:class:`.run.Playground` with the hexapod controls (see the module docstring).

    The selected leg is the base class's ``select`` (explore) / ``swing`` (step);
    ``leg_mode`` is random / aim / place. Aim and place both ride on the base
    class's aim mode (``aim`` True, so the sticks move the sphere); place also sets
    ``place``. ``planted`` marks which feet are on their footholds, in both modes
    (the base class only uses it in step); ``placed`` the legs a Cross in place
    mode left in the air. A placed foot stays there until its leg is planted
    again (sampling it, a full sample, Create in step), and is put back on its
    foothold when switching into step.
    """

    MODES = ("random", "aim", "place")

    def reset(self):
        self.leg_mode = getattr(self, "leg_mode", "random")
        self.place = False
        self.placed = set()     # legs left in the air by place (until re-planted)
        self.cursor = None      # step: the last selected leg (swing is cleared on commit)
        super().reset()

    # -- per tick ------------------------------------------------------------- #
    def _tick_explore(self, s, dt):
        p = s.pressed
        if not self.aim and self.target.update(s, dt):
            self.cand = self.kit.candidates(self.target.body())
            self.valid = None   # indexed by cand -- stale once cand changes
        self._tick_common(p)
        if "a" in p:
            self._sample_full() if self.select is None else self._sample_leg(self.select)
        if "b" in p:
            self._save()

    def _tick_step(self, s, dt):
        p = s.pressed
        prev = self.target.state()
        if not self.aim and self.target.update(s, dt):
            ok, post, c = self.kit.move_body(self.posture, self.stance,
                                             jnp.asarray(self.planted), self.target.body())
            self.bad = not bool(ok)
            if self.bad:
                self.target.set_state(prev)
                self.status = f"move rejected: {_fails(c)}"
            else:
                self.posture = post
                self.valid = None
                self.cand = self.kit.candidates(post.body)
        self._tick_common(p)
        if "a" in p:
            self._resample()
        if "b" in p:
            self._commit()
        if "back" in p:
            self._back()

    def _tick_common(self, p):
        if "left" in p or "right" in p:
            self._cycle_leg(1 if "right" in p else -1)
        if "x" in p:
            self.leg_mode = self.MODES[(self.MODES.index(self.leg_mode) + 1) % len(self.MODES)]
            self._apply_mode()
        if "up" in p or "down" in p:
            n = len(self.samplers.names)
            self.sampler = (self.sampler + (1 if "up" in p else -1)) % n

    # -- leg selection + mode --------------------------------------------------- #
    def _leg(self):
        return self.select if self.mode == "explore" else self.swing

    def _cycle_leg(self, d):
        """Select the next (``d = 1``) / previous leg; None (full / no leg) in between."""
        if self.posture is None:
            self.status = "sample a full stance first (Cross)"
            return
        order = [None] + list(range(self.L))
        cur = self.select if self.mode == "explore" else self.cursor
        new = order[(order.index(cur) + d) % len(order)]
        if self.mode == "explore":
            self.select = new
            self.status = "full stance" if new is None else f"leg {new}"
        elif not self._set_swing(new):
            return
        self._apply_mode()

    def _apply_mode(self):
        """Put the selected leg in ``leg_mode`` (aim / place show their sphere)."""
        self.aim = self.place = False
        i = self._leg()
        if i is None or self.posture is None:
            return
        if self.leg_mode != "random":
            self._start_aim(i)
        if self.leg_mode == "place":
            self._start_place(i)

    def _set_swing(self, i):
        """Step: put the swing leg down (unless placed), then lift leg ``i`` (or none)."""
        self.aim = self.place = False
        if not self._put_down():
            return False
        self.cursor = i
        if i is None:
            self.swing, self.valid = None, None
            self.status = "no leg lifted"
            return True
        if i in self.placed:            # already in the air
            self.swing, self.valid = i, None
            self.status = f"leg {i} (placed)"
            return True
        ok, post = self.kit.lift_leg(self.posture, self.stance, i, LIFT_HEIGHT)
        if not bool(ok):
            self.status = f"leg {i} can't lift here"
            self.swing = i
            return True
        planted = self.planted.copy()
        planted[i] = False
        c = self.kit.checks(post, self.stance, jnp.asarray(planted))
        self.posture, self.swing, self.valid, self.planted = post, i, None, planted
        fails = _fails(c)
        self.status = f"leg {i} lifted: move the body, Cross to re-plant" + (
            f"  (lifted pose invalid: {fails})" if fails else "")
        return True

    # -- the place step -------------------------------------------------------- #
    def _start_place(self, i):
        self.place = True
        self.point = Point(self.cfg, np.asarray(self.kit.robot.feet(self.posture)[i]))
        self.status = f"place leg {i}: move the sphere, Cross to put the foot there"

    def _place(self, i):
        """Put leg ``i``'s foot at the sphere, not planted."""
        ok, post = self.kit.place_leg(self.posture, i, jnp.asarray(self.point.pos))
        if not bool(ok):
            self.bad = True
            self.status = f"leg {i} can't reach the sphere"
            return
        planted = self.planted.copy()
        planted[i] = False
        fails = _fails(self.kit.checks(post, self.stance, jnp.asarray(planted)))
        self.posture, self.planted, self.bad = post, planted, bool(fails)
        self.placed.add(i)
        self.status = f"leg {i} placed (free)" + (f"  (invalid: {fails})" if fails else "")

    def _return_foot(self, i):
        """Put a placed leg ``i`` back on its foothold."""
        ok, post = self.kit.lift_leg(self.posture, self.stance, i, 0.0)
        if not bool(ok):
            self.bad = True
            self.status = f"leg {i} can't reach its foothold from here"
            return False
        self.posture, self.planted[i] = post, True
        self.placed.discard(i)
        return True

    def _planted_again(self, i):
        """Leg ``i`` stands on a foothold again; the other placed legs stay free."""
        self.placed.discard(i)
        self.planted[:] = True
        self.planted[list(self.placed)] = False

    # -- hooks into the base class --------------------------------------------- #
    def _put_down(self):
        if self.swing in self.placed:
            return True                 # placed: leave it in the air
        return super()._put_down()

    def _switch_mode(self):
        """Into step, placed feet go back to their footholds (path nodes are fully planted)."""
        self.place = False
        if self.mode == "explore":
            if self.posture is not None:
                for i in sorted(self.placed):
                    if not self._return_foot(i):
                        return
            self.cursor = None
        super()._switch_mode()
        if self.mode == "step":
            if self.posture is not None:
                self.status = "step: D-pad left / right to lift a leg"
        else:
            self._apply_mode()

    def _leg_base(self, i):
        """As the base class, but placed legs stay free (not tracked to their footholds)."""
        if self.mode == "step":
            return self.posture
        planted = self.planted.copy()
        planted[i] = False
        _, post, c = self.kit.move_body(self.posture, self.stance, jnp.asarray(planted),
                                        self.target.body())
        fails = _fails(c, [j for j in range(self.L) if j != i])
        if fails:
            self.bad = True
            self.status = f"other legs can't hold the ghost pose: {fails}"
            return None
        return post

    def _sample_full(self):
        super()._sample_full()
        if not self.bad:
            self.placed.clear()
            self.planted[:] = True

    def _sample_leg(self, i):
        if self.place:
            self._place(i)
            return
        super()._sample_leg(i)          # moves the body with the other feet held
        if not self.bad:
            self._planted_again(i)

    def _resample(self):
        if self.swing is None:
            self.status = "select a leg first (D-pad left / right)"
            return
        if self.place:
            self._place(self.swing)
            return
        super()._resample()             # on success sets every leg planted
        if self.planted.all():
            self._planted_again(self.swing)

    def _commit(self):
        super()._commit()               # on success clears swing and aim
        if self.swing is None:
            self.place = False

    def _back(self):
        self.place = False
        self.placed.clear()
        super()._back()
        self.cursor = None

    def _save(self):
        super()._save()
        if self.posture is not None:
            self.saved[-1]["planted"] = self.planted.copy()

    # -- display ---------------------------------------------------------------- #
    def highlighted(self):
        return self._leg()

    def markers(self):
        """In place mode: no sampling-weight sizing, the sphere in blue."""
        if not self.place:
            return super().markers()
        self.aim = False
        try:
            out = super().markers()
        finally:
            self.aim = True
        return out + [(self.point.pos, 0.02, RGBA_PLACE)]

    def text(self):
        i = self._leg()
        if self.mode == "explore":
            leg = "full" if i is None else str(i)
            head = (f"explore   leg: {leg}   mode: {self.leg_mode}   "
                    f"sampler: {self._sampler_name()}   saved: {len(self.saved)}")
        else:
            leg = "-" if i is None else str(i)
            head = f"step   leg: {leg}   mode: {self.leg_mode}   nodes: {len(self.path)}"
        if self.leg_mode == "aim":
            head += f"   sigma {self.sigma * 100:.1f} cm"
        if self.placed:
            head += f"   in air: {','.join(map(str, sorted(self.placed)))}"
        return head, self.status


def _relaunch_under_mjpython():
    """On macOS, re-exec as a module under mjpython (the passive viewer needs it)."""
    from mujoco import viewer as mj_viewer
    if sys.platform != "darwin" or getattr(mj_viewer, "_MJPYTHON", None) is not None:
        return
    mjpython = Path(sys.executable).with_name("mjpython")
    if not mjpython.exists():
        raise RuntimeError(f"`mjpython` not found next to {sys.executable}. "
                           "Run: uv run --extra mjx mjpython -m lab.gait_graph.run_hex")
    os.execv(str(mjpython), [str(mjpython), "-m", __spec__.name])


def write_jsonl(pg: Playground, kit: Kit, path: Path) -> int:
    """Write the saved postures and the committed path, one JSON object per line.

    Each line: ``kind`` ("saved" or "path"), ``index`` (within its kind),
    ``stance`` (foothold indices into this run's pool), ``footholds`` /
    ``normals`` (num_legs, 3) world frame, ``body`` (7,) as ``[qw, qx, qy, qz, x,
    y, z]`` (``SE3.wxyz_xyz``), ``thetas`` (num_legs, 3) joint angles in rad, and
    ``planted`` (num_legs,) -- False for a placed (free) leg, whose ``stance`` /
    ``footholds`` entry is stale.

    Returns:
        Number of lines written (nothing is written if 0).
    """
    fh = kit.footholds
    rows = [("saved", i, e) for i, e in enumerate(pg.saved)]
    rows += [("path", i, e) for i, e in enumerate(pg.path)]
    if not rows:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for kind, i, e in rows:
            st = np.asarray(e["stance"])
            f.write(json.dumps(dict(
                kind=kind, index=i, stance=st.tolist(),
                footholds=np.asarray(fh.position[st]).tolist(),
                normals=np.asarray(fh.normal[st]).tolist(),
                body=np.asarray(e["posture"].body.wxyz_xyz).tolist(),
                thetas=np.asarray(e["posture"].thetas).tolist(),
                planted=np.asarray(e.get("planted", np.ones(len(st), bool))).tolist()))
                    + "\n")
    return len(rows)


def build(cfg: HexCfg, seed: int = 0):
    """Robot, scene, foothold pool, the jitted :class:`Kit` and :class:`Samplers` (warmed up)."""
    key = jax.random.PRNGKey(seed)
    robot = make_robot(cfg)
    scene = terrain.make_scene(cfg.boxes)
    k_fh, key = jax.random.split(key)
    footholds = terrain.sample_footholds(k_fh, scene, cfg.num_footholds,
                                         edge_margin=cfg.edge_margin)
    kit = Kit(robot, scene, cfg, footholds)
    samplers = Samplers(kit, Scorer(robot, cfg))

    print("compiling ...", flush=True)
    t = time.time()
    body = Target(cfg).body()
    cand = kit.candidates(body)
    ok, post, st = kit.sample_stance(key, cand, body)
    planted = jnp.ones(cfg.num_legs, bool)
    kit.lift_leg(post, st, 0, LIFT_HEIGHT)
    kit.place_leg(post, 0, jnp.zeros(3))
    kit.move_body(post, st, planted, body)
    kit.resample_leg(key, cand, post, st, 0)
    kit.checks(post, st, planted)
    for name in samplers.names:
        samplers.full(name, key, cand, body)
        samplers.leg(name, key, cand, post, st, 0)
    samplers.leg_toward(key, cand, post, st, 0, np.zeros(3), cfg.aim_sigma)
    kit.leg_valid(key, cand, post, st, 0)
    print(f"compiled in {time.time() - t:.1f}s ({footholds.shape[0]} footholds)", flush=True)
    return robot, kit, samplers, key


def main():
    _relaunch_under_mjpython()
    from mujoco import viewer as mj_viewer
    from .input import DualSense

    cfg = HexCfg()
    robot, kit, samplers, key = build(cfg)
    xml = scene_xml(robot, cfg)
    Path(__file__).with_name("scene_hex.xml").write_text(xml)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "base")
    legs = leg_geoms(model, cfg.num_legs)
    mocap = model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target")]

    pad = DualSense()
    pg = HexPlayground(kit, samplers, cfg, key)
    print(__doc__.split("Controls:")[1].split("On exit")[0], flush=True)
    try:
        _loop(model, data, pg, pad, cfg, base, legs, mocap, mj_viewer)
    finally:
        pad.close()
        out = Path(__file__).with_name("runs") / time.strftime("hex_%Y%m%d_%H%M%S.jsonl")
        n = write_jsonl(pg, kit, out)
        print(f"wrote {n} postures to {out}" if n else "nothing saved", flush=True)


def _loop(model, data, pg, pad, cfg, base, legs, mocap, mj_viewer):
    """The viewer loop (as :func:`.run.main`)."""
    dt = 1.0 / cfg.control_hz
    drawn, shown = None, None
    rel_az, last_az = None, None    # camera azimuth relative to the body yaw (deg)
    with mj_viewer.launch_passive(model, data, show_left_ui=False) as viewer:
        while viewer.is_running():
            t0 = time.time()
            pg.cam = (np.deg2rad(viewer.cam.azimuth), np.deg2rad(viewer.cam.elevation))
            pg.tick(pad.poll(), dt)

            # Compute everything outside the viewer lock; inside, only assign.
            pos, quat = pg.target.mocap()
            ghost = pos if pg.mode == "explore" else np.array([0.0, 0.0, -5.0])  # hide in step
            qpos = to_qpos(pg.posture)[0] if pg.posture is not None else None
            sig = _marker_sig(pg) + (pg.place,)
            markers = pg.markers() if sig != drawn else None
            text = pg.text()
            hl = pg.highlighted()

            with viewer.lock():
                data.mocap_pos[mocap], data.mocap_quat[mocap] = ghost, quat
                if qpos is not None:
                    data.qpos[:] = qpos
                else:
                    data.qpos[:3] = [0, 0, -5]            # hide until sampled
                model.geom_rgba[base] = RGBA_BAD if pg.bad else RGBA_MATERIAL
                for i, ids in enumerate(legs):
                    model.geom_rgba[ids] = RGBA_SELECTED if i == hl else RGBA_MATERIAL
                mujoco.mj_forward(model, data)
                if markers is not None:
                    _draw(viewer.user_scn, markers)
                    drawn = sig
                # Camera rides with the body (see run.main).
                yaw = np.rad2deg(pg.target.yaw)
                if rel_az is None:
                    rel_az = viewer.cam.azimuth - yaw
                else:
                    rel_az += viewer.cam.azimuth - last_az
                viewer.cam.azimuth = last_az = rel_az + yaw
                viewer.cam.lookat[:] = pos
            if text != shown:
                viewer.set_texts([(None, None, *text)])
                shown = text
            viewer.sync()
            # always yield a little, so the viewer's render thread gets the lock
            time.sleep(max(0.002, dt - (time.time() - t0)))


if __name__ == "__main__":
    main()
