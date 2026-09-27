"""Gait-graph playground: sample stances and build a path of them with the DualSense.

The sim is visualization only (``qpos`` + ``mj_forward``, never stepped); all
validity is decided in code by :mod:`.stance`. Planted feet are treated as welded.

Run (macOS needs the MuJoCo viewer under mjpython):

    uv run --extra mjx mjpython -m lab.gait_graph.run

Controls:
    Triangle          switch mode: explore <-> step
    Options           reset (clears posture and path; the saved list is kept)

  explore (nothing committed; the ghost is the sampling pose):
    sticks / R2 / L2  move the ghost
    Square            cycle what to sample: full -> leg 0 -> aim 0 -> leg 1 -> aim 1
                      -> ... -> full (legs only once there is a posture)
    Cross             sample it at the ghost. A leg sample moves the body to the
                      ghost, keeps the other feet on their footholds, re-plants the leg
    Circle            add the current posture to the saved list
    D-pad up / down   cycle the sampler: prior (uniform among valid) / best (top score)

  step (build a path; no ghost, the sticks move the robot with its feet held):
    sticks / R2 / L2  move the body (rejected moves turn the body red)
    Square            cycle: none -> lift 0 -> aim 0 -> lift 1 -> aim 1 -> ... -> none
                      (the previous leg goes back down)
    Cross             sample a new foothold for the lifted leg (again to re-sample)
    Circle            commit the stance as the next node of the path
    D-pad left        jump back to the last committed node

  aim (either mode): an orange sphere appears at the leg's foothold and the sticks
  move it (instead of the body/ghost), in the camera frame: left stick = screen
  right/up, right stick up/down = depth, right stick left/right = screen right;
  R2 / L2 = world up/down. Cross then draws the leg's foothold
  with weight exp(-d^2 / sigma^2), d = distance to the sphere; L1 / R1 shrink /
  grow sigma (the translucent ball shows it).

Camera: follows the body (ghost in explore) and turns with its yaw; the mouse
orbits / zooms relative to it.

Foothold markers (for the selected / swing leg only): grey = its candidates (the
footholds in its shoulder's reach ball), green = where it could have gone, coloured = footholds of the committed path (colour per leg).
While aiming, the footholds the leg can reach are scaled by their sampling
probability (the most likely at a fixed max size, clipped below at a min size).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from controlkit.mujoco_viz import to_qpos

from . import terrain
from .config import Cfg
from .robot import (RGBA_BAD, RGBA_MATERIAL, RGBA_SELECTED, leg_geoms, make_robot,
                    scene_xml)
from .samplers import Samplers
from .scoring import Scorer
from .stance import Kit
from .target import Point, Target

LEG_RGBA = np.array([[0.95, 0.35, 0.2, 1], [0.95, 0.8, 0.2, 1],
                     [0.3, 0.8, 0.35, 1], [0.3, 0.55, 0.95, 1]])
LIFT_HEIGHT = 0.06


def _relaunch_under_mjpython():
    """On macOS, re-exec as a module under mjpython (the passive viewer needs it)."""
    from mujoco import viewer as mj_viewer
    if sys.platform != "darwin" or getattr(mj_viewer, "_MJPYTHON", None) is not None:
        return
    mjpython = Path(sys.executable).with_name("mjpython")
    if not mjpython.exists():
        raise RuntimeError(f"`mjpython` not found next to {sys.executable}. "
                           "Run: uv run --extra mjx mjpython -m lab.gait_graph.run")
    os.execv(str(mjpython), [str(mjpython), "-m", __spec__.name])


def _fails(c: dict, legs=None) -> str:
    """Failed checks as text, e.g. ``ankle[0,2], spacing``; per-leg ones only for ``legs``."""
    out = []
    for k, v in c.items():
        v = np.asarray(v)
        if v.ndim == 0:
            if not v:
                out.append(k)
            continue
        bad = [i for i in np.flatnonzero(~v) if legs is None or i in legs]
        if bad:
            out.append(f"{k}[{','.join(map(str, bad))}]")
    return ", ".join(out)


class Playground:
    """The interaction state machine (no viewer code)."""

    def __init__(self, kit: Kit, samplers: Samplers, cfg: Cfg, key):
        self.kit, self.samplers, self.cfg, self.key = kit, samplers, cfg, key
        self.sampler = 0        # index into samplers.names
        self.L = cfg.num_legs
        self.target = Target(cfg)
        self.saved = []         # [dict(stance, posture)], for saving later
        self.sigma = cfg.aim_sigma
        self.reset()

    def _key(self):
        self.key, k = jax.random.split(self.key)
        return k

    def reset(self):
        self.mode = "explore"
        self.select = None      # explore: None = full stance, i = leg i
        self.posture = None
        self.stance = None
        self.planted = np.ones(self.L, bool)
        self.swing = None       # step: the lifted / re-planted leg
        self.aim = False        # aiming the selected / swing leg with the sphere
        self.point = None       # the aiming sphere (a Point) while aiming
        self.aim_valid = None   # (K,) footholds the aimed leg can go to
        self.cam = (0.0, 0.0)   # viewer camera (azimuth, elevation) in rad; the sphere moves in its frame
        self.path = []          # step: [dict(stance, posture, edge_body)]
        self.valid = None       # (K,) where the last sampled leg could go, for display
        self.valid_leg = None   # ... which leg that was (valid indexes its candidate row)
        self.bad = False
        self.status = "explore: fly the ghost, Cross to sample a stance"
        self.cand = self.kit.candidates(self.target.body())

    # -- per tick ------------------------------------------------------------ #
    def tick(self, s, dt):
        p = s.pressed
        if p:
            self.bad = False    # any action clears the red body; a failing one sets it again
        if "start" in p:
            self.reset()
            return
        if "y" in p:
            self._switch_mode()
            return
        if "l1" in p or "r1" in p:
            c = self.cfg
            f = c.aim_sigma_step if "r1" in p else 1 / c.aim_sigma_step
            self.sigma = float(np.clip(self.sigma * f, *c.aim_sigma_range))
        if self.aim:
            self.point.update(s, dt, *self.cam)
        if self.mode == "explore":
            self._tick_explore(s, dt)
        else:
            self._tick_step(s, dt)

    def _tick_explore(self, s, dt):
        p = s.pressed
        if not self.aim and self.target.update(s, dt):
            self.cand = self.kit.candidates(self.target.body())
            self.valid = None   # indexed by cand -- stale once cand changes
        if "x" in p:
            self._cycle_select()
        if "a" in p:
            self._sample_full() if self.select is None else self._sample_leg(self.select)
        if "b" in p:
            self._save()
        if "up" in p or "down" in p:
            n = len(self.samplers.names)
            self.sampler = (self.sampler + (1 if "up" in p else -1)) % n

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
        if "x" in p:
            self._next_swing()
        if "a" in p:
            self._resample()
        if "b" in p:
            self._commit()
        if "left" in p:
            self._back()

    # -- mode switching ------------------------------------------------------ #
    def _switch_mode(self):
        if self.mode == "explore":
            if self.posture is None:
                self.status = "sample a stance first (Cross)"
                return
            self.mode = "step"
            self.path = [dict(stance=self.stance, posture=self.posture, edge_body=None)]
            self.swing, self.valid, self.aim = None, None, False
            self._sync_target()
            self.status = "step: Square to lift a leg"
        else:
            if not self._put_down():
                return
            self.mode, self.swing, self.valid, self.aim = "explore", None, None, False
            self.status = "explore: fly the ghost, Cross to sample"

    # -- explore ------------------------------------------------------------- #
    def _leg_base(self, i):
        """The posture leg ``i`` is re-planted from, or None if there is none.

        Step mode: the current posture. Explore mode: the body moved to the ghost
        with the other feet held (sets the status and ``bad`` if they can't).
        """
        if self.mode == "step":
            return self.posture
        planted = np.ones(self.L, bool)
        planted[i] = False
        _, post, c = self.kit.move_body(self.posture, self.stance, jnp.asarray(planted),
                                        self.target.body())
        # Leg i is about to be re-planted, so only the other legs (and the body) count.
        fails = _fails(c, [j for j in range(self.L) if j != i])
        if fails:
            self.bad = True
            self.status = f"other legs can't hold the ghost pose: {fails}"
            return None
        return post

    def _start_aim(self, i):
        """Aim leg ``i``: the sphere appears at its foothold and takes the sticks.

        The foothold, not the foot: a lifted foot hovers above it, and starting
        there would bias the distances by the lift height. The set of footholds
        the leg can go to is computed once here (the body can't move while aiming).
        """
        self.aim = True
        self.point = Point(self.cfg, np.asarray(self.kit.footholds.position[self.stance[i]]))
        base = self._leg_base(i)
        if base is None:
            self.aim_valid = np.zeros(self.cfg.k_per_leg, bool)
            return
        self.aim_valid = np.asarray(self.kit.leg_valid(self._key(), self.cand, base,
                                                       self.stance, i))
        self.status = (f"aim leg {i} ({int(self.aim_valid.sum())} reachable): "
                       "move the sphere, Cross to sample toward it")

    def _cycle_select(self):
        """full -> leg 0 -> aim 0 -> leg 1 -> aim 1 -> ... -> full."""
        if self.posture is None:
            self.select = None
            self.status = "sample a full stance first"
            return
        if self.select is None:
            self.select, self.aim = 0, False
        elif not self.aim:
            self._start_aim(self.select)
        else:
            self.aim = False
            self.select = None if self.select == self.L - 1 else self.select + 1

    def _sampler_name(self):
        return self.samplers.names[self.sampler]

    def _sample_full(self):
        ok, post, st, info = self.samplers.full(self._sampler_name(), self._key(),
                                                self.cand, self.target.body())
        self.bad = not bool(ok)
        if self.bad:
            self.status = "no valid stance at the ghost"
            return
        self.posture, self.stance, self.valid = post, st, None
        n = int(info["valid"].sum())
        self.status = f"stance sampled ({n} valid): {_score_text(info)}"

    def _sample_leg(self, i):
        """Move the body to the ghost with the other feet held, then re-plant leg ``i``."""
        post = self._leg_base(i)
        if post is None:
            return
        if self.aim:
            ok, post, st, info = self.samplers.leg_toward(
                self._key(), self.cand, post, self.stance, i, self.point.pos, self.sigma)
        else:
            ok, post, st, info = self.samplers.leg(self._sampler_name(), self._key(),
                                                   self.cand, post, self.stance, i)
        self.valid, self.valid_leg = np.asarray(info["valid"]), i
        self.bad = not bool(ok)
        if self.bad:
            self.status = f"no valid foothold for leg {i} at the ghost"
            return
        self.posture, self.stance = post, st
        self.status = f"leg {i} re-planted ({int(self.valid.sum())} options): {_score_text(info)}"

    def _save(self):
        if self.posture is None:
            self.status = "nothing to save"
            return
        self.saved.append(dict(stance=self.stance, posture=self.posture))
        self.status = f"saved posture {len(self.saved) - 1}"

    # -- step ---------------------------------------------------------------- #
    def _sync_target(self):
        t = np.asarray(self.posture.body.translation())
        rpy = self.posture.body.rotation().as_rpy_radians()
        self.target.set_state((float(t[0]), float(t[1]), float(t[2]),
                               float(rpy.pitch), float(rpy.yaw)))

    def _put_down(self):
        """Put the swing leg back on its (current) foothold."""
        if self.swing is None or self.planted[self.swing]:
            return True
        ok, post = self.kit.lift_leg(self.posture, self.stance, self.swing, 0.0)
        if not bool(ok):
            self.status = f"leg {self.swing} can't reach its foothold from here"
            return False
        self.posture, self.planted[self.swing] = post, True
        return True

    def _next_swing(self):
        """Cycle: none -> lift 0 -> aim 0 -> lift 1 -> aim 1 -> ... -> none."""
        if self.swing is not None and not self.aim:
            self._start_aim(self.swing)
            return
        self.aim = False
        if not self._put_down():
            return
        if self.swing == self.L - 1:
            self.swing, self.valid = None, None
            self.status = "no leg lifted"
            return
        i = 0 if self.swing is None else self.swing + 1
        ok, post = self.kit.lift_leg(self.posture, self.stance, i, LIFT_HEIGHT)
        if not bool(ok):
            self.status = f"leg {i} can't lift here"
            self.swing = i
            return
        planted = np.ones(self.L, bool)
        planted[i] = False
        c = self.kit.checks(post, self.stance, jnp.asarray(planted))
        self.posture, self.swing, self.valid, self.planted = post, i, None, planted
        fails = _fails(c)
        self.status = f"leg {i} lifted: move the body, Cross to re-plant" + (
            f"  (lifted pose invalid: {fails})" if fails else "")

    def _resample(self):
        if self.swing is None:
            self.status = "lift a leg first (Square)"
            return
        if self.aim:
            ok, post, st, info = self.samplers.leg_toward(
                self._key(), self.cand, self.posture, self.stance, self.swing,
                self.point.pos, self.sigma)
            valid = info["valid"]
        else:
            ok, post, st, valid = self.kit.resample_leg(
                self._key(), self.cand, self.posture, self.stance, self.swing)
        self.valid, self.valid_leg = np.asarray(valid), self.swing
        if not bool(ok):
            self.status = f"no valid foothold for leg {self.swing} from this body pose"
            return
        self.posture, self.stance = post, st
        self.planted[:] = True
        self.status = (f"leg {self.swing} re-planted ({int(valid.sum())} options): "
                       "Circle to commit, Cross to re-sample")

    def _commit(self):
        if not self.planted.all():
            self.status = "re-plant the lifted leg first (Cross)"
            return
        changed = int(np.sum(np.asarray(self.path[-1]["stance"]) != np.asarray(self.stance)))
        if changed == 0:
            self.status = "nothing to commit (stance unchanged)"
            return
        if changed > 1:
            # an edge moves exactly one leg (nodes share 3 footholds)
            self.status = f"{changed} legs changed, an edge moves one: D-pad left to go back"
            return
        self.path.append(dict(stance=self.stance, posture=self.posture,
                              edge_body=self.posture.body))
        self.swing, self.valid, self.aim = None, None, False
        self.status = f"committed node {len(self.path) - 1}"

    def _back(self):
        """Discard everything since the last committed node and return to it."""
        node = self.path[-1]
        self.posture, self.stance = node["posture"], node["stance"]
        self.planted = np.ones(self.L, bool)
        self.swing, self.valid, self.aim = None, None, False
        self._sync_target()
        self.status = f"back to node {len(self.path) - 1}"

    # -- display ------------------------------------------------------------- #
    def highlighted(self):
        """The leg to tint: the explore selection, or the step swing leg."""
        return self.select if self.mode == "explore" else self.swing

    def markers(self):
        """``[(pos, radius, rgba)]`` for the foothold markers."""
        fh = np.asarray(self.kit.footholds.position)
        out = []
        leg = self.highlighted()
        if leg is None:                            # candidates are per leg; none selected
            cand, valid = np.zeros(0, int), np.zeros(0, bool)
        else:
            m = np.asarray(self.cand.mask[leg])
            cand = np.asarray(self.cand.idx[leg])[m]
            valid = (self.valid[m] if self.valid is not None and self.valid_leg == leg
                     else np.zeros(len(cand), bool))
        radius = np.where(valid, 0.01, 0.005)
        if self.aim:
            # reachable footholds scaled by their sampling probability, max -> a
            # fixed radius, clipped below; the unreachable ones stay plain grey
            valid = self.aim_valid[m]
            c = self.cfg
            w = np.exp(-np.sum((fh[cand] - self.point.pos) ** 2, -1) / self.sigma**2)
            p = np.where(valid, w, 0.0)
            p = p / max(p.max(), 1e-300)
            radius = np.where(valid, np.clip(c.aim_marker_max * p, c.aim_marker_min,
                                             c.aim_marker_max),
                              0.75 * c.aim_marker_min)   # grey stays below the smallest green
        for j, v, r in zip(cand, valid, radius):
            out.append((fh[j], r, (0.2, 0.9, 0.3, 0.9) if v else (0.7, 0.7, 0.7, 0.5)))
        if self.aim:
            out.append((self.point.pos, 0.02, (1.0, 0.6, 0.1, 1.0)))
            out.append((self.point.pos, self.sigma, (1.0, 0.6, 0.1, 0.15)))
        if self.mode == "step":
            for node in self.path:
                for leg, j in enumerate(np.asarray(node["stance"])):
                    out.append((fh[j], 0.014, LEG_RGBA[leg]))
        return out

    def text(self):
        if self.mode == "explore":
            sel = "full" if self.select is None else f"leg {self.select}" + (
                " (aim)" if self.aim else "")
            head = (f"explore   sample: {sel}   sampler: {self._sampler_name()}   "
                    f"saved: {len(self.saved)}")
        else:
            swing = "-" if self.swing is None else str(self.swing)
            head = f"step   swing: {swing}   nodes: {len(self.path)}"
        if self.aim:
            head += f"   AIM  sigma {self.sigma * 100:.1f} cm"
        return head, self.status


def _score_text(info: dict) -> str:
    return (f"score {float(info['score']):.2f}  (|f| {float(info['force']):.1f} N, "
            f"|tau| {float(info['torque']):.2f} Nm, sigma {float(info['sigma']):.2f})")


def _marker_sig(pg: Playground) -> tuple:
    """Everything the markers depend on, by value (object ids can be reused)."""
    b = lambda a: None if a is None else np.asarray(a).tobytes()
    return (b(pg.cand.idx), b(pg.cand.mask), b(pg.valid), pg.valid_leg, b(pg.aim_valid),
            pg.highlighted(), len(pg.path), b(pg.stance), pg.mode, pg.aim, None if pg.point is None else pg.point.pos.tobytes(), pg.sigma)


def _draw(scn, markers):
    scn.ngeom = 0
    eye = np.eye(3).ravel()
    for pos, r, rgba in markers[: scn.maxgeom]:
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([r, 0, 0]),
                            np.asarray(pos, float), eye, np.asarray(rgba, np.float32))
        scn.ngeom += 1


def build(cfg: Cfg, seed: int = 0):
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

    cfg = Cfg()
    robot, kit, samplers, key = build(cfg)
    xml = scene_xml(robot, cfg)
    Path(__file__).with_name("scene.xml").write_text(xml)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "base")
    legs = leg_geoms(model, cfg.num_legs)
    mocap = model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target")]

    pad = DualSense()
    pg = Playground(kit, samplers, cfg, key)
    print(__doc__.split("Controls:")[1], flush=True)

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
            sig = _marker_sig(pg)
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
                # Camera rides with the body: the mouse sets the azimuth *relative*
                # to the body's yaw (whatever it changed since our last write is
                # the user's drag), elevation and distance as usual.
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
    pad.close()


if __name__ == "__main__":
    main()
