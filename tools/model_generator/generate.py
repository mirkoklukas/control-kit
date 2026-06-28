#!/usr/bin/env python3
"""Turn a hexapod YAML spec into a MuJoCo MJCF model.

    python tools/model_generator/generate.py output/hexapod_custom.yaml
    python tools/model_generator/generate.py spec.yaml -o models/myhex.xml

The YAML is the source of truth (hand-edit it or produce it from the web tool in
this directory); this script is the only thing that writes the XML. The geometry
recipe mirrors models/hexapod.xml: a hexagonal base with one leg per corner at
30/90/150/210/270/330 deg, replicated 6x so names index 0..5 as
FL,ML,BL,BR,MR,FR. Segment lengths are pure lengths along the leg-local +x axis
at the zero pose; the standing rest pose is set separately via joint angles and
written as a <keyframe>, with the base height chosen so the feet rest on z=0.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import yaml

# Fixed bits not exposed in the UI (kept as in models/hexapod.xml).
LEG_CAPSULE_RADIUS = 0.012  # fallback capsule radius if legs.thickness is unset
CORNER_ANGLES = (30, 90, 150, 210, 270, 330)  # leg 0..5 bearings


# ---------------------------------------------------------------- defaults / IO
def default_config() -> dict:
    """The baseline spec; the web tool starts from these too."""
    return {
        "name": "hexapod_custom",
        "radius": 0.12,  # circumradius: hexagon center -> vertex
        "body": {
            "density": 600.0,
            "half_thickness": 0.025,
            "color": [0.85, 0.5, 0.2],
            "head_color": [0.9, 0.6, 0.25],
        },
        "legs": {
            "density": 600.0,
            "thickness": 0.012,  # capsule radius, shared by coxa/femur/tibia
            "coxa": {"length": 0.06, "color": [0.85, 0.2, 0.2]},
            "femur": {"length": 0.10, "color": [0.2, 0.4, 0.85]},
            "tibia": {"length": 0.19, "color": [0.9, 0.8, 0.15]},
            "foot": {"size": 0.018, "density": 600.0, "color": [0.15, 0.15, 0.2]},
        },
        # Rest pose: hinge angles (degrees) applied to every leg, written to a
        # <keyframe>. coxa = yaw (about z), femur/tibia = lift/knee (about y).
        "rest": {"coxa": 0.0, "femur": 25.0, "tibia": 55.0},
        # Joint limits (degrees), written to each joint's <default> range.
        "joint_range": {"coxa": [-45.0, 45.0], "femur": [-75.0, 75.0], "tibia": [-110.0, 30.0]},
    }


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    return _merge(default_config(), cfg)


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        out[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


# ------------------------------------------------------------------ kinematics
def base_height(cfg: dict) -> float:
    """Body-origin height so the lowest point of a foot rests on z=0.

    At the zero pose the leg lies along +x; femur lift (about y) and tibia knee
    (about y, cumulative) drop the tip in -z. coxa yaw (about z) doesn't change
    height, so all six feet share one z.
    """
    lf = cfg["legs"]["femur"]["length"]
    lt = cfg["legs"]["tibia"]["length"]
    af = math.radians(cfg["rest"]["femur"])
    at = math.radians(cfg["rest"]["tibia"])
    foot_drop = lf * math.sin(af) + lt * math.sin(af + at)
    return foot_drop + cfg["legs"]["foot"]["size"]


# ------------------------------------------------------------------ formatting
def _rgba(color, alpha=1.0) -> str:
    r, g, b = color
    return f"{r:g} {g:g} {b:g} {alpha:g}"


def _lighten(color, t=0.35):
    return [c + (1.0 - c) * t for c in color]


def _hex_vertices(radius: float, half_t: float) -> str:
    """12 vertices (6 top + 6 bottom) for the hexagonal-prism convex hull."""
    rows = []
    for z in (half_t, -half_t):
        for ang in CORNER_ANGLES:
            a = math.radians(ang)
            rows.append(f"      {radius * math.cos(a): .5f} {radius * math.sin(a): .5f} {z: .5f}")
    return "\n".join(rows)


def _leg(i: int, ang_deg: float, R: float, lc: float, lf: float, lt: float,
         foot: float, foot_density: float) -> str:
    """One full leg subtree at corner `ang_deg`, names suffixed with index `i`.

    Written out explicitly (rather than via <replicate>) because <replicate>
    doesn't expand <keyframe> qpos, which we need for the rest pose.
    """
    a = math.radians(ang_deg)
    cx, cy = R * math.cos(a), R * math.sin(a)
    return f"""\
      <body name="coxa{i}" pos="{cx:g} {cy:g} 0" euler="0 0 {ang_deg:g}" childclass="hexapod">
        <joint name="coxa{i}" class="coxa"/>
        <geom class="coxa" fromto="0 0 0 {lc:g} 0 0"/>
        <body name="femur{i}" pos="{lc:g} 0 0">
          <joint name="femur{i}" class="femur"/>
          <geom class="femur" fromto="0 0 0 {lf:g} 0 0"/>
          <body name="tibia{i}" pos="{lf:g} 0 0">
            <joint name="tibia{i}" class="tibia"/>
            <geom class="tibia" fromto="0 0 0 {lt:g} 0 0"/>
            <geom name="foot{i}" type="sphere" pos="{lt:g} 0 0" size="{foot:g}" density="{foot_density:g}" material="foot"/>
            <site name="foot{i}" pos="{lt:g} 0 0" size="0.012"/>
          </body>
        </body>
      </body>"""


# ----------------------------------------------------------------- XML builder
def build_xml(cfg: dict) -> str:
    R = cfg["radius"]
    body = cfg["body"]
    legs = cfg["legs"]
    half_t = body["half_thickness"]
    lc = legs["coxa"]["length"]
    lf = legs["femur"]["length"]
    lt = legs["tibia"]["length"]
    foot = legs["foot"]["size"]
    foot_density = legs["foot"]["density"]
    rad = legs.get("thickness") or LEG_CAPSULE_RADIUS

    bz = base_height(cfg)

    legs_xml = "\n".join(
        _leg(i, ang, R, lc, lf, lt, foot, foot_density) for i, ang in enumerate(CORNER_ANGLES)
    )

    # <keyframe> qpos: free joint (pos 3 + quat 4) then coxa,femur,tibia per leg,
    # in leg order 0..5. Hinge angles in qpos are ALWAYS radians, regardless of
    # the compiler's angle=degree setting.
    ac, afr, atr = (math.radians(cfg["rest"][k]) for k in ("coxa", "femur", "tibia"))
    per_leg = f"{ac:g} {afr:g} {atr:g}"
    qpos = f"0 0 {bz:g} 1 0 0 0  " + "  ".join([per_leg] * 6)

    jr = cfg["joint_range"]
    rng = {k: f"{jr[k][0]:g} {jr[k][1]:g}" for k in ("coxa", "femur", "tibia")}

    head_color = body.get("head_color") or _lighten(body["color"])

    actuators = "\n".join(
        f'    <position name="{part}{i}" joint="{part}{i}" class="hexapod"/>'
        for i in range(6)
        for part in ("coxa", "femur", "tibia")
    )

    return f"""<mujoco model="{cfg['name']}">
  <compiler angle="degree" autolimits="true"/>
  <option timestep="0.004" integrator="implicitfast" iterations="10" ls_iterations="8"/>

  <!-- Generated by tools/model_generator from a YAML spec. Radial hexapod: a
       hexagonal base with one leg at each corner (30/90/150/210/270/330 deg),
       circumradius {R:g}. Per leg: coxa (yaw) -> femur (lift) -> tibia (knee),
       18 actuated DOF. Segment lengths are pure leg-local +x lengths at the zero
       pose; the standing rest pose lives in <keyframe name="home">. -->
  <default>
    <geom density="{legs['density']:g}" friction="1.0 0.5 0.5" contype="0" conaffinity="1"/>
    <default class="hexapod">
      <joint type="hinge" armature="0.008" damping="0.4" frictionloss="0.02"/>
      <geom type="capsule" size="{rad:g}"/>
      <position kp="25" kv="2.5" forcerange="-15 15"/>
      <default class="coxa">
        <joint axis="0 0 1" range="{rng['coxa']}"/>
        <geom material="coxa"/>
      </default>
      <default class="femur">
        <joint axis="0 1 0" range="{rng['femur']}"/>
        <geom material="femur"/>
      </default>
      <default class="tibia">
        <joint axis="0 1 0" range="{rng['tibia']}"/>
        <geom material="tibia"/>
      </default>
    </default>
  </default>

  <asset>
    <material name="base"  rgba="{_rgba(body['color'])}"/>
    <material name="head"  rgba="{_rgba(head_color)}"/>
    <material name="coxa"  rgba="{_rgba(legs['coxa']['color'])}"/>
    <material name="femur" rgba="{_rgba(legs['femur']['color'])}"/>
    <material name="tibia" rgba="{_rgba(legs['tibia']['color'])}"/>
    <material name="foot"  rgba="{_rgba(legs['foot']['color'])}"/>

    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.2 0.3 0.4" rgb2="0.1 0.15 0.2"
             width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="10 10" reflectance="0.2"/>
    <mesh name="hexbody" vertex="
{_hex_vertices(R, half_t)}"/>
  </asset>

  <worldbody>
    <light pos="0 0 2.0" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="grid" contype="1" conaffinity="1"/>

    <body name="base" pos="0 0 {bz:g}">
      <freejoint name="root"/>
      <geom name="base" type="mesh" mesh="hexbody" material="base" density="{body['density']:g}"/>
      <geom name="head" type="box" pos="{0.75 * R:g} 0 {half_t / 2:g}" size="0.02 0.03 0.02" material="head"/>

      <!-- 6 legs, one per corner. Index -> corner:
             0 = FL (30)   1 = ML (90)   2 = BL (150)
             3 = BR (210)  4 = MR (270)  5 = FR (330) -->
{legs_xml}
    </body>
  </worldbody>

  <actuator>
{actuators}
  </actuator>

  <keyframe>
    <key name="home" qpos="{qpos}"/>
  </keyframe>
</mujoco>
"""


def write_model(cfg: dict, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_xml(cfg))
    return out_path


# ------------------------------------------------------------------------- CLI
def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a hexapod MJCF from a YAML spec.")
    ap.add_argument("config", help="path to the YAML spec")
    ap.add_argument("-o", "--out", help="output .xml (default: spec path with .xml suffix)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = args.out or str(Path(args.config).with_suffix(".xml"))
    path = write_model(cfg, out)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
