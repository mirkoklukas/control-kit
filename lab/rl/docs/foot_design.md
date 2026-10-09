# Foot design

The magnetic foot used in `lab/rl` (robot and `test_foot`). Background and
sizing arguments: `docs/projects/magnetic-foot-sim.md`. Test results: `notes.md`.

## How the foot works

```
      tibia
        |
        o      ankle pivot: 2 passive hinges (Cardan), weak springs, hard stops
     [=pad=]   square magnetic pad
  ▁▁▁▁▁▁▁▁▁▁▁  surface (floor / wall)
```

- **Ankle.** Two passive hinges, both perpendicular to the tibia and meeting at one
  pivot, form a Cardan joint. The pad can roll and pitch relative to the tibia, but
  it can't twist about the tibia's axis (that twist is called yaw).
  - Weak springs pull the pad back to its neutral position, with its face
    perpendicular to the tibia.
  - Light damping keeps the pad from flapping during swing.
  - Hard stops at ±`ankle_range_deg`. Beyond those angles the tibia can lever the
    pad off the surface.
- **Landing flat.** Because the ankle is limp, the pad lies flat on the surface
  whenever the tibia meets it within ±`ankle_range_deg` of the normal. The leg
  doesn't have to control foot orientation (a 3-DOF leg can't).
- **Holding.** The magnet pulls the pad onto the surface. Pulling straight off
  needs the full adhesion force *A*. Pulling along the surface is limited by
  friction, μ·*A*. In between, Coulomb friction with adhesion gives the release
  force μ*A* / (sin θ + μ cos θ) for a pull at angle θ from the normal.
- **Load path.** Below the stops, the ankle transmits almost no moment, so the leg's
  force reaches the pad through the pivot. The pivot sits `pivot_height` +
  `pad_thickness` above the contact plane. So a shear force F also produces a peel
  moment of F × that height, which is why the pivot should be as low as possible.
- **Weak spots**
  - At the stops, the tibia (0.2 m) levers the pad directly, which is strong peel.
  - A light leg with a limp ankle swings dynamically and can overshoot into the stop
    even when the pull is inside the range (`test_foot`: free leg pulled at 40°).
    Ankle stiffness and damping matter.
- **Letting go.** The magnet is switched off: adhesion ctrl goes to 0, the way an
  electro-permanent magnet (EPM) would be. Without a switchable magnet, the foot
  would have to be levered off against the stops.

## Current parameters (`config.py`, `MjModelCfg`)

| Parameter | Value | Meaning |
|---|---|---|
| `pad_size` | 0.04 m | square pad side |
| `pad_cells` | 3 | pad = 3×3 cells, each with its own adhesion |
| `pad_thickness` | 0.006 m | |
| `pivot_height` | 0.005 m | pivot above the pad's back face |
| `mass_pad` | 0.05 kg | |
| `ankle_range_deg` | ±45° | hard stops, both hinges |
| `ankle_stiffness` | 0.05 N·m/rad | centering spring |
| `ankle_damping` | 0.005 N·m·s/rad | |
| `ankle_armature` | 1e-4 | numerical, regularizes the light pad |
| `pad_friction` | 0.5 | μ (painted steel ≈ 0.3–0.5) |
| `adhesion_gain` | 30 N | adhesion force per foot at ctrl 1, split over the cells |
| `adhesion_margin` | 0.002 m | contact margin on the pad boxes |

All are placeholders.

## MuJoCo model

Built in `model._add_foot` on each `foot{i}` body at the tibia's tip. The foot
frame's +x runs along the tibia.

- **Bodies and joints.**
  - `pad{i}` is a child of `foot{i}`, placed at the pivot.
  - Its hinges are `ankle{i}_a` (about the foot's y axis) and `ankle{i}_b` (about
    its z axis). Each has `stiffness`, `damping`, `armature` and a limited `range`.
- **Pad geometry and adhesion.** The pad is `pad_cells` × `pad_cells` cells. Each
  cell is its own child body `pad{i}_c{k}` (no joint), with a box geom and its own
  `adhesion` actuator `adhere{i}_{k}` (ctrl in [0, 1], gain `adhesion_gain` / N²).
  - Splitting the pad makes the force scale with the touching area; see "foot
    peeling problem" in `notes.md`.
  - `forcelimited` is off; otherwise the robot's servo `forcerange` default caps
    each actuator at 5 N.
  - Solver options: `cone="elliptic"` and `impratio=10`, as recommended for
    adhesion.
- **Collision details**
  - Tibia–pad collision is excluded.
  - The tibia's collision capsule ends 2 × `link_radius` short of the pivot, so its
    rounded end can't hit the ground or the pad when the ankle flexes. A thin
    visual-only capsule bridges the gap.

### Behavior to know about (MuJoCo 3.9.0)

- **Adhesion is all-or-nothing per body.** The full `gain × ctrl` is spread over
  however many contacts a body has, so a one-body pad holds as much on one corner
  as on the whole face. That's why the pad is split into cells: with 3×3, an edge
  holds about 1/3 and a corner about 1/9 of the force.
  - There is no area or air-gap dependence and no torque that pulls the pad flat.
  - Confirmed in `test_foot` (rigid section): with one body, flat, edge and corner
    all released at 30.2 N. With 3×3 cells: 30.6 / 10.6 / 4.0 N.
  - A pad starting on an edge therefore isn't flattened, and can roll onto its side
    face.
- **The margin makes the pad hover.** Contacts become active within `margin`, and
  their penetration is measured from `margin`, so the pad rests about 1.8 mm above
  the surface.
  - `gap` has no effect here, and the margins of the two geoms add up.
  - We accept the small hover.
  - Beyond the margin there is no adhesion at all.
- **MjSpec defaults to degrees.** Set `spec.compiler.degree = False` when building
  from scratch, since `_add_foot` passes radians. Missing this once made the ankle
  range ±0.8°.

### Possible improvements

1. Done: the pad is split into N×N cells (default 3×3). Sphere cells would give 1
   contact each instead of 4, which is cheaper.
2. Compute each cell's adhesion from its actual gap, with a falloff (a 1 mm gap
   leaves about 7% of the force). That adds a torque towards flat and realistic
   small-tilt peel. The walls are flat, so the distance is a dot product. The force
   is stiff, though, so it may need dt ≤ 1 ms or a force lag.
