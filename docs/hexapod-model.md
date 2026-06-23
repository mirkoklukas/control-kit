# Hexapod model notes

Working notes on [`models/hexapod.xml`](../models/hexapod.xml): a radial hexapod
with a hexagonal trunk and one 3-DOF leg at each corner (coxa → femur → tibia),
18 actuated joints. Everything below was checked against MuJoCo 3.9.0 by loading
the model and measuring; numbers are from those runs. Living doc, prune later.

## The rest pose: what "zero" means

"Rest position" can mean three different things in MJCF. Here they all coincide
at **0°**, which is why the model holds a clean standing pose with zero command:

1. **Kinematic zero — joint `ref`** → stored as `qpos0`. No joint sets `ref`, so
   every hinge's zero is 0°. This is the configuration the XML *itself draws*: the
   `<body>` positions and `euler` angles are the model at all-joints-zero. So the
   geometry in the file **is** the rest configuration.
2. **Passive spring rest — `springref` + joint `stiffness`** → there is none.
   Stiffness is 0, so nothing springs a joint back to a rest angle. The only
   passive terms are `damping="0.4"` and `frictionloss="0.02"`, which resist
   velocity, not displacement. Cut power and the legs fold under gravity.
3. **Commanded hold — the `position` actuator target** → defaults to 0. With
   `kp=25, kv=2.5` each servo drives its joint toward 0°. This is what actively
   holds the pose.

### Geometry of the zero pose

Per leg, in the leg-local frame (`+x` outward, `+z` up), walking the chain
outward gives coxa `0.06` + femur `0.10` + tibia foot offset `0.05` = **0.21 m
out** and **0.19 m down**, so each foot sits at leg-local `(0.21, 0, -0.19)`.

With the trunk freejoint at `z = 0.20`, the six feet land symmetrically at:

- world `z = 0.01` (i.e. right at the floor; the foot sphere radius is 0.018, so
  it just makes contact / penetrates ~8 mm)
- horizontal radius **0.33 m** (e.g. `foot_ML = (0, 0.33, 0.01)`), versus the
  trunk's 0.12 circumradius

The pose is 6-fold symmetric by construction. Total model mass is **1.76 kg**
(≈17.3 N weight).

## Joint ranges and their coordinate system

Each joint is a `hinge`, so its whole configuration is **one scalar**: the
rotation angle about its own `axis`. `range` is the allowed interval for that
scalar. Read it as:

- **Units: degrees** (`compiler angle="degree"`).
- **Zero = the rest pose.** The angle is measured relative to the joint `ref`
  (= 0), so `-45 45` means "±45° away from rest", not from any world axis or the
  ground.
- **It is a *relative* joint angle** in the chain (trunk→coxa, coxa→femur,
  femur→tibia), about that joint's axis, with **right-hand sign**. The `axis`
  vector is given in the **leg-local body frame** (after the corner `euler`), so
  the numbers are identical for all six legs even though the world-space motion
  differs per leg.

Signs verified by perturbing the FL leg one joint at a time:

| Joint | axis (leg-local)     | meaning                | `range`   | sign (measured)                                                |
| ----- | -------------------- | ---------------------- | --------- | -------------------------------------------------------------- |
| coxa  | `0 0 1` (= world up) | yaw / horizontal swing | `-45 45`  | **+** swings leg CCW seen from above; foot stays at constant z |
| femur | `0 1 0`              | lift                   | `-75 75`  | **−** raises foot, **+** lowers it                             |
| tibia | `0 1 0`              | knee                   | `-110 30` | **+** flexes (foot tucks in), **−** extends (foot out and up)  |

Hard numbers behind the signs: `femur_FL +20°` dropped the foot 40 mm in z;
`−20°` raised it 63 mm. `coxa_FL ±20°` moved the foot purely horizontally
(Δz = 0).

The one to watch is the **tibia: `-110 30` is not centered on rest** (only +30°
of flex past the standing pose, but −110° of extension). coxa and femur are
symmetric about 0.

## Per-segment colors

Colors hang off the existing `coxa` / `femur` / `tibia` default classes (which
the joints already reference), as a `<geom rgba=...>`:

```xml
<default class="coxa">  <joint .../>  <geom rgba="0.85 0.2 0.2 1"/></default>  <!-- red    -->
<default class="femur"> <joint .../>  <geom rgba="0.2 0.4 0.85 1"/></default>  <!-- blue   -->
<default class="tibia"> <joint .../>  <geom rgba="0.9 0.8 0.15 1"/></default>  <!-- yellow -->
```

Each segment geom then carries `class="coxa|femur|tibia"`. Because these classes
are nested inside the `hexapod` class, the geom still inherits `type="capsule"`
and `size` from the parent; the class only adds the color (and an explicit
per-geom `size` like the tibia's `0.010` still wins). Recoloring later is a
one-line change per segment instead of editing all six legs. Feet stay dark, the
trunk stays orange.

## Foot–ground friction

Friction is a **per-geom** property, set once on the root default:

```xml
<geom density="600" friction="1.0 0.5 0.5"/>
```

The three numbers are `[slide, torsion, roll]`, **not** x/y/z. Neither the foot
spheres nor the floor override it, so both inherit `[1.0, 0.5, 0.5]`.

**Combine rule.** For an auto-generated contact, MuJoCo takes the *elementwise
max* of the two geoms' friction, unless one geom has higher `priority` (then that
geom's values are used outright). Foot and floor are both `[1.0, 0.5, 0.5]`, so
the contact resolves to slide **μ = 1.0**.

**The catch — most of it is dormant.** The contact has `condim = 3` (the
default), which models only normal + 2 sliding directions. **Torsion and rolling
friction are inactive**; the `0.5 0.5` are carried into the contact vector but
never used. So effectively the feet have sliding friction μ = 1.0 and nothing
else (roll is moot anyway: the foot spheres are rigidly fixed to the tibia).
To activate spin/roll, raise `condim` to 4 (torsional) or 6 (rolling) on the foot
or floor geom; the contact takes the larger `condim`.

**Changing it** (the max rule is the gotcha — lowering only the feet won't help
if the floor is still μ = 1.0):

- *Single point of control:* put `friction` + `priority="1"` on the floor geom;
  the floor's values then govern every foot contact regardless of foot settings.
- *Explicit pairs:* `<contact><pair geom1="floor" geom2="foot_FL" friction="..."
  condim="4"/></contact>` for exact per-contact control, bypassing the combine
  rule.
- *Global:* just edit the `1.0` on the root default.

A reasonable walking range is μ ≈ 0.8–1.0; go lower to study slipping.

**Caveat — the rest stance is not a clean six-point contact.** At the rest pose
the collision check reports 9 contacts: the six feet (penetrating ~8 mm) plus the
**tibia capsule tips of FL, BL, BR grazing the floor** (dist ≈ 0). If you want a
clean six-point stance, nudge the trunk height up slightly or shorten the tibia
capsule's lower end.

## Magnetic / sticky foot

To model a foot that grips the ground and can be **switched on/off or dialed by a
parameter**, use MuJoCo's purpose-built **`<adhesion>` actuator**:

```xml
<adhesion name="mag_FL" body="tibia_FL" ctrlrange="0 1" gain="80"/>
```

```xml
<!-- give the foot a small reach so the magnet grabs near contact -->
<geom name="foot_FL" type="sphere" pos="0.05 0 -0.19" size="0.018"
      rgba="0.15 0.15 0.2 1" margin="0.02" gap="0.02"/>
```

How it works:

- It applies an attractive force **normal to every contact** of the named `body`,
  pulling it toward whatever it touches. Magnitude = `ctrl × gain` (N), so `ctrl`
  is the control parameter: `0/1` for a switch, or `ctrl ∈ [0,1]` for a dial-able
  holding force.
- It only acts **through existing contacts**. For "grab from a small gap" (real
  magnets reach a little), give the foot geom a `margin` (the reach) and a
  matching `gap` (suppresses the normal repulsion over that layer so the foot
  doesn't bounce while adhesion still pulls). Beyond `margin` the magnet releases.
- For a gait, add **one adhesion actuator per foot** (6 extra control channels);
  the controller turns each magnet on during stance, off during swing.

**Verified.** With `gain=80` on one foot and gravity flipped to `+z` (pulling the
robot off the floor at 9.81 m/s²): magnet **off**, the foot flew to z = 20 m in
2 s; magnet **on**, it stayed pinned at z ≈ 0.03 m while the trunk dangled up to
z = 0.35 m. One foot at gain 80 easily held the 17.3 N robot.

### Alternatives

| You want… | Use | Note |
|-----------|-----|------|
| Switchable finite force, dial-able strength | **`<adhesion>`** (above) | the right default |
| A true distance law (e.g. F ∝ 1/d²) toward a metal surface | compute in Python, apply via `data.xfrc_applied[body]` or a force at the foot `site` | full freedom, but it's code, not XML |
| A *rigid* latch — infinitely strong magnet that fully locks the foot | toggled equality: `<connect>` (position lock) or `<weld>` (full 6-DOF), flipped at runtime via `data.eq_active[i]` | discrete clamp, no compliance; you set the lock point on engage |

Caveat: adhesion sticks to **any** geom the foot contacts, not specifically
"metal". For a selective surface, gate `ctrl` in software (raise it only over the
magnetic surface) or restrict contacts with a `<pair>`. The `foot_XX` sites
already exist, which is handy for the custom-force route.

## Quick knob reference

| Want to change… | Where |
|-----------------|-------|
| Joint travel limits | `range` on the `coxa`/`femur`/`tibia` default classes |
| Limit the *commanded* target | add `ctrlrange` to the `position` actuators |
| Joint torque | `forcerange` (currently ±15 N·m) on the `hexapod` class |
| Passive resistance / rest spring | `damping`, `frictionloss`, `stiffness`+`springref` on `<joint>` |
| Segment color | `rgba` on the per-segment default classes |
| Ground friction | `friction` (+ `priority`) on floor/feet, or a `<pair>`; `condim` for spin/roll |
| Sticky/magnetic foot | `<adhesion>` actuator + foot `margin`/`gap` |
| Standing height | trunk freejoint `pos` z (currently 0.20) |
