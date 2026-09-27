# Magnetic Climbing Quadruped: Foot Design & MuJoCo Adhesion Modeling

Handoff document. Written 2026-09-22 to continue work in a new session.

---

## 0. Notes for the next assistant

- The user works at a robotics lab. They are proficient in Python and prefer crisp, to-the-point answers.
- MuJoCo has changed a lot recently (3.9 to 3.11, May to July 2026). **Verify version-specific claims against the docs for the installed version** before relying on them. Anything marked **[UNVERIFIED]** below was inferred, not confirmed.
- Check the installed version first: `python -c "import mujoco; print(mujoco.__version__)"`.

---

## 1. Project context

- **Goal:** a legged robot that climbs steel walls using magnetic feet.
- **Platform:** a quadruped, currently in the design phase.
- **Layouts under consideration:**
  - (a) A symmetric spider/insect-like layout: sprawled, with a yaw hip joint first.
  - (b) An ANYmal-like mammalian layout: roll hip joint first, legs under the body.

### Layout discussion so far (leaning spider/sprawled for wall-dominant use)

- **The key quantity is the peel moment.** On a vertical wall, the upper feet must resist pull-off of roughly `F_pull ≈ m·g·h / L`, where `h` is the COM height above the wall and `L` is the support length along gravity. A sprawled layout minimizes `h/L`. (The support length L is the distance, measured on the wall along the direction of gravity, between the upper and lower foot contacts.)
  - Example: 20 kg robot with L = 0.5 m. At h = 0.3 m the upper feet need about 118 N; at h = 0.1 m they need about 39 N.
- **Sprawled layout pros:** low COM, omnidirectional symmetry, large lateral workspace, easier concave-corner transitions.
- **Sprawled layout cons:** high joint torques on the floor, wide footprint.
- **Mammalian layout pros:** efficient on the ground, dynamic gaits, narrow footprint for girders and flanges, lots of reusable prior work.
- **Shear sizing:** total magnetic normal force must be at least `m·g / μ`, with μ around 0.3 to 0.5 on painted steel. This often sizes the magnets more than peel does.
- **Open questions for the user:**
  - What is the wall vs. floor time split?
  - What are the target structures (ship hulls, tanks, bridges)?
- **Relevant prior work:**
  - Magnecko (ETH, insect-style, EPM feet).
  - MARVEL (KAIST, EPM feet).
  - Uno et al., CLAWAR 2021, which compares joint topologies for climbing in simulation.

---

## 2. Foot design (current decision)

### Concept
- A square magnetic pad (a "thickened square").
- A **passive 2-DoF ankle**: roll and pitch free, **yaw blocked**.
- Implemented as a **universal (Cardan) joint**. The cross carries yaw torque inherently, it is stiff, and it is easy to add angle sensing to.

### Design requirements and notes
| Item | Decision / guidance |
|---|---|
| Joint | Cardan: two orthogonal hinges with intersecting axes |
| Range | About ±20–30° in roll and pitch, with hard stops (also usable for levering the foot off) |
| Centering | Light spring to neutral so the pad lands flat, plus damping to prevent flapping in swing |
| Pivot height | **As low as possible, ideally a few mm above the pad face.** Wall shear m·g × pivot height produces a moment that unloads one edge and starts a peel. A virtual pivot at the contact plane (spherical-shell bearing or remote-center linkage) is ideal. |
| Sensing | Hall sensors or small encoders on both axes, for touchdown detection, surface-normal estimation, and peel/slip warning |
| Pad | Steel pole pieces to concentrate flux. Keep any friction/compliant layer **thin**, because force drops steeply with air gap. Get compliance from the ankle, not from a rubber layer under the magnet. |
| Detachment | A passive ankle cannot peel by itself. Either lever off against the hard stops with a leg motion, or use EPMs (switchable). |

**Magnet sensitivity:** in the KAIST EPM foot, a 1 mm air gap reduces holding force to about 7% of maximum (Um et al. 2025). Air gap is the dominant effect to model.

**Open question for the user:** is "thickened" meant as compliance in the pad itself, or only structural thickness?

---

## 3. MuJoCo model of the foot

### Base MJCF (Cardan ankle with springs)

Notes on the model below:
- Two hinges in the same body with the same `pos` form a Cardan joint.
- `stiffness`/`springref` provide the centering spring; `damping` provides the damper.
- The pad is a 2×2 grid of box geoms instead of one box. This gives more and more stable contact points, a trick taken from the official adhesion example.

```xml
<mujoco>
  <compiler angle="degree"/>
  <option timestep="0.001" integrator="implicitfast" cone="elliptic" impratio="10"/>

  <default>
    <default class="pad">
      <!-- gap: detection buffer for adhesion (see Section 4 for semantics) -->
      <geom type="box" size="0.015 0.015 0.005" friction="0.5 0.005 0.0001"
            solref="0.005 1" solimp="0.95 0.99 0.001" gap="0.005"/>
    </default>
  </default>

  <worldbody>
    <body name="shank" pos="0 0 0.5">
      <!-- ... leg geoms/joints ... -->
      <body name="foot" pos="0 0 -0.25">
        <joint name="ankle_pitch" type="hinge" axis="0 1 0" pos="0 0 -0.005"
               range="-25 25" limited="true"
               stiffness="1.5" springref="0" damping="0.02" armature="0.0005"/>
        <joint name="ankle_roll"  type="hinge" axis="1 0 0" pos="0 0 -0.005"
               range="-25 25" limited="true"
               stiffness="1.5" springref="0" damping="0.02" armature="0.0005"/>
        <site name="foot_site" pos="0 0 -0.01"/>  <!-- at the pad face -->
        <geom class="pad" pos=" 0.015  0.015 -0.005" mass="0.04"/>
        <geom class="pad" pos=" 0.015 -0.015 -0.005" mass="0.04"/>
        <geom class="pad" pos="-0.015  0.015 -0.005" mass="0.04"/>
        <geom class="pad" pos="-0.015 -0.015 -0.005" mass="0.04"/>
      </body>
    </body>
  </worldbody>

  <sensor>
    <jointpos name="ankle_pitch_pos" joint="ankle_pitch"/>
    <jointpos name="ankle_roll_pos"  joint="ankle_roll"/>
    <force  name="foot_force"  site="foot_site"/>
    <torque name="foot_torque" site="foot_site"/>
  </sensor>
</mujoco>
```

### Tuning notes
- **Spring stiffness.** Static sag in swing is about `m_foot·g·r / k`, where r is the pivot-to-COM distance. Keep it to a few degrees.
  - Reference values: KAIST modeled their elastic ankle as a PD on a ball joint with Kp ≈ 0.05 and Kd ≈ 0.001 (foot about 0.2 kg). Their foot is 3-DoF, not Cardan.
- **Nonlinear springs.**
  - MuJoCo ≥ 3.7 supports polynomial stiffness/damping profiles on joints. **[UNVERIFIED]** Check the exact MJCF attribute in the XML reference.
  - Alternatively, set `stiffness="0"` and apply torque via `data.qfrc_applied` or `mjcb_passive`.
- **Timestep.** Use 0.0005–0.001 s. A light foot with springs and stiff contacts produces high-frequency dynamics.
- **Contact stiffness.** `solref` time constant should be at least 2× the timestep.

### Debugging the earlier bouncing/sliding
The user saw bouncing and sliding with adhesion. Likely causes:
1. **margin/gap misuse.** Semantics changed in 3.9 (see Section 4).
2. **Soft-contact creep.** Fix with `cone="elliptic"`, high `impratio`, and `noslip_iterations` (CPU only).
3. **Timestep too large.**
4. **Contact too soft** (`solref`).

To debug, log `data.ncon` and per-contact forces from `mj_contactForce`:
- Flickering contacts point to margin/gap or timestep problems.
- Drifting position under steady contact points to slip.

---

## 4. Margin/gap semantics (important; changed in MuJoCo 3.9.0, May 2026)

**New semantics (≥ 3.9):**
- Contacts are detected when `dist < margin + gap`.
- Contact forces are generated only when `dist < margin`.
- Contacts in `margin < dist ≤ margin + gap` are **inactive**: they appear in `mjData.contact` with no force. Adhesion uses exactly these.
- Old `margin="x" gap="x"` becomes new `margin="0" gap="x"`.

**Other changes:**
- Since 3.5, the margin/gap values of the two geoms are **summed**, not maxed.

**Takeaway:** for adhesion at a distance, use `margin="0" gap="0.005"` (or similar) on current versions.

---

## 5. Adhesion modeling options

### Option A: Adhesion actuator (the classic approach)

```xml
<actuator>
  <adhesion name="mag_FL" body="foot" ctrlrange="0 1" gain="150"/>
</actuator>
```

**Behavior:**
- Force = `gain × ctrl`, split among the body's active + inactive contacts. It is smooth in `ctrl`.
- It is **binary in distance**: full force anywhere inside the gap zone, zero outside. This is unrealistic, since real magnets fall off steeply with gap.
- It increases friction indirectly, by pulling geoms together and raising the contact normal force.

**Making it realistic:** scale `ctrl` each step by a force-gap curve, gate it on contact, and low-pass filter it for EPM switching time.

```python
import numpy as np
import mujoco

def magnet_scale(d, d0=0.0003):
    # Fit to your magnet's measured force-gap curve.
    # KAIST EPM: ~7% at 1 mm gap. Replace with np.interp over measured data.
    return 1.0 / (1.0 + max(d, 0.0) / d0) ** 2

def foot_min_dist(model, data, foot_body_id):
    dmin = np.inf
    for i in range(data.ncon):
        c = data.contact[i]
        b1 = model.geom_bodyid[c.geom1]
        b2 = model.geom_bodyid[c.geom2]
        if foot_body_id in (b1, b2):
            dmin = min(dmin, c.dist)
    return dmin

# each control step, before mj_step:
d = foot_min_dist(model, data, foot_id)
u_target = u_cmd * (magnet_scale(d) if np.isfinite(d) else 0.0)
u_filt += (u_target - u_filt) * dt / tau_switch   # EPM switching dynamics
data.ctrl[adh_id] = u_filt
```

**Gotcha:** distance-based scaling is uniform across the pad. It does not capture partial/tilted contact, so edge peel is only approximated through the uneven distribution of contact points.

**Also possible:** a `<general>` actuator with body transmission and `dyntype="filter"`, which handles the switching ramp internally. **[UNVERIFIED]** Syntax not checked.

### Option B: Adhesive contacts, `geom/adhesion` (new in MuJoCo 3.11.0, July 2026)

Per the 3.11 changelog:
- It is an adhesive force attached to a contact.
- Contacts can pull with up to the given force before breaking.
- The friction budget becomes `μ(f_N + adhesion)`.
- Combined with `gap`, it gives "adhesion at a distance", which the changelog explicitly suggests for magnets.
- Resting penetration is unaffected.
- `mj_contactForce` reports the net interface force, whose normal component can now be negative.

**Why it's attractive:**
- A built-in pull-off limit means peel and detachment emerge from the physics.
- Friction increases with adhesion, as it does with real magnetic preload.
- No actuator is needed.

**Sketch:**
```xml
<geom class="pad" ... adhesion="40" gap="0.001"/>
```

**[UNVERIFIED] items to check before use:**
- Units and whether the value is **per contact** (total ≈ n_contacts × value) or per geom pair.
- Whether it can be **switched per step** for EPM on/off. It is likely via the model field (probably `model.geom_adhesion`) or via `<pair>` elements; confirm the name.
- Whether it has **distance dependence** inside the gap, or is constant like the actuator.
- **MJX support**: it is not listed in the MJX parity table as of 3.11.

### Option C: Dynamic weld ("snap to wall")

Idea: when a foot touches down and the magnet is commanded ON, activate a weld equality that pins the foot pad to the wall at its current pose. Deactivate it on magnet OFF, or when the transmitted load exceeds a pull-off/peel/slip limit.

**Implementation:**
- Use a **mocap body per foot** as the anchor. Mocap pose lives in `data` (`mocap_pos`/`mocap_quat`), so it is easy to batch in MJX.
- Use **site-based welds** (MuJoCo ≥ 3.2.3). The two sites "snap together", so set the anchor to the foot pose *before* activating.

```xml
<worldbody>
  <body name="anchor_FL" mocap="true">
    <site name="anchor_FL_site"/>
  </body>
  <!-- robot ... foot has site "foot_site" at the pad face -->
</worldbody>
<equality>
  <weld name="stick_FL" site1="foot_site" site2="anchor_FL_site"
        active="false" solref="0.005 1"/>
</equality>
```

```python
eq_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "stick_FL")
site   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "foot_site")
mocap  = model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "anchor_FL")]

def attach(model, data):
    data.mocap_pos[mocap] = data.site_xpos[site]
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, data.site_xmat[site])
    data.mocap_quat[mocap] = q
    data.eq_active[eq_id] = 1

def detach(data):
    data.eq_active[eq_id] = 0

# Breaking rule (simplified; tune with hardware data).
# Use the foot force/torque sensors (site frame, ≈ load held by the weld, minus foot weight).
def should_break(F, M, F_max, mu, a):
    T = -F[2]                                   # tension away from wall (check sign convention)
    S = np.hypot(F[0], F[1])                    # shear
    peel = np.hypot(M[0], M[1]) > F_max * a/2   # crude edge-peel limit, a = pad side length
    return T > F_max or S > mu * max(F_max - T, 0) or peel
```

**Attach conditions** (following KAIST's gating):
- contact detected,
- magnet commanded ON,
- pad roughly flush, i.e. the ankle angles are consistent with the wall normal and the minimum distance is below a threshold,
- optionally, a Bernoulli success with p_attach (for example 0.85) for robustness training.

**Pros:**
- Rock-solid.
- No slip creep and no bouncing.
- Cheap.
- Works on CPU MuJoCo and MJX. `eq_active` is supported in MJX (since 3.2.5), and weld/connect equalities are supported by both backends.

**Cons:**
- No physics of adhesion itself. Pull-off, peel and slip are only as good as the hand-written breaking rule.
- Effectively infinite friction until broken.
- Risk of an optimistic sim-to-real gap.

**Verify:**
- `data.eq_active` location (it moved from model to data in the 3.x series).
- Weld site-frame conventions.
- Force/torque sensor sign conventions.

### Comparison

| | A: Adhesion actuator | B: geom/adhesion | C: Dynamic weld |
|---|---|---|---|
| Physical realism | Medium (with falloff hack) | Highest (pull-off limit, friction budget) | Low (rule-based) |
| Numerical robustness | Medium; sensitive to gap/solref | Unknown (new) | High |
| Peel/slip | Emergent, approximate | Emergent | Hand-coded rule |
| On/off switching | `ctrl` | [UNVERIFIED] | `eq_active` |
| CPU MuJoCo | ✓ | ✓ (≥ 3.11) | ✓ |
| MJX-Warp (NVIDIA) | ✓ (all transmissions) | [UNVERIFIED] | ✓ |
| MJX-JAX | ✗ (no BODY transmission) | [UNVERIFIED] | ✓ |

**Suggested path:**
1. Prototype A and B on CPU with a single foot on a wall.
2. Compare them against the measured force-gap curve and pull-off tests.
3. Use C as a robust fallback, or for early RL curriculum stages.

---

## 6. Reference: KAIST RL magnetic climbing (Um et al., 2025)

- **Paper:** "Reinforcement Learning-based Robust Wall Climbing Locomotion Controller in Ferromagnetic Environment", arXiv 2510.20174. It uses RaiSim, not MuJoCo, but the model is portable.
- **Robot:** 8 kg, EPM feet of about 0.2 kg each, about 697 N max holding force, about 5 ms switching.
- **Adhesion only if all of these hold:**
  - contact estimated with p ≥ 0.5,
  - magnet action ≥ 0.5,
  - full geometric alignment of pad and wall,
  - a stochastic success draw.
- **Curriculum:**
  1. Crawl on flat ground with adhesion disabled, plus an auxiliary reward for magnet timing.
  2. Rotate the gravity vector 0→90° over about 20k iterations. In MuJoCo this is just editing `model.opt.gravity`.
  3. Reduce p_attach from 1.0 to 0.85 to train recovery.
- **Ablations:** removing the realistic adhesion model or the stochastic failures badly hurt performance and recovery.

---

## 7. MJX notes (MuJoCo 3.11)

- There are two backends:
  - **MJX-Warp:** NVIDIA only, no autodiff, supports all transmissions.
  - **MJX-JAX:** transmissions limited to JOINT, JOINTINPARENT, SITE, TENDON.
- In MJX-Warp, contacts are private (`mjx.Data._impl`). Read them via **contact sensors**. The Python distance loop in Option A must become a `<contact>` sensor, e.g. reduce `mindist` with `dist` data. **[UNVERIFIED]** Exact attribute names.
- Neither backend supports `noslip`, so on GPU rely on the elliptic cone and `impratio`.
- In MJX-JAX, `implicitfast` is not supported together with fluid drag.

---

## 8. Open questions / next steps

1. **User inputs needed:**
   - robot mass,
   - pad size,
   - EPM vs. permanent magnet with mechanical release,
   - measured force-gap curve,
   - wall/floor split,
   - target structures.
2. Build a single-foot test scene: foot on a vertical steel plate, with pull-off, shear and peel tests. Compare Options A, B and C against hardware/datasheet numbers.
3. Confirm the [UNVERIFIED] items in Sections 3, 5 and 7 against the docs for the installed version.
4. Decide on the leg layout (read Uno et al. 2021; Magnecko is an insect-style reference).
5. Build the full quadruped MJCF with four feet and a curriculum environment.

---

## 9. References

- MuJoCo official adhesion example: https://github.com/google-deepmind/mujoco/blob/main/model/adhesion/active_adhesion.xml
- MuJoCo changelog (3.11: geom/adhesion; 3.9: margin/gap redesign): https://mujoco.readthedocs.io/en/3.11.0/changelog.html
- MJX feature parity (3.11): https://mujoco.readthedocs.io/en/3.11.0/mjx.html
- MJX adhesion issue #2667: https://github.com/google-deepmind/mujoco/issues/2667
- Um et al. 2025, RL magnetic wall climbing: https://arxiv.org/abs/2510.20174
- Magnecko (ETH): https://arxiv.org/abs/2504.13672
- Hong et al. 2022, MARVEL, Science Robotics 7(73): eadd1017
- Uno, Valsecchi, Hutter, Yoshida, "Simulation-Based Climbing Capability Analysis for Quadrupedal Robots", CLAWAR 2021: https://doi.org/10.3929/ethz-b-000501538
