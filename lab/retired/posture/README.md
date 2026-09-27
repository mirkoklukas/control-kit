# posture

Static posture / stance-force experiments for the hexapod.

- `kinematics.py` — types (`Foothold`, `Stance`, `Posture`), FK (`_joint_frames`,
  `_forward_model`), IK (`_infer_theta` foot-only, `_infer_theta2` foot+normal).
- `forces.py` — static stance forces/wrenches on the `6x4DOF.xml` model
  (`stance_forces` / `posture_forces`, `stance_wrenches` / `posture_wrenches`, …).
- `rerun_viz.py` — rerun logging (`log_posture`, `log_contacts`, `log_joint`).

## Note: gauging whether a stance can actually hold (force model)

`stance_forces` solves `M f = g[:6]` for the planted-foot forces `f`, where `g[:6]`
is the gravity wrench on the base and **`M` (6 × 3·nf) is the base map** — it sends
foot forces to the net body wrench `[Fx,Fy,Fz,Tx,Ty,Tz]`. It's the min-norm
least-squares solution, so it **always returns something**; the force/moment
magnitudes alone don't tell you if the stance is any good.

Two cheap diagnostics do:

1. **Rank / smallest singular value of `M`.** Key fact: `M = J_baseᵀ`, the transpose
   of the base contact Jacobian. So `rank(M) = rank(J_base)`, and a rank deficiency
   (`σ_min → 0`) has two faces that are *the same 6-vector* `w*`:
   - *statics:* `w*` is a body wrench no foot force can produce → the feet can't
     resist a disturbance along it;
   - *kinematics (dual):* `w*` read as a twist satisfies `J_base w* = 0` → a base
     motion that moves the body while every foot stays put → a free mode the
     (frozen-posture) feet cannot resist.

   `σ_min` is a continuous fragility score: `= 0` → free mode (unstable/degenerate);
   tiny → supportable only with `~1/σ_min ×` larger forces (fragile); healthy → well
   supported. E.g. three feet collinear/coincident in the pitch plane leave pitch
   (`Ty`) unsupported — `σ_min = 0`, and `w*` is rotation of the body about the
   contact line.

   Caveat: the 6 rows mix force (N) and torque (N·m), so raw σ's aren't comparable
   (force rows ≈ √nf, torque rows scale with lever arms in metres). Nondimensionalize
   the torque rows by a characteristic length (≈ body radius 0.15 m) before
   thresholding.

2. **Least-squares residual `‖M f − g‖`.** When `M` is rank-deficient, lstsq quietly
   drops the component it can't satisfy and returns a tidy-looking `f` that does *not*
   balance gravity. A nonzero residual flags exactly that silent failure.

**Force vs. wrench.** These are properties of the **force** model. `stance_wrenches`
uses a bigger map (6 × 6·nf, adding per-foot moment columns); those extra DOFs fill
otherwise-null directions, so `M_w` is typically full rank and `σ_min` looks healthy
**even for a geometrically degenerate stance** (e.g. two coincident feet). So `σ_min`
is a good check for the force model only; for the wrench model use an explicit
support-geometry / rank check on the contact points instead.
