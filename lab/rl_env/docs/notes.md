# rl_env — notes (scratchpad)

4x3-DOF spider (hip-yaw / hip-pitch / knee) with passive Cardan ankles and
magnetic pads. Plain MuJoCo on CPU for now. Goal right now: narrow down foot
design and actuator settings before any RL.

Background: `docs/projects/staged/RL-environment.md`, `docs/projects/magnetic-foot-sim.md`.
Foot: `foot_design.md` (how it works, parameters, MuJoCo model + quirks).

## Commands

```bash
uv run --extra mjx python -m lab.rl_env.test_tilt                            # standup: rest -> stand -> rest
uv run --extra mjx python -m lab.rl_env.test_tilt test=hold tilt_deg=90      # hold stand on a "wall", adhesion on
uv run --extra mjx python -m lab.rl_env.test_tilt test=hold tilt_deg=180     # ceiling
uv run --extra mjx python -m lab.rl_env.test_tilt test=sweep tilt_deg=180    # tilt gravity 0 -> 180, then hold
uv run --extra mjx python -m lab.rl_env.test_tilt model.ankle_range_deg=45 model.kp=30   # nested fields: dotted keys
uv run --extra mjx python -m lab.rl_env.test_foot                            # foot pull test, a row of fixtures
uv run --extra mjx --extra sb3 python -m lab.rl_env.test_policy steps=10e6 --tag=gait    # PPO walk forward (runkit run)
uv run --extra mjx --extra sb3 python -m lab.rl_env.test_policy --branch <id prefix> steps=15e6   # continue a run
uv run --extra mjx --extra sb3 python -m lab.rl_env.test_policy eval [<id prefix>]       # latest checkpoint -> its eval/
uv run --extra mjx --extra sb3 python -m lab.rl_env.test_policy viz [<id prefix>]        # summary of a run
uv run --extra mjx ctk play runs/rl_env/test_tilt_hold-90.npz               # replay a test (relaunches under mjpython)
uv run --extra mjx ctk play runs/test_policy/latest/checkpoints/current/eval/rollout.npz   # replay the latest policy eval
```

Outputs: `runs/rl_env/test_<name>[_<params>].npz` (e.g. `test_tilt_hold-90.npz`,
`test_foot.npz`) + the `.xml` model that run used; `test_policy` uses runkit run dirs.
`adhesion=<0..1>` sets the adhesion ctrl (default 0 for standup, 1 for hold/sweep).

Replay (`ctk play`) draws contact forces, a yellow gravity arrow (fixed above the
robot's start, pointing "down"), red external-force arrows (npz `force_pos` /
`force_vec`) and a live line plot (npz `plot`); `--no-contacts` / `--no-gravity` to
turn off.
Gravity is tilted, not the floor, so the floor always looks horizontal: read
"wall" / "ceiling" off the arrow.

Also written on every build: `lab/rl_env/models/scene_flat.xml` (inspection copy).

## Where things live

- `docs/` — these notes, `foot_design.md`.
- `config.py` — the shared configs: `ModelCfg` (robot model, keyframes), `ScriptedCfg`
  (stand-up / hold timing), `WalkEnvCfg` (task, reward, curriculum). A config lives with
  its only reader; shared ones here. Each experiment's own config sits next to its run
  function and nests these: `TiltCfg` (test_tilt), `FootTestCfg` (test_foot),
  `PolicyCfg` (test_policy). Nested fields via dotted keys: `model.kp=12`, `env.w_air=15`.
  Nothing hardcoded elsewhere.
- `model.py` — `to_mjcf` -> `MjSpec`, then edits: masses, ankle + pad + adhesion per
  foot, scene (floor, gravity tilt as a `build()` argument). Its docstring maps each
  `ModelCfg` field to where it lands in the model.
- `poses.py` — `rest` / `stand` keyframes via `Leg3DOF` IK; same footholds, only body
  height differs; ankles solved so pads lie flat.
- `scripted.py` — `standup` (rest -> stand -> rest via IK along body height) and `hold`
  (stand under tilted gravity, optional sweep); one `simulate` loop logs torques, pad
  forces (world frame), ankle angles, gravity.
- `test_tilt.py` — CLI + summary for the robot tests (standup, hold, sweep).
- `env.py` — `WalkEnv`, the Gymnasium env for the policy test: observation, action,
  reward composition (weights, curriculum, dt), contact history, termination.
- `rewards.py` — the reward terms of `policy-and-reward-terms.md` as pure functions of
  arrays (raw values, no weights / dt; per-foot terms vectorized over (4,)); numpy
  only, so a JAX port is mostly `np` -> `jnp`.
- `test_policy.py` — PPO (stable-baselines3, CPU) on `WalkEnv`, walking forward on the
  floor at `cmd_vx`. A runkit `Experiment` with run / eval / viz roles (`PolicyCfg` =
  `model` + `env` + steps, n_envs, `seed`; `config.yaml` is nested). Uses runkit's
  run-time API: `seed = random_seed()` (fresh per run, recorded; `seed=<n>` repeats a
  run exactly -- checked), `ctx.checkpoint("current")` every 20 PPO iterations + at the end,
  replacing the previous one to save disk (`checkpoints/current/` with `state/`: model.zip,
  vecnormalize.pkl, config.yaml (the scheduled config at that step); checkpoint.yaml
  info: it, steps, ep_return, vx; ~1.2 MB;
  `ctx.checkpoint()` without a name gives numbered ones, ~40 per 10M-step run), `ctx.progress`
  (steps in status.yaml, total rounded up to whole PPO iterations), `ctx.record` (one
  row per iteration in `metrics/run.jsonl`). The body's own files under `out/`:
  `progress.png`. eval takes a checkpoint (`evaluate(ckpt)`; default the latest run's
  latest), with the run's seed and full-strength config, and writes into its `eval/`:
  `eval.yaml`, `eval.jsonl`, `rollout.npz`, `gait.txt` -- emptied when the checkpoint is
  saved again. `--branch` continues a run from a checkpoint (weights, optimizer,
  normalization, step count; the schedule goes on). Older runs: model.zip at the
  checkpoint's top, the eval in `out/eval/`. `bench` stays outside runkit.
- `test_foot.py` — foot pull test: one scene, a row of fixtures (locked / free / rigid
  carriage + tibia + real foot), force ramp until release. Own runner; its docstring
  describes every section.

Parameter pointers (`ModelCfg` unless noted; see `model.py` docstring for the full list):
- ankle stop: `ankle_range_deg` (now 45; was 25, rest pose needed 32)
- ankle spring/damper: `ankle_stiffness`, `ankle_damping`
- pad: `pad_size`, `pad_cells` (N x N), `pad_thickness`, `pivot_height`, `pad_friction`
- adhesion: `adhesion_gain` (N per foot at ctrl 1, split over the cells), `adhesion_margin`
- servos: `kp`, `kv`, `forcerange`
- gravity tilt: `TiltCfg.tilt_deg` (0 flat, 90 wall, 180 ceiling)

Naming in the model: `leg{i}_j{k}`, `ankle{i}_a` / `ankle{i}_b`, `pad{i}` with cells
`pad{i}_c{k}`, `adhere{i}_{k}`. Actuators: 12 servos (leg-major), then the adhesion
actuators foot-major (`pad_cells**2` per foot).

## Issues

### Foot peeling problem (open, mitigated)

MuJoCo's `adhesion` actuator applies its full `gain x ctrl` to a body's contacts no
matter how many there are. With the pad as one body, a pad on an edge or a single
corner held the full 30 N and never peeled (`test_foot`, rigid section: flat / edge /
corner all 30.2 N). A policy would exploit that: sloppy, edge-first landings cost
nothing in sim and fail on hardware.

Mitigation (2026-09-23): the pad is `pad_cells` x `pad_cells` cell bodies, each with
its own adhesion actuator (gain A / N^2), so the force scales with the touching area.
Rigid section, pull straight off / shear (A = 30 N):

| N | flat | edge | corner | edge shear | corner shear | robot `mj_step` (wall hold) | contacts |
|---|------|------|--------|------------|--------------|-----------------------------|----------|
| 1 | 30.6 | 30.6 | 30.3 | 15.6 | 15.4 | 17 us | 16 |
| 2 | 30.6 | 15.6 | 8.1 | 8.2 | 4.5 | 40 us | 64 |
| 3 | 30.6 | 10.6 | 4.0 | 5.8 | 2.4 | 90 us | 144 |
| 4 | 30.6 | 8.1 | 2.6 | 4.5 | 1.7 | 165 us | 256 |

Default now N = 3 (corner ~13%, 2.3x the step cost of N = 2). Cost is ~linear in
contacts (4 per box cell on a plane).

Still open:
- No gap falloff: a cell within the margin (2 mm) pulls full force, beyond it zero.
  A real magnet at 1 mm gap has ~7%. Tilts of a few degrees are still too sticky.
- No torque towards flat beyond the margin: a pad starting on an edge isn't pulled flat.
- Cheaper cells: sphere cells would give 1 contact each (N^2 per foot instead of 4N^2).
- Proper fix if needed: per-cell force from the plane distance with a falloff
  (flat walls -> a dot product; but stiff -> may need dt <= 1 ms or a force lag).

## Test ladder (wall climbing)

Rotate gravity instead of the ground; real wall geometry only for transitions.

1. Foot alone on a fixture: pull-off, shear/slip, peel. Validates adhesion model + ankle.
2. Static hold, gravity sweep 0 -> 90 -> 180 deg: per-foot adhesion load, shear/normal
   vs mu, joint torques. Sizes actuators + magnets.
3. Three-foot hold on the wall (lift one leg): worst-case peel, adhesion margin.
4. Scripted crawl gait up the wall.
5. Transitions (floor -> wall, corners, openings) with real geometry.

## Findings / log

- 2026-09-23 — first model runs. Mass 2.8 kg. Flat stand: hip-pitch ~0.7-0.8 N m,
  knee ~0.3 N m (limit 5). 6.9 N per pad.
- Rest pose with tibia (0.20) > femur (0.15): tibia can't be near-vertical with belly
  on the ground -> ankle needs ~32 deg (peaks 34 during the motion). Raised stop to 45.
- Adhesion bug: the adhesion actuators inherited the servo `forcerange` (+/-5) from the
  `<position>` default -> capped at 5 N, robot fell off the wall. Fixed
  (`forcelimited=false`). With 30 N/foot: holds at 90 and 180 deg for 2 s,
  torques < 1 N m.

- Hold tests (30 N adhesion/foot, stand pose, 3 s): all hold, no torque > 1 N m.
  - wall (90): pad normal 35 N lower pair (legs 0, 3) / 25 N upper pair (peel moment), shear
    ~7 N/foot (= weight/4), shear/normal max 0.34 < mu 0.5. Drift 12 mm.
  - ceiling (180): normal 23 N/foot (30 adhesion - weight/4), shear/normal 0.07.
  - sweep 0 -> 180: shear/normal peaks 0.29 on the way.
- Standup (no adhesion): shear/normal hits 0.50 = mu -> feet slip at some point
  (legs push outward near rest). Check where in the replay.

- Adhesion probes (single pad, scratch scripts, MuJoCo 3.9.0):
  - A corner alone is enough: the `adhesion` actuator applies the full gain x ctrl,
    spread over however many contacts the pad body has (16 flat, 4 edge, 1 corner).
    No dependence on contact area or air gap -> unrealistically strong on edges.
  - Pads hover ~1.8 mm above the surface. The contact constraint activates at
    dist < `includemargin` and measures penetration relative to it, so `margin`
    effectively inflates the surface; rest = includemargin minus a small soft sink.
    Happens without adhesion too.
  - `gap` has no effect in 3.9.0 (includemargin stays = margin, not margin - gap), and
    margins of both geoms add (floor 2 mm + pad 2 mm -> 4 mm). Cause unconfirmed.
    Accepted for now: small hover is fine. (The margin==gap comment in `model.py` is
    what we intended, not what happens.)
  - Beyond the margin there is no adhesion at all: an on/off band, no falloff.

- Foot pull test — see the `test_foot.py` docstring for the full description.
  Summary of the setup:
  - One fixture = tibia rigidly attached at its top to a *carriage*, real foot at
    the bottom (ankle + pad + adhesion). Floor = wall, no gravity, pull = only load.
  - Carriage types: **locked** (blue): translates freely in x/y/z, cannot rotate ->
    tibia keeps the lean it was attached at (a leg holding its angle perfectly).
    **free** (orange): free body -> tibia also rotates, swings into line with the
    pull until the ankle stop. **rigid** (purple): locked carriage + ankle held by
    joint equalities -> pad kept flat / on an edge / on a corner.
  - Protocol: settle 0.5 s (adhesion on) -> force at the tibia top ramps 10 N/s ->
    pad moved 5 mm = release (force recorded, pull off).
  - Layout: fixtures in a row along x; pulls/leans in the y-z plane (0 = straight off
    the wall, 90 = shear along +y); gaps between sections, bigger gap between types.
- Foot pull test results (30 N, mu 0.5). Cases now stay within the ankle range
  (user's call); an earlier version with free pulls at 60/90 and a lean-60 edge
  start showed peel via the stop (~1.6-1.8 N) and a pad rolling onto its side face
  (adhesion only at contact points gives no torque towards flat).
  - locked / upright: pull-off 30.2; pull 30/60/shear 90 slip at 16.8 / 14.0 / 15.7 N
    = Coulomb mu*A / (sin + mu cos) (+~0.7 N detection lag).
  - locked / lean 30: 29.1 / 15.7 / 15.7 -> a leaning leg changes almost nothing (the
    ankle takes the lean).
  - free / upright: pull 0 = 30.2, pull 20 = 19.0 (theory 18.5; leg aligned to 20).
    Pull 40 = **2.1 N**: the light leg + limp ankle (0.05 N m/rad, damping 0.005)
    swing into line *dynamically* (0 -> 41 deg in ~0.15 s), overshoot towards the
    stop, tip the pad onto an edge, and whip it off. So even a pull inside the range
    effectively reaches the stop. Ankle damping / stiffness matter.
  - free / lean 20, 40: the leg straightens during the settle (to 7 / 14 deg), then
    pull-off at 30.2.
  - rigid / flat, edge, corner (pad held at a fixed tilt by stiff joint equalities):
    pull-off 30.2 / 30.2 / 30.2 N with 16 / 4 / 3 contacts; shear 15.7 / 15.7 / 15.4.
    **Flaw confirmed**: MuJoCo adhesion ignores contact area; a pad on an edge or a
    corner holds the full force and never peels. A policy would learn sloppy,
    edge-first landings that cost nothing in sim and fail on hardware -> fix the
    adhesion model (per-cell / gap falloff) before RL. First attempt with default
    equality stiffness gave corner 20.5 N: the pad yielded ~3-4 deg and peeled --
    equality solref now 2 * timestep.
  - Bugs found on the way: MjSpec defaults to degrees (`spec.compiler.degree=False`
    needed, else the ankle range was +/-0.8 deg); pads started at z = 0, i.e. 2 mm
    inside the margin -> now start at 0.9 * margin.

- Policy test (2026-09-23), PPO on the floor, 0.15 m/s forward, magnets off, pad_cells=1:
  - Throughput (SubprocVecEnv, random actions): 1 env 720, 8 envs 3.7k, 12 envs 5.1k,
    16 envs 5.7k env steps/s -> use 12. ~5k steps/s while training.
  - Deviations from the reward doc: `lin_sharpness` 20 instead of 4 (with 4, standing
    still earns 92% of the tracking reward at 0.15 m/s); `w_height` and `w_clear` 50
    (doc's 1.0 / 0.1 are negligible in metres^2); `w_term` 20 instead of 1 (else
    ending the episode is cheaper than early exploration costs); init policy std
    exp(-1) = 0.37.
  - First run `runs/rl_env/test_policy/20260923-235716` (stopped at 5.4M): vx 0.147
    of 0.15, no terminations, return ~35.
  - Contact history + gait terms (2026-09-24): the env keeps the last `contact_history`
    (100 = 2 s) policy steps of per-foot pad normal force, and air/planted timers
    (planted = force > 1 N). New terms: air time per touchdown (`w_air` 10, target
    0.25 s, capped 0.5 s) and minimum support (>= 3 feet planted, `w_support` 5), both
    curriculum-scaled. Progress lines show `duty` (planted fraction) and `air_s` (mean
    air time per step). `eval` saves the forces (`force` in the npz; the live plot in `ctk play` was dropped 2026-09-27).
  - Pitfalls found: air term without the dt factor was 20x the whole tracking reward
    (exploration makes feet chatter: ~0.04 s air per touchdown); with gait terms the
    early-episode costs (~-52 unscaled) exceeded `w_term` 20 -> gait terms now
    curriculum-scaled (~-18/episode) and `w_term` 50.
  - Kernel normalized (2026-09-25): K(x) = 4 / (e^x + 2 + e^-x), peak 1 instead of the
    paper's 0.25, so a tracking weight is the max reward per second. Weights rescaled
    to keep the rewards identical: `w_lin` 10 -> 2.5, `w_yaw` 6 -> 1.5 (verified on a
    seeded rollout). Reward doc updated.
  - Foot drag + yaw (2026-09-27): new cost `C_drag` (doc: "Foot drag"; `rewards.foot_drag`,
    `w_drag` 2, curriculum-scaled): feet touching without load that move along the floor.
    Three foot states now: in the air (< `contact_touch_min` 0.05 N) -> clearance;
    touching without load (0.05..1 N) -> drag; planted (> `contact_force_min` 1 N) ->
    slip, ankle. Clearance used to cover everything below 1 N. `w_yaw` 1.5 -> 3.0 (doc
    updated too).
  - Planted = loaded + flat (2026-09-27): a foot is planted when its normal force is
    > 1 N *and* its pad's face normal is within `pad_tilt_max_deg` = 3 deg of the floor
    normal (pad tilt, not ankle angle). Used everywhere (slip, drag, air time, support,
    duty). Measured on run d3b9f2f6 (checkpoint at 6.4M steps): loaded pads sit at
    0.29 deg median, 0.54 p90, 1.7% of loaded foot-steps > 3 deg; light touch
    (0.05-1 N) never occurs. With the new definition ~5% of the sliding moves from slip
    to drag, touchdowns unchanged (no flicker), duty 0.873 -> 0.858.
    Finding: the "dragging" is flat, loaded feet sliding (skating): 95% of the sliding
    of loaded feet, median 0.11 m/s. That is the slip term (`w_slip` 2, paper value;
    doc suggests 10), not drag -> raising `w_drag` would not change it. -> `w_slip`
    raised to 5.
  - Support: graded + ramped (2026-09-27). `w_support` 10 / 20 made the policy stop
    stepping (duty 0.92, vx ~0.01 at 0.7M steps): walking policies at `w_support` 5
    had < 3 planted feet ~half the time (duty ~0.6, trot-like), so a strong 0/1 penalty
    cost more than walking earns (+0.022/step at best). Now `C_support = max(0, 3 -
    n_planted)` (one foot short costs 1, two cost 2) and its own factor `k_support`:
    0 for `support_start` = 1M steps, then linear to 1 over `support_ramp` = 2M. Not
    scaled by `k_c` anymore. Progress lines show `k_sup`.
  - Curriculum never reached the envs (fixed 2026-09-27): the callback pushed `k_c` /
    `k_support` with `venv.set_attr`, which sets the attribute on the outer `Monitor`
    wrapper only. So all runs before the fix trained with a constant `k_c = kc0` (0.4;
    the rising k_c in the progress lines was the callback's own counter) and, since the
    support ramp, `k_support = 0` (support cost off). Now
    `env_method("set_wrapper_attr", ...)`, verified on the training stack (torque term
    x2.5 for k_c 0.4 -> 1.0). Old runs' results should be read with this in mind.
    Proposal for a general mechanism: `scheduled-config-proposal.md` (`ScheduledConfig`, `_schedule`).
  - ScheduledConfig implemented (2026-09-28, `scheduled_config.py`): `PolicyCfg._schedule`
    mirrors the config (`env.w_*: {op, kind, ...}`); `ScheduledConfig(cfg)(step)` returns
    the adjusted config. The progress callback pushes `sc(num_timesteps).env` into every
    env once per PPO iteration (`set_wrapper_attr("cfg", ...)`); envs start at step 0.
    `k_c`, `k_support`, `REG` and the WalkEnvCfg fields kc0 / kc_rate / support_start /
    support_ramp are gone; the default schedule reproduces them exactly (geometric
    x0 0.4, rate 0.997 per iteration on the ten shaping weights; linear 1M..3M on
    w_support; rewards identical to the old scaling at equal factors). `WalkEnv.cfg` is
    a property: setting it re-derives cmd / angle thresholds. Metrics: `schedule/<path>`.
    Note: runkit replaces a dict field on override instead of merging -> merged in
    `PolicyCfg.__post_init__`. Old run dirs have kc0 etc. in env: and no longer load.
  - Why `gymnasium.Env` and not a Brax env (2026-09-25): SB3 needs the Gymnasium API,
    and a stateful numpy/MuJoCo-C env is quick to write and debug on the Mac (contact
    loop, `mj_contactForce`, `mj_objectVelocity`, contact history). A Brax env is pure
    JAX on MJX (`reset(rng)`, `step(state, action)`, jit/vmap), fast on a GPU (100k+
    steps/s vs our ~5k) but slow on the Mac's CPU, and needs everything vectorized
    (no python loops, fixed-size arrays). Plan: prototype model + reward here; port to
    MJX (Brax `Env` or MuJoCo Playground `MjxEnv`) when we need GPU scale (wall,
    transitions, domain randomization). First check whether MJX 3.9 supports the
    `adhesion` actuator and our per-cell contacts. Easier port: keep reward terms as
    pure functions of arrays.

- Config restructure (2026-09-25): flat `Cfg` split into `ModelCfg` / `ScriptedCfg` /
  `WalkEnvCfg` (config.py) + one config per experiment next to its run function
  (`TiltCfg`, `FootTestCfg`, `PolicyCfg`), nesting the shared ones. `tilt_deg` moved out
  of the model (only test_tilt tilts): `build(cfg, tilt_deg=...)`. Verified: seeded
  reward rollout identical, test_tilt / test_foot results unchanged. Old run dirs have a
  flat `config.yaml` and no longer load into `PolicyCfg`. (A runkit bug that reset
  nested `default_factory` defaults on override is fixed in run-kit c463d87, so
  `PolicyCfg.model` uses a plain `default_factory` with `pad_cells=1`.)

## Open / ideas

- Dimensions are placeholders; adjust after viewing.
- More realistic adhesion, options:
  1. split each pad into 4 cells with one adhesion actuator each (face 4/4, edge 2/4,
     corner 1/4 of the force);
  2. compute per-cell adhesion from the actual gap each step, with a falloff
     (magnetic-foot-sim.md: 1 mm gap -> ~7% of max);
  3. `margin = 0`: no hover, adhesion only on real contact.
- Tilted-ramp curriculum for floor -> wall (from RL-environment.md).

## Unsorted

- For climbing, the policy should learn to be sure if a foot has been planted 
to a wall and is secured, that foot has contact and magnet holds it in place.