# Controller — remote planner (design sketch)

Run the sim + MPPI planner on the GPU box, keep the DualSense and the viewer
local. The point is to **scale the planner** (MJX on GPU: thousands of samples,
longer horizon, richer cost) past what the local CPU can do, while the control
stays interactive.

## Why this split

The pad is on a local USB port and you watch on a local display, so **input and
rendering are stuck local**. The sim + planner is the compute, so it goes
**remote**. The one rule that makes it usable:

- **Do not round-trip per tick inside the control loop.** Offloading only the
  MPPI rollout each tick puts a network RTT (20–100 ms over anything but LAN)
  inside the 16 ms control budget → stalls.
- **Keep the whole control loop on the remote.** Stream the controller state up
  and the sim state down. You feel one RTT of *input lag* (cloud-gaming style),
  but the loop runs at the remote's rate, unaffected by RTT.

```
 local  (mac, mjpython)                    remote (GPU box, headless)
 ───────────────────────                   ──────────────────────────
 DualSense.poll() ─ ControllerState ─UDP─▶ recv
                                           target.update(state)
 viewer.sync() ◀─ qpos, mocap ────────UDP─ planner(state, target)  [MJX, jit]
 (set qpos, mj_forward)                    mjx.step(real env)
```

- **local** = input + render only. Holds its own compiled copy of the model,
  sets `data.qpos` from the received state, `mj_forward`, `viewer.sync()`. Needs
  `mjpython` (viewer); does no physics.
- **remote** = the full sim+planner loop, headless. No viewer, so it runs under
  plain Python (sidesteps the whole `mjpython` main-thread problem entirely).

## Transport / protocol

- **UDP**, drop-tolerant: a late/lost packet is just skipped, no head-of-line
  stall. Two small fixed-layout messages (`struct` / `ndarray.tobytes`, no JSON).
- **up** (local→remote, ~60 Hz): sticks (4f), triggers (2f), button bitmask
  (L1/Options/…), seq. ~40 B.
- **down** (remote→local, planner rate): `qpos` (nq f), mocap pose (7f), optional
  diagnostics (cost, plan time), seq. ~200 B.
- Both directions carry a **seq**; the receiver drops stale packets and holds the
  last good one on a gap. Bandwidth is trivial (floats, not video).
- Exposure: bind to LAN, a VPN, or an **SSH tunnel** to the box — never open UDP
  to the internet.

## Decouple the two rates

The scaled planner may run slower than 60 Hz (bigger N/T). So:

- **remote** plans as fast as it can (its own rate = planner throughput).
- **local** renders at a steady 60 Hz from the latest received state (hold or
  interpolate between packets), so the view stays smooth even if planning is
  slower. Control rate and render rate are independent.

## Scaling the planner (the actual goal)

Port the rollout from numpy + `mujoco.rollout` (threaded CPU) to **MJX** (jitted,
GPU):

- `env_step = mjx.step`; the connect pins baked active (`eq_active0`); the
  position servos, `forcerange`, `armature`, `frictionloss` all carry over.
- **vmap** the rollout over the `N` candidates, **scan** over the horizon
  (`T·decim`), **jit** the whole planner. `N` into the thousands and a longer `T`
  become cheap on GPU after the one-time compile.
- Reuse `controlkit.mpc` (`make_rollout_sampler` / `make_mppi_planner`) — it is
  already abstract JAX; `lab/hexapod_mpc/env.py` is the working MJX-MPPI template.
- The cost and exploration knobs transfer unchanged (`w_pos`/`w_ori`/`w_vel`/
  `w_reg`; `noise_sigma` still under the **`kp·noise_sigma ≪ forcerange`** rule).

Caveats specific to this model:

- **float32.** MJX defaults to f32; the stiff constraints (`solref = 2·dt`, high
  `solimp`) and stiff servos may diverge in f32 (cf. the climb-bench f32
  divergence). Enable `jax_enable_x64` if the pins drift or blow up (slower), or
  soften `solref` for f32.
- **connect + `eq_active` in MJX.** Bake the pins active (`eq_active0`). Runtime
  toggling (future gaits: plant/lift feet) needs MJX `eq_active` support checked.
- **JIT warmup** at startup (seconds); keep `N`, `T`, shapes fixed so it doesn't
  recompile mid-run.
- **device↔host per tick** is only `qpos` + controller (tiny) — negligible; keep
  the real env state on device between steps.

## Components

```
lab/controller/
  remote/
    protocol.py   # message layouts (pack/unpack), ports
    serve.py      # remote: headless MJX sim + planner loop, UDP in/out
    client.py     # local: DualSense + viewer, UDP out/in (mjpython)
  policy_mjx.py   # the scaled MJX MPPI planner (or controlkit.mpc wiring)
```

Reused as-is: `robot.scene_xml` (same model → `mjx.put_model`), `config`,
`target`, `input`. `serve.py` runs `uv run --extra mjx python -m ...` on the box;
`client.py` runs `uv run --extra mjx mjpython -m ...` locally.

## Milestones

- **R0** — protocol + loopback: `serve` + `client` on localhost with the
  *existing* numpy planner. Proves the split and the render-from-state path.
- **R1** — move `serve` to the GPU box (LAN, or `lambda` via SSH tunnel), same
  numpy/mujoco planner. Proves the network path and the latency feel.
- **R2** — swap the planner to MJX; scale `N`/`T` on the GPU. The real payoff.
- **R3** — tune the bigger planner (samples, horizon, cost) and settle precision
  (f32 vs x64) against the stiff constraints.

## Open questions

- **Transport lib**: raw UDP (lowest latency) vs zmq (nicer ergonomics, PUB/SUB)
  vs websocket. Lean raw UDP or zmq.
- **Real env on remote**: run the "true" sim in MJX too (state on device, cleanest)
  vs a CPU `mujoco` step alongside the MJX planner. MJX-for-both preferred.
- **Target integration**: keep it on the remote as the single source of truth
  (simplest), vs mirror it locally for render-ahead prediction to mask lag.
- **Precision**: f32 default vs `jax_enable_x64` given the stiff pins/servos.
- **Reconnect / packet loss**: hold-last on a gap; reset protocol on resync.
- **Relation to the existing GPU path**: today `lambda` is used for *batch* MJX
  runs pulled back via `rsync` (see `lab/README.md`); this is a new *interactive*
  path, not batch.
```
