# model_generator

Interactive tool to design a hexapod and emit a MuJoCo MJCF model.

A live three.js preview (left) + control panel (right). **Save** writes a YAML
spec and then generates the XML from it, so the YAML is the source of truth and
`generate.py` is the only thing that writes MJCF.

## Run

```bash
ctk model-gen serve        # opens the tool at :8001  (add --no-open to skip the tab)
```

Or directly: `.venv/bin/python tools/model_generator/serve.py`.

Serving over http (not `file://`) is required so the page can POST to `/save`.
Needs internet (three.js from CDN).

## Controls

- **Geometry**: circumradius (center→vertex), coxa / femur / tibia length, foot size.
- **Rest pose**: coxa (yaw), femur (lift), tibia (knee) angles — the standing
  pose, written as `<keyframe name="home">`. The base height is auto-set so the
  feet rest on the floor.
- **Density**: body, legs (kg/m³; MuJoCo derives mass/inertia from geom shape).
- **Colors**: body, coxa, femur, tibia, foot.

Lengths are pure segment lengths (leg-local +x at the zero pose), independent of
the rest pose.

## Save / load

The name field (default `hexapod`) is the basename for both. **Save** writes,
under `output/` (git-ignored):

- `output/<name>.yaml` — the spec
- `output/<name>.xml`  — the MJCF, via `generate.py`

**Load** reads `output/<name>.yaml` back into all the controls and the preview.

## Generate from YAML directly

The web tool is optional; the YAML is hand-editable and the generator runs from
the CLI:

```bash
ctk model-gen build output/myhex.yaml                 # -> output/myhex.xml
ctk model-gen build spec.yaml -o models/myhex.xml
```

Or call the script directly: `.venv/bin/python tools/model_generator/generate.py spec.yaml`.

## Conventions (match models/hexapod.xml)

- 6 legs at corners 30/90/150/210/270/330 deg, indexed 0..5 =
  FL, ML, BL, BR, MR, FR.
- Per leg: `coxa{i}` (yaw, z) → `femur{i}` (lift, y) → `tibia{i}` (knee, y),
  18 actuated DOF, with `foot{i}` geom/site at the tip.
- Legs are written out explicitly (not via `<replicate>`) because `<replicate>`
  doesn't expand `<keyframe>` qpos, which the rest pose needs.

## Notes

- Preview is kinematic only (no physics); it mirrors the generator's geometry
  and FK, so it matches the saved model.
- Capsule (leg) radius and body half-thickness are fixed; edit `generate.py` to
  change them.
