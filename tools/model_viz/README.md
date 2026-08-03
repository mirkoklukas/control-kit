# model_viz

WebGL viewer for MuJoCo MJCF models (three.js, no build step).

## Run

```bash
.venv/bin/python tools/model_viz/serve.py                 # opens models/hexapod.xml
.venv/bin/python tools/model_viz/serve.py models/cartpole.xml
```

Serving over http lets the browser fetch the model; opening the `.html` directly
won't (browsers block local-file `fetch`).

## Load a model

- the `model` URL param (what `serve.py` sets), or
- the **load .xml** button, or
- **drag-and-drop** an `.xml` onto the window.

## Use

Orbit/pan/zoom with the mouse. The right panel has one **slider per hinge/slide
joint** to pose the kinematics, plus toggles for sites / wireframe / grid.

## Notes

- Kinematic viewer, not a simulator: no physics; the floating base is pinned.
- Needs internet (three.js from CDN).
- Geoms: box, sphere, capsule, cylinder, ellipsoid, plane, and inline-`vertex`
  meshes. `<mesh file="...">` (OBJ/STL) is skipped.
