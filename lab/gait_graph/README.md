# gait_graph

Interactive playground for building a path of stances (graph type A: a node is a
planted stance with a witness posture; two nodes are connected if they share 3 of
their 4 footholds). Background: `docs/projects/staged/gait-graph.md`.

Planted feet are treated as welded, so there is no stability check yet; validity
is kinematic + geometric only (`stance.checks`). The MuJoCo sim is visualization
only (never stepped).

Run (macOS needs the viewer under mjpython; DualSense over USB or Bluetooth):

    uv run --extra mjx mjpython -m lab.gait_graph.run

Controls: see the docstring of `run.py`.

## Leg numbering

Legs are indexed in `radial_mounts` order: shoulders at 45°, 135°, 225°, 315°,
counter-clockwise seen from above, with the body's +x forward (nose plate) and
+y to the left.

| leg | angle | position    |
|-----|-------|-------------|
| 0   | 45°   | front-left  |
| 1   | 135°  | rear-left   |
| 2   | 225°  | rear-right  |
| 3   | 315°  | front-right |

The same indices appear in the check messages (e.g. `ankle[0,2]`), the Square
cycle, the swing leg, and the path marker colours.
