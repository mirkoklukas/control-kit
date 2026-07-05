"""gait_graph — planning locomotion as a graph over stances.

A **stance** is a set of planted feet; a **posture** (body pose + joint angles)
connects two stances as a graph *edge*. We expand stances from a start node,
keep the reachable + statically-stable ones, and search the graph for a walk
that carries the body toward a goal -- that walk is a gait.

Pure jax (jaxlie ``SE3``); run under ``uv run --extra mjx``. Built on the
single-leg kinematics in ``lab.ik.core``. Starting model: ``models/weld0.xml``.
"""
