# LOG.md

Dead ends and rationale — the reasoning bootstrap.sh can't show.
Append as you go. What you tried, why it failed, what won. Not a transcript;
not routine successes.

---

## ros2
- Tried `apt-get install ros-jazzy-ros-base` before adding the repo → package
  not found.
- Root cause: the ROS apt source must be added first, now via the
  `ros2-apt-source` .deb (the old manual apt-key method is deprecated).
- Working path → bootstrap.sh.
- Gotcha: distro is locked to the Ubuntu version (Jazzy↔24.04). Hardcoding the
  wrong distro silently installs nothing usable.
- Gotcha: repo tests that import `rclpy` need `source /opt/ros/jazzy/setup.bash`
  in the same shell before `pass` runs.

## foo
- `pip install -e .` alone left the test deps missing → ModuleNotFoundError on
  pytest. The `[test]` extra in pyproject.toml has them; used `.[test]`.
