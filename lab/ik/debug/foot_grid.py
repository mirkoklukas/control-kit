"""Step 1 of foot-target generation: sample points on the world surface near the
robot and drop visual markers.

World: floor (z=0) + a 4x4x2 box (models/climb0.xml), near yz-face at x=1. We take
the part of the surface within MESH_DISTANCE of the body and uniformly (by area)
sample N points on it -- imaginary foot targets, no physics. Then render the robot
+ box + markers.

Plain MuJoCo: ``uv run python -m lab.ik.debug.foot_grid``.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg

ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / "models" / "climb0.xml"
SCRATCH = Path(__file__).resolve().parents[1] / "scratch"
SCRATCH.mkdir(exist_ok=True)

MESH_DISTANCE = 0.6
N = 500

# box: center (3,0,1), half (2,2,1) -> x in [1,5], y in [-2,2], z in [0,2].
BOX_NEAR = 1.0   # near yz-face; the floor is only exposed in front of it (x < BOX_NEAR)
# each surface rectangle: (corner, edge1, edge2); a point is corner + s*e1 + t*e2.
def _faces(base):
    floor_x0 = base[0] - 1.5
    return [
        # floor in FRONT of the box only (x from floor_x0 up to the box near face)
        (np.array([floor_x0, base[1]-1.5, 0.0]), np.array([BOX_NEAR - floor_x0, 0, 0]), np.array([0,3.,0])),
        (np.array([1.,-2,0]), np.array([0,4.,0]), np.array([0,0,2.])),                         # box near face x=1
        (np.array([1.,-2,2]), np.array([4.,0,0]), np.array([0,4.,0])),                         # box top z=2
        (np.array([1., 2,0]), np.array([4.,0,0]), np.array([0,0,2.])),                          # box side y=+2
        (np.array([1.,-2,0]), np.array([4.,0,0]), np.array([0,0,2.])),                          # box side y=-2
        (np.array([5.,-2,0]), np.array([0,4.,0]), np.array([0,0,2.])),                          # box back x=5
    ]


def sample_surface(base, n=N, mesh_distance=MESH_DISTANCE, seed=0):
    """Uniformly (by area) sample ``n`` points on the world surface within
    ``mesh_distance`` of ``base``. Returns (n, 3)."""
    rng = np.random.default_rng(seed)
    faces = _faces(base)
    areas = np.array([np.linalg.norm(np.cross(e1, e2)) for _, e1, e2 in faces])
    pts = []
    while len(pts) < n:
        c, e1, e2 = faces[rng.choice(len(faces), p=areas / areas.sum())]
        p = c + rng.random() * e1 + rng.random() * e2
        if np.linalg.norm(p - base) <= mesh_distance:
            pts.append(p)
    return np.array(pts)


def render(base, points, out=SCRATCH / "foot_grid.png"):
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    model.vis.global_.offwidth, model.vis.global_.offheight = 1000, 700
    data = mujoco.MjData(model)
    data.qpos[:] = model.key_qpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")]
    data.qpos[:3] = base                                   # stand the robot near the box
    mujoco.mj_forward(model, data)

    r = mujoco.Renderer(model, 700, 1000)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [base[0] + 0.3, 0.0, 0.45]              # between robot and box face
    cam.distance, cam.azimuth, cam.elevation = 1.8, 90.0, -12.0   # face the box (+x)
    r.update_scene(data, cam)
    scn = r.scene
    eye = np.eye(3).flatten()
    for p in points:                                        # markers as decorative spheres
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([0.008, 0, 0]), np.asarray(p, float), eye,
                            np.array([1, 0.9, 0.1, 1], np.float32))
        scn.ngeom += 1
    mpimg.imsave(out, r.render()); r.close()
    print(f"{len(points)} surface points within {MESH_DISTANCE} m -> {out}")


def _demo():
    base = np.array([0.6, 0.0, 0.241185])                   # standing, ~0.4 m in front of the box
    render(base, sample_surface(base))


if __name__ == "__main__":
    _demo()
