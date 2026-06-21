"""Minimal MJX example — batched rollout, viewing one sim in the classic viewer.

MJX (JAX) runs many sims in parallel but has no viewer. We step a batch of N
cartpoles together, then mirror sim 0 back into a classic MjData each frame with
mjx.get_data_into, which the normal MuJoCo viewer renders.

macOS needs mjpython for the viewer (GUI must own the main thread):
    uv run mjpython examples/00_minimal.py
"""

import time

import mujoco
import mujoco.viewer
from mujoco import mjx
import jax
import jax.numpy as jnp

model = mujoco.MjModel.from_xml_path("models/cartpole.xml")
model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST  # RK4 -> implicit (MJX needs this)
mx = mjx.put_model(model)

N = 100
key = jax.random.PRNGKey(0)


def make_one(key):
    dx = mjx.make_data(mx)
    theta0 = jax.random.uniform(key, minval=-0.1, maxval=0.1)  # small random tilt
    return dx.replace(qpos=dx.qpos.at[1].set(theta0))


# batch = jax.vmap(make_one)(jax.random.split(key, N))  # N independent sims


@jax.jit
@jax.vmap                                  # maps over batch + ctrl; mx is closed over
def step_mapped(dx, ctrl):
    return mjx.step(mx, dx.replace(ctrl=ctrl))

@jax.jit
@jax.vmap   
def score(taux):
    zs = taux.sites("tip").zpos
    return jnp.max(zs)



def policy(key, s, sigma=1.0):
    return sigma * jax.random.normal(key, model.nu)

def get_obs(dx):
    return jnp.concatenate([dx.qpos, dx.qvel])     # whatever your policy f expects

def rollout(key, dx0, T, policy):
    keys = jax.random.split(key, T)
    def step(dx, key):
        u = policy(key, None)                                    
        dx = mjx.step(mx, dx.replace(ctrl=u))       
        # carry forward dx, emit (s, a)
        return dx, (dx, u) 
    
    _, (dxs, us) = jax.lax.scan(step, dx0, keys, length=T)
    return dxs, us

# ctrl = jnp.zeros((N, model.nu))
# sigma = 2.5
# data = mujoco.MjData(model)    # classic data object the viewer renders (sim 0)

tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip")
dx0 = mjx.make_data(mx)
T = 10
key,_ = jax.random.split(key)
key,_ = jax.random.split(key)
key,_ = jax.random.split(key)
dxs, us = rollout(key, dx0, T, policy)

print(us)
print(dxs.site_xpos.shape)
print(dxs.site_xpos[:,tip_id,:])

# with mujoco.viewer.launch_passive(model, data) as viewer:
#     while viewer.is_running():

#         key, sub = jax.random.split(key)
#         ctrl = sigma * jax.random.normal(sub, ctrl.shape)  # random-walk control
#         batch = step_mapped(batch, ctrl)                   # all N sims advance together

#         dx0 = jax.tree_util.tree_map(lambda x: x[0], batch)       # pull sim 0 out of the batch
#         mjx.get_data_into(data, model, dx0)                       # -> classic MjData
#         viewer.sync()
#         time.sleep(model.opt.timestep)                            # rough real-time pacing


