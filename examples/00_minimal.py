import time

import mujoco
from mujoco import mjx
import jax
import jax.numpy as jnp


def policy(key, s, sigma=1.0):
    return sigma * jax.random.normal(key, model.nu)

def get_obs(dx):
    return jnp.concatenate([dx.qpos, dx.qvel]) 

def rollout(key, dx0, T, policy):
    keys = jax.random.split(key, T)
    def step(dx, key):
        u = policy(key, None)                                    
        dx = mjx.step(mx, dx.replace(ctrl=u))       
        return dx, (dx, u) 
    
    _, (dxs, us) = jax.lax.scan(step, dx0, keys, length=T)
    return dxs, us


T = 10
key = jax.random.PRNGKey(0)

model = mujoco.MjModel.from_xml_path("models/cartpole.xml")
model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST  # RK4 -> implicit (MJX needs this)
tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip")
mx = mjx.put_model(model)
dx = mjx.make_data(mx)

key,_ = jax.random.split(key)
dxs, us = rollout(key, dx, T, policy)


print(us)
print(dxs.site_xpos.shape)
print(dxs.site_xpos[:,tip_id,:])
