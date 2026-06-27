"""Compatibility shim: brax 0.14 vs jax 0.10.

Brax 0.14's training code still calls ``jax.device_put_replicated``, which jax
0.10 removed (part of the pmap deprecation). It is the only *hard* removal brax
hits here (``jax.pmap`` / ``lax.pmean`` are deprecated but still functional), so
we re-add a single-host equivalent: replicate each leaf along a leading device
axis and let ``pmap`` place the slices. Import this module before calling
``brax...ppo.train``.
"""
import jax
import jax.numpy as jnp


def _device_put_replicated(value, devices):
    n = len(devices)
    return jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(jnp.asarray(x), (n,) + jnp.asarray(x).shape), value
    )


if not hasattr(jax, "device_put_replicated"):
    jax.device_put_replicated = _device_put_replicated
