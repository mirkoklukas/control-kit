import jax.numpy as jnp
from jaxlie import SE3, SO3
from jax.scipy.spatial.transform import Rotation as Rot

# Let a batched SE3 be indexed/sliced like an array (jaxlie doesn't support it).
def _se3_getitem(self, index) -> SE3:
    if not isinstance(index, tuple):
        index = (index,)
    return SE3(self.wxyz_xyz[index + (slice(None),)])  # keep the trailing 7 axis
SE3.__getitem__ = _se3_getitem

def _se3_shape(self) -> tuple:
    return self.wxyz_xyz.shape[:-1]
SE3.shape = property(_se3_shape)

def _se3_reshape(self, shape) -> SE3:
    return SE3(self.wxyz_xyz.reshape(tuple(shape) + (7,)))
SE3.reshape = _se3_reshape

def _se3_broadcast_to(self, shape) -> SE3:
    return SE3(jnp.broadcast_to(self.wxyz_xyz, tuple(shape) + (7,)))
SE3.broadcast_to = _se3_broadcast_to

# Euler / RPY convention.
# > Intrinsic: XYZ (uppercase, moving axes)
# > Extrinsic: xyz (lowercase, fixed).
# jaxlie's SO3.from_rpy_radians(r, p, y) == Rz(y) @ Ry(p) @ Rx(r) (ZYX), and is
# exactly scipy's from_euler("xyz", [r, p, y]) / from_euler("ZYX", [y, p, r]).
def from_te(t, e, seq="xyz") -> SE3:
    """Create an SE3 from a translation ``t`` and Euler angles ``e`` (rpy=xyz)."""
    rpy = Rot.from_euler(seq, e, degrees=False).as_euler("xyz", degrees=False)
    return SE3.from_rotation_and_translation(SO3.from_rpy_radians(rpy[...,0], rpy[...,1], rpy[...,2]), t)
SE3.from_te = staticmethod(from_te)

def _se3_stack(tfs) -> SE3:
    """Create an SE3 from a translation ``t`` and Euler angles ``e`` (rpy=xyz)."""
    return SE3(wxyz_xyz=jnp.stack([tf.wxyz_xyz for tf in tfs], axis=0))
SE3.stack = staticmethod(_se3_stack)
