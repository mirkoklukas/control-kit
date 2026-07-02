"""SE(3) — the group of rigid-body transforms in 3D, backed by JAX.

`SE3` is the *mathematical* transform: a rotation + translation, with no
notion of named coordinate frames. Frame bookkeeping (parent/child, trees)
lives one layer up in `psilia.transforms.transform.Transform`, which is
intended to extend this class.

Representation. We store a translation ``(..., 3)`` and a quaternion
``(..., 4)`` in **xyzw** order (scipy convention, scalar-last). Both may
carry leading batch dimensions.

jaxlie as a transient helper. Group operations — composition, inverse,
exp/log, adjoint — are delegated to `jaxlie`, which we instantiate inside
the methods that need it. It is a helper, not the stored representation, so
we never hold a `jaxlie.SE3` around. Point application and rotation
conversions use `jax.scipy`'s `Rotation` directly (the hot path).

Conventions.
- Tangent vectors are 6-vectors ordered ``[v_translation, omega_rotation]``
  (matching jaxlie / Sophus).
- `exp` is the *group* exponential: the translation part is coupled to the
  rotation through the left Jacobian, so it is **not** ``(v, so3_exp(omega))``.
- `compose` (``@``) is matrix-consistent: ``(a @ b)(x) == a(b(x))``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jaxlie
from jax.scipy.spatial.transform import Rotation as Rot
from jax.tree_util import register_pytree_node_class


SCALAR_FIRST = False  # xyzw vs wxyz quaternion order; xyzw is scipy convention
 
@register_pytree_node_class
class SE3:
    """An element of SE(3): a rigid-body transform in 3D.

    Maps points from one frame to another via ``R @ x + t``. Stores a
    translation and an xyzw quaternion; both may carry leading batch
    dimensions.
    """

    def __init__(self, translation: jax.Array, quaternion: jax.Array):
        """
        Args:
            translation: Translation vector, shape ``(..., 3)``.
            quaternion: Unit quaternion in **xyzw** order, shape ``(..., 4)``.
        """
        self.translation: jax.Array = translation
        self.quaternion: jax.Array = quaternion

    # # # # # # # # # # # # # # # # # # # # # # #
    #
    #   Constructors
    #
    # # # # # # # # # # # # # # # # # # # # # # #
    @classmethod
    def from_matrix(cls, matrix: jax.Array) -> SE3:
        """Create from a ``(..., 4, 4)`` homogeneous matrix ``[R t; 0 1]``."""
        return cls._from_se3(jaxlie.SE3.from_matrix(matrix))

    @classmethod
    def from_tR(cls, t: jax.Array, R: jax.Array) -> SE3:
        """Create from a translation and a ``(..., 3, 3)`` rotation matrix."""
        return cls(t, Rot.from_matrix(R).as_quat())

    @classmethod
    def from_te(cls, t, e, *, seq: str = "xyz", degrees: bool = False) -> SE3:
        """Create from a translation and Euler angles."""
        return cls(t, Rot.from_euler(seq, e, degrees=degrees).as_quat())

    @classmethod
    def from_vw(cls, v: jax.Array, w: jax.Array) -> SE3:
        """Group exponential from the twist components ``v`` and ``w`` given separately.

        Convenience wrapper over `exp`: concatenates into a ``[v, w]`` tangent
        and applies the (coupled) group exponential. Same map as `exp` — ``v``
        is **not** the resulting translation.

        Args:
            v: Translational twist component, shape ``(..., 3)``.
            w: Rotational twist component (``omega``), shape ``(..., 3)``.
        """
        return cls.exp(jnp.concatenate([v, w], axis=-1))

    # # # # # # # # # # # # # # # # # # # # # # #
    #
    #   jaxlie bridge — transient; jaxlie is a helper, not the storage.
    #   jaxlie handles the xyzw <-> wxyz reorder internally.
    #
    # # # # # # # # # # # # # # # # # # # # # # #

    def _to_se3(self) -> jaxlie.SE3:
        return jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3.from_quaternion_xyzw(self.quaternion),
            self.translation,
        )

    @classmethod
    def _from_se3(cls, se3: jaxlie.SE3) -> SE3:
        return cls(se3.translation(), se3.rotation().as_quaternion_xyzw())

    # # # # # # # # # # # # # # # # # # # # # # #
    #
    #   Accessors
    #
    # # # # # # # # # # # # # # # # # # # # # # #
    @property
    def t(self) -> jax.Array:
        """Translation vector, shape ``(..., 3)``."""
        return self.translation

    @property
    def q(self) -> jax.Array:
        """Unit quaternion in xyzw order, shape ``(..., 4)``."""
        return self.quaternion

    @property
    def xyzw(self) -> jax.Array:
        """Quaternion in xyzw order (the stored layout)."""
        return self.quaternion

    @property
    def wxyz(self) -> jax.Array:
        """Quaternion in wxyz order (scalar-first; e.g. for jaxlie / ROS)."""
        q = self.quaternion
        return jnp.concatenate([q[..., -1:], q[..., :3]], axis=-1)

    @property
    def rot(self) -> Rot:
        """The rotation as a `jax.scipy` `Rotation`."""
        return Rot.from_quat(self.quaternion)

    @property
    def R(self) -> jax.Array:
        """Rotation matrix, shape ``(..., 3, 3)``."""
        return self.rot.as_matrix()

    def as_matrix(self) -> jax.Array:
        """Homogeneous matrix ``[R t; 0 1]``, shape ``(..., 4, 4)``."""
        return self._to_se3().as_matrix()

    # # # # # # # # # # # # # # # # # # # # # # #
    #
    #   Group operations
    #
    # # # # # # # # # # # # # # # # # # # # # # #
    def apply(self, xs: jax.Array) -> jax.Array:
        """Apply the transform to points: ``R @ xs + t``."""
        return self.rot.apply(xs) + self.translation

    def __call__(self, xs: jax.Array) -> jax.Array:
        return self.apply(xs)

    @classmethod
    def id(cls) -> SE3:
        """The identity transform."""
        return cls(jnp.zeros(3), jnp.array([0.0, 0.0, 0.0, 1.0]))

    identity = id  # alias for convenience

    def inv(self) -> SE3:
        """Inverse transform."""
        return SE3._from_se3(self._to_se3().inverse())

    def compose(self, other: SE3) -> SE3:
        """Compose two transforms.

        Matrix-consistent: ``(self.compose(other))(x) == self(other(x))`` and
        ``(self @ other).as_matrix() == self.as_matrix() @ other.as_matrix()``.
        """
        return SE3._from_se3(self._to_se3() @ other._to_se3())

    def __matmul__(self, other: SE3) -> SE3:
        return self.compose(other)

    def __rshift__(self, other: SE3) -> SE3:
        """``self >> other`` == ``other @ self`` — data flows through `self` first."""
        return other @ self

    def __lshift__(self, other: SE3) -> SE3:
        """``self << other`` == ``self @ other`` — data flows through `other` first."""
        return self @ other

    @classmethod
    def exp(cls, tangent: jax.Array) -> SE3:
        """Group exponential of a tangent vector.

        Args:
            tangent: Twist, shape ``(..., 6)``, ordered ``[v, omega]`` with the
                translational part first. The translation is coupled to the
                rotation through the SE(3) left Jacobian — this is **not** the
                same as ``SE3(v, so3_exp(omega))``.
        """
        return cls._from_se3(jaxlie.SE3.exp(tangent))

    def log(self) -> jax.Array:
        """Group logarithm: tangent ``(..., 6)`` ordered ``[v, omega]``."""
        return self._to_se3().log()

    def adjoint(self) -> jax.Array:
        """The ``(..., 6, 6)`` adjoint matrix."""
        return self._to_se3().adjoint()

    def interpolate(self, other: SE3, alpha: jax.Array) -> SE3:
        """Screw (geodesic) interpolation. ``alpha=0`` → self, ``alpha=1`` → other."""
        delta = (self.inv() @ other).log()
        return self @ SE3.exp(alpha * delta)

    # # # # # # # # # # # # # # # # # # # # # # #
    #
    #   JAX pytree registration (frameless → no aux data)
    #
    # # # # # # # # # # # # # # # # # # # # # # #

    def tree_flatten(self):
        return (self.translation, self.quaternion), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    # # # # # # # # # # # # # # # # # # # # # # #
    #
    #   Batching
    #
    # # # # # # # # # # # # # # # # # # # # # # #

    @property
    def shape(self) -> tuple:
        return self.translation.shape[:-1]

    def __len__(self) -> int:
        return 0 if self.translation.ndim == 1 else self.translation.shape[0]

    def __getitem__(self, index) -> SE3:
        return SE3(self.translation[index], self.quaternion[index])

    def __iter__(self):
        self._i = 0
        return self

    def __next__(self) -> SE3:
        if self._i < len(self):
            item = self[self._i]
            self._i += 1
            return item
        raise StopIteration

    @staticmethod
    def stack(items: list) -> SE3:
        """Stack a list of transforms along a new leading axis."""
        return SE3(
            jnp.stack([s.translation for s in items]),
            jnp.stack([s.quaternion for s in items]),
        )

    # # # # # # # # # # # # # # # # # # # # # # #
    #
    #   String representation
    #
    # # # # # # # # # # # # # # # # # # # # # # #

    def __repr__(self) -> str:
        e = self.rot.as_euler("XYZ", degrees=True).astype(int)
        return (
            "SE3("
            f"\n translation={self.translation},"
            f"\n quaternion={self.quaternion},"
            f"\n euler (XYZ°)={e}"
            "\n)"
        )
Transform = SE3  # alias for convenience