from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .utils import adjust_angle
from .se3 import SE3, SO3


#
#   Specifiying a multilegged robot's configuration and kinematics. 
#
NUM_LEGS = 6
RADIUS = 0.15  # circumradius of the hexagon formed by the shoulders

_SHOULDER_ANGLES = jnp.deg2rad(
    jnp.arange(360.0 / NUM_LEGS / 2, 360.0, 360.0 / NUM_LEGS)
)  

SHOULDERS = SE3.from_te(
    RADIUS * jnp.stack([
        jnp.cos(_SHOULDER_ANGLES), 
        jnp.sin(_SHOULDER_ANGLES),
        jnp.zeros(NUM_LEGS)], axis=-1
    ),
    _SHOULDER_ANGLES[:,None], "z"
)

NUM_JOINTS = 4
LENGTHS = jnp.array([0.025, 0.025, 0.2, 0.25])
ANKLE_LIMIT = jnp.pi/2
# joint axes in the order of the kinematic chain
# hip-yaw, hip-roll, hip-pitch, knee-pitch
AXES = ["z", "x", "y", "y"]  
JOINT_LIMITS = jnp.deg2rad(jnp.array([
    [-80.0,  80.0],    
    [-90.0,  90.0],    
    [-120.0,  120.0],  
    [  0.0, 120.0],  
]))





# Alternative names: ContactSite, SupportSite
@jax.tree_util.register_dataclass
@dataclass
class Foothold: 
    """Location in the environment that is suitable for placing a foot."""
    position: jax.Array
    normal: jax.Array 
    # Zero normal means no contact and a floating position, i.e. 
    # the foothold is not used, but we may store a foot position in there.

    def __getitem__(self, index) -> "Foothold":
        """Index a batched foothold (leading axis) -> a foothold."""
        return Foothold(self.position[index], self.normal[index])

    @property
    def shape(self) -> tuple[int, ...]:
        return self.position.shape[:-1]

    def __iter__(self):
        """Iterate over the leading axis of a batched foothold."""
        for i in range(self.shape[0]):
            yield self[i]

    @property
    def pos(self) -> jax.Array:
        return self.position


# Alternative names: mask = support
@jax.tree_util.register_dataclass
@dataclass
class Stance: 
    support: jax.Array
    feet: jax.Array
    normals: jax.Array

    @property
    def ids(self) -> jax.Array:
        """Indices of the stance's footholds in the pool."""
        return jnp.where(self.support)[0]
    
    def __getitem__(self, index) -> "Stance":
        """Index a batched stance (leading axis) -> a stance."""
        return Stance(self.support[index], self.feet[index], self.normals[index])

    @property
    def shape(self) -> tuple[int, ...]:
        return self.support.shape[:-1]

    def __iter__(self):
        """Iterate over the leading axis of a batched stance."""
        for i in range(self.shape[0]):
            yield self[i]


# Alternative names: Support
@jax.tree_util.register_dataclass
@dataclass
class Posture:
    body: SE3
    thetas: jax.Array
    stance: Stance

    @property
    def feet(self) -> jax.Array:
        """Indices of the stance's footholds in the pool."""
        return self.stance.feet

    def __getitem__(self, index) -> "Posture":
        """Index a batched posture (leading axis) -> a posture."""
        return Posture(self.body[index], self.thetas[index], self.stance[index])

    @property
    def shape(self) -> tuple[int, ...]:
        return self.body.shape

    def __iter__(self):
        """Iterate over the leading axis of a batched posture."""
        for i in range(self.shape[0]):
            yield self[i]

# 
#   Forward kinematics;
#   NOTE: All the links/segments are placed along the x axis. We could make that more general.
#
def _joint(x, theta, axis=AXES):
    return SE3.from_te(jnp.array([x, 0.0, 0.0]), theta, axis)

def _joint_frames(thetas, lengths=LENGTHS, axes=AXES):
    """Maps joint angles to the frames of the joints and the end-effector
    in the world frame. Assuming shoulder link is mounted at the origin of the world frame"""
    num_joints = len(lengths)

    tf = _joint(0.0, thetas[0], axes[0])
    frames = [tf]

    for i in jnp.arange(num_joints-1):
        tf = tf @ _joint(lengths[i], thetas[i+1], axes[i+1])
        frames.append(tf)
    tf = tf @ SE3.from_translation(jnp.array([lengths[-1], 0.0, 0.0]))
    frames.append(tf)
    
    return SE3.stack(frames)

def _forward_model(thetas, lengths=LENGTHS, axes=AXES):
    """Maps joint angles to the positions of the joints and the end-effector
    in the world frame. Assuming shoulder link is mounted at the origin of the world frame"""
    num_joints = len(lengths)

    tf = _joint(0.0, thetas[0], axes[0])
    xpos = [tf.translation()]

    for i in jnp.arange(num_joints-1):
        tf = tf @ _joint(lengths[i], thetas[i+1], axes[i+1])
        xpos.append(tf.translation())
    tf = tf @ SE3.from_translation(jnp.array([lengths[-1], 0.0, 0.0]))
    xpos.append(tf.translation())
    
    return jnp.stack(xpos, axis=0)

#
#   Inverse kinematics;
#
def _infer_theta_OLD(shoulder: SE3, foot: jax.Array, normal: jax.Array, lengths: jax.Array=LENGTHS) -> tuple[jax.Array, jax.Array]:
    """Compute joint angles that place the foot at ``foot``."""

    # Move the inputs to the shoulder frame.
    p = shoulder.inverse().apply(foot)
    # n = shoulder.rotation().inverse().apply(normal)          

    # HIP-YAW:
    # We pick the angle of the coxa joint to be in the range [-pi/2, pi/2]; 
    # This is enforcing a joint limit within [-pi/2, pi/2]. That is a modeling assumption to 
    # avoid the second theta_0 branch
    yaw = jnp.arctan2(p[1], p[0])
    yaw = jnp.where(p[0] < 0, yaw + jnp.pi, yaw)
    yaw = adjust_angle(yaw)

    # HIP-ROLL:
    # Choose roll such that the surface normal is in the flexion plane of the leg. 
    # Technically we have 1-parameter family of (yaw,roll) pairs.
    #
    # hip = _joint(0.0, yaw, "z")
    # n = hip.rotation().inverse().apply(n)
    # this should align the z-axis with the projected normal
    hip = _joint(0.0, yaw, "z") @ SE3.from_translation(jnp.array([lengths[0], 0.0, 0.0])) 
    v = hip.inverse().apply(p)
    roll = jnp.arctan2(v[2], v[1]) - jnp.pi/2 

    # HIP-PITCH and KNEE-PITCH:
    # We should now have a 2D problem in the plane of the leg. 
    # We can use the law of cosines to compute the angles.
    hip = ( 
        _joint(0.0, yaw, "z") @ 
        _joint(lengths[0], roll, "x") @ 
        SE3.from_translation(jnp.array([lengths[1], 0.0, 0.0])) 
    )
    u0, u1 = hip.inverse().apply(p)[jnp.array([0, 2])]
    
    # a=foot, b=femur, c=tibia
    a = jnp.sqrt(u0**2 + u1**2)
    b = lengths[2]
    c = lengths[3]

    reachable = (
        (a <= (b + c)) &
        (a >= jnp.abs(b - c))
    )

    offset = jnp.arctan2(u1, u0)
    alpha = jnp.arccos(
        (c**2 + b**2 - a**2 ) / (2 * c * b)
    )
    gamma = jnp.arccos(
        (b**2 + a**2 - c**2 ) / (2 * b * a)
    )

    # This orders the remaining branch by elbow down first. 
    theta = jnp.array([
        [yaw, roll,  -(offset + gamma), -(-jnp.pi + alpha)],
        [yaw, roll, -(offset - gamma),  -jnp.pi + alpha],
    ])

    return reachable, theta[0, :]


def _infer_theta(shoulder: SE3, foot: jax.Array, normal: jax.Array, lengths: jax.Array=LENGTHS) -> tuple[jax.Array, jax.Array]:
    """Like :func:`_infer_theta`, but yaw and roll are solved *jointly*.

    The leg's flexion plane (the x-z plane of the frame after yaw@roll) must contain
    both the foot and the surface normal, so its out-of-plane axis is
    ``y_hat = normalize(p x n)``. The femur base lies on the plane's x-axis (the
    coxa/roll offset is along ``x_hip`` and roll is about x, so it cancels), hence
    the plane passes through the shoulder origin and this is exact. yaw/roll are read
    back from ``y_hat``; the planar femur/knee solve is unchanged.

    Fixes the old bug where yaw (from the foot azimuth) and roll (from the normal)
    were picked independently, leaving the foot off the tilted plane by its
    out-of-plane component and mislocating it for non-vertical normals.
    """
    p = shoulder.inverse().apply(foot)
    n = shoulder.rotation().inverse().apply(normal)

    # Flexion-plane out-of-plane axis: perpendicular to both the foot and the normal.
    yhat = jnp.cross(p, n)
    yhat = yhat / (jnp.linalg.norm(yhat) + 1e-12)
    yhat = jnp.where(yhat[1] < 0, -yhat, yhat)       # sign -> coxa forward, yaw in [-pi/2, pi/2]

    # y_hat = [-sin(yaw)cos(roll), cos(yaw)cos(roll), sin(roll)]
    yaw = jnp.arctan2(-yhat[0], yhat[1])
    roll = jnp.arctan2(yhat[2], jnp.sqrt(yhat[0]**2 + yhat[1]**2))

    hip = (
        _joint(0.0, yaw, "z") @
        _joint(lengths[0], roll, "x") @
        _joint(lengths[1], 0.0, "y")
    )

    u0, u1 = hip.inverse().apply(p)[jnp.array([0, 2])]

    # a = reach in the flexion plane, b = femur, c = tibia
    a = jnp.sqrt(u0**2 + u1**2)
    b = lengths[2]
    c = lengths[3]

    reachable = (
        (a <= (b + c)) &
        (a >= jnp.abs(b - c))
    )

    offset = jnp.arctan2(u1, u0)
    alpha = jnp.arccos(
        (c**2 + b**2 - a**2) / (2 * c * b)
    )
    gamma = jnp.arccos(
        (b**2 + a**2 - c**2) / (2 * b * a)
    )

    # This orders the remaining branch by elbow down first.
    theta = jnp.array([
        [yaw, roll,  -(offset + gamma), -(-jnp.pi + alpha)],
        [yaw, roll, -(offset - gamma),  -jnp.pi + alpha],
    ])

    return reachable, theta[0, :]


def infer_theta(body, feet, normals, lengths=LENGTHS, shoulders=SHOULDERS) -> tuple[jax.Array, jax.Array]:
    reachable, theta = jax.vmap(_infer_theta, in_axes=(0, 0, 0, None))(
        body@shoulders, feet, normals, lengths)

    return reachable, theta



FEET_LIFTED = SHOULDERS.apply(_forward_model(
    jnp.deg2rad(jnp.array([0.0, 0.0, -80.0, 160.0])), 
    LENGTHS, AXES)[-1])


def infer_posture(body, stance: Stance, *, feet_lifted=FEET_LIFTED):
    """Infer the joint angles of a posture given the body pose and stance."""

    # Lifted legs have a home position. 
    feet = jnp.where(stance.support[:, None], 
                     stance.feet, 
                     body@feet_lifted)  
    normals = jnp.where(stance.support[:, None], 
                        stance.normals, 
                        body.apply(jnp.array([[0.0, 0.0, 1.0]]))) 

    reachable, theta = infer_theta(body, feet, normals) 

    # in_limits = (joint_limits[:,0] <= theta) & (theta <= joint_limits[:,1])  
    # in_ankle_limits = jax.vmap(angle_between)(infer_foot_vectors(body, theta), stance.normals)
    # valid = (
    #     jnp.all(reachable, where=stance.support) & 
    #     jnp.all(in_limits, where=stance.support[:,None]) &
    #     jnp.all(in_ankle_limits <= ankle_limit, where=stance.support)
    # )

    valid = jnp.all(reachable, where=stance.support)

    return valid, Posture(body, theta, Stance(stance.support, feet, stance.normals))