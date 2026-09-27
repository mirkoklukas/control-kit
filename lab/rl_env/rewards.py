"""Reward terms for the rl_env policies, as pure functions of arrays.

One function per term of docs/policy-and-reward-terms.md. Each takes plain arrays
(and parameters) and returns the term's *raw* value: rewards and costs both as
non-negative numbers, without weight, curriculum factor or dt. The env composes
them (see :meth:`.env.WalkEnv.step`): ``+w * reward``, ``-w * cost``, then the
curriculum and the dt.

No MuJoCo calls and no python loops over feet: the inputs are gathered by the env,
and per-foot terms are vectorized over a leading foot axis (4,). Only array
operations are used, so porting to JAX (MJX) is mostly ``np`` -> ``jnp``.

Conventions for per-foot inputs: shape (4,), foot i = leg i. ``planted`` is a bool
mask (foot in contact); swing = ``~planted``.
"""
import numpy as np


# ------------------------------------------------------------------ kernel
def kernel(x):
    """Bounded logistic kernel ``K(x) = 4 / (e^x + 2 + e^-x) = sech^2(x / 2)``.

    1 at ``x = 0``, falling towards 0 for large ``|x|``: normalized, so a tracking
    weight is the maximum reward. Bounded, so a large tracking error never makes the
    reward very negative (unlike a squared error). Near 0 it matches the Gaussian
    ``exp(-x^2 / 4)``; its tails are heavier (``~ e^-|x|``), which keeps a learning
    signal at large errors. Hwangbo et al. use it without the factor 4 (peak 0.25).

    Args:
        x: scaled tracking error (scalar or array).

    Returns:
        The kernel value in (0, 1], same shape as ``x``.
    """
    return 4.0 / (np.exp(x) + 2.0 + np.exp(-x))


# ------------------------------------------------------------------ tracking (rewards)
def lin_vel_tracking(v_xy, cmd_xy, sharpness=1.0):
    """``R_lin = K(sharpness * |v_xy - cmd_xy|)``: how well the base tracks the command.

    Args:
        v_xy: (2,) base velocity in the surface plane, body frame (m/s).
        cmd_xy: (2,) commanded velocity, same frame.
        sharpness: error scale (1 = the plain kernel). At 20, a 0.15 m/s error leaves
            ~20% of the max.

    Returns:
        Reward in (0, 1].
    """
    return kernel(sharpness * np.linalg.norm(v_xy - cmd_xy))


def yaw_rate_tracking(w_n, cmd_yaw, sharpness=1.0):
    """``R_yaw = K(sharpness * |w_n - cmd_yaw|)``: turning rate about the surface normal.

    Args:
        w_n: base angular velocity about the surface normal (rad/s).
        cmd_yaw: commanded yaw rate (rad/s).
        sharpness: error scale (1 = the plain kernel).

    Returns:
        Reward in (0, 1].
    """
    return kernel(sharpness * np.abs(w_n - cmd_yaw))


# ------------------------------------------------------------------ regularization (costs)
def torque(tau):
    """``C_tau = |tau|^2``: effort.

    Args:
        tau: (n,) actuator torques (N m).
    """
    return tau @ tau


def action_rate(a, a_prev):
    """``C_smooth = |a_t - a_t-1|^2``: jerky actions.

    Args:
        a, a_prev: (n,) this and the previous action.
    """
    return np.sum((a - a_prev) ** 2)


def joint_speed(qdot):
    """``C_phidot = |qdot|^2``: fast joints.

    Args:
        qdot: (n,) joint velocities (rad/s).
    """
    return np.sum(qdot ** 2)


def orientation(z_body, normal=(0.0, 0.0, 1.0)):
    """``C_orient = |n - z_body|``: body not parallel to the surface.

    Args:
        z_body: (3,) the body's z-axis in world coordinates (column 2 of the body's
            rotation matrix).
        normal: (3,) the surface normal (world). The floor's is +z.
    """
    return np.linalg.norm(np.asarray(normal) - z_body)


def body_height(d, d_hat):
    """``C_dist = (d_hat - d)^2``: base too high or too low above the surface.

    Args:
        d: base distance from the surface along the normal (m).
        d_hat: target distance (m).
    """
    return (d_hat - d) ** 2


# ------------------------------------------------------------------ feet (costs)
def foot_clearance(h, v_t, touching, h_hat):
    """``C_clear = sum_air (h_hat - h_i)^2 |v_t,i|``: feet in the air too low (or high).

    Weighted by the foot's speed along the surface, so it applies while a foot
    travels, not while it lifts or lowers in place.

    Args:
        h: (4,) height of each pad face above the surface (m).
        v_t: (4,) each foot's speed along the surface (m/s).
        touching: (4,) bool, foot in contact with the surface (these are skipped).
        h_hat: target clearance (m).
    """
    return np.sum(np.where(touching, 0.0, (h_hat - h) ** 2 * v_t))


def foot_drag(v_t, touching, planted):
    """``C_drag = sum_{touching, not planted} |v_t,i|``: feet scraping along the surface.

    A foot touching the surface without carrying load (on a wall: without being
    attached) should not move along it -- at lift-off, or a swinging foot hanging
    too low. Complements the other foot terms: slip covers planted feet, clearance
    feet in the air.

    Args:
        v_t: (4,) each foot's speed along the surface (m/s).
        touching: (4,) bool, foot in contact with the surface.
        planted: (4,) bool, foot loaded / attached (these are slip's, skipped here).
    """
    return np.sum(np.where(touching & ~planted, v_t, 0.0))


def foot_slip(v_t, w_n, planted, r_pad):
    """``C_slip = sum_contact (|v_t,i| + r_pad |w_n,i|)``: planted feet sliding or twisting.

    Args:
        v_t: (4,) each foot's speed along the surface (m/s).
        w_n: (4,) each foot's rotation rate about the surface normal (rad/s).
        planted: (4,) bool, foot in contact (only these count).
        r_pad: half the pad size (m); turns the twist rate into an edge speed.
    """
    return np.sum(np.where(planted, v_t + r_pad * np.abs(w_n), 0.0))


def ankle_range(q_ankle, planted, q_safe):
    """``C_ankle = sum_contact sum_ab max(0, |q| - q_safe)^2``: tibia near the stops.

    With the pad flat, the ankle angles equal the tibia's tilt from the normal; near
    the stops the tibia levers the pad (peel).

    Args:
        q_ankle: (4, 2) ankle hinge angles per foot (a, b) (rad).
        planted: (4,) bool, foot in contact (only these count).
        q_safe: angle without cost (rad).
    """
    excess = np.maximum(0.0, np.abs(q_ankle) - q_safe) ** 2        # (4, 2)
    return np.sum(np.where(planted[:, None], excess, 0.0))


# ------------------------------------------------------------------ gait
def min_support(planted, n_min):
    """``C_support = max(0, n_min - n_planted)``: how many feet short of the minimum.

    Graded, not 0/1: one foot short costs 1, two cost 2. A 0/1 penalty treats a
    near-crawl (occasionally one foot short) like a trot (often two short), which
    gives no gradient towards the crawl and, weighted strongly, makes not stepping at
    all the cheapest option.

    Args:
        planted: (4,) bool, foot planted.
        n_min: minimum number of planted feet (3 = crawl, at most one foot in the air).

    Returns:
        The shortfall, >= 0.
    """
    return np.maximum(0.0, n_min - np.sum(planted)).astype(float)


def air_time(touchdown, air_at_touchdown, target, t_max):
    """``R_air = sum_touchdown (min(t_air, t_max) - target)``: step length in time.

    Paid only on touchdown steps. Negative for steps shorter than ``target`` (so
    shuffling costs), positive for longer ones, capped at ``t_max`` (no lingering
    legs). Unlike the other rewards it can be negative.

    Args:
        touchdown: (4,) bool, foot touched down this step.
        air_at_touchdown: (4,) the air time that just ended (s); ignored where no
            touchdown.
        target: desired air time per step (s).
        t_max: air time counted at most this (s).
    """
    return np.sum(touchdown * (np.minimum(air_at_touchdown, t_max) - target))
