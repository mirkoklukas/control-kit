"""(Copied from ``lab/rl/rewards.py``; masks renamed planted -> attached, plus
:func:`magnet_switch` and :func:`phase_match`.)

Reward terms for the rl policies, as pure functions of arrays.

One function per term of docs/policy-and-reward-terms.md. Each takes plain arrays
(and parameters) and returns the term's *raw* value: rewards and costs both as
non-negative numbers, without weight, curriculum factor or dt. The env composes
them (see :meth:`.env.ClimbEnv.step`): ``+w * reward``, ``-w * cost``, then the
curriculum and the dt.

No MuJoCo calls and no python loops over feet: the inputs are gathered by the env,
and per-foot terms are vectorized over a leading foot axis (4,). Only array
operations are used, so porting to JAX (MJX) is mostly ``np`` -> ``jnp``.

Conventions for per-foot inputs: shape (4,), foot i = leg i. ``attached`` is a bool
mask (magnet on, pad flat and in contact; see :mod:`.env`); swing = ``~attached``.
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


def hard_clearance(h, v_t, h_min, v_tol):
    """``C_hard_clear = sum_{h_i < h_min} max(0, |v_t,i| - v_tol)``: feet low and moving.

    A foot below ``h_min`` (attached, touching, or skimming just above the surface)
    should not move along the surface beyond ``v_tol``. A hard step in height: no cost
    at or above ``h_min``. Overlaps slip (attached) and drag (touching); it adds the
    skimming foot that touches nothing, which clearance prices only weakly near h = 0.

    Args:
        h: (4,) height of each pad face above the surface (m).
        v_t: (4,) each foot's speed along the surface (m/s).
        h_min: feet below this height count (m).
        v_tol: speed allowed without cost (m/s).
    """
    return np.sum(np.where(h < h_min, np.maximum(0.0, v_t - v_tol), 0.0))


def foot_drag(v_t, touching, attached):
    """``C_drag = sum_{touching, not attached} |v_t,i|``: feet scraping along the surface.

    A foot touching the surface without carrying load (on a wall: without being
    attached) should not move along it -- at lift-off, or a swinging foot hanging
    too low. Complements the other foot terms: slip covers attached feet, clearance
    feet in the air.

    Args:
        v_t: (4,) each foot's speed along the surface (m/s).
        touching: (4,) bool, foot in contact with the surface.
        attached: (4,) bool, foot loaded / attached (these are slip's, skipped here).
    """
    return np.sum(np.where(touching & ~attached, v_t, 0.0))


def foot_slip(v_t, w_n, attached, r_pad):
    """``C_slip = sum_contact (|v_t,i| + r_pad |w_n,i|)``: attached feet sliding or twisting.

    Args:
        v_t: (4,) each foot's speed along the surface (m/s).
        w_n: (4,) each foot's rotation rate about the surface normal (rad/s).
        attached: (4,) bool, foot in contact (only these count).
        r_pad: half the pad size (m); turns the twist rate into an edge speed.
    """
    return np.sum(np.where(attached, v_t + r_pad * np.abs(w_n), 0.0))


def ankle_range(q_ankle, attached, q_safe):
    """``C_ankle = sum_contact sum_ab max(0, |q| - q_safe)^2``: tibia near the stops.

    With the pad flat, the ankle angles equal the tibia's tilt from the normal; near
    the stops the tibia levers the pad (peel).

    Args:
        q_ankle: (4, 2) ankle hinge angles per foot (a, b) (rad).
        attached: (4,) bool, foot in contact (only these count).
        q_safe: angle without cost (rad).
    """
    excess = np.maximum(0.0, np.abs(q_ankle) - q_safe) ** 2        # (4, 2)
    return np.sum(np.where(attached[:, None], excess, 0.0))


# ------------------------------------------------------------------ gait
def min_support(attached, n_min):
    """``C_support = max(0, n_min - n_attached)``: how many feet short of the minimum.

    Graded, not 0/1: one foot short costs 1, two cost 2. A 0/1 penalty treats a
    near-crawl (occasionally one foot short) like a trot (often two short), which
    gives no gradient towards the crawl and, weighted strongly, makes not stepping at
    all the cheapest option.

    Args:
        attached: (4,) bool, foot attached.
        n_min: minimum number of attached feet (3 = crawl, at most one foot in the air).

    Returns:
        The shortfall, >= 0.
    """
    return np.maximum(0.0, n_min - np.sum(attached)).astype(float)


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


def magnet_switch(cmd, cmd_prev):
    """``C_switch = sum_i 1[m_i,t != m_i,t-1]``: magnet toggles this step.

    An EPM spends energy only when it switches, not while it holds; without a cost the
    policy may flicker a magnet for free.

    Args:
        cmd: (4,) bool, magnet commands this step.
        cmd_prev: (4,) bool, magnet commands at the previous step.
    """
    return float(np.sum(cmd != cmd_prev))


def swing_height(stance, h, h_target):
    """``C_swing_height = sum_{swing} max(0, 1 - h_i / h_target)``: swing feet too low.

    The fraction of the target height each swing foot is missing: 1 on the floor, 0 at
    or above ``h_target``, linear in between. Not scaled by the foot's speed (unlike
    :func:`foot_clearance`), so a swing foot left low costs even when it does not move.

    Args:
        stance: (4,) bool per foot. True: the clock wants the foot in stance.
        h: (4,) pad face height above the floor (m).
        h_target: the swing height to reach (m).

    Returns:
        The summed shortfall over the swing feet, 0 ... 4.
    """
    short = np.clip(1.0 - np.asarray(h) / h_target, 0.0, 1.0)
    return float(np.sum(np.where(stance, 0.0, short)))


def swing_touch(stance, foot_touching):
    """``C_swing_touch = sum_i 1[foot i in swing and touching]``: swing feet not lifted.

    The clock has the foot in swing (its magnet off, in clock mode), but the foot still
    touches something. Pushes the policy to lift the swing foot, not just leave it
    resting on the floor with the magnet off.

    Args:
        stance: (4,) bool per foot. True: the clock wants the foot in stance.
        foot_touching: (4,) bool per foot. True: the foot touches anything.

    Returns:
        The number of swing feet that touch, 0 ... 4.
    """
    return float(np.sum(~stance & foot_touching))


def phase_match(stance, attached, magnet, touching):
    """``R_phase = mean_i 1[foot i matches the clock]``, in [0, 1].

    The crawl's contact pattern (reward-design-simple.md, "simpler alternative"),
    extended to the magnet: in desired stance the foot should be attached; in desired
    swing its magnet off and the pad off the surface.

    Args:
        stance: (4,) bool, the clock wants foot i in stance.
        attached: (4,) bool, foot attached (magnet on, pad flat and in contact).
        magnet: (4,) bool, magnet actually on.
        touching: (4,) bool, any pad cell in contact.
    """
    swing_ok = ~magnet & ~touching
    return float(np.mean(np.where(stance, attached, swing_ok)))
