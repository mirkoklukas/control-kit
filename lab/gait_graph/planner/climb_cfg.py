"""The planner's config for the climb robot: :class:`ClimbCfg`.

`gait_graph`'s :class:`~lab.gait_graph.config.Cfg` (stance validity, sampling, foothold
pool) with the climb robot's dimensions, taken from :class:`.climb_model.config.MjModelCfg`
by :func:`climb_cfg`, so the stance code and the MuJoCo model share one geometry.

Mapping (``Cfg`` field <- ``MjModelCfg``):

- ``num_legs``, ``mount_radius``, ``leg_lengths``, ``joint_limits_deg``,
  ``body_half_height``, ``link_radius``: as is.
- ``coxa_radius``, ``link_radius`` (the femur), ``tibia_radius``: ``link_radii``.
- ``foot_radius``: the ankle pivot's height above the pad face, ``pivot_height +
  pad_thickness``. The stance code places the "foot" (the pivot) that far above the
  foothold along its normal: a flat pad on the surface.
- ``ankle_limit_deg``: ``ankle_range_deg`` (a cone of that half-angle around the normal
  approximates the Cardan ankle's two +/- limits; conservative at the diagonals).
- ``body_height``: ``stand_height``.
"""
from dataclasses import dataclass, replace

from ..config import Cfg
from .climb_model.config import MjModelCfg


@dataclass
class ClimbCfg(Cfg):
    """`gait_graph`'s stance / sampling knobs, for the climb robot on flat ground (v0).

    The robot dimensions are filled in by :func:`climb_cfg`; the rest are defaults for its
    size (half of `gait_graph`'s robot).
    """
    coxa_radius: float = 0.015
    tibia_radius: float = 0.009

    # validity, scaled to the climb robot (gait_graph's robot is ~2x larger)
    min_foot_separation: float = 0.06    # pads are 40 mm wide
    foot_clearance_radius: float = 0.035 # ignore terrain contact this close to a planted
                                         # foot: the pad's half-diagonal (28 mm) + margin
    collision_delta: float = 0.01

    # terrain v0: flat ground only, top face at z = 0
    boxes: tuple = (((0.0, 0.0, -0.05), (2.0, 1.0, 0.05)),)
    num_footholds: int = 40_000     # ~1 per cm^2 on the 4 x 2 m ground
    k_per_leg: int = 512
    tries: int = 4_000


def climb_cfg(mj: MjModelCfg = None, **overrides) -> ClimbCfg:
    """A :class:`ClimbCfg` with the robot dimensions of ``mj`` (default: the defaults).

    Args:
        mj: the climb robot's model config.
        **overrides: any other :class:`ClimbCfg` field.

    Returns:
        The config.
    """
    mj = mj or MjModelCfg()
    dims = dict(
        num_legs=mj.num_legs, mount_radius=mj.mount_radius, leg_lengths=tuple(mj.leg_lengths),
        joint_limits_deg=tuple(tuple(l) for l in mj.joint_limits_deg),
        body_half_height=mj.body_half_height, coxa_radius=mj.link_radii[0],
        link_radius=mj.link_radii[1], tibia_radius=mj.link_radii[2],
        foot_radius=mj.pivot_height + mj.pad_thickness,
        ankle_limit_deg=mj.ankle_range_deg, body_height=mj.stand_height)
    return replace(ClimbCfg(), **(dims | overrides))
