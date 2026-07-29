"""The robot: a body with identical legs, and the pose helpers it needs.

Parallels :mod:`.leg` one level up. Where a :class:`~controlkit.kinematics.leg.Leg`
maps ``theta`` to frames, a :class:`Robot` maps a whole :class:`.Posture` (body pose
+ per-leg angles) to world frames. Concrete robot builds (specific dimensions) are
the caller's, not the library's.
"""

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from controlkit.kinematics.leg import Leg
from controlkit.kinematics.types import Foothold, Posture, Support
from controlkit.se3 import SE3


@jax.tree_util.register_dataclass
@dataclass
class Robot:
    """A body with ``num_legs`` identical legs mounted on it.

    Parallels :class:`Leg` one level up: where a leg maps ``theta`` to frames and
    samples a config on a foothold, a robot maps a whole :class:`Posture` (body pose
    + per-leg angles) to world frames and samples a posture given a support.

    Legs are homogeneous by construction: one :class:`Leg`, shared by every mount --
    which is what lets every method ``vmap`` over the leg axis.

    Args:
        mounts: (num_legs,) SE3 shoulder poses, in the body frame.
        leg: the leg carried at every mount.
    """

    mounts: SE3
    leg: Leg

    @property
    def num_legs(self) -> int:
        return self.mounts.shape[0]

    def shoulders(self, body: SE3) -> SE3:
        """World poses of the shoulders for a given body pose.

        Args:
            body: SE3 body pose.

        Returns:
            (num_legs,) SE3 shoulder poses in the world frame.
        """
        return body @ self.mounts

    def forward(self, posture: Posture) -> SE3:
        """Frames of every joint of every leg, in the world frame.

        The robot-level forward kinematics: each leg's chain, lifted from its base
        frame to the world by its shoulder.

        Args:
            posture: reads ``body`` and ``thetas``.

        Returns:
            (num_legs, n+1) SE3 world frames; ``[:, -1]`` are the feet.
        """
        mounts = posture.body @ self.mounts
        return jax.vmap(lambda m, t: m @ self.leg.forward(t))(mounts, posture.thetas)

    # TODO: i don't like this name. Maye joint positions? It is mujocos "xpos" right?
    def points(self, posture: Posture) -> jax.Array:
        """World positions of every joint and foot.

        The robot as a point cloud. Note this pins the body pose too: ``[:, 0]``
        are the shoulders, so three or more of them fix the body's position *and*
        orientation without a separate rotation term.

        Args:
            posture: the configuration.

        Returns:
            (num_legs, n+1, 3) positions; ``[:, -1]`` are the feet.
        """
        return self.forward(posture).translation()

    def feet(self, posture: Posture) -> jax.Array:
        """World foot positions.

        Args:
            posture: the configuration.

        Returns:
            (num_legs, 3) foot positions.
        """
        return self.points(posture)[:, -1]

    # TODO: i don't like this name either. Maybe joint positions? It is mujocos "xpos" right?
    def local_points(self, thetas: jax.Array) -> jax.Array:
        """Joint and foot positions in the *body* frame (no body pose).

        The body-frame counterpart of :meth:`points`: each leg's chain lifted only
        by its mount, ``mounts @ leg.forward(theta)``. Because it drops the body
        pose it depends on ``thetas`` alone -- which is exactly what a body-fixed
        self-collision test wants (the box and the legs both move with the body, so
        their relationship is pose-independent).

        Args:
            thetas: (num_legs, n) joint angles.

        Returns:
            (num_legs, n+1, 3) positions in the body frame; ``[:, 0]`` are the
            mounts, ``[:, -1]`` the feet.
        """
        return jax.vmap(lambda m, t: (m @ self.leg.forward(t)).translation())(
            self.mounts, thetas)

    def mount_box(self, *, margin: float = 0.0, half_height: float = 0.05):
        """Keep-out box around the body, spanned by the mounts. Body frame.

        The mounts' axis-aligned bounding box (they sit at ``z = 0``, so it is a
        flat rectangle), inflated by ``margin`` in x/y and given a half-thickness in
        z. Legs are not supposed to reach into this region -- it stands in for the
        body's own volume, a cheap self-collision proxy without modelling the body
        geometry.

        Args:
            margin: extra half-width added in x and y.
            half_height: half-extent in z (the mounts give none).

        Returns:
            ``(center, half)`` -- the box, in the convention
            :func:`..collision.box_sdf` expects.
        """
        t = self.mounts.translation()
        lo, hi = t.min(0), t.max(0)
        center = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo) + jnp.array([margin, margin, half_height])
        return center, half

    def self_collision(self, posture: Posture, *, margin: float = 0.02,
                       half_height: float = 0.04, radius: float = 0.0) -> jax.Array:
        """Do any joints or feet reach into the mount box? A cheap self-collision.

        Composes the two pieces: the :meth:`mount_box` and the :meth:`local_points`,
        checked with :func:`..collision.box_sdf`. The mounts themselves (``[:, 0]``)
        are dropped -- they span the box, so they always sit on it. Reads only
        ``posture.thetas``; the body pose is irrelevant (see :meth:`local_points`).

        Args:
            posture: the configuration (only ``thetas`` is used).
            margin: box half-width added in x/y.
            half_height: box half-extent in z.
            radius: link radius; a point is a hit when its distance is ``< radius``.

        Returns:
            Scalar bool: any joint or foot inside the (inflated) box.
        """
        from controlkit.kinematics import collision
        center, half = self.mount_box(margin=margin, half_height=half_height)
        pts = self.local_points(posture.thetas)[:, 1:, :]     # drop the mounts
        return (collision.box_sdf(pts, center, half) < radius).any()

    def point_cloud(self, posture: Posture, *, per_link: int = 4,
                    body_res=(3, 3, 1), body_margin: float = 0.0,
                    body_half_height: float = 0.02) -> jax.Array:
        """A dense **world-frame** point cloud of the whole robot: legs + body.

        For checking the robot against the environment: query a scene SDF at these
        points (``collision.sdf(scene, cloud) < radius``). The legs are sampled
        along their links (``collision.densify``); the body is a grid filling its
        box (``collision.grid_box``), lifted to the world by the body pose.

        Args:
            posture: the configuration.
            per_link: samples per leg link.
            body_res: grid resolution of the body box (flat by default).
            body_margin: x/y half-width of the body box.
            body_half_height: z half-extent of the body box.

        Returns:
            (P, 3) world points.
        """
        from controlkit.kinematics import collision
        legs = collision.densify(self.points(posture), per_link).reshape(-1, 3)
        center, half = self.mount_box(margin=body_margin, half_height=body_half_height)
        body = posture.body.apply(collision.grid_box(center, half, body_res))
        return jnp.concatenate([legs, body], axis=0)

    def collision_cloud(self, posture: Posture, *, delta: float, body: bool = True,
                        body_margin: float = 0.0, body_half_height: float = 0.02) -> jax.Array:
        """Dense **world-frame** collision cloud of the whole robot, ``delta``-spaced.

        Like :meth:`point_cloud`, but the legs are sampled with
        :meth:`Leg.collision_cloud` (spacing ``<= delta``, uniform density) and the
        body box is gridded at the same spacing. For an environment SDF check:
        ``collision.sdf(scene, cloud) < radius``.

        Args:
            posture: the configuration.
            delta: maximum point spacing (legs and body).
            body: include the body box.
            body_margin: x/y half-width of the body box.
            body_half_height: z half-extent of the body box.

        Returns:
            (P, 3) world points.
        """
        from controlkit.kinematics import collision
        sh = self.shoulders(posture.body)
        legs = jax.vmap(lambda s, t: s.apply(self.leg.collision_cloud(t, delta=delta)))(
            sh, posture.thetas)
        pts = legs.reshape(-1, 3)
        if body:
            # The body box and its grid depend only on the (constant) mounts, so
            # evaluate them at trace time; only the body-pose lift below is traced.
            with jax.ensure_compile_time_eval():
                center, half = self.mount_box(margin=body_margin,
                                              half_height=body_half_height)
                res = tuple(max(2, math.ceil(2 * float(half[i]) / delta) + 1)
                            for i in range(3))
                grid = collision.grid_box(center, half, res)
            pts = jnp.concatenate([pts, posture.body.apply(grid)], 0)
        return pts

    def arm_points(self, posture: Posture, legs=None, *, per_link: int = 4,
                   start: int = 1) -> jax.Array:
        """Dense **body-frame** points of some legs (the "arms").

        For arm-to-body collision -- check these against :meth:`mount_box` with
        ``collision.box_sdf``. Body frame, not world, because the body box lives
        there and the check is pose-independent (reads only ``thetas``).

        Use ``start`` to skip the proximal joints: a leg is attached to the body at
        its mount, so its hip and first links sit *inside* the body box no matter
        what -- that is the attachment, not a collision. For arm-to-body, start past
        the orienting joints (e.g. ``start=2`` to keep only the pitch links, which
        are the parts that actually swing into the body).

        Args:
            posture: the configuration (only ``thetas`` is used).
            legs: which legs to include (indices); None for all.
            per_link: samples per leg link.
            start: first joint index to include; drops joints ``[:start]``. The
                default 1 drops the mount+coxa (the attachment); ``start=0`` gives
                the full arm, e.g. for drawing.

        Returns:
            (Q, 3) body-frame points.
        """
        from controlkit.kinematics import collision
        local = self.local_points(posture.thetas)[:, start:, :]
        pts = collision.densify(local, per_link)
        if legs is not None:
            pts = pts[jnp.asarray(legs)]
        return pts.reshape(-1, 3)

    def leg_boxes(self, posture: Posture, *, radius: float = 0.015):
        """Oriented boxes for every link of every leg, in the **world** frame.

        Each leg's base-frame boxes (``leg.boxes``) lifted to the world by its
        shoulder. For leg-vs-obstacle or leg-vs-leg tests with
        :func:`..collision.overlap`.

        Args:
            posture: the configuration.
            radius: half-thickness of the links.

        Returns:
            :class:`..collision.OBB` of shape ``(num_legs, n)``.
        """
        from controlkit.kinematics import collision
        sh = self.shoulders(posture.body)

        def one(shoulder, theta):
            b = self.leg.boxes(theta, radius=radius)          # base-frame OBB (n,)
            rot = jnp.einsum("ij,njk->nik", shoulder.rotation().as_matrix(), b.rot)
            return collision.OBB(shoulder.apply(b.center), b.half, rot)

        return jax.vmap(one)(sh, posture.thetas)

    def body_box(self, posture: Posture, *, margin: float = 0.0,
                 half_height: float = 0.05):
        """The body as one oriented box, **world** frame.

        The :meth:`mount_box` placed and oriented at the body pose. For body-vs-
        obstacle, or leg-vs-body with :func:`..collision.overlap`.

        Args:
            posture: the configuration.
            margin: x/y half-width added to the body box.
            half_height: z half-extent of the body box.

        Returns:
            A single :class:`..collision.OBB`.
        """
        from controlkit.kinematics import collision
        center, half = self.mount_box(margin=margin, half_height=half_height)
        return collision.OBB(posture.body.apply(center), half,
                             posture.body.rotation().as_matrix())

    def features(self, posture: Posture, weights=None) -> jax.Array:
        """Flatten a posture into a vector whose Euclidean distance is meaningful.

        The point cloud of :meth:`points`, flattened. For two postures,

            ``||features(P1) - features(P2)||`` = ``sqrt(sum_m w_m ||dp_m||^2)``

        so plain Euclidean distance on these vectors compares postures in *metres* --
        no radian-vs-metre weights to invent (see :mod:`..metrics`). Divide by
        ``sqrt(M)`` (unweighted) or ``sqrt(sum w)`` to read it as an RMSD.

        ``vmap`` this over a stack to get the ``(N, 3M)`` matrix :func:`..metrics.knn`
        wants.

        Args:
            posture: the configuration.
            weights: per-point weights, broadcastable to ``(num_legs, n+1)``; pass
                ``w[:, None]`` for per-leg. Folded in as ``sqrt(w)``, which keeps the
                distance Euclidean -- so masking free legs (weight 0) or emphasising
                feet costs nothing. None weights every point equally.

        Returns:
            (3 * num_legs * (n+1),) feature vector.
        """
        p = self.points(posture)
        if weights is not None:
            p = p * jnp.sqrt(jnp.asarray(weights))[..., None]
        return p.reshape(-1)

    def contact_vectors(self, posture: Posture) -> jax.Array:
        """Unit vectors at the feet, pointing back along the last link.

        Compare against surface normals to score how squarely each foot meets its
        surface.

        Args:
            posture: the configuration.

        Returns:
            (num_legs, 3) unit vectors in the world frame.
        """
        sh = self.shoulders(posture.body)
        return jax.vmap(lambda s, t: s @ self.leg.contact_vector(t))(sh, posture.thetas)

    def reach_mask(self, key: jax.Array, body: SE3, footholds: Foothold, *,
                   samples: int = 16, **kwargs) -> jax.Array:
        """Which legs can stand on which footholds, at a given body pose.

        For every ``(foothold, leg)`` pair, whether the leg can place its foot on
        that foothold, via :meth:`Leg.reach`. That is a *sampled* test (it draws
        plants and asks if any is valid), so this is stochastic and its fidelity
        rises with ``samples``.

        Footholds are in the **world** frame; each is pulled into a leg's base frame
        by that leg's shoulder before the leg is asked.

        Args:
            key: PRNG key.
            body: SE3 body pose.
            footholds: (M,) footholds to test, world frame.
            samples: plant attempts per pair (forwarded to :meth:`Leg.reach`).
            **kwargs: forwarded to :meth:`Leg.reach` -> :meth:`Leg.sample_planted`
                (e.g. :meth:`Leg4DOF.sample_planted`'s ``sharpness``).

        Returns:
            (M, num_legs) bool: ``[j, i]`` is True iff leg ``i`` can reach
            foothold ``j``.
        """
        shoulders = self.shoulders(body)

        def per_foothold(k, foothold):
            leg_keys = jax.random.split(k, self.num_legs)
            return jax.vmap(
                lambda lk, sh: self.leg.reach(
                    lk, foothold.transform(sh.inverse()), samples=samples, **kwargs)
            )(leg_keys, shoulders)

        return jax.vmap(per_foothold)(                            # over footholds -> rows
            jax.random.split(key, footholds.shape[0]), footholds)

    def sample_posture(self, key: jax.Array, body: SE3, support: Support, *,
                       xyz_limits=None, rpy_limits=None, seq: str = "xyz"):
        """Sample a posture around a body pose, with each leg's config drawn.

        The body is sampled first -- a pose near ``body`` within ``xyz_limits`` /
        ``rpy_limits`` (:func:`sample_pose`). The footholds stay fixed in the world,
        so perturbing the body explores which nearby body poses the stance admits;
        planted feet still land on their sites (when reachable).

        Then every leg starts with a free config (:meth:`Leg.sample_free`); the
        ``k`` supported legs overwrite theirs with a config standing on their site
        (``sample_planted``), each site expressed in its leg's base frame.

        Args:
            key: PRNG key.
            body: SE3 body pose to sample around (the centre).
            support: the ``k`` planted legs and their footholds.
            xyz_limits: (3, 2) ``[lo, hi]`` translation bounds, in the body's local
                frame. None means no positional perturbation.
            rpy_limits: (3, 2) ``[lo, hi]`` euler-angle bounds. None means no
                rotational perturbation.
            seq: euler sequence for ``rpy_limits`` (as :meth:`SE3.from_te`).

        Returns:
            ``(valid, posture)`` -- ``valid`` is true when every supported leg found
            a reachable, in-limit config; the posture carries the *sampled* body.
            Free legs never invalidate.
        """
        key_body, key_free, key_plant = jax.random.split(key, 3)
        body = sample_pose(key_body, body, xyz_limits, rpy_limits, seq)
        shoulders = self.shoulders(body)

        # Every leg free by default...
        thetas = jax.vmap(self.leg.sample_free)(
            jax.random.split(key_free, self.num_legs))

        # ...then overwrite the k planted legs.
        contact_sh = shoulders[support.ids]                       # (k,) gather
        def plant(key, shoulder, site):
            local = site.transform(shoulder.inverse())            # world -> leg base
            return self.leg.sample_planted(key, local)

        oks, planted = jax.vmap(plant)(
            jax.random.split(key_plant, support.k), contact_sh, support.sites)
        thetas = thetas.at[support.ids].set(planted)              # scatter

        return jnp.all(oks), Posture(body, thetas)

    def to_mjcf(self, *, name: str = "robot", link_radius: float = 0.015,
                foot_radius: float = 0.02, body_margin: float = 0.0,
                body_half_height: float = 0.05, base_pos=(0.0, 0.0, 0.0),
                weld_feet: bool = True, body_density: float = None,
                leg_density: float = None, foot_density: float = None) -> str:
        """A MuJoCo MJCF (XML) of the robot's kinematics, as a string.

        A direct transcription of the chain: each leg body is ``mounts[i]`` then the
        leg's ``offsets`` (body ``pos``/``quat``), each joint carries the leg's
        ``axes[k]`` and limit ``range``, each link a capsule, each foot a sphere. The
        body is one box spanning the mounts (:meth:`mount_box`). Since a ``Robot`` is
        pure kinematics, this is a *collision/kinematic* model -- geometry and joints,
        with no actuators or damping.

        Mass comes from geom density: each part's density defaults to ``None`` (leave
        it to MuJoCo's own default of 1000), or pass ``body_density`` / ``leg_density``
        / ``foot_density`` for realistic inertia -- which is what makes force
        magnitudes off this model meaningful (see :func:`controlkit.forces.model_from_robot`).

        With ``weld_feet``, each foot gets a ``<site>`` and an inactive
        ``<weld body1="foot{i}">`` equality (cf. ``models/weld0.xml``). MuJoCo cannot
        add equalities to a compiled model, so planting a foot by activating its weld
        needs the constraint declared up front; ``torquescale=0`` makes it a point
        pin (the foot may still pivot).

        Pure (no ``mujoco`` import): returns the XML. Compile it with
        :meth:`to_mujoco`, or ``mujoco.MjModel.from_xml_string`` yourself.

        Args:
            name: model name.
            link_radius: capsule radius for the links.
            foot_radius: sphere radius for the feet.
            body_margin: x/y half-width added to the body box (:meth:`mount_box`).
            body_half_height: z half-extent of the body box.
            base_pos: initial world position of the (free-jointed) base body.
            weld_feet: emit foot sites and inactive foot-to-world weld equalities, so
                a planted foot can be pinned by activating ``weld_foot{i}``.
            body_density: density (kg/m^3) of the body box; None -> MuJoCo default.
            leg_density: density of the link capsules; None -> MuJoCo default.
            foot_density: density of the foot spheres; None -> MuJoCo default.

        Returns:
            The MJCF as a string.
        """
        import numpy as np

        def vec(a):
            return " ".join(f"{float(v):.9g}" for v in np.asarray(a).ravel())

        def dens(d):
            return f' density="{d:.9g}"' if d is not None else ""

        body_d, leg_d, foot_d = dens(body_density), dens(leg_density), dens(foot_density)

        leg = self.leg
        n = leg.num_joints
        axes = np.asarray(leg.axes)                               # (n, 3)
        limits = np.asarray(leg.limits)                           # (n, 2), radians
        off_t = np.asarray(leg.offsets.translation())            # (n, 3)
        off_q = np.asarray(leg.offsets.rotation().wxyz)          # (n, 4) wxyz
        tool_t = np.asarray(leg.tool.translation())              # (3,)
        tool_q = np.asarray(leg.tool.rotation().wxyz)            # (4,)
        # Leg body 0 sits at the mount composed with the leg's first offset.
        b0 = jax.vmap(lambda m: m @ leg.offsets[0])(self.mounts)
        b0_t = np.asarray(b0.translation())                      # (num_legs, 3)
        b0_q = np.asarray(b0.rotation().wxyz)                    # (num_legs, 4)
        # The link on body k runs to the next body's origin (the foot for the last).
        child_t = [off_t[k + 1] if k < n - 1 else tool_t for k in range(n)]

        def leg_xml(i):
            s = []
            for k in range(n):
                if k == 0:
                    s.append(f'<body name="leg{i}_0" pos="{vec(b0_t[i])}" quat="{vec(b0_q[i])}">')
                else:
                    s.append(f'<body name="leg{i}_{k}" pos="{vec(off_t[k])}" quat="{vec(off_q[k])}">')
                s.append(f'  <joint name="leg{i}_j{k}" type="hinge" axis="{vec(axes[k])}" '
                         f'range="{limits[k, 0]:.9g} {limits[k, 1]:.9g}"/>')
                s.append(f'  <geom type="capsule" fromto="0 0 0 {vec(child_t[k])}" size="{link_radius:.9g}"{leg_d}/>')
            s.append(f'<body name="foot{i}" pos="{vec(tool_t)}" quat="{vec(tool_q)}">')
            s.append(f'  <geom name="foot{i}" type="sphere" size="{foot_radius:.9g}"{foot_d}/>')
            if weld_feet:
                s.append(f'  <site name="foot{i}" size="{0.5 * foot_radius:.9g}"/>')
            s.append("</body>")                                   # foot
            s.append("</body>" * n)                               # the n joint bodies
            return "\n".join(s)

        center, half = self.mount_box(margin=body_margin, half_height=body_half_height)
        legs = "\n".join(leg_xml(i) for i in range(self.num_legs))
        equality = ""
        if weld_feet:
            welds = "\n".join(
                f'    <weld name="weld_foot{i}" body1="foot{i}" torquescale="0" active="false"/>'
                for i in range(self.num_legs))
            equality = f'  <equality>\n{welds}\n  </equality>\n'
        return (
            f'<mujoco model="{name}">\n'
            f'  <compiler angle="radian" autolimits="true"/>\n'
            f'  <worldbody>\n'
            f'    <body name="base" pos="{vec(base_pos)}">\n'
            f'      <freejoint name="root"/>\n'
            f'      <geom name="base" type="box" pos="{vec(center)}" size="{vec(half)}"{body_d}/>\n'
            f'{legs}\n'
            f'    </body>\n'
            f'  </worldbody>\n'
            f'{equality}'
            f'</mujoco>\n'
        )

    def to_mujoco(self, **kwargs):
        """Compile :meth:`to_mjcf` into a ``mujoco.MjModel``.

        Lazy-imports ``mujoco`` (the ``mjx`` extra), so it stays optional -- the rest
        of ``kinematics`` is pure JAX. Takes the same keywords as :meth:`to_mjcf`.

        Returns:
            A compiled ``mujoco.MjModel``.
        """
        import mujoco
        return mujoco.MjModel.from_xml_string(self.to_mjcf(**kwargs))


def sample_pose(key: jax.Array, center: SE3, xyz_limits=None, rpy_limits=None,
                seq: str = "xyz") -> SE3:
    """Sample an SE3 pose near ``center``, offset within the given bounds.

    The offset is composed in ``center``'s local frame (``center @ offset``): the
    translation is within ``xyz_limits`` along ``center``'s own axes, the rotation
    within ``rpy_limits`` (euler ``seq``). Symmetric bounds sample *around* the
    centre; ``None`` (or zero-width) bounds leave that part untouched, so with both
    ``None`` this returns ``center`` unchanged.

    Args:
        key: PRNG key.
        center: the pose to sample around.
        xyz_limits: (3, 2) ``[lo, hi]`` translation bounds; None -> no offset.
        rpy_limits: (3, 2) ``[lo, hi]`` euler-angle bounds; None -> no offset.
        seq: euler sequence for ``rpy_limits`` (as :meth:`SE3.from_te`).

    Returns:
        The sampled SE3 pose.
    """
    if xyz_limits is None:
        xyz_limits = jnp.zeros((3, 2))
    if rpy_limits is None:
        rpy_limits = jnp.zeros((3, 2))
    key_t, key_r = jax.random.split(key)
    t = jax.random.uniform(key_t, (3,), minval=xyz_limits[:, 0], maxval=xyz_limits[:, 1])
    e = jax.random.uniform(key_r, (3,), minval=rpy_limits[:, 0], maxval=rpy_limits[:, 1])
    return center @ SE3.from_te(t, e, seq)


def radial_mounts(num_legs: int, radius: float, *, z: float = 0.0) -> SE3:
    """Shoulder poses evenly spaced around a circle, each facing outward.

    Angles are offset by half a step so no leg sits on the +x axis, which keeps the
    layout symmetric about the body's fore-aft axis for even ``num_legs``.

    Args:
        num_legs: number of mounts.
        radius: circumradius of the polygon the mounts form.
        z: height of the mounts in the body frame.

    Returns:
        (num_legs,) SE3 mount poses, +x pointing radially outward.
    """
    angles = jnp.deg2rad(
        jnp.arange(360.0 / num_legs / 2, 360.0, 360.0 / num_legs)
    )
    positions = jnp.stack(
        [radius * jnp.cos(angles), radius * jnp.sin(angles), jnp.full(num_legs, z)],
        axis=-1,
    )
    return SE3.from_te(positions, angles[:, None], "z")
