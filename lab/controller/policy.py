"""CPU MPPI regulator: drive the leg servos so the pinned body reaches a target.

With the four feet pinned (connect equalities), the body moves only through the
16 leg-joint servos, so the planner absorbs the inverse kinematics -- it searches
servo targets and scores each rollout by how close the body's free-joint pose
lands to the target pose. No balance term is needed: the pins hold the body up
whatever the joints do.

Rollouts run on ``mujoco.rollout`` (threaded C++), so there is no JIT wait and it
stays plain-CPU alongside the viewer. Everything else is numpy MPPI: sample noisy
control sequences around the running plan, roll out, softmax-weight by cost,
update the plan, apply its head, shift.
"""
from __future__ import annotations

import os

import numpy as np
import mujoco
from mujoco import rollout

from .config import Cfg

_FULLPHYS = mujoco.mjtState.mjSTATE_FULLPHYSICS


class MPPI:
    """Sampling-based receding-horizon regulator to a body pose.

    Args:
        model: the actuated scene model with the feet pinned (connect).
        cfg: MPPI knobs (horizon, samples, sigma, weights, ...).
        rest_ctrl: (nu,) servo targets that hold the rest posture; the initial
            plan and the exploration anchor.
    """

    def __init__(self, model, cfg: Cfg, rest_ctrl):
        self.model = model
        self.cfg = cfg
        self.nu = model.nu
        self.nq = model.nq
        self.lo = model.actuator_ctrlrange[:, 0].copy()
        self.hi = model.actuator_ctrlrange[:, 1].copy()
        self.rest = np.asarray(rest_ctrl, float)
        self.U = np.tile(self.rest, (cfg.horizon, 1))                      # (T, nu)

        # one MjData per worker thread for the threaded rollout. The foot pins are
        # active on the model (eq_active0, set by pin_all_feet), so every rollout
        # inherits them on reset -- no per-data patching needed.
        nthread = min(cfg.samples, max(1, os.cpu_count() or 1))
        self.data = [mujoco.MjData(model) for _ in range(nthread)]
        self.nstate = mujoco.mj_stateSize(model, _FULLPHYS)

    def reset(self, rest_ctrl):
        """Reset the running plan to hold ``rest_ctrl``."""
        self.U[:] = np.asarray(rest_ctrl, float)

    def __call__(self, state0, target_pos, target_quat):
        """Plan from ``state0`` toward the target pose; return the head control.

        Args:
            state0: (nstate,) FULLPHYSICS state of the live sim.
            target_pos: (3,) desired body position (world).
            target_quat: (4,) desired body orientation, wxyz.

        Returns:
            (nu,) servo targets to apply for the next control step.
        """
        cfg, T, nu = self.cfg, self.cfg.horizon, self.nu
        N, decim = cfg.samples, cfg.decimation

        eps = cfg.noise_sigma * np.random.randn(N, T, nu)
        cand = np.clip(self.U[None] + eps, self.lo, self.hi)              # (N, T, nu)
        ctrl = np.repeat(cand, decim, axis=1)                            # (N, T*decim, nu)
        init = np.tile(np.asarray(state0, float), (N, 1))                # (N, nstate)

        state, _ = rollout.rollout(self.model, self.data, init, ctrl)    # (N, nstep, nstate)
        # FULLPHYSICS state is [time, qpos, qvel, act]; slice qpos then qvel out.
        qpos = state[:, :, 1:1 + self.nq]
        qvel = state[:, :, 1 + self.nq:1 + self.nq + self.model.nv]
        pos, quat = qpos[:, :, 0:3], qpos[:, :, 3:7]

        cost = self._cost(pos, quat, qvel, target_pos, target_quat, cand)  # (N,)
        # a blown-up rollout scores nan/+inf; map it to a large finite cost so it
        # gets ~zero softmax weight instead of poisoning the weighted average.
        cost = np.nan_to_num(cost, nan=1e12, posinf=1e12)
        w = _softmax(-(cost - cost.min()) / cfg.lam)                     # (N,)
        self.U = np.clip(np.einsum("n,ntu->tu", w, cand), self.lo, self.hi)

        u0 = self.U[0].copy()
        # shift the plan one step for next tick's warm start
        self.U[:-1] = self.U[1:]
        return u0

    def _cost(self, pos, quat, qvel, tpos, tquat, cand):
        cfg = self.cfg
        e_pos = ((pos - tpos) ** 2).sum(-1)                              # (N, nstep)
        dot = (quat * tquat).sum(-1)                                     # (N, nstep)
        e_ori = 1.0 - dot ** 2                                           # in [0, 1]
        e_vel = (qvel ** 2).sum(-1)                                      # (N, nstep)
        c = (cfg.w_pos * e_pos + cfg.w_ori * e_ori + cfg.w_vel * e_vel).sum(1)
        if cfg.w_reg:
            c = c + cfg.w_reg * ((cand - self.rest) ** 2).sum((1, 2))
        if cfg.w_ctrl:
            c = c + cfg.w_ctrl * (np.diff(cand, axis=1) ** 2).sum((1, 2))
        return c


def _softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()
