"""Reward terms used only by the ZSL1 rough-v6 task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_width(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_width: float = 0.30,
    softness: float = 0.05,
) -> torch.Tensor:
    """One-sided stance-width reward in the body frame.

    Ordered feet: [FR, FL, RR, RL]. Width is the mean lateral distance of the
    front and rear foot pairs expressed in the body frame, so world yaw does
    not matter. Returns 1.0 whenever both pairs are at least ``target_width``
    apart; narrower (adducted) stances decay with a squared-hinge kernel. The
    URDF default pose puts the feet 0.319 m apart, so the term is saturated
    at the natural stance and only acts against crossing the legs inward.
    """
    robot = env.scene[asset_cfg.name]
    feet_w = robot.data.body_pos_w[:, asset_cfg.body_ids, :] - robot.data.root_pos_w[:, None, :]
    # world -> body: rotate by the conjugate of the root quaternion (w,x,y,z)
    qw = robot.data.root_quat_w[:, 0:1].unsqueeze(-1)   # (N,1,1)
    qvec = -robot.data.root_quat_w[:, 1:4].unsqueeze(1)  # (N,1,3), broadcasts per foot
    t = 2.0 * torch.cross(qvec, feet_w, dim=-1)
    feet_b = feet_w + qw * t + torch.cross(qvec, t, dim=-1)
    y = feet_b[..., 1]
    width = 0.5 * (torch.abs(y[:, 1] - y[:, 0]) + torch.abs(y[:, 3] - y[:, 2]))
    deficit = torch.clamp(target_width - width, min=0.0)
    return torch.exp(-torch.square(deficit / softness))
