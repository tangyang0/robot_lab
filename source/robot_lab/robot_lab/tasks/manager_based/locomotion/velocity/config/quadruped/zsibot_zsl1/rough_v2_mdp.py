"""Reward terms used only by the ZSL1 rough-v2 task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_slide_world(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize horizontal foot motion relative to the static world ground.

    A planted foot has zero world velocity even while the base moves. Unlike
    the legacy reward, this deliberately does not subtract base velocity.
    Matching ordered body names in both SceneEntityCfg objects is required.
    The short contact history prevents a one-step contact flicker from
    immediately removing the slip penalty.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    robot = env.scene[asset_cfg.name]
    contact = (
        sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .amax(dim=1)
        > contact_threshold
    )
    horizontal_speed = robot.data.body_lin_vel_w[:, asset_cfg.body_ids, :2].norm(dim=-1)
    return torch.sum(horizontal_speed * contact, dim=1)
