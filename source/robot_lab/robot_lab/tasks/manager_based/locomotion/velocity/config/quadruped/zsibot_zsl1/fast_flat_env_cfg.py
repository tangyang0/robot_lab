# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""High-speed flat-ground configuration for the ZSL1."""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

from robot_lab.tasks.manager_based.locomotion.velocity import mdp

from .flat_env_cfg import ZsibotZSL1FlatEnvCfg


@configclass
class ZsibotZSL1FastFlatEnvCfg(ZsibotZSL1FlatEnvCfg):
    """Flat-ground policy focused on faster forward locomotion.

    The command curriculum starts at the previous policy's 1.0 m/s range and
    expands to 2.0 m/s only when velocity tracking is already good.  This
    keeps the fast policy from immediately collapsing when resumed from the
    smooth policy checkpoint.
    """

    def __post_init__(self):
        super().__post_init__()

        # The final curriculum range is +/-2.0 m/s.  The curriculum starts at
        # 50% of this range ( +/-1.0 m/s ) and grows by 0.1 m/s per successful
        # episode until it reaches the final limit.
        self.commands.base_velocity.ranges.lin_vel_x = (-2.0, 2.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.curriculum.command_levels_lin_vel = CurrTerm(
            func=mdp.command_levels_lin_vel,
            params={
                "reward_term_name": "track_lin_vel_xy_exp",
                "range_multiplier": (0.5, 1.0),
            },
        )
        self.curriculum.command_levels_ang_vel = None

        # Keep smoothness shaping, but reduce the penalties relative to the
        # smooth-only policy so they do not suppress a faster gait.
        self.rewards.ang_vel_xy_l2.weight = -0.07
        self.rewards.joint_acc_l2.weight = -5.0e-7
        self.rewards.action_rate_l2.weight = -0.025
        self.rewards.feet_slide.weight = -0.12
        self.rewards.track_lin_vel_xy_exp.weight = 4.0

        # This subclass does not match the parent's class-name guard.
        self.disable_zero_weight_rewards()
