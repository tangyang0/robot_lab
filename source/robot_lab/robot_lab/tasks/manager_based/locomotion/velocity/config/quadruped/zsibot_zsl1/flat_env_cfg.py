# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from .rough_env_cfg import ZsibotZSL1RoughEnvCfg


@configclass
class ZsibotZSL1FlatEnvCfg(ZsibotZSL1RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override rewards
        self.rewards.base_height_l2.params["sensor_cfg"] = None
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # The flat policy is optimized for a nominal, smooth walking gait.  The
        # rough-terrain random pushes and actuator-gain randomization are useful
        # for robustness, but they encourage abrupt corrective actions on a
        # uniform plane.
        self.events.randomize_apply_external_force_torque = None
        self.events.randomize_push_robot = None
        self.events.randomize_actuator_gains = None

        # Increase smoothness shaping for the flat-only policy.
        self.rewards.ang_vel_xy_l2.weight = -0.10
        self.rewards.joint_acc_l2.weight = -1.0e-6
        self.rewards.action_rate_l2.weight = -0.05
        self.rewards.feet_slide.weight = -0.20

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "ZsibotZSL1FlatEnvCfg":
            self.disable_zero_weight_rewards()
