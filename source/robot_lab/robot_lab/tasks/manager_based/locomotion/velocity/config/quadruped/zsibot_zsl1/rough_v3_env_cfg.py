"""Independent v3 candidate: v2 terrain/feet plus straight-command precision."""

import copy

from isaaclab.utils import configclass

from .rough_v2_env_cfg import ZsibotZSL1RoughV2EnvCfg
from .rough_v3_mdp import StraightMixtureVelocityCommandCfg


@configclass
class ZsibotZSL1RoughV3EnvCfg(ZsibotZSL1RoughV2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        old = self.commands.base_velocity
        command = StraightMixtureVelocityCommandCfg()
        for key, value in vars(old).items():
            if not key.startswith("_") and key != "class_type":
                setattr(command, key, copy.deepcopy(value))
        command.heading_command = False
        command.rel_heading_envs = 0.0
        command.ranges.heading = None
        command.rel_standing_envs = 0.05
        command.straight_fraction = 0.50
        command.straight_vx_range = (0.20, 0.90)
        self.commands.base_velocity = command

        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.35
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.25
        # Weights stay 3.0 / 1.5. Terrain, reset, observation/action schemas,
        # actuator settings and every foot reward remain inherited from v2.
