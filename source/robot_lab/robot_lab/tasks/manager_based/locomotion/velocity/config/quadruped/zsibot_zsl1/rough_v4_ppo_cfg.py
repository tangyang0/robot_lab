"""Separate experiment for the optional rough-texture continuation."""

from isaaclab.utils import configclass

from .rough_v3_ppo_cfg import ZsibotZSL1RoughV3PPORunnerCfg


@configclass
class ZsibotZSL1RoughV4PPORunnerCfg(ZsibotZSL1RoughV3PPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "zsibot_zsl1_rough_v4"
        self.run_name = "rough_texture_refinement"
