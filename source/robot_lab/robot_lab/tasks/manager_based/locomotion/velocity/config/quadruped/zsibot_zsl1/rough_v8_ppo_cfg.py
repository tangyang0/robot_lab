"""Stance/push continuation of V5 with unchanged PPO and symmetry settings."""

from isaaclab.utils import configclass
from .rough_v5_ppo_cfg import ZsibotZSL1RoughV5PPORunnerCfg


@configclass
class ZsibotZSL1RoughV8PPORunnerCfg(ZsibotZSL1RoughV5PPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "zsibot_zsl1_rough_v8"
        self.run_name = "widetrack_from_v7_2100"
