"""Stance/push continuation of V5 with unchanged PPO and symmetry settings."""

from isaaclab.utils import configclass
from .rough_v5_ppo_cfg import ZsibotZSL1RoughV5PPORunnerCfg


@configclass
class ZsibotZSL1RoughV6PPORunnerCfg(ZsibotZSL1RoughV5PPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "zsibot_zsl1_rough_v6"
        self.run_name = "stance_push_from_v5_1600"
