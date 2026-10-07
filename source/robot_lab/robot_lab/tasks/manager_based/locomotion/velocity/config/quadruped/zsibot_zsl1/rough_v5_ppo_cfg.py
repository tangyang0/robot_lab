"""Large rollout batch with unchanged per-minibatch raw sample count."""

from isaaclab.utils import configclass
from .rough_v4_ppo_cfg import ZsibotZSL1RoughV4PPORunnerCfg
from .rough_v5_symmetry import compute_symmetric_states


@configclass
class ZsibotZSL1RoughV5PPORunnerCfg(ZsibotZSL1RoughV4PPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "zsibot_zsl1_rough_v5"
        self.run_name = "deployment_limits_rough_32k"
        self.algorithm.num_mini_batches = 16
        self.algorithm.symmetry_cfg.data_augmentation_func = compute_symmetric_states
