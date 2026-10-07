"""V3 PPO candidate with optional left/right augmentation and mirror loss."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlSymmetryCfg

from .rough_v2_ppo_cfg import ZsibotZSL1RoughV2PPORunnerCfg
from .rough_v3_symmetry import compute_symmetric_states


@configclass
class ZsibotZSL1RoughV3PPORunnerCfg(ZsibotZSL1RoughV2PPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "zsibot_zsl1_rough_v3"
        self.run_name = "straight_lr_symmetry"
        self.algorithm.entropy_coef = 0.003
        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=True,
            data_augmentation_func=compute_symmetric_states,
            mirror_loss_coeff=0.10,
        )


@configclass
class ZsibotZSL1RoughV3NoSymPPORunnerCfg(ZsibotZSL1RoughV3PPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "zsibot_zsl1_rough_v3_no_symmetry"
        self.run_name = "straight_precision"
        self.algorithm.symmetry_cfg = None
