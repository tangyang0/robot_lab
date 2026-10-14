"""Self-contained ZSL1 rough-v11 PPO config (flattened from the V2->V11 chain)."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlSymmetryCfg

from .agents.rsl_rl_ppo_cfg import ZsibotZSL1RoughPPORunnerCfg
from .rough_v11_symmetry import compute_symmetric_states


@configclass
class ZsibotZSL1RoughV11PPORunnerCfg(ZsibotZSL1RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        # (V2) conservative warm-start settings
        self.experiment_name = "zsibot_zsl1_rough_v11"
        self.run_name = "fresh_fullstack_from_smooth"
        self.max_iterations = 10000
        self.save_interval = 100
        self.algorithm.learning_rate = 3.0e-4
        self.policy.init_noise_std = 0.4
        self.resume = False
        # (V3+V5) exploration and left/right augmentation
        self.algorithm.entropy_coef = 0.003
        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=True,
            data_augmentation_func=compute_symmetric_states,
            mirror_loss_coeff=0.10,
        )
        # (V5) large rollout batch, unchanged per-minibatch raw samples
        self.algorithm.num_mini_batches = 16
