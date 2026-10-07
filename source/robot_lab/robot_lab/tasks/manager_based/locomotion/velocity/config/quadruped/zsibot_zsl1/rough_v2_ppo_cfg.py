"""Independent PPO experiment with conservative actor warm-start settings."""

from isaaclab.utils import configclass

from .agents.rsl_rl_ppo_cfg import ZsibotZSL1RoughPPORunnerCfg


@configclass
class ZsibotZSL1RoughV2PPORunnerCfg(ZsibotZSL1RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "zsibot_zsl1_rough_v2"
        self.run_name = "upright_terrain_curriculum"
        self.max_iterations = 10000
        self.save_interval = 100
        self.algorithm.learning_rate = 3.0e-4
        self.policy.init_noise_std = 0.4
        # Warm-starting only the successful flat actor is handled by the
        # training entry point; normal resume must not load the failed rough.
        self.resume = False
