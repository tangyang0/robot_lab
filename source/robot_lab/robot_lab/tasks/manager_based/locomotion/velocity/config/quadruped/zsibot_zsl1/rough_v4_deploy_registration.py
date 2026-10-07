"""Register the optional refinement without modifying existing task IDs."""

import gymnasium as gym


def register_rough_v4_deploy_task():
    package = __package__
    gym.register(
        id="RobotLab-Isaac-Velocity-Rough-V4-Deploy-Zsibot-ZSL1-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{package}.rough_v4_deploy_env_cfg:ZsibotZSL1RoughV4DeployEnvCfg",
            "rsl_rl_cfg_entry_point": f"{package}.rough_v4_ppo_cfg:ZsibotZSL1RoughV4PPORunnerCfg",
        },
    )
