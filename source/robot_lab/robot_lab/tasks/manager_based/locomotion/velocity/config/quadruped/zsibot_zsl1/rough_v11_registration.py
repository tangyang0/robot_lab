"""Register a separate V11 task without replacing earlier experiments."""

import gymnasium as gym


def register_rough_v11_task():
    package = __package__
    gym.register(
        id="RobotLab-Isaac-Velocity-Rough-V11-Zsibot-ZSL1-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{package}.rough_v11_env_cfg:ZsibotZSL1RoughV11EnvCfg",
            "rsl_rl_cfg_entry_point": f"{package}.rough_v11_ppo_cfg:ZsibotZSL1RoughV11PPORunnerCfg",
        },
    )
