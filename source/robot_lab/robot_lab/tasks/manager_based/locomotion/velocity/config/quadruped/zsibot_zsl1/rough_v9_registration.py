"""Register a separate V9 task without replacing earlier experiments."""

import gymnasium as gym


def register_rough_v9_task():
    package = __package__
    gym.register(
        id="RobotLab-Isaac-Velocity-Rough-V9-Zsibot-ZSL1-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{package}.rough_v9_env_cfg:ZsibotZSL1RoughV9EnvCfg",
            "rsl_rl_cfg_entry_point": f"{package}.rough_v9_ppo_cfg:ZsibotZSL1RoughV9PPORunnerCfg",
        },
    )
