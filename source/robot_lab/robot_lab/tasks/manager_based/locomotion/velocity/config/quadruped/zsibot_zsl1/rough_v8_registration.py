"""Register a separate V8 task without replacing earlier experiments."""

import gymnasium as gym


def register_rough_v8_task():
    package = __package__
    gym.register(
        id="RobotLab-Isaac-Velocity-Rough-V8-Zsibot-ZSL1-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{package}.rough_v8_env_cfg:ZsibotZSL1RoughV8EnvCfg",
            "rsl_rl_cfg_entry_point": f"{package}.rough_v8_ppo_cfg:ZsibotZSL1RoughV8PPORunnerCfg",
        },
    )
