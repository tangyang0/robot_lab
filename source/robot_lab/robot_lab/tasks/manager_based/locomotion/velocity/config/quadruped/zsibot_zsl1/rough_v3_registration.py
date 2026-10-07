"""Import and call from the existing ZSL1 package only after choosing v3."""


def register_rough_v3_tasks():
    import gymnasium as gym

    package = __package__
    for suffix, runner in (
        ("", "ZsibotZSL1RoughV3PPORunnerCfg"),
        ("-NoSym", "ZsibotZSL1RoughV3NoSymPPORunnerCfg"),
    ):
        gym.register(
            id=f"RobotLab-Isaac-Velocity-Rough-V3{suffix}-Zsibot-ZSL1-v0",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{package}.rough_v3_env_cfg:ZsibotZSL1RoughV3EnvCfg",
                "rsl_rl_cfg_entry_point": f"{package}.rough_v3_ppo_cfg:{runner}",
            },
        )
