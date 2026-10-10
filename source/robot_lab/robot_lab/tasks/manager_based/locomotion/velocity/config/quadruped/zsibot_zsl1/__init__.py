# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="RobotLab-Isaac-Velocity-Flat-Zsibot-ZSL1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:ZsibotZSL1FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ZsibotZSL1FlatPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Fast-Flat-Zsibot-ZSL1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fast_flat_env_cfg:ZsibotZSL1FastFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ZsibotZSL1FastFlatPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-Zsibot-ZSL1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:ZsibotZSL1RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ZsibotZSL1RoughPPORunnerCfg",
    },
)


gym.register(
    id="RobotLab-Isaac-Velocity-Rough-V2-Zsibot-ZSL1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_v2_env_cfg:ZsibotZSL1RoughV2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.rough_v2_ppo_cfg:ZsibotZSL1RoughV2PPORunnerCfg",
    },
)

# Independent straight-command/symmetry continuation of the rough-v2 task.
from .rough_v3_registration import register_rough_v3_tasks
register_rough_v3_tasks()

# Independent refinement for stronger random roughness.
from .rough_v4_registration import register_rough_v4_task
register_rough_v4_task()

# Independent deployment-limit and rough-exposure refinement.
from .rough_v5_registration import register_rough_v5_task
register_rough_v5_task()

# Playback with the real deployment controller target limits.
from .rough_v4_deploy_registration import register_rough_v4_deploy_task
register_rough_v4_deploy_task()

# Independent stance-width and lateral-push refinement.
from .rough_v6_registration import register_rough_v6_task
register_rough_v6_task()

# Softer-push continuation targeting both rough gains and flat 0.8 straightness.
from .rough_v7_registration import register_rough_v7_task
register_rough_v7_task()

# Wide-track/smoothness/reversal continuation of V7.
from .rough_v8_registration import register_rough_v8_task
register_rough_v8_task()

# Orientation-fix continuation of V8.
from .rough_v9_registration import register_rough_v9_task
register_rough_v9_task()
