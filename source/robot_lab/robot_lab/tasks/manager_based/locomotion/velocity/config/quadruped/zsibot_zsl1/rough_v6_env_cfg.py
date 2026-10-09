"""Stance-width and lateral-push robustness refinement of the V5 task.

The observation/action interface, deployment joint limits, terrain mixture
and every other reward weight are identical to V5. Only training incentives
change, targeting two reported real-robot defects: an adducted (narrow)
stance with the feet nearly touching, and easy lateral tipping.
"""

import isaaclab.envs.mdp as isaac_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .rough_v5_env_cfg import ZsibotZSL1RoughV5EnvCfg
from .rough_v6_mdp import feet_width


@configclass
class ZsibotZSL1RoughV6EnvCfg(ZsibotZSL1RoughV5EnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # 1) Stance width. Reuse the declared-but-disabled foot-geometry term
        # slot (V2 zeroed feet_height_body) so the reward manager sees it.
        self.rewards.feet_height_body = RewTerm(
            func=feet_width,
            weight=0.5,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=["FR_FOOT_LINK", "FL_FOOT_LINK", "RR_FOOT_LINK", "RL_FOOT_LINK"],
                    preserve_order=True,
                ),
                "target_width": 0.30,
                "softness": 0.05,
            },
        )

        # 2) Do not pull ABAD toward zero while the width reward pushes the
        # stance out; HIP/KNEE default-pose regularization is unchanged.
        self.rewards.joint_pos_penalty.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_HIP_JOINT", ".*_KNEE_JOINT"]
        )

        # 3) Lateral robustness. Pushes were disabled in V2; without them the
        # policy never trained lateral balance recovery, matching the real
        # robot tipping over easily. Same magnitude as the legacy rough task.
        self.events.randomize_push_robot = EventTerm(
            func=isaac_mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(10.0, 15.0),
            params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
        )
