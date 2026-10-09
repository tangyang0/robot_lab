"""Wider walking track, smoother actions, more command-reversal exposure.

Probe findings this addresses (see 实验记录 D7): the V5/V7 walking gait keeps
the feet on a ~0.12 m centerline track (standing is 0.32 m) — the V6/V7 width
reward never fired because its squared hinge (softness 0.05) has no gradient
at a 0.17 m deficit. Action jerk is also marginally higher in V7.
"""

from isaaclab.utils import configclass

from .rough_v7_env_cfg import ZsibotZSL1RoughV7EnvCfg


@configclass
class ZsibotZSL1RoughV8EnvCfg(ZsibotZSL1RoughV7EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # 1) Soften and strengthen the width kernel so the 0.17 m walking
        #    deficit gets useful gradient: exp(-(0.17/0.15)^2) ~ 0.28 instead
        #    of ~0. Keep the 0.30 m target and one-sided saturation.
        self.rewards.feet_height_body.weight = 1.0
        self.rewards.feet_height_body.params.update(target_width=0.30, softness=0.15)
        # 2) Smoothness: jerk was slightly up in V7 vs V5 (walk action delta
        #    0.166 vs 0.158); tripling toward the smooth_ft-proven value.
        self.rewards.action_rate_l2.weight = -0.03
        # 3) Reversal exposure: commands previously resampled every 10 s;
        #    4-6 s makes vx sign flips a routine training event. (The sim
        #    reversal probe shows 0/16 falls already, so this is margin.)
        self.commands.base_velocity.resampling_time_range = (4.0, 6.0)
