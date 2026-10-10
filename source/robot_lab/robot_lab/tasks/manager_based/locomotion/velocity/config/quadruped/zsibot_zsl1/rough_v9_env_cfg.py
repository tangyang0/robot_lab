"""Stronger orientation penalty: kill the direction-coupled base lean.

Pitch probe (see 实验记录 D8): V5-era models lean ~2 deg toward the travel
direction (forward = nose-down, backward = nose-up), so a direction switch
must swing the lean across zero and tips the real robot. V8 halved the
steady swing; V9 triples flat_orientation_l2 (-0.2 -> -0.6) to remove it.
"""

from isaaclab.utils import configclass

from .rough_v8_env_cfg import ZsibotZSL1RoughV8EnvCfg


@configclass
class ZsibotZSL1RoughV9EnvCfg(ZsibotZSL1RoughV8EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.flat_orientation_l2.weight = -0.6
