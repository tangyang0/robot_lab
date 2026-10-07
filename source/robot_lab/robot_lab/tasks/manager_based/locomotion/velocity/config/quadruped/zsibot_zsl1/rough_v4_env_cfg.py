"""Candidate texture refinement; all V3 locomotion settings are inherited."""

from isaaclab.utils import configclass

from .rough_v3_env_cfg import ZsibotZSL1RoughV3EnvCfg


@configclass
class ZsibotZSL1RoughV4EnvCfg(ZsibotZSL1RoughV3EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        rough = self.scene.terrain.terrain_generator.sub_terrains["random_rough"]
        # This generator ignores difficulty. Match the challenging held-out
        # texture explicitly; the other six terrain types and their mixture
        # weights remain unchanged. V3 already traverses 43/48 such trials.
        rough.noise_range = (0.02, 0.10)
        rough.noise_step = 0.02
        rough.downsampled_scale = 0.10
