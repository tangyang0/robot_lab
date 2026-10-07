"""Deployment target limits and more varied rough-terrain exposure."""

from isaaclab.utils import configclass
from .rough_v4_env_cfg import ZsibotZSL1RoughV4EnvCfg


@configclass
class ZsibotZSL1RoughV5EnvCfg(ZsibotZSL1RoughV4EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Exact float32 SDK windows after the deployment's inward 1e-3 margin.
        # JointPositionAction clips processed absolute targets; raw action
        # history remains unchanged, as in policy_deploy.py.
        self.actions.joint_pos.clip = {
            ".*_ABAD_JOINT": (-0.4790000021457672, 0.4790000021457672),
            ".*_HIP_JOINT": (-1.1489999294281006, 2.9690001010894775),
            ".*_KNEE_JOINT": (-2.8990001678466797, -0.6509999632835388),
        }
        terrain = self.scene.terrain.terrain_generator
        terrain.num_cols = 40
        proportions = {
            "flat": 0.20,
            "pyramid_stairs": 0.10,
            "pyramid_stairs_inv": 0.15,
            "boxes": 0.10,
            "random_rough": 0.40,
            "hf_pyramid_slope": 0.025,
            "hf_pyramid_slope_inv": 0.025,
        }
        assert abs(sum(proportions.values()) - 1.0) < 1e-12
        assert set(proportions) == set(terrain.sub_terrains)
        for name, proportion in proportions.items():
            terrain.sub_terrains[name].proportion = proportion
        # The 65,536-environment probe exceeded the inherited patch/stack
        # capacities. These allocations prevent dropped contacts at scale.
        self.sim.physx.gpu_max_rigid_patch_count = 2**20
        self.sim.physx.gpu_collision_stack_size = 2**27
