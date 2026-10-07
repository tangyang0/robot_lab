"""V4 playback with the exact target limits used by the deployment controller."""

from isaaclab.utils import configclass
from .rough_v4_env_cfg import ZsibotZSL1RoughV4EnvCfg


@configclass
class ZsibotZSL1RoughV4DeployEnvCfg(ZsibotZSL1RoughV4EnvCfg):
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
